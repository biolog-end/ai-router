import base64
import json
import struct
import unittest
from types import SimpleNamespace
from unittest import mock

from helpers import Clients, ai, daily_quota, gemini_error, reset_dirs, router

import gemini_budget as gb
from ai_router import keys, tts, voices


def pcm(samples=2400):
    return struct.pack(f'<{samples}h', *([1000] * samples))


def audio_payload(data=None, mime='audio/l16;codec=pcm;rate=24000', finish='STOP'):
    data = pcm() if data is None else data
    return {'candidates': [{'content': {'parts': [{'inlineData': {'mimeType': mime,
                                                                  'data': base64.b64encode(data).decode()}}]},
                            'finishReason': finish}],
            'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 50, 'totalTokenCount': 60}}


class FakeApi:
    """Stands in for client._api_client; script items are payload dicts or exceptions."""

    def __init__(self, script, stream_script=None):
        self.script, self.stream_script, self.calls = list(script), stream_script, []

    def _next(self):
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def request(self, method, path, body):
        self.calls.append((method, path, json.loads(json.dumps(body))))
        return SimpleNamespace(body=json.dumps(self._next()))

    def request_streamed(self, method, path, body):
        self.calls.append((method, path, json.loads(json.dumps(body))))
        for chunk in self.stream_script:
            if isinstance(chunk, BaseException):
                raise chunk
            yield SimpleNamespace(body=json.dumps(chunk))


def fake(script, stream_script=None):
    return SimpleNamespace(_api_client=FakeApi(script, stream_script))


class SpeakTests(unittest.TestCase):
    def setUp(self):
        reset_dirs()
        self.k1 = keys.add('gemini', 'AIza-k1', label='Первый')
        self.k2 = keys.add('gemini', 'AIza-k2', label='Второй')

    def run_with(self, clients, fn=ai.speak, **kwargs):
        with mock.patch.object(router, 'client_for', clients):
            result = fn(**kwargs)
            # speak_stream is a generator: it must run while the fake client is in place.
            return list(result) if fn is ai.speak_stream else result

    def body(self, client, index=0):
        return client._api_client.calls[index][2]

    def test_modern_request_shape_and_wav_result(self):
        client = fake([audio_payload()])
        result = self.run_with(Clients(**{'AIza-k1': client, 'AIza-k2': client}),
                               text='Привет! <laugh> Как дела?', voice='kore', style='cheerful')
        method, path, body = client._api_client.calls[0]
        self.assertEqual(path, 'models/gemini-3.8-flash-tts:generateContent')
        part = body['contents'][0]['parts'][0]
        self.assertEqual(part, {'text': 'Привет! <laugh> Как дела?', 'speech_metadata': {'style': 'cheerful'}})
        self.assertEqual(body['generationConfig']['speechConfig'], {'voiceConfig': {'voice': 'Kore'}})
        self.assertNotIn('responseFormat', body['generationConfig'])
        self.assertEqual(result.audio[:4], b'RIFF')
        self.assertEqual(result.duration_s, 0.1)
        self.assertEqual(result.voices, {'': 'Kore'})
        self.assertIsNone(result.data)
        self.assertNotIn('audio', result.to_dict())
        used = sum(row['requests_today'] for key in ('AIza-k1', 'AIza-k2')
                   for row in gb.status(key) if row['model'] == 'gemini-3.8-flash-tts')
        self.assertEqual(used, 1)

    def test_legacy_square_tags_become_style_segments_on_38(self):
        client = fake([audio_payload()])
        self.run_with(Clients(**{'AIza-k1': client, 'AIza-k2': client}), voice='Puck', style='fast',
                      text='[excitement] Мы выиграли! [whispers] Тихо. [confusion] А где билет?')
        parts = self.body(client)['contents'][0]['parts']
        self.assertEqual(parts, [
            {'text': 'Мы выиграли! <whispers> Тихо.', 'speech_metadata': {'style': 'fast, excitement'}},
            {'text': 'А где билет?', 'speech_metadata': {'style': 'fast, confusion'}}])

    def test_chain_falls_back_to_legacy_with_translated_tags(self):
        modern = fake([gemini_error(429, daily_quota())])
        legacy = fake([audio_payload()])

        class ByModel(FakeApi):
            def request(self, method, path, body):
                target = modern if '3.8' in path else legacy
                return target._api_client.request(method, path, body)

        client = SimpleNamespace(_api_client=ByModel([]))
        events = []
        result = self.run_with(Clients(**{'AIza-k1': client, 'AIza-k2': client}),
                               text='<whispers> Тише... <laugh> шучу! |ага|', voice='Charon', style='scared',
                               chain=['gemini-3.8-flash-tts', 'gemini-3.1-flash-tts-preview'], on_event=events.append)
        self.assertEqual(result.model, 'gemini-3.1-flash-tts-preview')
        self.assertTrue(result.fallback_used)
        body = legacy._api_client.calls[0][2]
        self.assertEqual(body['contents'][0]['parts'][0]['text'], '[scared, whispers] Тише... [laughs] шучу!')
        self.assertEqual(body['generationConfig']['speechConfig'],
                         {'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': 'Charon'}}})
        self.assertTrue(any('подхваты' in note for note in result.notes))
        self.assertTrue(any(e['type'] == 'next_model' for e in events))

    def test_legacy_substitutes_custom_voice_by_gender(self):
        voices.remember('voice_ded', self.k2['id'], gender='male', name='Дед')
        client = fake([audio_payload()])
        result = self.run_with(Clients(**{'AIza-k1': client, 'AIza-k2': client}), text='Жили-были',
                               voice='voice_ded', model='gemini-3.1-flash-tts-preview')
        voice = result.voices['']
        self.assertEqual(voices.PREBUILT[voice][0], 'male')
        self.assertTrue(any('voice_ded' in note and voice in note for note in result.notes))

    def test_custom_voice_is_pinned_to_its_key(self):
        voices.remember('voice_ded', self.k2['id'], gender='male')
        one, two = fake([audio_payload()]), fake([audio_payload()])
        result = self.run_with(Clients(**{'AIza-k1': one, 'AIza-k2': two}), text='Привет', voice='voice_ded')
        self.assertEqual(result.key_id, self.k2['id'])
        self.assertEqual(one._api_client.calls, [])
        self.assertEqual(self.body(two)['generationConfig']['speechConfig'], {'voiceConfig': {'voice': 'voice_ded'}})

    def test_unknown_custom_voice_walks_keys_and_learns_owner(self):
        missing = gemini_error(404, message='Voice voice_x not found.', status='NOT_FOUND')
        one, two = fake([missing]), fake([audio_payload()])
        result = self.run_with(Clients(**{'AIza-k1': one, 'AIza-k2': two}), text='Привет', voice='voice_x')
        self.assertEqual(result.key_id, self.k2['id'])
        self.assertEqual(voices.owner('voice_x'), self.k2['id'])

    def test_dialogue_shapes(self):
        client = fake([audio_payload()])
        turns = [{'speaker': 'Anya', 'text': 'Вставай!', 'style': 'excited'},
                 {'speaker': 'Boris', 'text': '<yawn> Шесть утра...'}]
        self.run_with(Clients(**{'AIza-k1': client, 'AIza-k2': client}), turns=turns,
                      speakers={'Anya': 'Kore', 'Boris': 'charon'})
        body = self.body(client)
        self.assertEqual(body['contents'][0]['parts'][1], {'text': '<yawn> Шесть утра...',
                                                           'speech_metadata': {'speaker': 'Boris'}})
        self.assertEqual(body['generationConfig']['speechConfig']['multiSpeakerVoiceConfig']['speakerVoiceConfigs'][1],
                         {'speaker': 'Boris', 'voiceConfig': {'voice': 'Charon'}})
        legacy = fake([audio_payload()])
        self.run_with(Clients(**{'AIza-k1': legacy, 'AIza-k2': legacy}), turns=turns,
                      speakers={'Anya': 'Kore', 'Boris': 'Charon'}, model='gemini-2.5-flash-preview-tts')
        self.assertEqual(self.body(legacy)['contents'][0]['parts'][0]['text'],
                         'Anya: [excited] Вставай!\nBoris: [yawns] Шесть утра...')

    def test_three_speakers_are_synthesised_per_turn(self):
        client = fake([audio_payload(pcm(240))])
        result = self.run_with(Clients(**{'AIza-k1': client, 'AIza-k2': client}),
                               turns=[{'speaker': 'A', 'text': 'раз'}, {'speaker': 'B', 'text': 'два'},
                                      {'speaker': 'C', 'text': 'три'}],
                               speakers={'A': 'Kore', 'B': 'Puck', 'C': 'Leda'})
        self.assertEqual(len(client._api_client.calls), 3)
        gap = int(tts.SAMPLE_RATE * tts.TURN_GAP_S) * 2
        self.assertEqual(len(result.pcm), 3 * 480 + 2 * gap)
        self.assertEqual(result.voices, {'A': 'Kore', 'B': 'Puck', 'C': 'Leda'})

    def test_formats(self):
        client = fake([audio_payload(b'\xff' * 800, mime='audio/mulaw;rate=8000')])
        result = self.run_with(Clients(**{'AIza-k1': client, 'AIza-k2': client}), text='Алло', format='mulaw',
                               sample_rate=8000, chain=['gemini-3.1-flash-tts-preview', 'gemini-3.8-flash-tts'])
        self.assertEqual(result.model, 'gemini-3.8-flash-tts')
        self.assertEqual(self.body(client)['generationConfig']['responseFormat'],
                         {'audio': {'mimeType': 'AUDIO_MULAW', 'sampleRate': 8000}})
        self.assertEqual(result.duration_s, 0.1)
        with self.assertRaises(ValueError):
            ai.speak('x', format='mulaw', model='gemini-2.5-flash-preview-tts')
        client = fake([audio_payload()])
        result = self.run_with(Clients(**{'AIza-k1': client, 'AIza-k2': client}), text='x', format='pcm')
        self.assertEqual(result.audio, pcm())
        self.assertEqual(result.mime_type, 'audio/l16;rate=24000')

    def test_empty_audio_explains_finish_reason(self):
        client = fake([{'candidates': [{'finishReason': 'OTHER'}]}])
        with self.assertRaises(ai.EmptyResponseError) as caught:
            self.run_with(Clients(**{'AIza-k1': client, 'AIza-k2': client}), text='[шёпотом]',
                          model='gemini-3.1-flash-tts-preview')
        self.assertIn('OTHER', caught.exception.message)

    def test_safety_finish(self):
        client = fake([{'candidates': [{'finishReason': 'SAFETY'}]}])
        with self.assertRaises(ai.SafetyError):
            self.run_with(Clients(**{'AIza-k1': client, 'AIza-k2': client}), text='x')

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            ai.speak()
        with self.assertRaises(ValueError):
            ai.speak(turns=[{'speaker': 'Anya', 'text': 'x'}])
        with self.assertRaises(ValueError):
            ai.speak(turns=[{'speaker': 'Ann a', 'text': 'x'}], speakers={'Ann a': 'Kore'})
        with self.assertRaises(ValueError):
            ai.speak('x', format='mp3')

    def test_stream_yields_pcm_and_final_wav(self):
        chunks = [audio_payload(pcm(100)), audio_payload(pcm(50))]
        client = fake([], stream_script=chunks)
        events = self.run_with(Clients(**{'AIza-k1': client, 'AIza-k2': client}), fn=ai.speak_stream,
                               text='Привет', voice='Kore')
        deltas = [e for e in events if e['type'] == 'delta']
        self.assertEqual([len(d['audio']) for d in deltas], [200, 100])
        self.assertIn('streamGenerateContent?alt=sse', client._api_client.calls[0][1])
        result = events[-1]['result']
        self.assertEqual(result.audio[:4], b'RIFF')
        self.assertEqual(len(result.pcm), 300)

    def test_stream_on_25_is_one_chunk(self):
        client = fake([audio_payload(pcm(100))])
        events = self.run_with(Clients(**{'AIza-k1': client, 'AIza-k2': client}), fn=ai.speak_stream,
                               text='Привет', model='gemini-2.5-flash-preview-tts', format='pcm')
        self.assertEqual(len([e for e in events if e['type'] == 'delta']), 1)
        self.assertTrue(client._api_client.calls[0][1].endswith(':generateContent'))
        self.assertEqual(events[-1]['result'].audio, pcm(100))

    def test_extra_is_merged(self):
        client = fake([audio_payload()])
        self.run_with(Clients(**{'AIza-k1': client, 'AIza-k2': client}), text='x',
                      extra={'generationConfig': {'temperature': 1.5}})
        config = self.body(client)['generationConfig']
        self.assertEqual(config['temperature'], 1.5)
        self.assertIn('speechConfig', config)


class VoicesTests(unittest.TestCase):
    def setUp(self):
        reset_dirs()
        self.k1 = keys.add('gemini', 'AIza-k1', label='Первый')

    def test_design_registers_owner_and_returns_preview(self):
        sample = base64.b64encode(tts.wav(pcm(10))).decode()
        client = fake([{'id': 'voice_abc', 'type': 'prompted', 'display_name': 'Дед', 'gender': 'male',
                        'language_code': 'ru-RU', 'sample_audio': {'mime_type': 'audio/wav', 'data': sample}}])
        with mock.patch.object(router, 'client_for', Clients(**{'AIza-k1': client})):
            voice = ai.voices.design('хриплый дед', name='Дед', gender='male', language='ru-RU')
        method, path, body = client._api_client.calls[0]
        self.assertEqual((method, path), ('post', 'voices'))
        self.assertEqual(body['voice']['prompted'], {'input': 'хриплый дед'})
        self.assertEqual(body['voice']['gender'], 'male')
        self.assertTrue(body['store'])
        self.assertEqual(voice['sample_audio'][:4], b'RIFF')
        self.assertEqual(voices.owner('voice_abc'), self.k1['id'])
        self.assertEqual(voices.custom()[0]['name'], 'Дед')

    def test_replicate_requires_wav(self):
        with self.assertRaises(ValueError):
            ai.voices.replicate(b'ID3 mp3 data', b'RIFF....')

    def test_list_builds_query(self):
        client = fake([{'voices': [{'id': 'pl-pl-tutor-1', 'type': 'prebuilt', 'gender': 'male'}]}])
        with mock.patch.object(router, 'client_for', Clients(**{'AIza-k1': client})):
            rows = ai.voices.list(language='pl-PL', gender=['male', 'neutral'], limit=10)
        path = client._api_client.calls[0][1]
        self.assertIn('languageCode=pl-PL', path)
        self.assertIn('gender=male&gender=neutral', path)
        self.assertEqual(rows[0]['id'], 'pl-pl-tutor-1')

    def test_substitute(self):
        self.assertEqual(voices.substitute('Kore')[0], 'Kore')
        voices.remember('voice_girl', None, gender='female', pitch='high')
        self.assertEqual(voices.PREBUILT[voices.substitute('voice_girl')[0]][:2], ('female', 'high'))
        self.assertEqual(voices.substitute('voice_unknown')[0], voices.DEFAULT_VOICE)


class GuideTests(unittest.TestCase):
    def test_sections(self):
        names = tts.sections()
        self.assertGreater(len(names), 5)
        self.assertIn('## ', ai.tts_guide(names[1].split()[0]))
        with self.assertRaises(ValueError):
            ai.tts_guide('нет такого раздела')


if __name__ == '__main__':
    unittest.main()
