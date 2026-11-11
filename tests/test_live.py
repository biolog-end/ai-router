"""Live models (bidiGenerateContent) through generate()/stream(), on a fake websocket session."""

import asyncio
import base64
import contextlib
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from helpers import (HANG, Clients, FakeGemini, FakeLive, ai, gemini_response, live_error, live_messages,
                     reset_dirs, router)

import gemini_budget as gb
from ai_router import errors, keys, live

QUOTA = 'You exceeded your current quota, please check your plan and billing details.'


class LiveModelTests(unittest.TestCase):
    def setUp(self):
        reset_dirs()
        self.k1 = keys.add('gemini', 'AIza-k1', label='Первый')
        self.k2 = keys.add('gemini', 'AIza-k2', label='Второй')

    def generate(self, clients, **kwargs):
        with mock.patch.object(router, 'client_for', clients):
            return ai.generate(**kwargs)

    def stream(self, clients, **kwargs):
        with mock.patch.object(router, 'client_for', clients):
            return list(ai.stream(**kwargs))

    def test_is_live_model(self):
        for model in ('gemini-3.8-live', 'models/gemini-3.1-flash-live-preview',
                      'gemini-2.5-flash-native-audio-preview-12-2025', 'gemini-9-live'):
            self.assertTrue(ai.is_live_model(model), model)
        for model in ('gemini-3.8-flash', 'gpt-5.4-mini', 'lyria-realtime-exp', 'gemini-3.8-flash-tts'):
            self.assertFalse(ai.is_live_model(model), model)

    def test_generate_sends_conversation_and_collects_transcription(self):
        fake = FakeLive([live_messages('Вас зовут ', 'Арсений.')])
        image = base64.b64encode(b'png-bytes').decode()
        result = self.generate(
            Clients(**{'AIza-k1': fake, 'AIza-k2': fake}), model='gemini-3.8-live', system='Ты кот',
            temperature=0.3, top_k=20, max_output_tokens=50, search=True, thinking='high', safety='off',
            gemini={'enable_affective_dialog': True, 'candidate_count': 1},
            messages=[{'role': 'user', 'parts': [{'text': 'Я Арсений'}, {'image_base64': image, 'mime_type': 'image/png'}]},
                      {'role': 'model', 'parts': [{'text': 'Привет'}]}],
            prompt='Как меня зовут?')
        self.assertEqual(result.text, 'Вас зовут Арсений.')
        self.assertEqual(result.finish_reason, 'STOP')
        self.assertEqual(result.usage['input_tokens'], 570)
        self.assertNotIsInstance(result, ai.SpeechResult)
        call = fake.calls[0]
        self.assertEqual(call['model'], 'gemini-3.8-live')
        self.assertTrue(call['turn_complete'])
        self.assertTrue(call['closed'])
        self.assertEqual([turn.role for turn in call['turns']], ['user', 'model', 'user'])
        self.assertEqual(call['turns'][0].parts[1].inline_data.data, b'png-bytes')
        config = call['config']
        self.assertEqual([m.value for m in config.response_modalities], ['AUDIO'])
        self.assertIsNotNone(config.output_audio_transcription)
        self.assertEqual(config.system_instruction, 'Ты кот')
        self.assertEqual((config.temperature, config.top_k), (0.3, 20))
        self.assertTrue(config.enable_affective_dialog)
        self.assertIsNone(config.max_output_tokens)
        self.assertIsNone(config.tools)
        note = next(n for n in result.notes if 'Live API' in n)
        for dropped in ('max_output_tokens', 'search', 'thinking', 'safety', 'candidate_count'):
            self.assertIn(dropped, note)
        self.assertEqual(sum(row['requests_today'] for key in ('AIza-k1', 'AIza-k2')
                             for row in gb.status(key) if row['model'] == 'gemini-3.8-live'), 1)

    def test_stream_yields_text_and_audio(self):
        pcm = b'\x01\x00' * 2400
        fake = FakeLive([live_messages('Раз, ', 'два.', audio=pcm)])
        events = self.stream(Clients(**{'AIza-k1': fake, 'AIza-k2': fake}), prompt='Считай',
                             model='gemini-3.8-live', audio=True, voice='puck')
        deltas = [e for e in events if e['type'] == 'delta']
        self.assertEqual(''.join(e.get('text', '') for e in deltas), 'Раз, два.')
        self.assertEqual(b''.join(e.get('audio', b'') for e in deltas), pcm * 2)
        result = events[-1]['result']
        self.assertIsInstance(result, ai.SpeechResult)
        self.assertEqual(result.text, 'Раз, два.')
        self.assertEqual(result.audio[:4], b'RIFF')
        self.assertEqual(result.pcm, pcm * 2)
        self.assertEqual(result.sample_rate, 24000)
        self.assertEqual(result.duration_s, 0.2)
        self.assertEqual(result.voices, {'': 'Puck'})
        voice = fake.calls[0]['config'].speech_config.voice_config.prebuilt_voice_config.voice_name
        self.assertEqual(voice, 'Puck')

    def test_text_only_stream_has_no_audio(self):
        fake = FakeLive([live_messages('Да.', audio=b'\x00\x00')])
        events = self.stream(Clients(**{'AIza-k1': fake, 'AIza-k2': fake}), prompt='x', model='gemini-3.8-live')
        self.assertFalse(any('audio' in e for e in events if e['type'] == 'delta'))
        self.assertIsNone(events[-1]['result'].data)

    def test_unknown_voice_is_dropped(self):
        fake = FakeLive([live_messages('Да.')])
        result = self.generate(Clients(**{'AIza-k1': fake, 'AIza-k2': fake}), prompt='x', model='gemini-3.8-live',
                               audio=True, voice='voices/custom-123')
        self.assertIsNone(fake.calls[0]['config'].speech_config)
        self.assertTrue(any('базовых голосов' in n for n in result.notes))
        self.assertEqual(result.voices, {})

    def test_audio_on_ordinary_model_is_a_note(self):
        fake = FakeGemini([gemini_response('ok')])
        result = self.generate(Clients(**{'AIza-k1': fake, 'AIza-k2': fake}), prompt='x', model='gemini-3.6-flash',
                               audio=True)
        self.assertNotIsInstance(result, ai.SpeechResult)
        self.assertTrue(any('только у Live-моделей' in n for n in result.notes))

    def test_json_mode_and_unsuitable_models_move_the_chain_on(self):
        fake_live = FakeLive([live_messages('нет')])
        fake = FakeGemini([gemini_response('{"ok": true}')])
        fake.aio = fake_live.aio
        clients = Clients(**{'AIza-k1': fake, 'AIza-k2': fake})
        for model in ('gemini-3.5-live-translate-preview', 'gemini-3.8-live-extended-thinking'):
            with self.assertRaises(ai.RequestError):
                self.generate(clients, prompt='x', model=model)
        with self.assertRaises(ai.RequestError) as caught:
            self.generate(clients, prompt='x', model='gemini-3.8-live', json_mode=True)
        self.assertIn('JSON', caught.exception.message)
        result = self.generate(clients, prompt='x', chain=['gemini-3.8-live', 'gemini-3.6-flash'],
                               json_schema={'type': 'object'})
        self.assertEqual(result.model, 'gemini-3.6-flash')
        self.assertEqual(result.data, {'ok': True})
        self.assertEqual(fake_live.calls, [])

    def test_quota_close_rotates_key_and_pauses_model(self):
        busy = FakeLive([live_error(1011, QUOTA)])
        good = FakeLive([live_messages('ok')])
        result = self.generate(Clients(**{'AIza-k1': busy, 'AIza-k2': good}), prompt='x', model='gemini-3.8-live')
        self.assertEqual(result.key_id, self.k2['id'])
        self.assertEqual(result.attempts[0].kind, 'quota')
        self.assertIn('минутная квота', result.attempts[0].message)
        self.assertTrue(gb.status('AIza-k1', 'gemini-3.8-live')['blocked'])

    def test_bad_key_close_rejects_key(self):
        invalid = FakeLive([live_error(1007, 'API key not valid. Please pass a valid API key.')])
        good = FakeLive([live_messages('ok')])
        self.generate(Clients(**{'AIza-k1': invalid, 'AIza-k2': good}), prompt='x', model='gemini-3.8-live')
        row = next(k for k in ai.status()['keys'] if k['id'] == self.k1['id'])
        self.assertTrue(row['rejected'])

    def test_abnormal_close_retries_same_key(self):
        flaky = FakeLive([live_error(1006, 'abnormal closure [internal]'), live_messages('ok')])
        clients = Clients(**{'AIza-k1': flaky, 'AIza-k2': flaky})
        result = self.generate(clients, prompt='x', model='gemini-3.8-live')
        self.assertEqual(result.text, 'ok')
        self.assertEqual(len(flaky.calls), 2)
        self.assertEqual(len(set(clients.used)), 1)

    def test_empty_answer(self):
        fake = FakeLive([live_messages()])
        with self.assertRaises(ai.EmptyResponseError):
            self.generate(Clients(**{'AIza-k1': fake, 'AIza-k2': fake}), prompt='x', model='gemini-3.8-live')

    def test_idle_timeout_moves_to_next_key(self):
        silent = FakeLive([[HANG]])
        good = FakeLive([live_messages('ok')])
        started = time.time()
        result = self.generate(Clients(**{'AIza-k1': silent, 'AIza-k2': good}), prompt='x', model='gemini-3.8-live',
                               timeout_s=1)
        self.assertLess(time.time() - started, 5)
        self.assertEqual(result.attempts[0].kind, 'timeout')
        self.assertEqual(result.text, 'ok')
        self.assertTrue(silent.calls[0]['closed'])

    def test_stop_event_ends_a_hanging_answer(self):
        messages = live_messages('Начало')
        fake = FakeLive([[messages[0], HANG]])
        stop = threading.Event()
        with mock.patch.object(router, 'client_for', Clients(**{'AIza-k1': fake, 'AIza-k2': fake})):
            with self.assertRaises(ai.StoppedError):
                for event in ai.stream('x', model='gemini-3.8-live', stop_event=stop):
                    if event['type'] == 'delta':
                        stop.set()
        self.assertTrue(fake.calls[0]['closed'])

    def test_works_inside_a_running_event_loop(self):
        fake = FakeLive([live_messages('ok')])

        async def caller():
            return self.generate(Clients(**{'AIza-k1': fake, 'AIza-k2': fake}), prompt='x', model='gemini-3.8-live')

        self.assertEqual(asyncio.run(caller()).text, 'ok')


class LiveClassificationTests(unittest.TestCase):
    def test_close_codes(self):
        cases = [
            (1007, 'API key not valid. Please pass a valid API key.', 'bad_key', '*'),
            (1008, 'models/gemini-9-live is not found for API version v1beta, or is not supported for '
                   'bidiGenerateContent.', 'model_unavailable', 'gemini-3.8-live'),
            (1011, QUOTA, 'quota', 'gemini-3.8-live'),
            (1011, 'Internal error encountered.', 'server', None),
            (1006, 'abnormal closure [internal]', 'server', None),
            (1007, 'Request contains an invalid argument.', 'bad_request', None),
        ]
        for code, reason, kind, scope in cases:
            with self.subTest(code=code, kind=kind):
                failure = errors.classify_gemini(live_error(code, reason), 'gemini-3.8-live', now=1000)
                self.assertEqual(failure.kind, kind)
                self.assertEqual(failure.scope, scope)
                self.assertIn(f'Live API {code}', failure.raw)
        quota = errors.classify_gemini(live_error(1011, QUOTA), 'gemini-3.8-live', now=1000)
        self.assertEqual(quota.until, 1060)


class _TurnByTurnLive:
    """Like google.genai AsyncSession: receive() stops after each turn_complete; once the
    script is exhausted it waits, as a live socket with nothing to say does."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.aio = SimpleNamespace(live=SimpleNamespace(connect=self._connect))

    @contextlib.asynccontextmanager
    async def _connect(self, model, config):
        async def receive():
            if not self.turns:
                await asyncio.Event().wait()
            for message in self.turns.pop(0):
                await asyncio.sleep(0)
                yield message

        async def send_realtime_input(**kwargs):
            pass

        yield SimpleNamespace(receive=receive, send_realtime_input=send_realtime_input)


class StreamSessionTests(unittest.TestCase):
    def test_session_survives_turn_complete(self):
        fake = _TurnByTurnLive([live_messages('Алло?'), live_messages('Да, слышу.')])
        session = live.StreamSession(fake, 'gemini-3.8-live', None, idle_s=2)
        session.start()
        texts = []
        try:
            for event in session.events():
                if event['type'] == 'text':
                    texts.append(event['text'])
                if event['type'] in ('closed', 'error') or len(texts) == 2:
                    break
        finally:
            session.close()
        self.assertEqual(texts, ['Алло?', 'Да, слышу.'])


if __name__ == '__main__':
    unittest.main()
