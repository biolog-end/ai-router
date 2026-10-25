import base64
import threading
import unittest
from unittest import mock

from helpers import (Clients, FakeGemini, FakeOpenAI, ai, daily_quota, gemini_error, gemini_response,
                     openai_error, openai_response, reset_dirs, router, types, zero_quota)

import gemini_budget as gb
import openai_budget as ob
from ai_router import keys, state


class KeyVaultTests(unittest.TestCase):
    def setUp(self):
        reset_dirs()

    def test_add_list_and_dedupe(self):
        first = keys.add('gemini', 'AIza-secret-one', label='Проект 1')
        again = keys.add('gemini', ' AIza-secret-one ')
        self.assertEqual(first['id'], again['id'])
        rows = keys.list_keys()
        self.assertEqual(len(rows), 1)
        self.assertNotIn('key', rows[0])
        self.assertNotIn('AIza-secret-one', str(rows))
        self.assertEqual(keys.secret(first['id']), 'AIza-secret-one')

    def test_admin_key_is_refused(self):
        with self.assertRaises(ValueError):
            keys.add('openai', 'sk-admin-abc')

    def test_replace_keeps_ids_and_secrets(self):
        a = keys.add('gemini', 'AIza-a')
        keys.add('gemini', 'AIza-b')
        o = keys.add('openai', 'sk-proj-o')
        rows = keys.replace('gemini', [{'id': a['id'], 'label': 'Новое имя', 'enabled': False}, {'key': 'AIza-c'}])
        self.assertEqual([r['label'] for r in rows][0], 'Новое имя')
        self.assertEqual(keys.secrets('gemini'), ['AIza-a', 'AIza-c'])
        self.assertFalse(keys.get(a['id'])['enabled'])
        self.assertEqual(keys.get(o['id'])['provider'], 'openai')

    def test_import_once_does_not_resurrect(self):
        self.assertEqual(keys.import_once('file:x', [{'provider': 'gemini', 'key': 'AIza-x'}], source='app'), 1)
        keys.remove(keys.list_keys()[0]['id'])
        self.assertIsNone(keys.import_once('file:x', [{'provider': 'gemini', 'key': 'AIza-x'}], source='app'))
        self.assertEqual(keys.list_keys(), [])

    def test_import_env(self):
        env = {'GOOGLE_API_KEYS': 'AIza-1, AIza-2', 'OPENAI_API_KEY': 'sk-proj-1', 'OPENAI_API_KEYS': 'sk-admin-x'}
        self.assertEqual(keys.import_env(env), 3)
        self.assertEqual(keys.import_env(env), 0)


class GenerateGeminiTests(unittest.TestCase):
    def setUp(self):
        reset_dirs()
        self.k1 = keys.add('gemini', 'AIza-k1', label='Первый')
        self.k2 = keys.add('gemini', 'AIza-k2', label='Второй')

    def run_with(self, clients, **kwargs):
        with mock.patch.object(router, 'client_for', clients):
            return ai.generate(**kwargs)

    def test_success_maps_request_and_counts_quota(self):
        fake = FakeGemini([gemini_response('Ответ', thought='мысль')])
        clients = Clients(**{'AIza-k1': fake, 'AIza-k2': fake})
        result = self.run_with(clients, prompt='Привет', model='gemini-3.8-flash', system='Ты кот',
                               temperature=0.5, thinking='off', search=True, include_thoughts=True,
                               gemini={'candidate_count': 1})
        self.assertEqual(result.text, 'Ответ')
        self.assertEqual(result.thoughts, 'мысль')
        self.assertEqual(result.usage['input_tokens'], 12)
        config = fake.calls[0]['config']
        self.assertEqual(config.system_instruction, 'Ты кот')
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.thinking_config.thinking_level, types.ThinkingLevel.MINIMAL)
        self.assertTrue(config.tools[0].google_search is not None)
        self.assertEqual(config.candidate_count, 1)
        self.assertTrue(any('платном проекте' in note for note in result.notes))
        used = sum(row['requests_today'] for key in ('AIza-k1', 'AIza-k2')
                   for row in gb.status(key) if row['model'] == 'gemini-3.8-flash')
        self.assertEqual(used, 1)

    def test_budget_thinking_for_25(self):
        fake = FakeGemini([gemini_response()])
        self.run_with(Clients(**{'AIza-k1': fake, 'AIza-k2': fake}), prompt='x', model='gemini-2.5-flash',
                      thinking='high')
        self.assertEqual(fake.calls[0]['config'].thinking_config.thinking_budget, -1)

    def test_daily_quota_rotates_and_is_remembered(self):
        bad = FakeGemini([gemini_error(429, daily_quota())])
        good = FakeGemini([gemini_response('ok')])
        clients = Clients(**{'AIza-k1': bad, 'AIza-k2': good})
        events = []
        result = self.run_with(clients, prompt='x', model='gemini-3.8-flash', on_event=events.append)
        self.assertEqual(result.key_id, self.k2['id'])
        self.assertEqual([a.outcome for a in result.attempts], ['failed', 'ok'])
        self.assertEqual(result.attempts[0].kind, 'quota')
        self.assertTrue(any(e['type'] == 'failure' for e in events))
        self.assertTrue(gb.status('AIza-k1', 'gemini-3.8-flash')['blocked'])
        # The paused key is skipped by gemini_budget next time, without an HTTP call.
        bad.calls.clear()
        result = self.run_with(clients, prompt='x', model='gemini-3.8-flash')
        self.assertEqual(bad.calls, [])
        self.assertEqual(result.attempts[0].outcome, 'skipped')

    def test_paid_model_zero_quota_pauses_in_router_state(self):
        zero = FakeGemini([gemini_error(429, zero_quota())])
        clients = Clients(**{'AIza-k1': zero, 'AIza-k2': zero})
        with self.assertRaises(ai.AllFailedError) as caught:
            self.run_with(clients, prompt='x', model='gemini-3.7-flash')
        self.assertEqual({a.kind for a in caught.exception.attempts}, {'zero_quota'})
        with self.assertRaises(ai.KeysBusyError) as caught:
            self.run_with(clients, prompt='x', model='gemini-3.7-flash')
        self.assertIn('квота модели для этого проекта равна 0', caught.exception.message)
        self.assertIn('gemini-3.7-flash', ai.status()['blocked_models'])
        ai.reset_pauses()
        self.assertNotIn('gemini-3.7-flash', ai.status()['blocked_models'])

    def test_bad_key_is_rejected_for_everyone(self):
        invalid = FakeGemini([gemini_error(400, [{'@type': 'type.googleapis.com/google.rpc.ErrorInfo',
                                                  'reason': 'API_KEY_INVALID'}], 'API key not valid.', 'INVALID_ARGUMENT')])
        good = FakeGemini([gemini_response()])
        clients = Clients(**{'AIza-k1': invalid, 'AIza-k2': good})
        self.run_with(clients, prompt='x', model='gemini-3.6-flash')
        row = next(k for k in ai.status()['keys'] if k['id'] == self.k1['id'])
        self.assertTrue(row['rejected'])
        self.assertFalse(row['available'])
        self.assertNotIn('AIza-k1', str(ai.status()))

    def test_server_error_retries_same_key(self):
        flaky = FakeGemini([gemini_error(503, status='UNAVAILABLE', message='overloaded'), gemini_response('ok')])
        clients = Clients(**{'AIza-k1': flaky, 'AIza-k2': flaky})
        result = self.run_with(clients, prompt='x', model='gemini-3.6-flash')
        self.assertEqual(len(flaky.calls), 2)
        self.assertEqual(len({c for c in clients.used}), 1)
        self.assertEqual(result.text, 'ok')

    def test_safety_and_empty_stop_the_model(self):
        blocked = FakeGemini([gemini_response(finish='SAFETY')])
        clients = Clients(**{'AIza-k1': blocked, 'AIza-k2': blocked})
        with self.assertRaises(ai.SafetyError) as caught:
            self.run_with(clients, prompt='x', model='gemini-3.6-flash')
        self.assertIn('SAFETY', caught.exception.message)
        self.assertEqual(len(blocked.calls), 1)
        empty = FakeGemini([gemini_response(text='', finish='MAX_TOKENS')])
        with self.assertRaises(ai.EmptyResponseError):
            self.run_with(Clients(**{'AIza-k1': empty, 'AIza-k2': empty}), prompt='x', model='gemini-3.6-flash')
        prompt_block = FakeGemini([gemini_response(block='PROHIBITED_CONTENT')])
        with self.assertRaises(ai.SafetyError):
            self.run_with(Clients(**{'AIza-k1': prompt_block, 'AIza-k2': prompt_block}), prompt='x',
                          model='gemini-3.6-flash')

    def test_bad_request_raises_without_trying_other_keys(self):
        bad = FakeGemini([gemini_error(400, message='Invalid value', status='INVALID_ARGUMENT')])
        clients = Clients(**{'AIza-k1': bad, 'AIza-k2': bad})
        with self.assertRaises(ai.RequestError):
            self.run_with(clients, prompt='x', model='gemini-3.6-flash')
        self.assertEqual(len(bad.calls), 1)
        with self.assertRaises(ai.AllFailedError):
            self.run_with(clients, prompt='x', model='gemini-3.6-flash', try_all_keys=True)

    def test_pinned_key_ignores_router_pause(self):
        state.pause(keys.fingerprint('AIza-k1'), '*', router.time.time() + 600, 'quota', 'тест')
        good = FakeGemini([gemini_response()])
        result = self.run_with(Clients(**{'AIza-k1': good}), prompt='x', model='gemini-3.6-flash',
                               key_id=self.k1['id'])
        self.assertEqual(result.key_id, self.k1['id'])

    def test_json_schema_and_legacy_parts(self):
        fake = FakeGemini([gemini_response('{"ok": true}')])
        image = base64.b64encode(b'png-bytes').decode()
        result = self.run_with(Clients(**{'AIza-k1': fake, 'AIza-k2': fake}), model='gemini-3.6-flash',
                               messages=[{'role': 'user', 'parts': [{'text': 'смотри'},
                                                                    {'image_base64': image, 'mime_type': 'image/png',
                                                                     'sticker_id': 5}]},
                                         {'role': 'model', 'parts': [{'text': 'вижу'}]}],
                               prompt='ещё', json_schema={'type': 'object'})
        self.assertEqual(result.data, {'ok': True})
        contents = fake.calls[0]['contents']
        self.assertEqual([c.role for c in contents], ['user', 'model', 'user'])
        self.assertEqual(contents[0].parts[1].inline_data.data, b'png-bytes')
        self.assertEqual(fake.calls[0]['config'].response_json_schema, {'type': 'object'})

    def test_stop_event(self):
        stop = threading.Event()
        stop.set()
        with self.assertRaises(ai.StoppedError):
            self.run_with(Clients(), prompt='x', model='gemini-3.6-flash', stop_event=stop)

    def test_no_keys(self):
        with self.assertRaises(ai.NoKeysError):
            ai.generate('x', model='gpt-5.4-mini')

    def test_retired_model(self):
        with self.assertRaises(ai.RequestError) as caught:
            ai.generate('x', model='gemini-2.0-flash')
        self.assertIn('больше не поддерживается', caught.exception.message)


class OpenAIAndChainTests(unittest.TestCase):
    def setUp(self):
        reset_dirs()
        keys.add('gemini', 'AIza-g')
        self.o1 = keys.add('openai', 'sk-proj-1', label='OA1')
        self.o2 = keys.add('openai', 'sk-proj-2', label='OA2')

    def test_chain_falls_through_to_openai(self):
        gemini = FakeGemini([gemini_error(429, daily_quota())])
        oa = FakeOpenAI([openai_response('Hi')])
        events = []
        with mock.patch.object(router, 'client_for', Clients(**{'AIza-g': gemini, 'sk-proj-1': oa, 'sk-proj-2': oa})):
            result = ai.generate('x', chain=['gemini-3.8-flash', 'gpt-5.4-mini'], system='sys', on_event=events.append)
        self.assertEqual(result.model, 'gpt-5.4-mini')
        self.assertEqual(result.position, 2)
        self.assertTrue(result.fallback_used)
        self.assertIn('next_model', [e['type'] for e in events])
        method, kwargs = oa.calls[0]
        self.assertEqual(kwargs['instructions'], 'sys')
        self.assertEqual(kwargs['input'][0]['content'][0], {'type': 'input_text', 'text': 'x'})
        self.assertEqual(ob.status()['groups']['small']['used'], 120)

    def test_chain_all_fail(self):
        gemini = FakeGemini([gemini_error(429, daily_quota())])
        oa = FakeOpenAI([openai_error(401, {'error': {'message': 'Incorrect API key', 'type': 'invalid_request_error',
                                                      'code': 'invalid_api_key'}})])
        with mock.patch.object(router, 'client_for', Clients(**{'AIza-g': gemini, 'sk-proj-1': oa, 'sk-proj-2': oa})):
            with self.assertRaises(ai.AllFailedError) as caught:
                ai.generate('x', chain=['gemini-3.8-flash', 'gpt-5.4-mini'])
        self.assertEqual(len(caught.exception.errors), 2)
        self.assertIn('gpt-5.4-mini', caught.exception.message)

    def test_no_credits_pauses_whole_key(self):
        broke = FakeOpenAI([openai_error(429, {'error': {'message': 'You exceeded your current quota',
                                                         'type': 'insufficient_quota', 'code': 'insufficient_quota'}})])
        good = FakeOpenAI([openai_response()])
        with mock.patch.object(router, 'client_for', Clients(**{'sk-proj-1': broke, 'sk-proj-2': good})):
            result = ai.generate('x', model='gpt-5.4-mini')
        self.assertEqual(result.key_id, self.o2['id'])
        row = next(k for k in ai.status()['keys'] if k['id'] == self.o1['id'])
        self.assertEqual(row['cooldowns'][0]['model'], '*')
        self.assertFalse(row['available'])

    def test_free_pool_refusal_and_paid_override(self):
        ob.record('gpt-5.4-mini', 2_495_000, 0)
        oa = FakeOpenAI([openai_response()])
        clients = Clients(**{'sk-proj-1': oa, 'sk-proj-2': oa})
        with mock.patch.object(router, 'client_for', clients):
            with self.assertRaises(ai.BudgetError):
                ai.generate('x', model='gpt-5.4-mini')
            self.assertEqual(oa.calls, [])
            result = ai.generate('x', model='gpt-5.4-mini', allow_paid=True)
        self.assertTrue(any('платно' in note for note in result.notes))

    def test_openai_parameter_mapping(self):
        oa = FakeOpenAI([openai_response()])
        with mock.patch.object(router, 'client_for', Clients(**{'sk-proj-1': oa, 'sk-proj-2': oa})):
            result = ai.generate('x', model='gpt-5.4', temperature=0.3, top_k=5, cache_key='chat-1', search=True)
            _, kwargs = oa.calls[-1]
            self.assertEqual(kwargs['reasoning'], {'effort': 'none'})
            self.assertEqual(kwargs['temperature'], 0.3)
            self.assertEqual(kwargs['prompt_cache_key'], 'chat-1')
            self.assertNotIn('tools', kwargs)
            self.assertTrue(any('top_k' in n for n in result.notes))
            result = ai.generate('x', model='gpt-5-mini', temperature=0.3, thinking='high')
            _, kwargs = oa.calls[-1]
            self.assertNotIn('temperature', kwargs)
            self.assertEqual(kwargs['reasoning'], {'effort': 'high'})
            ai.generate('x', model='gpt-5-mini', thinking=False, openai={'store': False})
            _, kwargs = oa.calls[-1]
            self.assertEqual(kwargs['reasoning'], {'effort': 'minimal'})
            self.assertIs(kwargs['store'], False)

    def test_refusal_is_safety(self):
        oa = FakeOpenAI([openai_response(refusal='Не могу')])
        with mock.patch.object(router, 'client_for', Clients(**{'sk-proj-1': oa, 'sk-proj-2': oa})):
            with self.assertRaises(ai.SafetyError) as caught:
                ai.generate('x', model='gpt-5.4-mini')
        self.assertIn('refusal', caught.exception.message)

    def test_pydantic_schema_uses_parse(self):
        from pydantic import BaseModel

        class Letter(BaseModel):
            text: str

        response = openai_response('{"text": "hi"}')
        response.output_parsed = Letter(text='hi')
        oa = FakeOpenAI([response])
        with mock.patch.object(router, 'client_for', Clients(**{'sk-proj-1': oa, 'sk-proj-2': oa})):
            result = ai.generate('x', model='gpt-5.4-mini', json_schema=Letter)
        self.assertEqual(oa.calls[0][0], 'parse')
        self.assertIs(oa.calls[0][1]['text_format'], Letter)
        self.assertEqual(result.data.text, 'hi')


class LowLevelAndStreamTests(unittest.TestCase):
    def setUp(self):
        reset_dirs()
        keys.add('gemini', 'AIza-1', label='A')
        keys.add('gemini', 'AIza-2', label='B')

    def test_call_gets_context(self):
        seen = []

        def request(client, ctx):
            seen.append((ctx.model, ctx.provider, ctx.key_label))
            return client.models.generate_content(model=ctx.model, contents='x')

        fake = FakeGemini([gemini_response('raw')])
        with mock.patch.object(router, 'client_for', Clients(**{'AIza-1': fake, 'AIza-2': fake})):
            result = ai.call(request, model='gemini-3.6-flash', input_tokens=10)
        self.assertEqual(result.text, 'raw')
        self.assertEqual(seen[0][:2], ('gemini-3.6-flash', 'gemini'))
        self.assertIsNotNone(result.response)

    def test_unknown_exception_propagates(self):
        def request(client, ctx):
            raise ZeroDivisionError('bug')

        with mock.patch.object(router, 'client_for', Clients(**{'AIza-1': None, 'AIza-2': None})):
            with self.assertRaises(ZeroDivisionError):
                ai.call(request, model='gemini-3.6-flash')

    def test_stream_switches_key_before_first_delta(self):
        chunk = gemini_response('Пр', finish=None)
        last = gemini_response('ивет')
        failing = FakeGemini([[gemini_error(429, daily_quota())]])
        working = FakeGemini([[chunk, last]])
        clients = Clients(**{'AIza-1': failing, 'AIza-2': working})
        with mock.patch.object(router, 'client_for', clients):
            events = list(ai.stream('x', model='gemini-3.6-flash'))
        deltas = ''.join(e['text'] for e in events if e['type'] == 'delta')
        self.assertEqual(deltas, 'Привет')
        self.assertEqual(events[-1]['result'].text, 'Привет')
        self.assertIn('failure', [e['type'] for e in events])

    def test_stream_error_after_delta_is_raised(self):
        broken = FakeGemini([[gemini_response('Пр', finish=None), gemini_error(503, status='UNAVAILABLE')]])
        with mock.patch.object(router, 'client_for', Clients(**{'AIza-1': broken, 'AIza-2': broken})):
            stream = ai.stream('x', model='gemini-3.6-flash')
            with self.assertRaises(ai.RouterError) as caught:
                for _ in stream:
                    pass
        self.assertIn('оборвался', caught.exception.message)

    def test_stream_call_passes_custom_events(self):
        def request(client, ctx):
            yield {'type': 'delta', 'text': 'a'}
            yield {'type': 'meta', 'value': 1}
            yield {'type': 'usage', 'input_tokens': 3, 'total_tokens': 5}

        with mock.patch.object(router, 'client_for', Clients(**{'AIza-1': object(), 'AIza-2': object()})):
            events = list(ai.stream_call(request, model='gemini-3.6-flash'))
        self.assertEqual([e['type'] for e in events if e['type'] in ('delta', 'meta', 'done')], ['delta', 'meta', 'done'])
        self.assertEqual(events[-1]['result'].usage['input_tokens'], 3)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        reset_dirs()

    def test_default_chain_is_editable(self):
        self.assertEqual(ai.default_chain(), ai.DEFAULT_CHAIN)
        ai.set_default_chain(['gemini-3.8-flash', 'models/gpt-5-mini'])
        self.assertEqual(ai.default_chain(), ['gemini-3.8-flash', 'gpt-5-mini'])

    def test_models_of_both_providers(self):
        rows = {row['id']: row for row in ai.models()}
        self.assertEqual(rows['gpt-5.4-mini']['provider'], 'openai')
        self.assertTrue(rows['gemini-3.8-flash']['free_tier'])
        self.assertFalse(rows['gemini-3.7-flash']['free_tier'])
        self.assertEqual(ai.model_info('gpt-5.4')['free_group'], 'large')
        self.assertEqual(ai.light_model('openai'), 'gpt-5-mini')


if __name__ == '__main__':
    unittest.main()
