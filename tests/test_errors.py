"""Gemini/OpenAI error classification and rotation cases recorded from real API answers."""

import unittest
from types import SimpleNamespace
from unittest import mock

from helpers import (Clients, FakeGemini, ai, gemini_error, gemini_response, openai_error, reset_dirs,
                     router, types)

import gemini_budget as gb
from ai_router import errors, keys, state


def violation(quota_id, **extra):
    return {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{
        'quotaId': quota_id, 'quotaDimensions': {'model': 'gemini-backend-alias'}, **extra}]}


class GeminiClassificationTests(unittest.TestCase):
    def test_quota_report_and_wait(self):
        minute = violation('GenerateContentInputTokensPerModelPerMinute-FreeTier')
        day = violation('GenerateRequestsPerDayPerProjectPerModel-FreeTier')
        retry = {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '12.5s'}
        cases = [
            ([minute, retry], 'quota', 'TPM', 1012.5),
            ([day, retry], 'quota', 'RPD', 9000),
            ([minute, day, retry], 'quota', 'RPD', 9000),
            ([violation('InputTokensPerDay')], 'quota', 'TPD', 9000),
            ([violation('RequestsPerMinute')], 'quota', 'RPM', 1060),
            ([violation('RequestsPerMinute', quotaValue='0'), retry], 'zero_quota', 'равна 0', 2800),
            ([{'@type': 'type.googleapis.com/google.rpc.ErrorInfo', 'reason': 'RATE_LIMIT_EXCEEDED',
               'metadata': {'quota_limit_value': '0', 'quota_metric': 'input_tokens',
                            'quota_unit': '1/min/{project}/{model}', 'model': 'gemini-backend-alias'}}],
             'zero_quota', 'равна 0', 2800),
            ([], 'quota', 'без типа квоты', 0),
        ]
        with mock.patch.object(errors, 'next_pacific_midnight', return_value=9000):
            for details, kind, label, deadline in cases:
                with self.subTest(label=label, details=details):
                    failure = errors.classify_gemini(gemini_error(details=details), 'gemini-requested', now=1000)
                    self.assertEqual(failure.kind, kind)
                    self.assertEqual(failure.scope, 'gemini-requested')
                    self.assertEqual(failure.until, deadline)
                    self.assertIn(label, failure.message)
                    self.assertIn('status=RESOURCE_EXHAUSTED', failure.detail)
                    if label == 'без типа квоты':
                        self.assertNotIn('RPM', failure.message)
        legacy = gemini_error(message='Quota exceeded, limit: 0.5, model: gemini-requested')
        self.assertFalse(errors.quota_report(legacy)['zero_limit'])
        legacy.message = 'Quota exceeded, limit: 0, model: gemini-requested. Please retry in 5s.'
        self.assertTrue(errors.quota_report(legacy)['zero_limit'])
        raw = gemini_error()
        raw.details = [violation('GenerateContentInputTokensPerModelPerMinute-FreeTier'),
                       {'retryDelay': {'seconds': '7', 'nanos': 500000000}}]
        self.assertEqual(errors.quota_report(raw)['retry_s'], 7.5)

    def test_report_survives_in_attempt_detail(self):
        failure = errors.classify_gemini(gemini_error(details=[violation('RequestsPerMinute', quotaValue='0',
                                                                         subject='full-report-subject')]), 'm')
        self.assertIn('full-report-subject', failure.detail)
        self.assertIn('quotaValue', failure.detail)

    def test_access_and_key_errors(self):
        cases = [(403, 'Model gemini-test: access denied', 'model_unavailable'),
                 (404, 'models/gemini-test is not found for API version v1beta', 'model_unavailable'),
                 (404, 'This model models/gemini-2.5-pro is no longer available to new users.', 'model_unavailable'),
                 (404, 'File files/missing was not found', 'bad_request'),
                 (403, 'Permission denied on file files/private', 'bad_request'),
                 (503, 'overloaded', 'server')]
        for code, message, kind in cases:
            with self.subTest(code=code, message=message):
                self.assertEqual(errors.classify_gemini(gemini_error(code, message=message), 'gemini-test').kind, kind)
        invalid = gemini_error(400, details=[{'reason': 'API_KEY_INVALID'}])
        self.assertEqual(errors.classify_gemini(invalid, 'gemini-test').kind, 'bad_key')
        self.assertIn('новых проектов', errors.classify_gemini(gemini_error(
            404, message='This model models/gemini-2.5-pro is no longer available to new users.'), 'x').message)

    def test_network_errors(self):
        import httpx
        self.assertEqual(errors.classify(httpx.ReadTimeout('slow'), 'gemini', 'm').kind, 'timeout')
        self.assertEqual(errors.classify(httpx.ConnectError('down'), 'gemini', 'm').kind, 'network')
        self.assertIsNone(errors.classify(ValueError('bug'), 'gemini', 'm'))


class OpenAIClassificationTests(unittest.TestCase):
    def test_kinds(self):
        no_credits = openai_error(429, {'error': {'message': 'You exceeded your current quota',
                                                  'type': 'insufficient_quota', 'code': 'insufficient_quota'}})
        self.assertEqual(errors.classify_openai(no_credits, 'gpt-5').kind, 'no_credits')
        self.assertEqual(errors.classify_openai(no_credits, 'gpt-5').scope, '*')
        rate = openai_error(429, {'error': {'message': 'Rate limit reached. Please try again in 1m30s.',
                                            'type': 'requests', 'code': 'rate_limit_exceeded'}})
        failure = errors.classify_openai(rate, 'gpt-5', now=0)
        self.assertEqual((failure.kind, failure.scope, failure.until), ('quota', 'gpt-5', 90))
        bad = openai_error(401, {'error': {'message': 'Incorrect API key', 'type': 'invalid_request_error',
                                           'code': 'invalid_api_key'}})
        self.assertEqual(errors.classify_openai(bad, 'gpt-5').kind, 'bad_key')
        server = openai_error(500, {'error': {'message': 'boom', 'type': 'server_error', 'code': None}})
        self.assertEqual(errors.classify_openai(server, 'gpt-5').kind, 'server')


class RotationTests(unittest.TestCase):
    def setUp(self):
        reset_dirs()
        self.a = keys.add('gemini', 'fake-a', label='project-a')
        self.b = keys.add('gemini', 'fake-b', label='project-b')

    def call(self, clients, request, model, **kwargs):
        with mock.patch.object(router, 'client_for', clients):
            return ai.call(request, model=model, **kwargs)

    def test_independent_projects_and_requested_alias(self):
        calls = []

        def request(client, ctx):
            calls.append(ctx.key_label)
            if len(calls) == 1:
                raise gemini_error(details=[violation('RequestsPerMinute', quotaValue='5')])
            return 'ОК'

        clients = Clients(**{'fake-a': 'a', 'fake-b': 'b'})
        self.assertEqual(self.call(clients, request, 'gemini-requested').response, 'ОК')
        pauses = state.snapshot([keys.fingerprint('fake-a')])[keys.fingerprint('fake-a')]['cooldowns']
        self.assertIn('gemini-requested', pauses)
        self.assertNotIn('gemini-backend-alias', pauses)
        self.assertNotIn('*', pauses)
        self.call(clients, request, 'gemini-other')
        self.call(clients, request, 'gemini-requested')
        self.assertEqual(calls, ['project-a', 'project-b', 'project-a', 'project-b'])

    def test_model_access_moves_on_but_file_errors_stop(self):
        for code, message, expected_calls in ((404, 'models/gemini-test is not found for API version v1beta', 2),
                                              (404, 'File files/missing was not found', 1)):
            reset_dirs()
            keys.add('gemini', 'fake-a', label='a')
            keys.add('gemini', 'fake-b', label='b')
            calls = []

            def request(client, ctx):
                calls.append(ctx.key_label)
                if ctx.key_label == 'a':
                    raise gemini_error(code, message=message)
                return 'ОК'

            clients = Clients(**{'fake-a': 'a', 'fake-b': 'b'})
            with self.subTest(message=message):
                if expected_calls == 2:
                    self.assertEqual(self.call(clients, request, 'gemini-test').response, 'ОК')
                else:
                    with self.assertRaises(ai.RequestError):
                        self.call(clients, request, 'gemini-test')
                self.assertEqual(len(calls), expected_calls)
                self.assertTrue(all(k['enabled'] for k in keys.list_keys()))

    def test_paid_pro_models_do_not_consume_free_quota(self):
        with mock.patch.object(gb, 'reserve', side_effect=AssertionError('free quota used')):
            for model in ('gemini-2.5-pro', 'gemini-3.1-pro-preview', 'gemini-3.7-flash'):
                self.assertEqual(self.call(Clients(**{'fake-a': 1, 'fake-b': 1}), lambda c, ctx: 'OK', model).response,
                                 'OK')

    def test_failed_and_empty_replies_release_counters(self):
        model = 'gemini-3.8-flash'
        replies = [gemini_error(504, status='DEADLINE_EXCEEDED'), gemini_error(), gemini_response(text=''),
                   gemini_response(text='', thought='мысли'), gemini_response('нет', finish='SAFETY'),
                   gemini_response('ответ')]
        for index, reply in enumerate(replies):
            fake = FakeGemini([reply])
            with mock.patch.object(router, 'client_for', Clients(**{'fake-a': fake, 'fake-b': fake})):
                try:
                    ai.generate('x', model=model, key_id=self.a['id'], retry_server_errors=False)
                except ai.RouterError:
                    pass
            row = gb.status('fake-a', model)
            success = int(index == len(replies) - 1)
            self.assertEqual((row['requests_today'], row['requests_last_minute'], row['in_flight']),
                             (success, success, 0), index)
            self.assertEqual(row['input_tokens_last_minute'], 12 if success else 0)

    def test_skips_exhausted_key_and_stores_real_prompt_usage(self):
        model = 'gemini-3.8-flash'
        gb.correct('fake-a', model, {'rpm': 5, 'tpm': 250000, 'rpd': 20, 'requests_today': 20,
                                     'requests_last_minute': 0, 'input_tokens_last_minute': 0})
        calls = []

        def request(client, ctx):
            calls.append(ctx.key_label)
            return SimpleNamespace(usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=123, total_token_count=150))

        result = self.call(Clients(**{'fake-a': 1, 'fake-b': 1}), request, 'models/' + model, input_tokens=50)
        self.assertEqual(calls, ['project-b'])
        self.assertEqual(result.attempts[0].outcome, 'skipped')
        row = gb.status('fake-b', model)
        self.assertEqual((row['requests_today'], row['input_tokens_last_minute']), (1, 123))


if __name__ == '__main__':
    unittest.main()
