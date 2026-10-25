"""Isolated state dirs and fake SDK clients. Import before ai_router in every test module."""

import os
import shutil
import tempfile
import time
from types import SimpleNamespace

ROOT = tempfile.mkdtemp(prefix='ai_router_tests_')
for name in ('AI_ROUTER_DIR', 'GEMINI_BUDGET_DIR', 'OPENAI_BUDGET_DIR'):
    os.environ[name] = os.path.join(ROOT, name.lower())
os.environ.pop('OPENAI_ADMIN_KEY', None)

import httpx  # noqa: E402
import openai  # noqa: E402
from google.genai import errors as genai_errors, types  # noqa: E402

import ai_router as ai  # noqa: E402
from ai_router import router  # noqa: E402


def reset_dirs():
    for name in ('AI_ROUTER_DIR', 'GEMINI_BUDGET_DIR', 'OPENAI_BUDGET_DIR'):
        shutil.rmtree(os.environ[name], ignore_errors=True)
        os.makedirs(os.environ[name], exist_ok=True)
    import openai_budget.core as obcore
    obcore._config_cache = None
    router.SERVER_RETRY_DELAYS = (0, 0, 0)


def gemini_error(code=429, details=None, message='Check your plan and billing details.', status='RESOURCE_EXHAUSTED'):
    return genai_errors.APIError(code, {'error': {'code': code, 'message': message, 'status': status,
                                                  'details': details or []}})


def daily_quota():
    return [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{
        'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier',
        'quotaDimensions': {'model': 'backend-alias'}}]}]


def zero_quota():
    return [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{
        'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '0'}]}]


def openai_error(status, body):
    request = httpx.Request('POST', 'https://api.openai.com/v1/responses')
    response = httpx.Response(status, request=request, json=body)
    cls = {429: openai.RateLimitError, 401: openai.AuthenticationError, 400: openai.BadRequestError,
           500: openai.InternalServerError}[status]
    return cls(body['error']['message'], response=response, body=body['error'])


def gemini_response(text='Привет', finish='STOP', prompt_tokens=12, thought=None, block=None):
    if block:
        return types.GenerateContentResponse(candidates=[], prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=block))
    parts = ([types.Part(text=thought, thought=True)] if thought else []) + ([types.Part(text=text)] if text else [])
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role='model', parts=parts), finish_reason=finish)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens, candidates_token_count=5, total_token_count=prompt_tokens + 5))


def openai_response(text='Hello', status='completed', reason=None, refusal=None, input_tokens=100, output_tokens=20):
    content = [SimpleNamespace(type='refusal', refusal=refusal)] if refusal else \
        [SimpleNamespace(type='output_text', text=text, annotations=[])]
    return SimpleNamespace(
        id='resp_1', output_text=text if not refusal else '', status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        output=[SimpleNamespace(type='message', content=content)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                              total_tokens=input_tokens + output_tokens,
                              input_tokens_details=SimpleNamespace(cached_tokens=0),
                              output_tokens_details=SimpleNamespace(reasoning_tokens=0)))


class FakeGemini:
    """script: list of responses/exceptions returned by consecutive calls (last one repeats)."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.models = SimpleNamespace(generate_content=self._generate, generate_content_stream=self._stream,
                                      list=lambda: [SimpleNamespace(name='models/gemini-3.8-flash',
                                                                    supported_actions=['generateContent'])])

    def _next(self):
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        return self._next()

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        item = self._next()
        for chunk in item:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


class FakeOpenAI:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.responses = SimpleNamespace(create=self._create, parse=self._parse)
        self.models = SimpleNamespace(list=lambda: [SimpleNamespace(id='gpt-5.4-mini')])

    def _next(self):
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def _create(self, **kwargs):
        self.calls.append(('create', kwargs))
        return self._next()

    def _parse(self, **kwargs):
        self.calls.append(('parse', kwargs))
        return self._next()


class Clients:
    """Replaces router.client_for; clients are looked up by secret."""

    def __init__(self, **by_secret):
        self.by_secret = by_secret
        self.used = []

    def __call__(self, provider, secret, timeout_s=None):
        self.used.append(secret)
        return self.by_secret[secret]


def now():
    return time.time()
