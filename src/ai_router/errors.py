"""Router exceptions and the normalisation of Gemini/OpenAI API errors into one shape.

Messages are Russian because every consumer shows them in a Russian UI as is.
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

DEFAULT_MINUTE_COOLDOWN_S = 60
# Zero quota and missing access need a settings change, not retries every minute.
ZERO_QUOTA_COOLDOWN_S = 30 * 60
MODEL_ACCESS_COOLDOWN_S = 5 * 60
# OpenAI "no credits": re-checked every half hour so a top-up is picked up by itself.
NO_CREDITS_COOLDOWN_S = 30 * 60
FALLBACK_DAILY_COOLDOWN_S = 3600

# What the attempt loop does with each failure kind.
NEXT_KEY = frozenset({'quota', 'zero_quota', 'no_credits', 'bad_key', 'model_unavailable', 'budget', 'timeout'})
RETRY_SAME_KEY = frozenset({'server', 'network'})
STOP_MODEL = frozenset({'bad_request', 'safety', 'empty', 'other'})

KIND_LABELS = {
    'quota': 'лимит квоты', 'zero_quota': 'квота 0', 'no_credits': 'нет кредитов',
    'bad_key': 'ключ отклонён', 'model_unavailable': 'модель недоступна', 'budget': 'локальный лимит',
    'timeout': 'таймаут', 'server': 'ошибка сервера', 'network': 'нет связи',
    'bad_request': 'неверный запрос', 'safety': 'фильтр безопасности', 'empty': 'пустой ответ',
    'other': 'ошибка',
}


@dataclass
class Attempt:
    """One try of one key (or a model-level check). Safe to show: no secrets."""
    model: str
    provider: str
    key_id: str = None
    key_label: str = None
    outcome: str = 'failed'          # ok | failed | skipped
    kind: str = None
    message: str = ''
    until: float = None
    detail: str = ''                 # raw API status/details for logs
    at: float = field(default_factory=time.time)
    duration_s: float = 0.0

    def to_dict(self):
        return dict(self.__dict__)


class RouterError(Exception):
    """Base class. `kind` tells what happened, `attempts` how the router got there."""
    kind = 'failed'

    def __init__(self, message, *, kind=None, model=None, provider=None, attempts=None, retry_at=None,
                 errors=None):
        super().__init__(message)
        self.message = message
        self.kind = kind or self.kind
        self.model = model
        self.provider = provider
        self.attempts = list(attempts or [])
        self.retry_at = retry_at
        self.errors = list(errors or [])

    @property
    def retry_in_s(self):
        return max(0, int(self.retry_at - time.time())) if self.retry_at else None

    def to_dict(self):
        return {'kind': self.kind, 'message': self.message, 'model': self.model, 'provider': self.provider,
                'retry_at': self.retry_at, 'attempts': [a.to_dict() for a in self.attempts]}


class NoKeysError(RouterError):
    kind = 'no_keys'


class KeysBusyError(RouterError):
    """No key could even be tried: all paused, rejected or refused by the local quota."""
    kind = 'busy'


class BudgetError(RouterError):
    """The free OpenAI pool (openai_budget) would be exceeded; no request was sent."""
    kind = 'budget'


class RequestError(RouterError):
    """The provider rejected the request itself; another key will not fix it."""
    kind = 'bad_request'


class SafetyError(RouterError):
    kind = 'safety'


class EmptyResponseError(RouterError):
    kind = 'empty'


class StoppedError(RouterError):
    kind = 'stopped'


class AllFailedError(RouterError):
    """Every key (or every model of a chain) was tried and failed."""
    kind = 'failed'


ERROR_CLASSES = {'no_keys': NoKeysError, 'busy': KeysBusyError, 'budget': BudgetError,
                 'bad_request': RequestError, 'safety': SafetyError, 'empty': EmptyResponseError,
                 'stopped': StoppedError}


@dataclass
class Failure:
    """A classified failure of one attempt. scope: model id or '*' to pause, None for no pause."""
    kind: str
    message: str
    code: int = None
    scope: str = None
    until: float = 0.0
    report: dict = None
    raw: str = ''
    detail: str = ''


def format_wait(seconds):
    seconds = max(int(seconds), 0)
    if seconds >= 3600:
        return f'{seconds // 3600} ч {(seconds % 3600) // 60} мин'
    if seconds >= 90:
        return f'{seconds // 60} мин'
    return f'{seconds} с'


def next_pacific_midnight(now=None):
    """Google resets daily free quotas at midnight America/Los_Angeles."""
    now = time.time() if now is None else now
    try:
        from zoneinfo import ZoneInfo
        zone = ZoneInfo('America/Los_Angeles')
    except Exception:
        return now + FALLBACK_DAILY_COOLDOWN_S
    current = datetime.fromtimestamp(now, zone)
    return (current + timedelta(days=1)).replace(hour=0, minute=0, second=30, microsecond=0).timestamp()


# --------------------------------------------------------------------------
# Network errors (both SDKs run on httpx)
# --------------------------------------------------------------------------

def _network_failure(exc):
    name = type(exc).__name__.lower()
    module = type(exc).__module__ or ''
    is_timeout = isinstance(exc, TimeoutError) or 'timeout' in name
    if module.startswith(('httpx', 'httpcore', 'openai')) or isinstance(exc, (ConnectionError, TimeoutError)):
        if is_timeout:
            return Failure('timeout', 'Провайдер не ответил до таймаута запроса.', raw=str(exc))
        if 'connect' in name or 'network' in name or 'transport' in name or 'protocol' in name \
                or isinstance(exc, ConnectionError):
            return Failure('network', 'Не удалось связаться с провайдером.', raw=str(exc))
    return None


# --------------------------------------------------------------------------
# Gemini (google.genai.errors.APIError — not a GoogleAPIError subclass)
# --------------------------------------------------------------------------

# Websocket close codes of the Live API → the HTTP status with the same meaning.
# 1011 is 429 when its reason mentions the quota.
LIVE_CLOSE_CODES = {1000: 503, 1001: 503, 1006: 503, 1007: 400, 1008: 404, 1011: 503, 1012: 503, 1013: 503,
                    1014: 503}


def _is_genai_error(exc):
    try:
        from google.genai import errors as genai_errors
    except ImportError:
        return False
    return isinstance(exc, genai_errors.APIError)


def gemini_error_items(exc):
    """The SDK keeps the whole error body in APIError.details; details[] lists are accepted too."""
    payload = getattr(exc, 'details', None)
    if isinstance(payload, dict) and isinstance(payload.get('error'), dict):
        payload = payload['error']
    if isinstance(payload, dict):
        payload = payload.get('details', [])
    return [item for item in payload or [] if isinstance(item, dict)] if isinstance(payload, list) else []


def _parse_duration(value):
    if isinstance(value, dict):
        try:
            return max(float(value.get('seconds', 0)) + float(value.get('nanos', 0)) / 1e9, 0)
        except (TypeError, ValueError):
            return None
    match = re.match(r'^\s*(\d+(?:\.\d+)?)\s*s?\s*$', str(value or ''))
    return float(match.group(1)) if match else None


def quota_report(exc):
    """Google's quota report: model, period, explicit zero value and RetryInfo.

    A missing quotaValue does not mean zero. A daily violation outranks a per-minute one.
    """
    report = {'model': None, 'scope': 'unknown', 'retry_s': None, 'zero_limit': False,
              'violations': [], 'reasons': []}
    for item in gemini_error_items(exc):
        if item.get('reason'):
            report['reasons'].append(str(item['reason']))
        violations = item.get('violations') or []
        metadata = item.get('metadata')
        if isinstance(metadata, dict) and any(k in metadata for k in ('quota_metric', 'quota_limit', 'quota_limit_value')):
            violation = {'quotaMetric': metadata.get('quota_metric'), 'quotaId': metadata.get('quota_limit'),
                         'quotaDimensions': {'model': metadata.get('model'), 'unit': metadata.get('quota_unit')}}
            if 'quota_limit_value' in metadata:
                violation['quotaValue'] = metadata['quota_limit_value']
            violations = [*violations, violation]
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            report['violations'].append({k: violation[k] for k in ('quotaId', 'quotaMetric', 'quotaDimensions', 'quotaValue')
                                         if k in violation})
            dims = violation.get('quotaDimensions') if isinstance(violation.get('quotaDimensions'), dict) else {}
            report['model'] = report['model'] or dims.get('model')
            metric = ' '.join(str(violation.get(k) or '') for k in ('quotaId', 'quotaMetric')) + ' ' + str(dims.get('unit') or '')
            if re.search(r'per[ _]*day|/d(?:ay)?\b', metric, re.IGNORECASE):
                report['scope'] = 'day'
            elif report['scope'] == 'unknown' and re.search(r'per[ _]*minute|/min\b', metric, re.IGNORECASE):
                report['scope'] = 'minute'
            if 'quotaValue' in violation:
                try:
                    report['zero_limit'] |= float(violation['quotaValue']) == 0
                except (TypeError, ValueError):
                    pass
        if 'retryDelay' in item:
            delay = _parse_duration(item['retryDelay'])
            if delay is not None:
                report['retry_s'] = max(report['retry_s'] or 0, delay)

    # Older answers carry quotaId, limit and retryDelay only in the text.
    text = str(getattr(exc, 'message', '') or exc)
    if report['scope'] == 'unknown':
        if re.search(r'per\s*day|PerDay', text, re.IGNORECASE):
            report['scope'] = 'day'
        elif re.search(r'per\s*minute|PerMinute', text, re.IGNORECASE):
            report['scope'] = 'minute'
    report['zero_limit'] |= bool(re.search(r'\blimit\s*[:=]\s*0(?:\.0+)?(?=\s|[,;]|\.?(?:\s|$))', text, re.IGNORECASE))
    if report['retry_s'] is None:
        match = re.search(r'retry\w*[\'"]?\s*(?:[:=]|in)\s*[\'"]?(\d+(?:\.\d+)?)\s*s', text, re.IGNORECASE)
        if match:
            report['retry_s'] = float(match.group(1))
    if report['model'] is None:
        match = re.search(r'model[\'"]?\s*:\s*[\'"]?([\w.\-]+)', text)
        if match:
            report['model'] = match.group(1)
    return report


def _gemini_quota(exc, report, now):
    """(kind, until, reason) for a Gemini 429."""
    if report['zero_limit']:
        return 'zero_quota', now + ZERO_QUOTA_COOLDOWN_S, (
            'квота модели для этого проекта равна 0 (только платный тариф или нет доступа). '
            'Перепроверка через 30 мин; сброс пауз в хранилище ключей снимает её сразу')
    scope = report['scope']
    matching = [v for v in report['violations'] if scope == 'unknown' or re.search(
        r'per[ _]*day|/d(?:ay)?\b' if scope == 'day' else r'per[ _]*minute|/min\b',
        str(v.get('quotaId') or '') + ' ' + str(v.get('quotaMetric') or '') + ' ' + str(v.get('quotaDimensions') or ''),
        re.IGNORECASE)]
    metrics = ' '.join(str(v.get('quotaMetric') or v.get('quotaId') or '') for v in matching) \
        or str(getattr(exc, 'message', '') or '')
    tokens, requests = bool(re.search(r'token', metrics, re.I)), bool(re.search(r'request', metrics, re.I))
    unit = 'tokens' if tokens and not requests else 'requests' if requests and not tokens else ''
    if scope == 'day':
        name = {'tokens': 'TPD', 'requests': 'RPD'}.get(unit, 'тип метрики не указан')
        return 'quota', next_pacific_midnight(now), f'исчерпана суточная квота ({name}); сброс в полночь по Тихоокеанскому времени'
    retry_s = report['retry_s']
    if scope == 'unknown' and retry_s is None:
        return 'quota', 0.0, 'Google вернул 429 без типа квоты и срока повтора; пауза не назначена'
    wait = max(retry_s, 1.0) if retry_s is not None else DEFAULT_MINUTE_COOLDOWN_S
    if scope == 'minute':
        name = {'tokens': 'TPM', 'requests': 'RPM'}.get(unit, 'тип метрики не указан')
        return 'quota', now + wait, f'исчерпана минутная квота ({name}); повтор через {format_wait(wait)}'
    return 'quota', now + wait, f'Google вернул 429 без типа квоты; повтор через {format_wait(wait)}'


def _looks_like_bad_key(code, message, reasons):
    if code not in (400, 401, 403):
        return False
    text = (message or '').lower()
    markers = ('api key not valid', 'api_key_invalid', 'invalid api key', 'api key expired',
               'api key was reported as leaked')
    return any(m in text for m in markers) or bool(set(reasons) & {'API_KEY_INVALID', 'API_KEY_EXPIRED'})


def classify_gemini(exc, model, now=None):
    failure = _classify_gemini(exc, model, now)
    if failure is not None:
        items = gemini_error_items(exc)
        failure.detail = f"status={getattr(exc, 'status', None)}; details={json.dumps(items, ensure_ascii=False)}"
        if failure.report:
            failure.detail += f'; report={json.dumps(failure.report, ensure_ascii=False)}'
    return failure


def _classify_gemini(exc, model, now=None):
    if not _is_genai_error(exc):
        return None
    now = time.time() if now is None else now
    code = getattr(exc, 'code', None)
    status = getattr(exc, 'status', None)
    raw = getattr(exc, 'message', None) or str(exc)
    live = code in LIVE_CLOSE_CODES
    if live:
        # The Live API closes the websocket instead of answering with an HTTP status.
        raw = f'Live API {code}: {exc.details if isinstance(exc.details, str) else raw}'
        code = 429 if code == 1011 and 'quota' in raw.lower() else LIVE_CLOSE_CODES[code]
    items = gemini_error_items(exc)
    reasons = {str(item.get('reason') or '') for item in items}
    if code == 429:
        report = quota_report(exc)
        if live and report['scope'] == 'unknown':
            # Live models have no daily quota on the free tier, only TPM.
            report['scope'] = 'minute'
        kind, until, reason = _gemini_quota(exc, report, now)
        # Google may name an internal alias; the requested model is what gets paused.
        return Failure(kind, f'Gemini 429: {reason}.', code, model, until, report, raw)
    if _looks_like_bad_key(code, raw, reasons):
        return Failure('bad_key', f'Gemini {code}: Google отклонил ключ ({raw}).', code, '*', 0.0, raw=raw)
    model_error = reasons & {'MODEL_NOT_FOUND', 'MODEL_ACCESS_DENIED', 'MODEL_NOT_AVAILABLE'} or re.search(
        r'(?:\bmodels?[-/\s][\w.\-]+|\bmodel\b)[^\n]*(?:not found|not available|no longer available|not supported|'
        r'does not exist|access denied|permission denied)', raw, re.IGNORECASE) or (
        code in (403, 404) and re.search(r'\bno longer available to new users\b|\bnot available to new users\b',
                                          raw, re.IGNORECASE))
    if code in (403, 404) and model_error or (code == 400 and status == 'FAILED_PRECONDITION'):
        explanation = 'модель недоступна этому проекту'
        if 'new users' in raw.lower():
            explanation += '; Google закрыл её для новых проектов — выберите более новую модель'
        elif status == 'FAILED_PRECONDITION':
            explanation = 'проекту недоступен этот запрос (регион, биллинг или доступ к модели)'
        return Failure('model_unavailable', f'Gemini {code}: {explanation}: {raw}. Перепроверка через 5 мин.',
                       code, model, now + MODEL_ACCESS_COOLDOWN_S, raw=raw)
    if code in (500, 502, 503, 504):
        hint = {503: 'сервис перегружен', 504: 'модель не успела ответить'}.get(code, 'внутренняя ошибка сервера')
        return Failure('server', f'Gemini {code}: {hint} ({raw}).', code, raw=raw)
    if code == 404:
        return Failure('bad_request', f'Gemini 404: модель или ресурс не найдены ({raw}).', code, raw=raw)
    details = f' details={json.dumps(items, ensure_ascii=False)}' if items else ''
    return Failure('bad_request' if code and 400 <= code < 500 else 'other',
                   f'Gemini {code} {status or ""}: {raw}.{details}'.replace('  ', ' '), code, raw=raw)


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------

def _openai_wait_seconds(exc):
    """retry-after header, or 'Please try again in 1.5s / 20s / 1m30s' in the text."""
    headers = getattr(getattr(exc, 'response', None), 'headers', None)
    raw = headers.get('retry-after') if headers else None
    if raw and str(raw).strip().replace('.', '', 1).isdigit():
        return float(raw)
    text = str(getattr(exc, 'message', '') or exc)
    match = re.search(r'try again in\s*(?:(\d+)m)?\s*(\d+(?:\.\d+)?)\s*(ms|s)\b', text, re.IGNORECASE)
    if match:
        value = float(match.group(2))
        return int(match.group(1) or 0) * 60 + (value / 1000.0 if match.group(3).lower() == 'ms' else value)
    return None


def classify_openai(exc, model, now=None):
    try:
        import openai
    except ImportError:
        return None
    now = time.time() if now is None else now
    if isinstance(exc, openai.APITimeoutError):
        return Failure('timeout', 'OpenAI не ответил до таймаута запроса.', raw=str(exc))
    if isinstance(exc, openai.APIConnectionError):
        return Failure('network', 'Не удалось связаться с OpenAI.', raw=str(exc))
    if not isinstance(exc, openai.APIStatusError):
        return None
    failure = _classify_openai_status(exc, model, now)
    failure.detail = f"type={getattr(exc, 'type', None)}; code={getattr(exc, 'code', None)}"
    return failure


def _classify_openai_status(exc, model, now):
    code = getattr(exc, 'status_code', None)
    raw = getattr(exc, 'message', None) or str(exc)
    err_code = str(getattr(exc, 'code', '') or '')
    err_type = str(getattr(exc, 'type', '') or '')
    if code == 429:
        if err_type == 'insufficient_quota' or err_code in ('insufficient_quota', 'credit_balance_exhausted') \
                or 'insufficient_quota' in raw:
            return Failure('no_credits', 'OpenAI 429: нет кредитов (insufficient_quota) — пополните баланс на '
                           'platform.openai.com/settings/organization/billing. Перепроверка через 30 мин.',
                           code, '*', now + NO_CREDITS_COOLDOWN_S, raw=raw)
        wait = _openai_wait_seconds(exc) or DEFAULT_MINUTE_COOLDOWN_S
        return Failure('quota', f'OpenAI 429: лимит запросов (RPM/TPM), повтор через {format_wait(wait)}.',
                       code, model, now + max(wait, 1.0), raw=raw)
    if code == 401:
        return Failure('bad_key', f'OpenAI 401: ключ отклонён ({raw}).', code, '*', raw=raw)
    if code in (403, 404):
        return Failure('model_unavailable', f'OpenAI {code}: модель недоступна этому ключу или региону ({raw}). '
                       'Перепроверка через 5 мин.', code, model, now + MODEL_ACCESS_COOLDOWN_S, raw=raw)
    if code is not None and code >= 500:
        return Failure('server', f'OpenAI {code}: ошибка сервера ({raw}).', code, raw=raw)
    return Failure('bad_request', f'OpenAI {code}: {raw}', code, raw=raw)


def classify(exc, provider, model, now=None):
    """Failure for a known API/network error, None for anything else (a bug to re-raise)."""
    failure = classify_openai(exc, model, now) if provider == 'openai' else classify_gemini(exc, model, now)
    return failure or _network_failure(exc)
