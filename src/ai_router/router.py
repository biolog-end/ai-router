"""The attempt loop: model chain → keys of the model's provider → one request each.

Every public entry point (call, stream_call, generate, stream) runs the same generator,
`_execute`, which yields progress events and finally {'type': 'done', 'result': Result}.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import gemini_budget as gb
import openai_budget as ob

from . import catalog, keys, live, request as rq, state, voices
from .errors import (NEXT_KEY, RETRY_SAME_KEY, AllFailedError, Attempt, BudgetError, EmptyResponseError,
                     ERROR_CLASSES, KIND_LABELS, KeysBusyError, NoKeysError, RequestError, RouterError,
                     SafetyError, StoppedError, classify, format_wait)

log = logging.getLogger('ai_router')

SERVER_RETRY_DELAYS = (2, 4, 8)
DEFAULT_TIMEOUT_S = 300
DEFAULT_EXPECTED_OUTPUT = 1200

_clients = {}
_clients_lock = threading.Lock()


@dataclass
class Result:
    """A successful answer plus how the router got it. Never contains a secret."""
    text: str = ''
    data: Any = None
    thoughts: str = ''
    model: str = ''
    provider: str = ''
    key_id: str = None
    key_label: str = None
    response: Any = None
    usage: dict = field(default_factory=dict)
    finish_reason: str = ''
    truncated: bool = False
    sources: list = field(default_factory=list)
    response_id: str = None
    position: int = 1
    attempts: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def fallback_used(self):
        return self.position > 1

    def to_dict(self):
        return {name: value for name, value in self.__dict__.items() if name not in ('response', 'data', 'attempts')} | {
            'attempts': [attempt.to_dict() for attempt in self.attempts]}


@dataclass
class Context:
    """What the request function needs to know about the current attempt."""
    model: str
    provider: str
    key_id: str
    key_label: str
    position: int
    total: int


@dataclass
class _Run:
    """Routing options shared by all entry points."""
    key_id: str = None
    prefer_key: str = None
    try_all_keys: bool = False
    retry_server_errors: bool = True
    allow_paid: bool = False
    strict_budget: bool = False
    stop_event: Any = None
    input_tokens: int = None
    openai_estimate: int = None
    expected_output_tokens: int = DEFAULT_EXPECTED_OUTPUT
    timeout_s: float = DEFAULT_TIMEOUT_S
    accept: Any = None
    streaming: bool = False
    pin_keys: dict = None


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------

def client_for(provider, secret, timeout_s=DEFAULT_TIMEOUT_S):
    """Cached SDK client of one key. SDK retries are off: they would hide 429/5xx."""
    cache_key = (provider, keys.fingerprint(secret), int(timeout_s or DEFAULT_TIMEOUT_S))
    with _clients_lock:
        client = _clients.get(cache_key)
        if client is None:
            if provider == 'openai':
                import openai
                client = openai.OpenAI(api_key=secret, max_retries=0, timeout=float(timeout_s or DEFAULT_TIMEOUT_S))
            else:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=secret, http_options=types.HttpOptions(
                    timeout=int(float(timeout_s or DEFAULT_TIMEOUT_S) * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1)))
            _clients[cache_key] = client
        return client


def any_client(provider):
    """A client of the first enabled key, e.g. for models.list(); None without keys."""
    for entry in keys.entries(provider):
        if entry.get('enabled', True):
            return client_for(provider, entry['key'])
    return None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _event(event_type, text, **data):
    return {'type': event_type, 'text': text, **data}


def _scrub(text, secret):
    return str(text).replace(secret, '[КЛЮЧ]') if secret else str(text)


def _models(model, chain):
    if chain is True:
        return catalog.default_chain()
    if chain:
        result = [catalog.canonical(m) for m in ([chain] if isinstance(chain, str) else chain) if catalog.canonical(m)]
        if not result:
            raise ValueError('Цепочка моделей пуста.')
        return result
    return [catalog.canonical(model) or catalog.default_model('gemini')]


def _check_stop(run, model, provider, attempts):
    if run.stop_event is not None and run.stop_event.is_set():
        raise StoppedError('Генерация остановлена.', model=model, provider=provider, attempts=attempts)


def _wait(run, seconds):
    if run.stop_event is not None:
        return run.stop_event.wait(seconds)
    time.sleep(seconds)
    return False


def _openai_budget_check(model, run, notes):
    """Raises BudgetError when the free pool would be exceeded and paid requests are not allowed."""
    if ob.group_for(model) is None:
        return
    estimate = run.openai_estimate
    if estimate is None:
        estimate = int(run.input_tokens or 0) + int(run.expected_output_tokens or 0)
    try:
        ok, info = ob.can_spend(model, estimate)
    except Exception as exc:
        if run.strict_budget:
            raise BudgetError(f'Бесплатный остаток OpenAI не удалось проверить: {exc}', model=model,
                              provider='openai') from exc
        notes.append(f'Бесплатный остаток OpenAI не проверен: {exc}')
        return
    if ok:
        return
    if run.allow_paid:
        notes.append(f"OpenAI: {info.get('reason')}. Платные запросы разрешены — запрос идёт платно.")
        return
    raise BudgetError(f"Запрос к OpenAI отменён: {info.get('reason') or 'бесплатный лимит исчерпан'}",
                      model=model, provider='openai', retry_at=time.time() + int(info.get('reset_in_s') or 0))


def _blocker_text(blocker):
    if blocker.get('kind') == 'bad_key':
        return f"ключ отклонён: {blocker.get('reason')}"
    wait = f"; освободится через {format_wait(blocker['until'] - time.time())}" if blocker.get('until') else ''
    return f"{blocker.get('reason') or KIND_LABELS.get(blocker.get('kind'), '')}{wait}"


def _candidates(model, provider, run, tracked):
    """(usable entries in order, blocked [(entry, blocker)])."""
    entries = keys.entries(provider)
    key_id = (run.pin_keys or {}).get(model) or run.key_id
    if key_id:
        pinned = [entry for entry in entries if entry['id'] == key_id]
        if not pinned:
            raise NoKeysError(f'Ключ {key_id} не найден в хранилище ai_router.', model=model, provider=provider)
        return pinned, []
    enabled = [entry for entry in entries if entry.get('enabled', True)]
    if not entries:
        raise NoKeysError(f'Нет ключей {keys.LABELS[provider]}. Добавьте ключ в хранилище ai_router '
                          f'(интерфейс программы или «ai-router add {provider} КЛЮЧ»).', model=model, provider=provider)
    if not enabled:
        raise NoKeysError(f'Все ключи {keys.LABELS[provider]} выключены. Включите хотя бы один.',
                          model=model, provider=provider)
    now = time.time()
    snapshot = state.snapshot([keys.fingerprint(entry['key']) for entry in enabled])
    usable, blocked = [], []
    for entry in enabled:
        row = snapshot[keys.fingerprint(entry['key'])]
        blocker = None
        if row['rejected']:
            blocker = {'model': '*', 'kind': 'bad_key', 'until': None, **row['rejected']}
        else:
            for scope in ('*', model):
                pause = row['cooldowns'].get(scope)
                if pause and pause['until'] > now:
                    blocker = {'model': scope, **pause}
                    break
        if blocker:
            blocked.append((entry, blocker))
        else:
            usable.append((row['last_used'] or 0, entry))
    usable.sort(key=lambda pair: pair[0])
    ordered = [entry for _, entry in usable]
    if run.prefer_key:
        ordered.sort(key=lambda entry: entry['id'] != run.prefer_key)
    return ordered, blocked


def _busy_error(model, provider, blocked, attempts):
    label = keys.LABELS[provider]
    waits = [b for _, b in blocked if b.get('until')] + [{'until': a.until, 'reason': a.message}
                                                        for a in attempts if a.until]
    soonest = min(waits, key=lambda b: b['until']) if waits else None
    reasons = '; '.join(f"{entry['label']}: {_blocker_text(b)}" for entry, b in blocked)
    skipped = '; '.join(a.message for a in attempts if a.outcome == 'skipped')
    detail = '; '.join(part for part in (reasons, skipped) if part)
    message = f'Модель {model} сейчас недоступна ни на одном ключе {label}: {detail or "причина неизвестна"}.'
    if soonest:
        message += f" Ближайшая попытка через {format_wait(soonest['until'] - time.time())}."
    return KeysBusyError(message, model=model, provider=provider, attempts=attempts,
                         retry_at=soonest['until'] if soonest else None)


# --------------------------------------------------------------------------
# One model, all its keys
# --------------------------------------------------------------------------

def _run_model(model, provider, position, total, perform, run, notes, attempts):
    """Yields events; returns Result. Raises a RouterError when the model cannot answer."""
    retired = catalog.is_retired(model)
    if retired:
        raise RequestError(f"Модель '{model}' больше не поддерживается Google. {retired}", model=model,
                           provider=provider, attempts=attempts)
    if provider == 'openai':
        _openai_budget_check(model, run, notes)
    tracked = provider == 'gemini' and gb.has_free_quota(model)
    usable, blocked = _candidates(model, provider, run, tracked)
    for entry, blocker in blocked:
        attempts.append(Attempt(model, provider, entry['id'], entry['label'], 'skipped', blocker.get('kind'),
                                f"{entry['label']}: {_blocker_text(blocker)}", blocker.get('until')))
    if not usable:
        raise _busy_error(model, provider, blocked, [])

    fast_failover = run.try_all_keys or not run.retry_server_errors
    retry_delays = [] if fast_failover else list(SERVER_RETRY_DELAYS)
    model_attempts = []
    for entry in usable:
        secret, label, fp = entry['key'], entry['label'], keys.fingerprint(entry['key'])
        while True:
            _check_stop(run, model, provider, attempts)
            reservation = None
            if tracked:
                try:
                    reservation, reason = gb.reserve(secret, model, int(run.input_tokens or 0))
                except Exception as exc:
                    raise RouterError(f'Общий учёт gemini_budget недоступен: {exc}', kind='other', model=model,
                                      provider=provider, attempts=attempts) from exc
                if reservation is None:
                    attempt = Attempt(model, provider, entry['id'], label, 'skipped', 'budget', f'{label}: {reason}')
                    attempts.append(attempt)
                    model_attempts.append(attempt)
                    yield _event('skip', attempt.message, model=model, provider=provider, key_id=entry['id'],
                                 key_label=label, kind='budget')
                    break

            ctx = Context(model, provider, entry['id'], label, position, total)
            yield _event('attempt', f'Запрос {model}, ключ «{label}».', model=model, provider=provider,
                         key_id=entry['id'], key_label=label, position=position, total=total)
            started = time.time()
            emitted, final, usage_event = False, None, {}
            iterator = None
            try:
                client = client_for(provider, secret, run.timeout_s)
                iterator = perform(client, ctx)
                for event in iterator:
                    kind = event.get('type')
                    if kind == '_final':
                        final = event
                    elif kind == 'usage':
                        usage_event = event
                    else:
                        emitted |= kind == 'delta' and bool(event.get('text') or event.get('audio'))
                        yield event
                if final is None:
                    final = {'response': None, 'parsed': None}
                parsed = final.get('parsed')
                accepted = parsed is not None or run.accept is None or bool(run.accept(final.get('response')))
                if not accepted:
                    raise EmptyResponseError(f'Модель {model} не вернула готового ответа.', model=model,
                                             provider=provider)
            except GeneratorExit:
                # The consumer stopped listening; whatever was streamed is already billed.
                if tracked:
                    gb.finish(secret, model, reservation, success=emitted)
                raise
            except (SafetyError, EmptyResponseError) as exc:
                if tracked:
                    gb.finish(secret, model, reservation, success=False)
                message = _scrub(exc.message, secret)
                state.record(fp, provider, model, ok=False, error=message)
                attempt = Attempt(model, provider, entry['id'], label, 'failed', exc.kind, f'{label}: {message}',
                                  duration_s=time.time() - started)
                attempts.append(attempt)
                yield _event('failure', attempt.message, model=model, provider=provider, key_id=entry['id'],
                             key_label=label, kind=exc.kind)
                # Another key gets the same answer from the same model.
                raise type(exc)(message, model=model, provider=provider, attempts=attempts) from exc
            except Exception as exc:
                failure = classify(exc, provider, model)
                if failure is None:
                    if tracked:
                        gb.finish(secret, model, reservation, success=False)
                    raise
                message = _scrub(failure.message, secret)
                if tracked:
                    budget_failure = None
                    if failure.kind in ('quota', 'zero_quota', 'model_unavailable') and failure.until:
                        budget_failure = {'kind': 'model_unavailable' if failure.kind == 'model_unavailable' else 'quota',
                                          'until': failure.until, 'reason': message, 'report': failure.report or {}}
                    gb.finish(secret, model, reservation, success=emitted, failure=budget_failure)
                state.record(fp, provider, model, ok=False, error=message)
                attempt = Attempt(model, provider, entry['id'], label, 'failed', failure.kind, f'{label}: {message}',
                                  failure.until or None, duration_s=time.time() - started,
                                  detail=_scrub(failure.detail, secret))
                attempts.append(attempt)
                model_attempts.append(attempt)
                log.warning('ai_router %s, ключ «%s»: %s', model, label, message)
                yield _event('failure', attempt.message, model=model, provider=provider, key_id=entry['id'],
                             key_label=label, kind=failure.kind, until=failure.until or None)
                if emitted:
                    raise RouterError(f'Ответ оборвался: {message}', kind=failure.kind, model=model,
                                      provider=provider, attempts=attempts) from exc

                if failure.kind == 'bad_key':
                    state.reject(fp, message)
                    break
                if failure.kind in NEXT_KEY:
                    # Free Gemini models are paused per model by gemini_budget itself.
                    if failure.scope and failure.until > time.time() and not (tracked and failure.scope == model):
                        state.pause(fp, failure.scope, failure.until, failure.kind, message)
                    break
                if failure.kind in RETRY_SAME_KEY:
                    if retry_delays:
                        wait = retry_delays.pop(0)
                        yield _event('retry', f'{label}: повтор через {wait} с.', model=model, provider=provider,
                                     key_id=entry['id'], key_label=label, wait_s=wait)
                        if _wait(run, wait):
                            raise StoppedError('Генерация остановлена.', model=model, provider=provider,
                                               attempts=attempts)
                        continue
                    break
                if run.try_all_keys:
                    break
                error_class = ERROR_CLASSES.get(failure.kind, RequestError)
                raise error_class(message, kind=failure.kind, model=model, provider=provider,
                                  attempts=attempts) from exc

            response, parsed = final.get('response'), final.get('parsed') or {}
            usage = parsed.get('usage') or rq.usage_of(response) or {
                k: usage_event.get(k) for k in ('input_tokens', 'output_tokens', 'total_tokens') if k in usage_event}
            if tracked:
                gb.finish(secret, model, reservation, success=True, input_tokens=usage.get('input_tokens'))
            if provider == 'openai':
                try:
                    if response is not None and getattr(response, 'usage', None) is not None:
                        ob.record_response(model, response)
                    elif usage.get('input_tokens') is not None or usage.get('output_tokens') is not None:
                        ob.record(model, usage.get('input_tokens') or 0, usage.get('output_tokens') or 0,
                                  usage.get('cached_tokens') or 0)
                except Exception as exc:
                    notes.append(f'Ответ получен, но расход не записан в openai_budget: {exc}')
            state.record(fp, provider, model, ok=True, tokens=usage.get('total_tokens') or 0)
            attempt = Attempt(model, provider, entry['id'], label, 'ok', message=f'{label}: ответ получен.',
                              duration_s=time.time() - started)
            attempts.append(attempt)
            yield _event('success', f'Ответ {model} получен (ключ «{label}»).', model=model, provider=provider,
                         key_id=entry['id'], key_label=label)
            return Result(text=parsed.get('text') if 'text' in parsed else rq.text_of(response),
                          data=parsed.get('data'), thoughts=parsed.get('thoughts', ''), model=model,
                          provider=provider, key_id=entry['id'], key_label=label, response=response, usage=usage,
                          finish_reason=parsed.get('finish_reason', ''), truncated=parsed.get('truncated', False),
                          sources=parsed.get('sources', []), response_id=parsed.get('response_id'),
                          position=position, notes=notes)

    if all(attempt.outcome == 'skipped' for attempt in model_attempts):
        raise _busy_error(model, provider, blocked, model_attempts)
    detail = '; '.join(attempt.message for attempt in model_attempts)
    waits = [attempt.until for attempt in model_attempts if attempt.until]
    raise AllFailedError(f'Все доступные ключи {keys.LABELS[provider]} для {model} не сработали. {detail}',
                         model=model, provider=provider, attempts=attempts, retry_at=min(waits) if waits else None)


def _execute(models, perform, run, notes):
    """Yields events and finally {'type': 'done', 'result': Result}; raises a RouterError."""
    attempts, errors = [], []
    total = len(models)
    for position, model in enumerate(models, 1):
        provider = catalog.provider_for(model)
        if total > 1:
            yield _event('model', f'Модель {position}/{total}: {model}.', model=model, provider=provider,
                         position=position, total=total)
        emitted = False
        try:
            runner = _run_model(model, provider, position, total, perform, run, notes, attempts)
            while True:
                try:
                    event = next(runner)
                except StopIteration as stop:
                    result = stop.value
                    break
                emitted |= event.get('type') == 'delta'
                yield event
            result.attempts = list(attempts)
            yield {'type': 'done', 'result': result}
            return
        except StoppedError:
            raise
        except RouterError as exc:
            exc.attempts = list(attempts)
            if emitted or total == 1:
                raise
            errors.append(exc)
            if position < total:
                yield _event('next_model', f'{model}: {exc.message} Пробую следующую модель.', model=model,
                             provider=provider, kind=exc.kind)
    waits = [error.retry_at for error in errors if error.retry_at]
    raise AllFailedError('Все модели цепочки не сработали. ' + '; '.join(f'{e.model}: {e.message}' for e in errors),
                         attempts=attempts, retry_at=min(waits) if waits else None, errors=errors)


def _drive(events, on_event):
    result = None
    for event in events:
        if event['type'] == 'done':
            result = event['result']
        elif on_event is not None:
            on_event(event)
    return result


def _stream(events, on_event):
    for event in events:
        if on_event is not None and event['type'] not in ('delta', 'done'):
            on_event(event)
        yield event


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

def _routing(kwargs, streaming=False):
    names = {'key_id', 'prefer_key', 'try_all_keys', 'retry_server_errors', 'allow_paid', 'strict_budget',
             'stop_event', 'input_tokens', 'expected_output_tokens', 'timeout_s', 'accept'}
    run = _Run(streaming=streaming, **{name: kwargs.pop(name) for name in list(kwargs) if name in names})
    if run.timeout_s is None:
        run.timeout_s = DEFAULT_TIMEOUT_S
    return run


def call(fn, model=None, *, chain=None, on_event=None, **routing):
    """Runs fn(client, ctx) → raw SDK response with key rotation and an optional model chain.

    fn builds the provider's request itself (ctx.provider / ctx.model say which one).
    Routing options: key_id, prefer_key, try_all_keys, retry_server_errors, allow_paid,
    strict_budget, stop_event, input_tokens, expected_output_tokens, timeout_s,
    accept(response) -> bool. Returns Result (result.response is fn's return value).
    """
    run = _routing(routing)
    if routing:
        raise TypeError(f'Неизвестные параметры call(): {", ".join(routing)}')

    def perform(client, ctx):
        yield {'type': '_final', 'response': fn(client, ctx), 'parsed': None}

    return _drive(_execute(_models(model, chain), perform, run, []), on_event)


def stream_call(fn, model=None, *, chain=None, on_event=None, **routing):
    """Like call(), but fn(client, ctx) yields events: {'type': 'delta', 'text'}, optionally
    {'type': 'usage', input_tokens, output_tokens, total_tokens} and {'type': 'response', 'response'};
    anything else is passed through. Keys and models switch only before the first delta.
    """
    run = _routing(routing, streaming=True)
    if routing:
        raise TypeError(f'Неизвестные параметры stream_call(): {", ".join(routing)}')

    def perform(client, ctx):
        response = None
        for event in fn(client, ctx):
            if event.get('type') == 'response':
                response = event.get('response')
                continue
            yield event
        yield {'type': '_final', 'response': response, 'parsed': None}

    return _stream(_execute(_models(model, chain), perform, run, []), on_event)


def _prepare(prompt, messages, params):
    run = _routing(params)
    options = rq.Options(**{name: params.pop(name) for name in list(params) if name in rq.Options.__dataclass_fields__})
    if params:
        raise TypeError(f'Неизвестные параметры: {", ".join(params)}')
    neutral = rq.normalize_messages(messages, prompt)
    if not neutral:
        raise ValueError('Пустой запрос: нет ни prompt, ни messages.')
    options.allow_paid, options.timeout_s = run.allow_paid, run.timeout_s
    if run.input_tokens is None:
        run.input_tokens = rq.estimate_gemini_input(neutral, options.system)
    run.openai_estimate = rq.estimate_openai(neutral, options.system,
                                             options.max_output_tokens or run.expected_output_tokens)
    return neutral, options, run


def _with_audio(result, options):
    """SpeechResult when a Live model answered with audio=True; a note when another model answered."""
    if not (options.audio or options.voice):
        return result
    if not live.is_live_model(result.model):
        result.notes.append('Звук (audio/voice) есть только у Live-моделей — ответ без звука.')
        return result
    if not options.audio:
        return result
    from .tts import _speech_result, wav
    voice = voices.canonical(options.voice) if options.voice and voices.is_prebuilt(options.voice) else None
    return _speech_result(result, wav(result.data or b'', live.SAMPLE_RATE), 'wav', live.SAMPLE_RATE,
                          {'': voice} if voice else {})


def _finish_audio(events, options):
    for event in events:
        if event['type'] == 'done':
            event = {'type': 'done', 'result': _with_audio(event['result'], options)}
        yield event


def generate(prompt=None, *, model=None, chain=None, messages=None, on_event=None, **params):
    """One text (or JSON) answer from Gemini or OpenAI, built from a neutral request.

    Live models (gemini-3.8-live, …) answer here too; audio=True (and voice=) on them returns
    a SpeechResult with the spoken answer. See README for every parameter; unsupported ones
    are dropped and listed in result.notes.
    """
    neutral, options, run = _prepare(prompt, messages, params)
    notes = []

    def perform(client, ctx):
        if live.is_live_model(ctx.model):
            yield from live.perform(client, ctx, neutral, options, notes, run, streaming=False)
        elif ctx.provider == 'openai':
            method, kwargs = rq.openai_request(ctx.model, neutral, options, notes)
            response = getattr(client.responses, method)(**kwargs)
            yield {'type': '_final', 'response': response, 'parsed': rq.parse_openai(response, ctx.model, options, method)}
        else:
            response = client.models.generate_content(**rq.gemini_request(ctx.model, neutral, options, notes))
            yield {'type': '_final', 'response': response, 'parsed': rq.parse_gemini(response, ctx.model, options)}

    return _with_audio(_drive(_execute(_models(model, chain), perform, run, notes), on_event), options)


def stream(prompt=None, *, model=None, chain=None, messages=None, on_event=None, **params):
    """Streaming generate(): yields {'type': 'delta', 'text'}, progress events and a final
    {'type': 'done', 'result': Result}. Keys and models switch only before the first delta.
    A Live model with audio=True also yields {'type': 'delta', 'audio': PCM, 'sample_rate'}.
    """
    neutral, options, run = _prepare(prompt, messages, params)
    if options.json_schema is not None:
        raise ValueError('Структурированный ответ (json_schema) не поддерживается в потоковом режиме.')
    run.streaming = True
    notes = []

    def perform(client, ctx):
        if live.is_live_model(ctx.model):
            yield from live.perform(client, ctx, neutral, options, notes, run, streaming=True)
            return
        if ctx.provider == 'openai':
            _, kwargs = rq.openai_request(ctx.model, neutral, options, notes)
            with client.responses.stream(**kwargs) as events:
                for event in events:
                    if getattr(event, 'type', '') == 'response.output_text.delta' and event.delta:
                        yield {'type': 'delta', 'text': event.delta}
                response = events.get_final_response()
            yield {'type': '_final', 'response': response, 'parsed': rq.parse_openai(response, ctx.model, options)}
            return
        request = rq.gemini_request(ctx.model, neutral, options, notes)
        text, finish, usage, blocked = [], '', None, ''
        for chunk in client.models.generate_content_stream(**request):
            delta, chunk_finish, chunk_usage, chunk_blocked = rq.gemini_chunk(chunk)
            finish, usage, blocked = chunk_finish or finish, chunk_usage or usage, chunk_blocked or blocked
            if delta:
                text.append(delta)
                yield {'type': 'delta', 'text': delta}
        if blocked:
            raise SafetyError(f'Заблокировано Gemini ({blocked}).', model=ctx.model, provider='gemini')
        if finish in rq.SAFETY_FINISH_REASONS:
            raise SafetyError(f'Ответ остановлен фильтром безопасности Gemini. Причина завершения: {finish}',
                              model=ctx.model, provider='gemini')
        joined = ''.join(text).strip()
        if not joined:
            raise EmptyResponseError(f'Модель {ctx.model} вернула пустой ответ.'
                                     + (f' Причина завершения: {finish}.' if finish else ''),
                                     model=ctx.model, provider='gemini')
        yield {'type': '_final', 'response': None, 'parsed': {
            'text': joined, 'finish_reason': finish, 'truncated': finish == 'MAX_TOKENS',
            'usage': rq._gemini_usage(usage)}}

    return _finish_audio(_stream(_execute(_models(model, chain), perform, run, notes), on_event), options)
