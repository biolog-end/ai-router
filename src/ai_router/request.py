"""Neutral request → Gemini GenerateContent / OpenAI Responses payloads, and responses back.

Messages are a list of {'role': 'user'|'model'|'assistant'|'developer', 'parts': [...]};
a part is {'text': …} or binary data in any of the common shapes (bytes, base64,
data-URL, *_base64 keys of older projects) plus google.genai Content/Part objects.
"""

import base64
import json
from dataclasses import dataclass, field
from typing import Any

import gemini_budget as gb
import openai_budget as ob

from .errors import EmptyResponseError, SafetyError

SAFETY_FINISH_REASONS = ('SAFETY', 'PROHIBITED_CONTENT', 'IMAGE_SAFETY', 'BLOCKLIST', 'SPII')
GEMINI_MEDIA_TOKENS = 1024
OPENAI_PLACEHOLDERS = {
    'audio': '[голосовое сообщение — эта модель его не воспринимает]',
    'video': '[видео — эта модель его не воспринимает]',
    'other': '[вложение]',
}
_DEFAULT_MIME = {'image_base64': 'image/jpeg', 'file_base64': 'application/pdf', 'audio_base64': 'audio/ogg',
                 'video_base64': 'video/mp4', 'image': 'image/jpeg', 'file': 'application/pdf',
                 'audio': 'audio/ogg', 'video': 'video/mp4'}
_ROLES = {'user': 'user', 'model': 'model', 'assistant': 'model', 'developer': 'developer', 'system': 'developer'}


@dataclass
class Options:
    """Generation parameters of generate()/stream(); None means "provider default"."""
    system: str = None
    temperature: float = None
    top_p: float = None
    top_k: int = None
    max_output_tokens: int = None
    stop: Any = None
    seed: int = None
    presence_penalty: float = None
    frequency_penalty: float = None
    thinking: Any = None
    include_thoughts: bool = False
    search: bool = False
    json_schema: Any = None
    json_mode: bool = False
    safety: Any = None
    cache_key: str = None
    image_detail: str = 'auto'
    timeout_s: float = None
    allow_paid: bool = False
    gemini: dict = field(default_factory=dict)
    openai: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Neutral messages
# --------------------------------------------------------------------------

def _decode(value, mime=None):
    """bytes | base64 | data-URL → (bytes, mime)."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), mime
    text = str(value)
    if text.startswith('data:'):
        header, _, payload = text.partition(',')
        return base64.b64decode(payload), mime or header[5:].split(';', 1)[0] or None
    return base64.b64decode(text), mime


def _part(raw):
    """One part in any accepted shape → {'text'} | {'data', 'mime_type'} | {'uri', 'mime_type'} | None."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return {'text': raw} if raw else None
    if not isinstance(raw, dict):
        text = getattr(raw, 'text', None)
        if isinstance(text, str):
            return {'text': text} if text and not getattr(raw, 'thought', False) else None
        inline = getattr(raw, 'inline_data', None)
        if inline is not None and getattr(inline, 'data', None) is not None:
            return {'data': inline.data, 'mime_type': inline.mime_type}
        file_data = getattr(raw, 'file_data', None)
        if file_data is not None:
            return {'uri': file_data.file_uri, 'mime_type': file_data.mime_type}
        return None
    if raw.get('type') == 'input_text':
        return {'text': raw.get('text')} if raw.get('text') else None
    if 'text' in raw:
        return {'text': raw['text']} if raw['text'] else None
    for name in ('image_base64', 'file_base64', 'audio_base64', 'video_base64', 'image', 'file', 'audio', 'video', 'data'):
        if raw.get(name) is not None:
            data, mime = _decode(raw[name], raw.get('mime_type'))
            return {'data': data, 'mime_type': mime or _DEFAULT_MIME.get(name, 'application/octet-stream'),
                    'name': raw.get('name') or raw.get('filename')}
    inline = raw.get('inline_data')
    if isinstance(inline, dict) and inline.get('data') is not None:
        data, mime = _decode(inline['data'], inline.get('mime_type'))
        return {'data': data, 'mime_type': mime}
    url = raw.get('image_url')
    if url:
        if str(url).startswith('data:'):
            data, mime = _decode(url)
            return {'data': data, 'mime_type': mime or 'image/jpeg'}
        return {'uri': url, 'mime_type': raw.get('mime_type') or 'image/jpeg'}
    if raw.get('uri'):
        return {'uri': raw['uri'], 'mime_type': raw.get('mime_type') or 'application/octet-stream'}
    return None


def normalize_messages(messages, prompt=None):
    """Any accepted message list (+ an optional trailing user prompt) → neutral messages."""
    if isinstance(messages, str):
        messages = [{'role': 'user', 'parts': [{'text': messages}]}]
    result = []
    for item in messages or []:
        if isinstance(item, dict):
            role = _ROLES.get(str(item.get('role') or 'user').lower(), 'user')
            raw_parts = item.get('parts')
            if raw_parts is None:
                content = item.get('content')
                raw_parts = [content] if isinstance(content, str) else content or []
        else:
            role = _ROLES.get(str(getattr(item, 'role', None) or 'user').lower(), 'user')
            raw_parts = getattr(item, 'parts', None) or []
        parts = [part for part in map(_part, raw_parts) if part]
        if parts:
            result.append({'role': role, 'parts': parts})
    if prompt:
        result.append({'role': 'user', 'parts': [{'text': prompt}]})
    return result


def plain_text(messages, system=None):
    """All text of the request, for token estimates and logs."""
    chunks = [system or '']
    chunks += [part['text'] for message in messages for part in message['parts'] if 'text' in part]
    return '\n'.join(chunk for chunk in chunks if chunk)


def media_count(messages):
    return sum(1 for message in messages for part in message['parts'] if 'text' not in part)


def estimate_gemini_input(messages, system=None):
    """Pre-request TPM estimate; Google's prompt_token_count replaces it after the reply."""
    return (len(plain_text(messages, system).encode('utf-8')) + 2) // 3 + media_count(messages) * GEMINI_MEDIA_TOKENS


def estimate_openai(messages, system=None, expected_output=1200):
    images = sum(1 for message in messages for part in message['parts']
                 if str(part.get('mime_type') or '').startswith('image/'))
    return ob.estimate_tokens(plain_text(messages, system), images=images, expected_output=expected_output)


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------

def _level(thinking, style):
    if thinking is True or thinking == -1:
        return 'HIGH'
    if thinking in (False, None, 'off', 'none', 0):
        return style['off']
    if isinstance(thinking, (int, float)) and not isinstance(thinking, bool):
        return 'LOW' if thinking < 2048 else 'MEDIUM' if thinking < 8192 else 'HIGH'
    level = str(thinking).upper()
    if level == 'MINIMAL' and style['off'] != 'MINIMAL':
        return style['off']
    return level if level in ('MINIMAL', 'LOW', 'MEDIUM', 'HIGH') else 'HIGH'


def _budget(thinking, style):
    if thinking is True or thinking == 'high':
        return -1
    if thinking in (False, None, 'off', 'none', 'minimal', 0):
        return style['off']
    if isinstance(thinking, (int, float)) and not isinstance(thinking, bool):
        return -1 if thinking < 0 else max(int(thinking), style['off'])
    return {'low': 1024, 'medium': 8192}.get(str(thinking).lower(), -1)


def gemini_contents(messages):
    from google.genai import types
    contents = []
    for message in messages:
        parts = []
        for part in message['parts']:
            if 'text' in part:
                parts.append(types.Part.from_text(text=part['text']))
            elif 'data' in part:
                parts.append(types.Part.from_bytes(data=part['data'], mime_type=part['mime_type']))
            elif 'uri' in part:
                parts.append(types.Part.from_uri(file_uri=part['uri'], mime_type=part['mime_type']))
        if parts:
            contents.append(types.Content(role='model' if message['role'] == 'model' else 'user', parts=parts))
    return contents


def gemini_config(model, options, notes):
    from google.genai import types
    kw = {}
    if options.system:
        kw['system_instruction'] = options.system
    for name in ('temperature', 'top_p', 'top_k', 'max_output_tokens', 'seed', 'presence_penalty', 'frequency_penalty'):
        if getattr(options, name) is not None:
            kw[name] = getattr(options, name)
    if options.stop:
        kw['stop_sequences'] = [options.stop] if isinstance(options.stop, str) else list(options.stop)

    style = gb.thinking_style(model)
    if options.thinking is not None or options.include_thoughts:
        if style is None:
            if options.thinking not in (None, False, 'off', 'none', 0):
                notes.append(f'{model} не поддерживает размышления — параметр thinking пропущен.')
        else:
            thinking = {'include_thoughts': True} if options.include_thoughts else {}
            if options.thinking is not None:
                if style['style'] == 'level':
                    thinking['thinking_level'] = _level(options.thinking, style)
                else:
                    thinking['thinking_budget'] = _budget(options.thinking, style)
            kw['thinking_config'] = types.ThinkingConfig(**thinking)

    if options.search:
        info = gb.model_info(model)
        if info is not None and info['search'] is None:
            notes.append(f'У {model} нет Google Search в API — инструмент пропущен.')
        else:
            kw['tools'] = [types.Tool(google_search=types.GoogleSearch())]
            if info is not None and info['search'] == 'paid':
                notes.append(f'Google Search у {model} работает только на платном проекте; '
                             'на бесплатном ключе запрос может получить 429.')

    if options.json_schema is not None:
        kw['response_mime_type'] = 'application/json'
        kw['response_json_schema' if isinstance(options.json_schema, dict) else 'response_schema'] = options.json_schema
    elif options.json_mode:
        kw['response_mime_type'] = 'application/json'

    if options.safety == 'off':
        kw['safety_settings'] = [types.SafetySetting(category=category, threshold='OFF') for category in (
            'HARM_CATEGORY_HARASSMENT', 'HARM_CATEGORY_HATE_SPEECH',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'HARM_CATEGORY_DANGEROUS_CONTENT')]
    elif options.safety:
        kw['safety_settings'] = options.safety

    if options.timeout_s:
        kw['http_options'] = types.HttpOptions(timeout=int(float(options.timeout_s) * 1000),
                                               retry_options=types.HttpRetryOptions(attempts=1))
    kw.update(options.gemini or {})
    return types.GenerateContentConfig(**kw)


def gemini_request(model, messages, options, notes):
    """{'model', 'contents', 'config'} ready for client.models.generate_content(**request)."""
    return {'model': model, 'contents': gemini_contents(messages), 'config': gemini_config(model, options, notes)}


def _name(value):
    return getattr(value, 'name', None) or (str(value) if value else '')


def _gemini_usage(usage):
    if usage is None:
        return {}
    return {'input_tokens': getattr(usage, 'prompt_token_count', None),
            'output_tokens': getattr(usage, 'candidates_token_count', None),
            'thinking_tokens': getattr(usage, 'thoughts_token_count', None),
            'cached_tokens': getattr(usage, 'cached_content_token_count', None),
            'total_tokens': getattr(usage, 'total_token_count', None)}


def _gemini_sources(candidate):
    chunks = getattr(getattr(candidate, 'grounding_metadata', None), 'grounding_chunks', None) or []
    return [{'title': getattr(chunk.web, 'title', None), 'url': getattr(chunk.web, 'uri', None)}
            for chunk in chunks if getattr(chunk, 'web', None)]


def _json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def parse_gemini(response, model, options):
    """Text, data, usage… of a GenerateContentResponse; raises on safety blocks and empty replies."""
    candidates = getattr(response, 'candidates', None) or []
    if not candidates:
        feedback = getattr(response, 'prompt_feedback', None)
        if feedback is not None and getattr(feedback, 'block_reason', None):
            reason = getattr(feedback, 'block_reason_message', None) or _name(feedback.block_reason)
            raise SafetyError(f'Заблокировано Gemini ({_name(feedback.block_reason)}): {reason}', model=model,
                              provider='gemini')
        raise EmptyResponseError(f'Gemini ({model}) вернул ответ без кандидатов.', model=model, provider='gemini')
    candidate = candidates[0]
    finish = _name(getattr(candidate, 'finish_reason', None))
    parts = getattr(getattr(candidate, 'content', None), 'parts', None) or []
    text = ''.join(p.text for p in parts if isinstance(getattr(p, 'text', None), str) and not getattr(p, 'thought', False))
    thoughts = ''.join(p.text for p in parts if isinstance(getattr(p, 'text', None), str) and getattr(p, 'thought', False))
    if finish in SAFETY_FINISH_REASONS:
        raise SafetyError(f'Ответ остановлен фильтром безопасности Gemini. Причина завершения: {finish}',
                          model=model, provider='gemini')
    data = None
    if options is not None and (options.json_schema is not None or options.json_mode):
        data = getattr(response, 'parsed', None)
        if data is None:
            data = _json(text)
    if not text.strip() and data is None:
        reason = f'Модель {model} вернула пустой ответ.'
        if finish:
            reason += f' Причина завершения: {finish}.'
        if finish == 'MAX_TOKENS':
            reason += ' Лимит токенов израсходован до текста (часто на размышления).'
        raise EmptyResponseError(reason, model=model, provider='gemini')
    if options is not None and (options.json_schema is not None or options.json_mode) and data is None:
        raise EmptyResponseError(f'Модель {model} вернула не JSON.', model=model, provider='gemini')
    return {'text': text.strip(), 'thoughts': thoughts.strip(), 'data': data, 'finish_reason': finish,
            'truncated': finish == 'MAX_TOKENS', 'usage': _gemini_usage(getattr(response, 'usage_metadata', None)),
            'sources': _gemini_sources(candidate), 'response_id': getattr(response, 'response_id', None)}


def gemini_chunk(chunk):
    """(text delta, finish_reason, usage, blocked_reason) of one streamed chunk."""
    candidates = getattr(chunk, 'candidates', None) or []
    feedback = getattr(chunk, 'prompt_feedback', None)
    blocked = _name(feedback.block_reason) if feedback is not None and getattr(feedback, 'block_reason', None) else ''
    if not candidates:
        return '', '', getattr(chunk, 'usage_metadata', None), blocked
    candidate = candidates[0]
    parts = getattr(getattr(candidate, 'content', None), 'parts', None) or []
    text = ''.join(p.text for p in parts if isinstance(getattr(p, 'text', None), str) and not getattr(p, 'thought', False))
    return text, _name(getattr(candidate, 'finish_reason', None)), getattr(chunk, 'usage_metadata', None), blocked


# --------------------------------------------------------------------------
# OpenAI (Responses API)
# --------------------------------------------------------------------------

def openai_input(messages, image_detail='auto'):
    items = []
    for message in messages:
        if message['role'] == 'model':
            # Past assistant turns are accepted as plain text only.
            text = '\n'.join(part['text'] for part in message['parts'] if 'text' in part)
            if text:
                items.append({'role': 'assistant', 'content': text})
            continue
        content = []
        for part in message['parts']:
            mime = str(part.get('mime_type') or '')
            if 'text' in part:
                content.append({'type': 'input_text', 'text': part['text']})
            elif 'data' in part and mime.startswith('image/'):
                encoded = base64.b64encode(part['data']).decode('ascii')
                content.append({'type': 'input_image', 'detail': image_detail, 'image_url': f'data:{mime};base64,{encoded}'})
            elif 'data' in part and mime == 'application/pdf':
                encoded = base64.b64encode(part['data']).decode('ascii')
                content.append({'type': 'input_file', 'filename': part.get('name') or 'document.pdf',
                                'file_data': f'data:application/pdf;base64,{encoded}'})
            elif 'uri' in part and mime.startswith('image/') and str(part['uri']).startswith('http'):
                content.append({'type': 'input_image', 'detail': image_detail, 'image_url': part['uri']})
            else:
                kind = 'audio' if mime.startswith('audio/') else 'video' if mime.startswith('video/') else 'other'
                content.append({'type': 'input_text', 'text': OPENAI_PLACEHOLDERS[kind]})
        if content:
            items.append({'role': 'developer' if message['role'] == 'developer' else 'user', 'content': content})
    return items


def _effort(model, thinking, notes):
    if thinking is None:
        return None
    if not ob.supports_reasoning(model):
        if thinking not in (False, 'off', 'none', 0):
            notes.append(f'{model} не принимает reasoning — параметр thinking пропущен.')
        return None
    if thinking in (False, 'off', 0):
        return ob.reasoning_off(model)
    if thinking is True or thinking == -1:
        return 'high'
    if isinstance(thinking, (int, float)) and not isinstance(thinking, bool):
        return 'low' if thinking < 2048 else 'medium' if thinking < 8192 else 'high'
    return str(thinking).lower()


def openai_request(model, messages, options, notes):
    """(method, kwargs): method is 'create' or 'parse' (structured output with a pydantic model)."""
    kw = {'model': model, 'input': openai_input(messages, options.image_detail)}
    if options.system:
        kw['instructions'] = options.system
    if options.max_output_tokens is not None:
        kw['max_output_tokens'] = options.max_output_tokens
    effort = _effort(model, options.thinking, notes)
    reasoning = {}
    if effort:
        reasoning['effort'] = effort
    if options.include_thoughts and ob.supports_reasoning(model):
        reasoning['summary'] = 'auto'
    if options.temperature is not None or options.top_p is not None:
        mode = ob.temperature_mode(model)
        if mode == 'without_reasoning' and reasoning.get('effort') in (None, 'none'):
            reasoning['effort'] = 'none'
        if mode == 'always' or (mode == 'without_reasoning' and reasoning.get('effort') == 'none'):
            if options.temperature is not None:
                kw['temperature'] = options.temperature
            if options.top_p is not None:
                kw['top_p'] = options.top_p
        else:
            notes.append(f'{model} с текущими размышлениями не принимает temperature/top_p — параметр пропущен.')
    if reasoning:
        kw['reasoning'] = reasoning
    for name, label in (('top_k', 'top_k'), ('seed', 'seed'), ('stop', 'stop'),
                        ('presence_penalty', 'presence_penalty'), ('frequency_penalty', 'frequency_penalty'),
                        ('safety', 'safety')):
        if getattr(options, name):
            notes.append(f'OpenAI Responses API не принимает {label} — параметр пропущен.')
    if options.cache_key:
        kw['prompt_cache_key'] = str(options.cache_key)
    if options.search:
        if options.allow_paid:
            kw['tools'] = [{'type': 'web_search'}]
        else:
            notes.append('Веб-поиск OpenAI оплачивается отдельно от бесплатных токенов — пропущен '
                         '(разрешите платные запросы, чтобы включить).')
    method = 'create'
    if options.json_schema is not None:
        if isinstance(options.json_schema, dict):
            kw['text'] = {'format': {'type': 'json_schema', 'name': 'response',
                                     'schema': options.json_schema, 'strict': False}}
        else:
            method, kw['text_format'] = 'parse', options.json_schema
    elif options.json_mode:
        kw['text'] = {'format': {'type': 'json_object'}}
    if options.timeout_s:
        kw['timeout'] = float(options.timeout_s)
    kw.update(options.openai or {})
    return method, kw


def _openai_usage(usage):
    if usage is None:
        return {}
    get = (lambda obj, name: obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None))
    return {'input_tokens': get(usage, 'input_tokens'), 'output_tokens': get(usage, 'output_tokens'),
            'thinking_tokens': get(get(usage, 'output_tokens_details') or {}, 'reasoning_tokens'),
            'cached_tokens': get(get(usage, 'input_tokens_details') or {}, 'cached_tokens'),
            'total_tokens': get(usage, 'total_tokens')}


def parse_openai(response, model, options, method='create'):
    for item in getattr(response, 'output', None) or []:
        for content in getattr(item, 'content', None) or []:
            if getattr(content, 'type', '') == 'refusal':
                raise SafetyError(f"Отказ модели '{model}' (refusal): {getattr(content, 'refusal', '') or 'refusal'}",
                                  model=model, provider='openai')
    status = getattr(response, 'status', None)
    details = getattr(response, 'incomplete_details', None)
    reason = getattr(details, 'reason', None) if details else None
    if reason == 'content_filter':
        raise SafetyError(f'Ответ OpenAI ({model}) остановлен фильтром (content_filter).', model=model, provider='openai')
    text = getattr(response, 'output_text', '') or ''
    data = None
    if method == 'parse':
        data = getattr(response, 'output_parsed', None)
    elif options is not None and (options.json_schema is not None or options.json_mode):
        data = _json(text)
    if not text.strip() and data is None:
        message = f"Модель '{model}' вернула пустой ответ. Статус: {status}"
        if reason:
            message += f', причина: {reason}'
        if reason == 'max_output_tokens':
            message += ' — в max_output_tokens входят и токены размышлений'
        raise EmptyResponseError(message + '.', model=model, provider='openai')
    if options is not None and (options.json_schema is not None or options.json_mode) and data is None:
        raise EmptyResponseError(f'Модель {model} вернула не JSON.', model=model, provider='openai')
    thoughts, sources = [], []
    for item in getattr(response, 'output', None) or []:
        if getattr(item, 'type', '') == 'reasoning':
            thoughts += [getattr(s, 'text', '') for s in getattr(item, 'summary', None) or []]
        for content in getattr(item, 'content', None) or []:
            for note in getattr(content, 'annotations', None) or []:
                if getattr(note, 'type', '') == 'url_citation':
                    sources.append({'title': getattr(note, 'title', None), 'url': getattr(note, 'url', None)})
    return {'text': text.strip(), 'thoughts': '\n'.join(t for t in thoughts if t).strip(), 'data': data,
            'finish_reason': reason or status or '', 'truncated': reason == 'max_output_tokens',
            'usage': _openai_usage(getattr(response, 'usage', None)), 'sources': sources,
            'response_id': getattr(response, 'id', None)}


def usage_of(response):
    """Token usage of a raw Gemini or OpenAI response; {} for anything else."""
    if getattr(response, 'usage_metadata', None) is not None:
        return _gemini_usage(response.usage_metadata)
    if getattr(response, 'usage', None) is not None:
        return _openai_usage(response.usage)
    return {}


def text_of(response):
    """Best-effort reply text of a raw Gemini or OpenAI response; '' when unknown."""
    try:
        if hasattr(response, 'output_text'):
            return (response.output_text or '').strip()
        if getattr(response, 'candidates', None):
            parts = getattr(response.candidates[0].content, 'parts', None) or []
            return ''.join(p.text for p in parts if isinstance(getattr(p, 'text', None), str)
                           and not getattr(p, 'thought', False)).strip()
    except Exception:
        pass
    return ''
