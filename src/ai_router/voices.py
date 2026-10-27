"""Gemini TTS voices: 30 prebuilt ones, the extended library, designed and replicated voices.

A custom voice (`voice_…`) lives in the Google project of the key that created it; other
projects answer 404. The registry (~/.ai_router/voices.json) remembers that key, and
speak() sends requests with such a voice through it.
"""

import base64
import time
from pathlib import Path
from urllib.parse import urlencode

from . import keys
from ._storage import file_lock, read_json, state_dir, write_json
from .errors import NoKeysError, RequestError

DESIGN_MODEL = 'gemini-3.8-flash-tts'
LIBRARY_TTL_S = 7 * 24 * 3600

# name: (gender, pitch, short descriptor). Gender/pitch as Google's voice library reports them.
PREBUILT = {
    'Zephyr': ('female', 'high', 'Bright'), 'Puck': ('male', 'low', 'Upbeat'),
    'Charon': ('male', 'low', 'Informative'), 'Kore': ('female', 'medium', 'Firm'),
    'Fenrir': ('male', 'low', 'Excitable'), 'Leda': ('female', 'high', 'Youthful'),
    'Orus': ('male', 'low', 'Firm'), 'Aoede': ('female', 'medium', 'Breezy'),
    'Callirrhoe': ('female', 'medium', 'Easy-going'), 'Autonoe': ('female', 'medium', 'Bright'),
    'Enceladus': ('male', 'low', 'Breathy'), 'Iapetus': ('male', 'low', 'Clear'),
    'Umbriel': ('male', 'low', 'Easy-going'), 'Algieba': ('male', 'low', 'Smooth'),
    'Despina': ('female', 'medium', 'Smooth'), 'Erinome': ('female', 'medium', 'Clear'),
    'Algenib': ('male', 'low', 'Gravelly'), 'Rasalgethi': ('male', 'medium', 'Informative'),
    'Laomedeia': ('female', 'high', 'Upbeat'), 'Achernar': ('female', 'high', 'Soft'),
    'Alnilam': ('male', 'low', 'Firm'), 'Schedar': ('male', 'low', 'Even'),
    'Gacrux': ('female', 'medium', 'Mature'), 'Pulcherrima': ('neutral', 'medium', 'Forward'),
    'Achird': ('male', 'low', 'Friendly'), 'Zubenelgenubi': ('male', 'low', 'Casual'),
    'Vindemiatrix': ('female', 'medium', 'Gentle'), 'Sadachbia': ('male', 'low', 'Lively'),
    'Sadaltager': ('male', 'medium', 'Knowledgeable'), 'Sulafat': ('female', 'medium', 'Warm'),
}
DEFAULT_VOICE = 'Kore'
_BY_LOWER = {name.lower(): name for name in PREBUILT}

# The phrase the owner of a voice reads into consent_audio, verbatim (Google's list has 30 locales).
CONSENT_PHRASES = {
    'ru-RU': 'Я являюсь владельцем этого голоса и даю согласие Google на использование этого голоса '
             'для создания модели синтетического голоса.',
    'en-US': 'I am the owner of this voice and I consent to Google using this voice to create a '
             'synthetic voice model.',
}


def is_prebuilt(voice):
    return str(voice or '').strip().lower() in _BY_LOWER


def is_custom(voice):
    return str(voice or '').strip().startswith(('voice_', 'voicekey_'))


def canonical(voice):
    """'kore' → 'Kore'; any other id is returned as is."""
    voice = str(voice or '').strip()
    return _BY_LOWER.get(voice.lower(), voice)


def prebuilt():
    """The 30 voices every TTS model knows, with gender, pitch and Google's descriptor."""
    return [{'id': name, 'gender': g, 'pitch': p, 'descriptor': d} for name, (g, p, d) in PREBUILT.items()]


# --------------------------------------------------------------------------
# Registry of custom voices
# --------------------------------------------------------------------------

def _registry_path():
    return state_dir() / 'voices.json'


def _load():
    data = read_json(_registry_path(), None)
    return data if isinstance(data, dict) and isinstance(data.get('voices'), dict) else {'version': 1, 'voices': {}}


def custom():
    """Custom voices this machine knows about: id, name, owner key, gender, language, expiry."""
    with file_lock(_registry_path()):
        rows = _load()['voices']
    return sorted(rows.values(), key=lambda row: row.get('created_at') or 0, reverse=True)


def owner(voice_id):
    """key_id of the key whose project holds the voice, or None when unknown."""
    with file_lock(_registry_path()):
        row = _load()['voices'].get(str(voice_id))
    return row.get('key_id') if row else None


def remember(voice_id, key_id, **info):
    with file_lock(_registry_path()):
        data = _load()
        row = data['voices'].setdefault(str(voice_id), {'id': str(voice_id), 'created_at': time.time()})
        row.update({name: value for name, value in info.items() if value is not None})
        if key_id:
            entry = keys.get(key_id)
            row.update(key_id=key_id, key_label=entry.get('label') if entry else None)
        write_json(_registry_path(), data)
    return dict(row)


def forget(voice_id):
    with file_lock(_registry_path()):
        data = _load()
        removed = data['voices'].pop(str(voice_id), None)
        write_json(_registry_path(), data)
    return removed is not None


# --------------------------------------------------------------------------
# HTTP helpers (voices endpoints are not covered by the SDK this project pins)
# --------------------------------------------------------------------------

def raw_request(client, method, path, body=None, stream=False):
    """JSON request to the Gemini API through the SDK's own HTTP client.

    It carries the key, timeout and error mapping (google.genai.errors.APIError), so
    failures are classified exactly like generate_content ones.
    """
    import json
    api = client._api_client
    if stream:
        return (json.loads(chunk.body) for chunk in api.request_streamed(method, path, body or {}) if chunk.body)
    response = api.request(method, path, body or {})
    return json.loads(response.body) if response.body else {}


def _keys(key_id=None):
    entries = [entry for entry in keys.entries('gemini') if entry.get('enabled', True)]
    if key_id:
        entries = [entry for entry in keys.entries('gemini') if entry['id'] == key_id]
    if not entries:
        raise NoKeysError('Нет включённых ключей Gemini в хранилище ai_router.', provider='gemini')
    return entries


def _on_some_key(method, path, body=None, key_id=None):
    """(payload, key_id) from the first key that answers; voices endpoints cost no quota."""
    from .router import client_for
    last = None
    for entry in _keys(key_id):
        try:
            return raw_request(client_for('gemini', entry['key']), method, path, body), entry['id']
        except Exception as exc:
            last = exc
    raise RequestError(f'Запрос {method.upper()} /{path.split("?")[0]} не прошёл ни на одном ключе Gemini: '
                       f'{str(last).replace(entry["key"], "[КЛЮЧ]")}', provider='gemini') from last


def _query(filters, page_size, page_token=None):
    params = []
    for name, value in filters.items():
        if value is None:
            continue
        for item in ([value] if isinstance(value, str) else value):
            params.append((name, item))
    params.append(('pageSize', page_size))
    if page_token:
        params.append(('pageToken', page_token))
    return 'voices?' + urlencode(params)


def _row(voice, key_id=None):
    row = {name: voice.get(name) for name in ('id', 'type', 'display_name', 'language_code', 'region_code',
                                              'accent', 'gender', 'pitch', 'persona', 'context', 'description',
                                              'expire_time', 'model')
           if voice.get(name) is not None}
    if voice.get('prompted'):
        row['prompt'] = voice['prompted'].get('input')
    if key_id and row.get('type') in ('prompted', 'replicated'):
        row['key_id'] = key_id
    return row


def list(*, language=None, gender=None, pitch=None, accent=None, region=None, persona=None, context=None,
         type=None, search=None, limit=100, key_id=None):
    """Voices matching the filters, custom voices of the key's project first.

    A list value means "any of"; different filters combine with AND. language is a
    BCP-47 tag ('en-GB'), gender 'female'|'male'|'neutral', pitch 'low'|'medium'|'high',
    type 'prebuilt'|'prompted'|'replicated', search is a substring of name/description.
    The library has no ru-RU voices: Russian is spoken by any of them.
    """
    filters = {'languageCode': language, 'gender': gender, 'pitch': pitch, 'accent': accent,
               'regionCode': region, 'persona': persona, 'context': context, 'type': type, 'search': search}
    rows, token, used_key = [], None, key_id
    while limit is None or len(rows) < limit:
        size = 1000 if limit is None else min(1000, limit - len(rows))
        payload, used_key = _on_some_key('get', _query(filters, size, token), key_id=used_key)
        rows += [_row(voice, used_key) for voice in payload.get('voices') or []]
        token = payload.get('next_page_token') or payload.get('nextPageToken')
        if not token:
            break
    return rows


def _library_path():
    return state_dir() / 'voice_library.json'


def library(refresh=False):
    """The whole prebuilt catalog (~2000 voices), cached for a week."""
    with file_lock(_library_path()):
        cached = read_json(_library_path(), None)
    if not refresh and isinstance(cached, dict) and time.time() - cached.get('at', 0) < LIBRARY_TTL_S:
        return cached['voices']
    voices = list(type='prebuilt', limit=None)
    write_json(_library_path(), {'at': time.time(), 'voices': voices})
    return voices


def describe(voice):
    """{'gender', 'pitch', …} of any voice id, or None when nothing is known about it."""
    voice = canonical(voice)
    if voice in PREBUILT:
        gender, pitch, descriptor = PREBUILT[voice]
        return {'id': voice, 'gender': gender, 'pitch': pitch, 'description': descriptor}
    with file_lock(_registry_path()):
        row = _load()['voices'].get(voice)
    if row:
        return row
    if is_custom(voice):
        return None
    try:
        return next((row for row in library() if row.get('id') == voice), None)
    except Exception:
        return None


def substitute(voice):
    """(prebuilt name, why) closest to a library or custom voice: same gender, then same pitch."""
    info = describe(voice) or {}
    gender, pitch = info.get('gender'), info.get('pitch')
    if not gender:
        return DEFAULT_VOICE, 'пол голоса неизвестен'
    same = [name for name, (g, p, _) in PREBUILT.items() if g == gender]
    exact = [name for name in same if PREBUILT[name][1] == pitch]
    choice = (exact or same or [DEFAULT_VOICE])[0]
    return choice, f'{gender}, {pitch or "высота неизвестна"}'


# --------------------------------------------------------------------------
# Creating and deleting custom voices
# --------------------------------------------------------------------------

def _audio(value, what):
    if isinstance(value, (str, Path)) and not str(value).startswith('data:') and Path(value).exists():
        value = Path(value).read_bytes()
    if isinstance(value, str):
        value = base64.b64decode(value.split(',', 1)[-1])
    data = bytes(value)
    if data[:4] != b'RIFF':
        raise ValueError(f'{what}: нужен WAV (лучше 24 кГц, моно, 16 бит). Перевести можно так: '
                         f'ffmpeg -i запись.m4a -ar 24000 -ac 1 -sample_fmt s16 запись.wav')
    return base64.b64encode(data).decode('ascii')


def _created(result, store, extra):
    payload = result.response or {}
    voice_id = payload.get('id') or payload.get('voice_key') or payload.get('voiceKey')
    if not voice_id:
        raise RequestError(f'Google не вернул id голоса: {str(payload)[:300]}', provider='gemini')
    sample = payload.get('sample_audio') or payload.get('sampleAudio') or {}
    row = {'name': payload.get('display_name') or extra.get('name'), 'type': payload.get('type') or extra['type'],
           'gender': payload.get('gender') or extra.get('gender'),
           'language_code': payload.get('language_code') or extra.get('language'),
           'expire_time': payload.get('expire_time'), 'model': result.model, 'prompt': extra.get('prompt')}
    if store:
        remember(voice_id, result.key_id, **row)
    return {'id': voice_id, **row, 'key_id': result.key_id, 'key_label': result.key_label,
            'sample_audio': base64.b64decode(sample['data']) if sample.get('data') else None,
            'usage': payload.get('usage'), 'attempts': result.attempts}


def design(description, *, name=None, gender=None, language=None, model=DESIGN_MODEL, on_event=None, **routing):
    """Creates a voice from a text description; returns {'id', 'sample_audio' (WAV bytes), …}.

    Describe what never changes — age, gender, timbre, texture, accent ("хрипловатый мужчина
    лет семидесяти, тёплый, неторопливый"). Mood belongs to speak(style=…). The voice is
    stored in the project of the key that created it for a year after its last use
    (200 custom voices per project).
    """
    from .router import call
    voice = {'model': model, 'type': 'prompted', 'prompted': {'input': str(description).strip()}}
    for field, value in (('display_name', name), ('gender', gender), ('language_code', language)):
        if value:
            voice[field] = value
    result = call(lambda client, ctx: raw_request(client, 'post', 'voices', {'store': True, 'voice': {**voice, 'model': ctx.model}}),
                  model=model, on_event=on_event, input_tokens=len(str(description).encode('utf-8')) // 3 + 50,
                  **routing)
    return _created(result, True, {'type': 'prompted', 'name': name, 'gender': gender, 'language': language,
                                   'prompt': str(description).strip()})


def replicate(source_audio, consent_audio, *, name=None, store=True, model=DESIGN_MODEL, on_event=None, **routing):
    """Clones a real voice from 10–30 s of clean speech plus a recording of the consent phrase.

    Both recordings: the same adult speaker, the same microphone and room, WAV. The consent
    recording reads CONSENT_PHRASES[locale] verbatim. store=False returns a `voicekey_…`
    that lives 7 days and is not kept by Google; store=True a `voice_…` for a year.

    Google gates voice replication behind **Paid Quota Tier 1 or higher**: on a free key
    this returns 403 PERMISSION_DENIED. Enable billing on the key's project to unlock it.
    """
    from .errors import RequestError
    from .router import call
    voice = {'model': model, 'type': 'replicated', 'replicated': {
        'source_audio': {'mime_type': 'audio/wav', 'data': _audio(source_audio, 'source_audio')},
        'consent_audio': {'mime_type': 'audio/wav', 'data': _audio(consent_audio, 'consent_audio')}}}
    if name:
        voice['display_name'] = name
    try:
        result = call(lambda client, ctx: raw_request(client, 'post', 'voices',
                                                      {'store': bool(store), 'voice': {**voice, 'model': ctx.model}}),
                      model=model, on_event=on_event, input_tokens=1000, **routing)
    except RequestError as exc:
        if 'Paid Quota Tier' in exc.message or 'requires Paid' in exc.message:
            raise RequestError(
                'Google разрешает клонирование голоса только на платном тарифе (Paid Quota Tier 1+). '
                'На бесплатных ключах возвращается 403. Включите биллинг хотя бы на одном ключе '
                'Gemini и повторите запрос — созданный voice_id будет работать и для платных, '
                'и для бесплатных вызовов speak(), пока голос не истечёт.',
                provider='gemini', attempts=exc.attempts) from exc
        raise
    return _created(result, store, {'type': 'replicated', 'name': name})


def get(voice_id):
    """Google's record of a custom voice (with sample_audio for designed ones)."""
    payload, key_id = _on_some_key('get', f'voices/{voice_id}', key_id=owner(voice_id))
    row = _row(payload, key_id)
    sample = payload.get('sample_audio') or {}
    if sample.get('data'):
        row['sample_audio'] = base64.b64decode(sample['data'])
    return row


def delete(voice_id):
    """Deletes a stored custom voice from its Google project and the local registry."""
    if str(voice_id).startswith('voicekey_'):
        forget(voice_id)
        return True
    _on_some_key('delete', f'voices/{voice_id}', key_id=owner(voice_id))
    forget(voice_id)
    return True


def sync():
    """Asks every key's project for its custom voices and records which key owns each one.

    Voices made in AI Studio become usable by speak() without guessing the key.
    """
    from .router import client_for
    found = []
    for entry in _keys():
        try:
            payload = raw_request(client_for('gemini', entry['key']), 'get',
                                  _query({'type': ['prompted', 'replicated']}, 1000))
        except Exception:
            continue
        for voice in payload.get('voices') or []:
            row = _row(voice, entry['id'])
            remember(row['id'], entry['id'], name=row.get('display_name'), type=row.get('type'),
                     gender=row.get('gender'), language_code=row.get('language_code'),
                     expire_time=row.get('expire_time'), prompt=row.get('prompt'))
            found.append(row)
    return found
