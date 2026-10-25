"""Shared key vault: one list of Gemini and OpenAI keys for every project of this user.

Secrets live in plain JSON in the user profile (~/.ai_router/keys.json), like the
OpenAI Admin key of openai_budget. Status views never contain them; `secret()` and
`secrets()` are the only explicit ways out.
"""

import hashlib
import os
import re
import time

from ._storage import file_lock, read_json, state_dir, write_json

PROVIDERS = ('gemini', 'openai')
LABELS = {'gemini': 'Gemini', 'openai': 'OpenAI'}

ENV_NAMES = {
    'gemini': ('GEMINI_API_KEYS', 'GEMINI_API_KEY', 'GOOGLE_API_KEYS', 'GOOGLE_API_KEY'),
    'openai': ('OPENAI_API_KEYS', 'OPENAI_API_KEY'),
}


def path():
    return state_dir() / 'keys.json'


def fingerprint(secret):
    """Same value as gemini_budget.fingerprint, so both stores agree on a key."""
    return hashlib.sha256(secret.strip().encode('utf-8')).hexdigest()


def mask(secret):
    if not secret:
        return ''
    return f'{secret[:6]}…{secret[-4:]}' if len(secret) > 12 else secret[:3] + '…'


def _clean_provider(value):
    provider = str(value or '').strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f'Неизвестный провайдер ключа: {value!r}. Допустимо: gemini, openai.')
    return provider


def _clean_secret(secret, provider):
    secret = str(secret or '').strip()
    if not secret or len(secret) > 512 or any(ch.isspace() for ch in secret):
        raise ValueError('Ключ API должен быть одной строкой без пробелов.')
    if provider == 'openai' and secret.startswith('sk-admin-'):
        raise ValueError('Это Admin-ключ OpenAI: он задаётся в openai_budget (openai-budget set-key), '
                         'а не как ключ для запросов.')
    return secret


def _load():
    data = read_json(path(), None)
    if data is None:
        return {'version': 1, 'keys': [], 'imports': {}}
    # A damaged vault must never be silently replaced: that would drop every key.
    if not isinstance(data, dict) or not isinstance(data.get('keys'), list):
        raise ValueError(f'Хранилище ключей ai_router повреждено: {path()}')
    data.setdefault('imports', {})
    return data


def _save(data):
    write_json(path(), data)


def _public(entry):
    return {'id': entry['id'], 'provider': entry['provider'], 'label': entry.get('label', ''),
            'enabled': bool(entry.get('enabled', True)), 'masked': mask(entry['key']),
            'fingerprint': fingerprint(entry['key']), 'source': entry.get('source', ''),
            'added_at': entry.get('added_at')}


def _new_entry(data, provider, secret, label, enabled, source):
    fp = fingerprint(secret)
    count = sum(1 for item in data['keys'] if item['provider'] == provider) + 1
    return {'id': provider[0] + fp[:9], 'provider': provider, 'key': secret,
            'label': str(label or '').strip() or f'{LABELS[provider]} {count}',
            'enabled': bool(enabled), 'added_at': time.time(), 'source': source}


def entries(provider=None):
    """Copies with secrets, in vault order. For the router and server-side code only."""
    with file_lock(path()):
        return [dict(item) for item in _load()['keys'] if provider is None or item['provider'] == provider]


def list_keys(provider=None):
    """Keys without secrets: id, provider, label, enabled, masked, fingerprint, source."""
    return [_public(item) for item in entries(provider)]


def get(key_id):
    return next((_public(item) for item in entries() if item['id'] == key_id), None)


def secret(key_id):
    for item in entries():
        if item['id'] == key_id:
            return item['key']
    raise KeyError(f'Ключ {key_id} не найден в хранилище ai_router.')


def secrets(provider):
    """Raw keys of one provider in vault order, e.g. for a one-key-per-line editor."""
    return [item['key'] for item in entries(_clean_provider(provider))]


def add(provider, secret, *, label='', enabled=True, source='manual'):
    """Adds a key and returns its public row; an existing key is returned unchanged."""
    provider = _clean_provider(provider)
    secret = _clean_secret(secret, provider)
    with file_lock(path()):
        data = _load()
        fp = fingerprint(secret)
        for item in data['keys']:
            if fingerprint(item['key']) == fp:
                return _public(item)
        entry = _new_entry(data, provider, secret, label, enabled, source)
        data['keys'].append(entry)
        _save(data)
        return _public(entry)


def update(key_id, *, label=None, enabled=None, secret=None):
    with file_lock(path()):
        data = _load()
        for item in data['keys']:
            if item['id'] != key_id:
                continue
            if label is not None:
                item['label'] = str(label).strip() or item['label']
            if enabled is not None:
                item['enabled'] = bool(enabled)
            if secret is not None:
                value = _clean_secret(secret, item['provider'])
                if any(other is not item and fingerprint(other['key']) == fingerprint(value)
                       for other in data['keys']):
                    raise ValueError('Такой ключ уже есть в хранилище.')
                item['key'] = value
            _save(data)
            return _public(item)
    raise KeyError(f'Ключ {key_id} не найден в хранилище ai_router.')


def remove(key_id):
    with file_lock(path()):
        data = _load()
        kept = [item for item in data['keys'] if item['id'] != key_id]
        if len(kept) == len(data['keys']):
            return False
        data['keys'] = kept
        _save(data)
        return True


def reorder(key_ids):
    """Moves the listed keys to the front in this order; the rest keep their order."""
    position = {key_id: index for index, key_id in enumerate(key_ids)}
    with file_lock(path()):
        data = _load()
        data['keys'].sort(key=lambda item: position.get(item['id'], len(position)))
        _save(data)
        return [_public(item) for item in data['keys']]


def replace(provider, items, *, source='manual'):
    """Makes the provider's key list exactly `items`, in order; other providers stay.

    Each item is {'id'?, 'key'?, 'label'?, 'enabled'?}. Without 'key' the secret of the
    existing key with that id is kept, so an editor can post masked rows back.
    """
    provider = _clean_provider(provider)
    with file_lock(path()):
        data = _load()
        current = {item['id']: item for item in data['keys'] if item['provider'] == provider}
        by_fp = {fingerprint(item['key']): item for item in current.values()}
        result, seen = [], set()
        for raw in items:
            raw = raw if isinstance(raw, dict) else {'key': raw}
            if raw.get('key'):
                value = _clean_secret(raw['key'], provider)
                old = by_fp.get(fingerprint(value)) or (current.get(raw.get('id')) if raw.get('id') else None)
            elif raw.get('id') in current:
                old, value = current[raw['id']], current[raw['id']]['key']
            else:
                continue
            fp = fingerprint(value)
            if fp in seen:
                continue
            seen.add(fp)
            entry = dict(old) if old else _new_entry(data, provider, value, raw.get('label'), True, source)
            if old and fingerprint(old['key']) != fp:
                entry = {**_new_entry(data, provider, value, raw.get('label'), True, source),
                         'label': old.get('label')}
            entry['key'] = value
            if raw.get('label') is not None and str(raw['label']).strip():
                entry['label'] = str(raw['label']).strip()
            if raw.get('enabled') is not None:
                entry['enabled'] = bool(raw['enabled'])
            result.append(entry)
        others = [item for item in data['keys'] if item['provider'] != provider]
        data['keys'] = others + result
        _save(data)
        return [_public(item) for item in result]


def import_keys(items, *, source):
    """Merges keys (dicts with provider/key/label/enabled); duplicates are skipped. Returns count."""
    added = 0
    with file_lock(path()):
        data = _load()
        known = {fingerprint(item['key']) for item in data['keys']}
        for raw in items:
            try:
                provider = _clean_provider(raw.get('provider') or 'gemini')
                value = _clean_secret(raw.get('key'), provider)
            except ValueError:
                continue
            if fingerprint(value) in known:
                continue
            known.add(fingerprint(value))
            data['keys'].append(_new_entry(data, provider, value, raw.get('label'),
                                           raw.get('enabled', True), source))
            added += 1
        if added:
            _save(data)
    return added


def import_once(marker, items, *, source):
    """Like import_keys, but only the first time for this marker (e.g. an old key file).

    A key the user deleted from the vault is therefore not resurrected on the next start.
    Returns None when the marker was imported before.
    """
    with file_lock(path()):
        data = _load()
        if marker in data['imports']:
            return None
        added = import_keys(items, source=source)
        data = _load()
        data['imports'][marker] = {'at': time.time(), 'added': added}
        _save(data)
        return added


def import_env(environ=None):
    """Imports GEMINI/GOOGLE/OPENAI_API_KEY(S) once per key. Returns how many were added."""
    environ = os.environ if environ is None else environ
    added = 0
    for provider, names in ENV_NAMES.items():
        for name in names:
            for value in re.split(r'[,\s]+', environ.get(name, '') or ''):
                if not value.strip():
                    continue
                marker = f'env:{fingerprint(value)[:16]}'
                result = import_once(marker, [{'provider': provider, 'key': value, 'label': f'{LABELS[provider]} из {name}'}],
                                     source='env')
                added += result or 0
    return added
