"""Shared runtime state of keys: pauses, rejected keys, counters, last error.

Keyed by fingerprint, so every project sees that a key is paused or rejected. Gemini
models with a free quota are paused in gemini_budget instead; this file covers the rest.
"""

import time
from datetime import datetime, timedelta, timezone

from ._storage import file_lock, read_json, state_dir, write_json

RECENT_WINDOW_S = 60


def path():
    return state_dir() / 'state.json'


def day_stamp(provider, now=None):
    """The provider's billing day: 00:00 UTC for OpenAI, Pacific midnight for Google."""
    now = time.time() if now is None else now
    if provider == 'openai':
        return datetime.fromtimestamp(now, timezone.utc).strftime('%Y-%m-%d')
    try:
        from zoneinfo import ZoneInfo
        zone = ZoneInfo('America/Los_Angeles')
    except Exception:
        zone = timezone(timedelta(hours=-8))
    return datetime.fromtimestamp(now, zone).strftime('%Y-%m-%d')


def _load():
    data = read_json(path(), None)
    if not isinstance(data, dict) or not isinstance(data.get('keys'), dict):
        # Losing pauses and counters is harmless: the next API answer restores them.
        return {'version': 1, 'keys': {}}
    return data


def _row(data, fp):
    row = data['keys'].setdefault(fp, {})
    row.setdefault('cooldowns', {})
    row.setdefault('rejected', None)
    row.setdefault('ok', 0)
    row.setdefault('fail', 0)
    return row


def _roll(row, provider, now):
    day = day_stamp(provider, now)
    if row.get('day') != day:
        row.update(day=day, requests_today=0, tokens_today=0, tokens_by_model={})
    row['recent'] = [at for at in row.get('recent', []) if at > now - RECENT_WINDOW_S]


def _active(entry, now):
    return entry and entry.get('until', 0) > now


def snapshot(fps):
    """{fp: row} with expired pauses dropped. Pass {fp: provider} to roll daily counters."""
    now = time.time()
    providers = fps if isinstance(fps, dict) else dict.fromkeys(fps)
    with file_lock(path()):
        data = _load()
    result = {}
    for fp, provider in providers.items():
        row = data['keys'].get(fp) or {}
        if provider and row.get('day') != day_stamp(provider, now):
            row = {**row, 'requests_today': 0, 'tokens_today': 0, 'tokens_by_model': {}}
        result[fp] = {
            'cooldowns': {model: dict(entry) for model, entry in (row.get('cooldowns') or {}).items()
                          if _active(entry, now)},
            'rejected': row.get('rejected'),
            'ok': row.get('ok', 0), 'fail': row.get('fail', 0),
            'last_error': row.get('last_error', ''), 'last_error_at': row.get('last_error_at'),
            'last_used': row.get('last_used', 0),
            'day': row.get('day'), 'requests_today': row.get('requests_today', 0),
            'tokens_today': row.get('tokens_today', 0),
            'tokens_by_model': dict(row.get('tokens_by_model') or {}),
            'requests_last_minute': sum(1 for at in row.get('recent', []) if at > now - RECENT_WINDOW_S),
        }
    return result


def blocker(fp, model, now=None):
    """The pause or rejection that keeps this key from serving `model`, or None."""
    now = time.time() if now is None else now
    row = snapshot([fp])[fp]
    if row['rejected']:
        return {'model': '*', 'kind': 'bad_key', 'until': None, **row['rejected']}
    for scope in ('*', model):
        entry = row['cooldowns'].get(scope)
        if entry and entry['until'] > now:
            return {'model': scope, **entry}
    return None


def pause(fp, scope, until, kind, reason):
    """Pauses a key for one model (scope = model id) or for everything (scope = '*')."""
    with file_lock(path()):
        data = _load()
        row = _row(data, fp)
        row['cooldowns'][scope] = {'until': float(until), 'kind': kind, 'reason': reason, 'at': time.time()}
        write_json(path(), data)


def reject(fp, reason):
    """The API refused the key itself; it stays out until reset or a successful probe."""
    with file_lock(path()):
        data = _load()
        _row(data, fp)['rejected'] = {'reason': reason, 'at': time.time()}
        write_json(path(), data)


def reset(fps=None, model=None):
    """Clears pauses (of one model or all) and, without a model, rejections too."""
    with file_lock(path()):
        data = _load()
        for fp, row in data['keys'].items():
            if fps is not None and fp not in fps:
                continue
            if model is None:
                row['cooldowns'] = {}
                row['rejected'] = None
            else:
                (row.get('cooldowns') or {}).pop(model, None)
        write_json(path(), data)


def record(fp, provider, model, *, ok, tokens=0, error=''):
    """One finished attempt: counters for the status view and LRU order for rotation."""
    now = time.time()
    with file_lock(path()):
        data = _load()
        row = _row(data, fp)
        _roll(row, provider, now)
        row['last_used'] = now
        if ok:
            row['ok'] += 1
            row['requests_today'] += 1
            row['recent'].append(now)
            row['last_error'] = ''
            if tokens:
                row['tokens_today'] += int(tokens)
                row['tokens_by_model'][model] = row['tokens_by_model'].get(model, 0) + int(tokens)
        else:
            row['fail'] += 1
            row['last_error'] = str(error)[:2000]
            row['last_error_at'] = now
        write_json(path(), data)
