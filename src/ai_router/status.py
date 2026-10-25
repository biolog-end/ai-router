"""Status for key-manager UIs, pause reset and explicit key probes."""

import time

import gemini_budget as gb
import openai_budget as ob

from . import catalog, keys, state
from .errors import RouterError, classify
from . import router
from .router import generate


def _free_tier(tokens_by_model, groups):
    usage = {group: {'used': 0, 'limit': info['limit'], 'label': info['label']} for group, info in groups.items()}
    for model, tokens in tokens_by_model.items():
        group = ob.group_for(model)
        if group in usage:
            usage[group]['used'] += int(tokens)
    return usage


def status(provider=None, *, with_openai_budget=True):
    """Keys (without secrets) with pauses, counters and Gemini quota rows; provider totals;
    models blocked on every available key; the shared OpenAI budget."""
    now = time.time()
    entries = keys.entries(provider)
    rows = state.snapshot({keys.fingerprint(entry['key']): entry['provider'] for entry in entries})
    errors = []
    quota = {}
    gemini_entries = [entry for entry in entries if entry['provider'] == 'gemini']
    if gemini_entries:
        try:
            quota = gb.snapshots([{'id': entry['id'], 'key': entry['key']} for entry in gemini_entries])
        except Exception as exc:
            errors.append(f'Общий учёт gemini_budget недоступен: {exc}')
    openai_status = None
    if with_openai_budget and (provider in (None, 'openai')):
        try:
            openai_status = ob.status()
        except Exception as exc:
            errors.append(f'openai_budget недоступен: {exc}')

    items = []
    for entry in entries:
        row = rows[keys.fingerprint(entry['key'])]
        cooldowns = [{'model': model, 'left_s': int(pause['until'] - now), 'kind': pause.get('kind'),
                      'reason': pause.get('reason', ''), 'source': 'ai_router'}
                     for model, pause in row['cooldowns'].items()]
        item = {**keys._public(entry), 'rejected': row['rejected'], 'cooldowns': cooldowns,
                'last_error': row['last_error'], 'last_error_at': row['last_error_at'],
                'last_used': row['last_used'] or None, 'ok': row['ok'], 'fail': row['fail'],
                'requests_today': row['requests_today'], 'requests_last_minute': row['requests_last_minute'],
                'tokens_today': row['tokens_today'], 'tokens_by_model': row['tokens_by_model']}
        if entry['provider'] == 'gemini':
            item['quota'] = []
            for quota_row in quota.get(entry['id'], []):
                free = gb.has_free_quota(quota_row['model'])
                item['quota'].append({**quota_row, 'task': gb.MODEL_TASKS.get(quota_row['model']), 'free': free})
                if free and quota_row['blocked']:
                    cooldowns.append({'model': quota_row['model'], 'left_s': quota_row['left_s'], 'kind': 'quota',
                                      'reason': quota_row['reason'], 'source': 'gemini_budget'})
        elif openai_status is not None:
            item['free_tier'] = _free_tier(row['tokens_by_model'], openai_status['groups'])
        item['available'] = item['enabled'] and not row['rejected'] and not any(c['model'] == '*' for c in cooldowns)
        items.append(item)

    providers = {}
    blocked_models = {}
    for name in keys.PROVIDERS:
        mine = [item for item in items if item['provider'] == name]
        usable = [item for item in mine if item['available']]
        providers[name] = {'total': len(mine), 'enabled': sum(item['enabled'] for item in mine),
                           'available': len(usable)}
        per_model = {}
        for item in usable:
            for pause in item['cooldowns']:
                if pause['model'] != '*':
                    per_model.setdefault(pause['model'], []).append(pause)
        for model, pauses in per_model.items():
            if len(pauses) == len(usable):
                blocked_models[model] = {'provider': name, 'left_s': min(p['left_s'] for p in pauses),
                                         'reason': pauses[0]['reason']}
    return {'keys': items, 'providers': providers, 'blocked_models': blocked_models, 'errors': errors,
            'openai_budget': openai_status, 'default_chain': catalog.default_chain(),
            'gemini_profile': {'date': gb.PROFILE_DATE, 'source': gb.PROFILE_SOURCE}}


def reset_pauses(key_id=None, model=None):
    """Clears pauses (one model or all) of one key or every key; without a model also
    rejections. Use it after fixing billing or replacing a key."""
    entries = [entry for entry in keys.entries() if key_id is None or entry['id'] == key_id]
    state.reset({keys.fingerprint(entry['key']) for entry in entries}, model)
    for entry in entries:
        if entry['provider'] == 'gemini':
            gb.clear_pause(entry['key'], model)
    return len(entries)


def test_key(key_id, model=None):
    """Explicit probe of one key: a tiny request to `model`, or models.list() without one."""
    entry = next((item for item in keys.entries() if item['id'] == key_id), None)
    if entry is None:
        raise KeyError(f'Ключ {key_id} не найден в хранилище ai_router.')
    fp = keys.fingerprint(entry['key'])
    if model:
        try:
            result = generate('Ответь одним словом: OK', model=model, key_id=key_id, thinking='off',
                              retry_server_errors=False)
        except RouterError as exc:
            return {'ok': False, 'model': model, 'kind': exc.kind, 'message': exc.message}
        state.reset({fp})
        return {'ok': True, 'model': model, 'message': 'Запрос принят моделью: квота генерации доступна.',
                'text': result.text}
    try:
        client = router.client_for(entry['provider'], entry['key'])
        count = len(list(client.models.list()))
    except Exception as exc:
        failure = classify(exc, entry['provider'], None)
        if failure is None:
            raise
        message = failure.message.replace(entry['key'], '[КЛЮЧ]')
        if failure.kind == 'bad_key':
            state.reject(fp, message)
        return {'ok': False, 'kind': failure.kind, 'message': message}
    state.reset({fp})
    return {'ok': True, 'models': count,
            'message': 'Ключ принят, список моделей доступен. Квоты генерации проверяются отдельно.'}


def live_models(provider, force=False):
    """Text model ids offered by the API to the first enabled key (cached 10 min)."""
    for entry in keys.entries(provider):
        if entry.get('enabled', True):
            return catalog.live_models(provider, router.client_for(provider, entry['key']), force)
    return []
