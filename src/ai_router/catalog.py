"""One model catalog for both providers, the default chain and live model lists."""

import logging
import threading
import time

import gemini_budget as gb
import openai_budget as ob

from ._storage import file_lock, read_json, state_dir, write_json

log = logging.getLogger('ai_router')

DEFAULT_CHAIN = ['gemini-3.8-flash', 'gemini-3.6-flash', 'gemini-3.5-flash-lite', 'gpt-5.4', 'gpt-5.4-mini']
LIVE_TTL_S = 600

_live_lock = threading.Lock()
_live_cache = {}


def provider_for(model):
    """'gpt-5-mini' → 'openai'; everything else (gemini-*, gemma-*, unknown) → 'gemini'."""
    return 'openai' if ob.is_openai_model(model) else 'gemini'


def canonical(model):
    return (model or '').strip().removeprefix('models/')


def model_info(model):
    """Catalog row of either provider with a 'provider' field, or None."""
    if provider_for(model) == 'openai':
        info = ob.model_info(model)
        return {**info, 'provider': 'openai', 'free_group': ob.group_for(model),
                'free_tier': ob.group_for(model) is not None} if info else None
    info = gb.model_info(model)
    if info is None:
        return None
    return {**info, 'provider': 'gemini', 'free_tier': gb.has_free_quota(info['id'])}


def models(provider=None, task='chat'):
    """Catalog rows of both providers. OpenAI rows only exist for task='chat' (or None)."""
    rows = []
    if provider in (None, 'gemini'):
        rows += [{**row, 'provider': 'gemini'} for row in gb.models(task=task)]
    if provider in (None, 'openai') and task in (None, 'chat'):
        rows += [{**row, 'free_tier': row['free_group'] is not None} for row in ob.models()]
    return rows


def is_retired(model):
    return gb.is_retired(model) if provider_for(model) == 'gemini' else None


def is_text_model(model):
    return ob.is_text_model(model) if provider_for(model) == 'openai' else gb.is_text_model(model)


def default_model(provider='gemini'):
    return ob.DEFAULT_MODEL if provider == 'openai' else gb.DEFAULT_CHAT_MODEL


def light_model(provider='gemini'):
    """A cheap model of the same provider for summaries and memory."""
    return ob.LIGHT_MODEL if provider == 'openai' else gb.LIGHT_CHAT_MODEL


def _config_path():
    return state_dir() / 'config.json'


def default_chain():
    """The chain used when a call names neither a model nor a chain. Editable for everyone."""
    with file_lock(_config_path()):
        chain = (read_json(_config_path(), {}) or {}).get('default_chain')
    return list(chain) if isinstance(chain, list) and chain else list(DEFAULT_CHAIN)


def set_default_chain(chain):
    chain = [canonical(model) for model in chain if canonical(model)]
    if not chain:
        raise ValueError('Цепочка моделей пуста.')
    with file_lock(_config_path()):
        config = read_json(_config_path(), {}) or {}
        config['default_chain'] = chain
        write_json(_config_path(), config)
    return chain


def live_models(provider, client, force=False):
    """Text model ids the API offers to this client (cached for 10 min); [] on error."""
    now = time.time()
    with _live_lock:
        cached = _live_cache.get(provider)
        if cached and not force and now - cached[0] < LIVE_TTL_S:
            return list(cached[1])
    ids = []
    try:
        if provider == 'openai':
            ids = [m.id for m in client.models.list() if getattr(m, 'id', None)]
        else:
            for model in client.models.list():
                actions = getattr(model, 'supported_actions', None)
                if actions and 'generateContent' not in actions:
                    continue
                if getattr(model, 'name', None):
                    ids.append(canonical(model.name))
    except Exception as exc:
        log.warning('Не удалось получить список моделей %s: %s', provider, exc)
        return []
    # OpenAI serves per-day snapshots too (gpt-5.4-mini-2026-03-17); collapse them to the
    # catalog family before filtering so a snapshot does not drop the whole model.
    collapsed = []
    for model in ids:
        if provider == 'openai':
            model = ob.base_model(model)
        if is_text_model(model):
            collapsed.append(model)
    ids = sorted(dict.fromkeys(collapsed))
    with _live_lock:
        _live_cache[provider] = (now, ids)
    return list(ids)
