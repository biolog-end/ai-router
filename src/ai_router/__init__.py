"""ai_router — shared keys and fault-tolerant requests to Gemini and OpenAI.

    import ai_router as ai

    ai.keys.add('gemini', 'AIza…')                   # once, for every project of this user
    result = ai.generate('Привет!', model='gemini-3.8-flash', temperature=0.7)
    print(result.text, result.model, result.key_label)

    result = ai.generate(messages=history, system=prompt, chain=['gemini-3.8-flash', 'gpt-5.4-mini'])

Accounting goes to gemini_budget (per-key Gemini quotas) and openai_budget (the free
OpenAI pool); both are shared by every project on the machine.
"""

from . import keys
from .catalog import (DEFAULT_CHAIN, default_chain, default_model, is_retired, is_text_model, light_model,
                      model_info, models, provider_for, set_default_chain)
from .errors import (AllFailedError, Attempt, BudgetError, EmptyResponseError, KeysBusyError, NoKeysError,
                     RequestError, RouterError, SafetyError, StoppedError)
from .keys import LABELS, PROVIDERS
from .request import normalize_messages
from .router import Context, Result, any_client, call, client_for, generate, stream, stream_call
from .status import live_models, reset_pauses, status, test_key

__version__ = '0.1.0'

__all__ = [
    'AllFailedError', 'Attempt', 'BudgetError', 'Context', 'DEFAULT_CHAIN', 'EmptyResponseError',
    'KeysBusyError', 'LABELS', 'NoKeysError', 'PROVIDERS', 'RequestError', 'Result', 'RouterError',
    'SafetyError', 'StoppedError', 'any_client', 'call', 'client_for', 'default_chain', 'default_model',
    'generate', 'is_retired', 'is_text_model', 'keys', 'light_model', 'live_models', 'model_info', 'models',
    'normalize_messages', 'provider_for', 'reset_pauses', 'set_default_chain', 'status', 'stream',
    'stream_call', 'test_key',
]
