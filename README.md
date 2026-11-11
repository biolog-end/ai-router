# ai_router

**English** | [Русский](README.ru.md)

Shared API-key vault for Gemini and OpenAI, plus fault-tolerant requests: key rotation
on 429/5xx, per-model paused keys, an optional model chain (`gemini-3.8-flash → … →
gpt-5.4-mini`), and quota accounting through [gemini-budget](https://github.com/biolog-end/gemini-budget)
and [openai-budget](https://github.com/biolog-end/openai-budget). Every project of this
user shares the same vault, counters and pauses.

## Install

`gemini-budget` and `openai-budget` are required and are not on PyPI, so install them
first, in this order:

```bat
pip install git+https://github.com/biolog-end/gemini-budget.git
pip install git+https://github.com/biolog-end/openai-budget.git
pip install git+https://github.com/biolog-end/ai-router.git
pip install google-genai openai        :: the SDKs ai_router drives
```

For development, install editable copies of the three libraries:

```bat
pip install -e %USERPROFILE%\gemini_budget -e %USERPROFILE%\openai_budget -e %USERPROFILE%\ai_router
```

Keys live in `~/.ai_router/keys.json`. Set `AI_ROUTER_DIR` to move it. The admin key of
openai_budget stays separate (`~/.openai_budget/config.json`); the OpenAI key for
requests is a project key (`sk-proj-…`), and `ai_router` refuses an `sk-admin-…` with a
clear error.

## Quick start

```python
import ai_router as ai

ai.keys.add("gemini", "AIza…", label="Проект 1")
ai.keys.add("openai", "sk-proj-…", label="OpenAI")

result = ai.generate("Привет!", model="gemini-3.8-flash", temperature=0.7)
print(result.text, result.model, result.key_label)
```

Streaming:

```python
for event in ai.stream("Расскажи сказку", model="gemini-3.6-flash"):
    if event["type"] == "delta":
        print(event["text"], end="")
```

Chain across providers (one call tries every model until one answers):

```python
result = ai.generate(messages=history, system=prompt,
                     chain=["gemini-3.8-flash", "gpt-5.4-mini"])
```

Low-level: build the provider's own request yourself and let ai_router rotate keys.

```python
def my_request(client, ctx):
    return client.responses.create(model=ctx.model, input=..., tools=my_tools)

result = ai.call(my_request, model="gpt-5.4", input_tokens=200)
```

## What ai_router does

- **One key vault** across every project of this user. `ai.keys.add`, `remove`, `update`,
  `replace`, `list_keys`, `secret`, `secrets`, `import_env`. Secrets never leave server-
  side helpers; `list_keys` returns only masked rows and fingerprints.
- **Attempts.** For one `generate()` or `stream()`, `ai_router` runs the model chain in
  order; for each model it tries that provider's keys least-recently-used first,
  skipping keys that are paused (per-model or whole-key), rejected by the provider, or
  short on their free Gemini quota. A failed key records a pause (minute/day/zero) for
  that `(key, model)` pair and the next key is tried immediately. A key that already
  streamed text is never replaced mid-reply.
- **Normalised errors.** The provider SDK error turns into `RouterError` subclasses the
  UI already knows: `NoKeysError`, `KeysBusyError`, `BudgetError`, `SafetyError`,
  `EmptyResponseError`, `RequestError`, `AllFailedError`, `StoppedError`. Every
  exception carries `attempts` (one `Attempt` per key tried) and `retry_at` where known.
  Nothing in these objects contains a secret.
- **Request shapes.** `ai.generate(prompt=..., messages=...)` accepts neutral messages
  (`{"role": "user"|"model"|"assistant"|"developer", "parts": [{"text": ...} | binary]}`)
  and builds the provider's own payload. Options the SDK rejects for this model are
  dropped and listed in `result.notes` (e.g. `top_k` for OpenAI, `reasoning` for
  non-reasoning models). Reasoning setting: `thinking=True/False/"low"/"medium"/"high"`
  or a token budget. Google Search via `search=True`. JSON mode via `json_mode=True` or
  `json_schema=<pydantic.BaseModel | dict>`.
- **Pauses are shared.** Gemini quota (minute/day/zero) is stored in `gemini_budget`
  per `(key, model)` and seen by every project of this user. Other pauses (whole-key
  rejections, OpenAI "no credits", OpenAI per-model quota) live in `~/.ai_router/state.json`.
- **OpenAI free-pool guard.** For a model in a free group (`openai_budget.group_for`),
  `generate()` refuses a request that would spill into paid billing. Pass `allow_paid=True`
  to override; the setting lives inside the request, not globally. `allow_paid=True` also
  turns on OpenAI web_search; otherwise `search=True` is dropped with a note.
- **No SDK retries.** OpenAI's `max_retries` is set to 0 and Gemini's `attempts=1`,
  because the SDKs would hide 429/5xx from the pool.

## Status

```python
status = ai.status()                                     # list of keys, pauses, blocked models
status = ai.status(provider="gemini")                    # only Gemini
ai.reset_pauses()                                        # clear pauses after fixing billing
ai.reset_pauses(key_id, model="gemini-3.8-flash")        # one row only
ai.test_key(key_id, model="gpt-5.4-mini")                # explicit probe
```

The status row carries both the Gemini `quota` snapshot (per free chat model) and the
`cooldowns` list; `blocked_models` names the models unavailable on every usable key of
their provider. Reading `status()` never performs network calls — it's a snapshot.

## Models

```python
ai.models()                                 # Gemini + OpenAI catalog rows
ai.models(provider="gemini", task="chat")   # task is 'chat' | 'tts' | 'transcription' | ...
ai.model_info("gpt-5.4-mini")               # catalog row of one model
ai.is_text_model(name)
ai.is_retired("gemini-2.0-flash")           # text explaining the Google retirement, or None
ai.live_models("gemini")                    # text model ids this user's first enabled key sees
ai.default_model("openai")                  # provider defaults for the UI
ai.light_model("gemini")                    # a cheap model of the same provider (memory, summaries)
```

## Live models

`gemini-3.8-live`, `gemini-3.1-flash-live-preview` and `gemini-2.5-flash-native-audio-*`
work through the Live API (a websocket session) but answer `generate()`/`stream()` like any
other model, in a chain too. The free tier counts only TPM for them (65K for 3.8), with no
RPM/RPD limit.

```python
result = ai.generate("Привет!", model="gemini-3.8-live")
for event in ai.stream("Расскажи сказку", model="gemini-3.8-live"):
    ...
speech = ai.generate("Привет!", model="gemini-3.8-live", audio=True, voice="Kore")
speech.text, speech.save("answer.wav")                   # SpeechResult: the text plus its audio
ai.is_live_model("gemini-3.8-live")                       # True
```

- A Live model always speaks; the text is the transcription of that speech and arrives at
  speaking pace (a 300-word answer takes about 100 s). Code and JSON come as text without speech.
- Every call is one session: the whole conversation (images included) goes in as one turn.
- `system`, `temperature`, `top_p`, `top_k`, `seed`, `voice` (30 prebuilt voices) are applied;
  `max_output_tokens`, `stop`, penalties, `safety`, `search`, `thinking` are dropped into
  `result.notes`. `json_mode`/`json_schema` raise `RequestError`, so a chain moves on.
- With `audio=True`, `stream()` also yields `{'type': 'delta', 'audio': PCM, 'sample_rate': 24000}`.
- `timeout_s` is the longest silence between two messages, not the length of the answer.
- `*-live-translate-*`, `*-transcribe-live` (audio only) and `*-extended-thinking` (does not
  answer text) raise `RequestError`.

## Default model chain

The shared chain is used when a call passes `chain=True` without a list.

```python
ai.default_chain()                                       # ['gemini-3.8-flash', ...]
ai.set_default_chain(["gemini-3.8-flash", "gpt-5.4-mini"])
```

Edit it from the command line: `ai-router chain gemini-3.8-flash gpt-5.4-mini`.

## Command line

```bat
ai-router keys                               :: list of keys (no secrets)
ai-router add gemini AIza…
ai-router add openai sk-proj-…
ai-router enable|disable <key_id>
ai-router remove <key_id>
ai-router reset [<key_id>]                   :: clear pauses after fixing billing
ai-router status                             :: pauses, counters, blocked models, budgets
ai-router models --provider gemini --free    :: catalog filtered by provider / task / free tier
ai-router chain [MODEL ...]                  :: show or set the default chain
ai-router test --key <key_id> [--model …]    :: explicit probe
ai-router ask "привет" [--chain]             :: send a request through the router
ai-router path                               :: where the vault and state live
```

## Isolation

Every project opens the vault via `AI_ROUTER_DIR`, defaulting to `~/.ai_router`. Set the
same variable in tests to point every library at a temporary directory. The vault file
is written atomically with an OS lock, so parallel programs cannot corrupt it.
