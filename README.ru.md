# ai_router

[English](README.md) | **Русский**

Общее хранилище API-ключей Gemini и OpenAI плюс устойчивый вызов нейросетей: ротация
ключей при 429/5xx, паузы по модели, опциональная цепочка моделей
(`gemini-3.8-flash → … → gpt-5.4-mini`) и учёт квот через
[gemini-budget](https://github.com/biolog-end/gemini-budget) и
[openai-budget](https://github.com/biolog-end/openai-budget). У всех проектов текущего
пользователя одно хранилище ключей, одни счётчики и одни паузы.

## Установка

`gemini-budget` и `openai-budget` обязательны и на PyPI их нет, поэтому сначала ставятся
они, именно в таком порядке:

```bat
pip install git+https://github.com/biolog-end/gemini-budget.git
pip install git+https://github.com/biolog-end/openai-budget.git
pip install git+https://github.com/biolog-end/ai-router.git
pip install google-genai openai        :: SDK, которые дёргает ai_router
```

Для разработки — редактируемые копии трёх библиотек:

```bat
pip install -e %USERPROFILE%\gemini_budget -e %USERPROFILE%\openai_budget -e %USERPROFILE%\ai_router
```

Ключи лежат в `~/.ai_router/keys.json`. Переменная `AI_ROUTER_DIR` переносит хранилище.
Admin-ключ `openai_budget` живёт отдельно (`~/.openai_budget/config.json`); ключ для
запросов OpenAI — проектный (`sk-proj-…`), `sk-admin-…` библиотека принципиально не
принимает.

## Быстрый старт

```python
import ai_router as ai

ai.keys.add("gemini", "AIza…", label="Проект 1")
ai.keys.add("openai", "sk-proj-…", label="OpenAI")

result = ai.generate("Привет!", model="gemini-3.8-flash", temperature=0.7)
print(result.text, result.model, result.key_label)
```

Потоковый ответ:

```python
for event in ai.stream("Расскажи сказку", model="gemini-3.6-flash"):
    if event["type"] == "delta":
        print(event["text"], end="")
```

Цепочка между провайдерами (один вызов пробует модели по очереди):

```python
result = ai.generate(messages=history, system=prompt,
                     chain=["gemini-3.8-flash", "gpt-5.4-mini"])
```

Низкий уровень: сам соберёшь запрос провайдера, ai_router только перебирает ключи.

```python
def my_request(client, ctx):
    return client.responses.create(model=ctx.model, input=..., tools=my_tools)

result = ai.call(my_request, model="gpt-5.4", input_tokens=200)
```

## Что делает ai_router

- **Единое хранилище ключей** для всех программ одного пользователя: `ai.keys.add`,
  `remove`, `update`, `replace`, `list_keys`, `secret`, `secrets`, `import_env`.
  Секреты наружу не отдаются; `list_keys` возвращает только маски и отпечатки.
- **Попытки.** На один `generate()` или `stream()` ai_router идёт по цепочке моделей,
  внутри модели — по ключам её провайдера от самого «отдохнувшего». Пропускает ключи
  в кулдауне (на модель или целиком), отклонённые API, а для бесплатных моделей Gemini —
  и те, у кого кончилась квота. При ответе 429 ключ ставится на паузу на эту модель
  (минутную, суточную или до правки лимитов). Ключ, который уже начал потоково
  отдавать текст, не меняется на лету.
- **Нормализованные ошибки.** Из любого SDK-исключения получается `RouterError`-подкласс,
  которого ждёт интерфейс: `NoKeysError`, `KeysBusyError`, `BudgetError`, `SafetyError`,
  `EmptyResponseError`, `RequestError`, `AllFailedError`, `StoppedError`. У каждой
  ошибки есть `attempts` (по одной `Attempt` на попытку) и `retry_at`, если известно.
  Секретов в этих объектах нет.
- **Формат запроса.** `ai.generate(prompt=..., messages=...)` принимает нейтральные
  сообщения (`{"role": "user"|"model"|"assistant"|"developer", "parts": [{"text": ...} |
  бинарник]}`) и сам собирает полезную нагрузку провайдера. Параметры, которые эта
  модель не принимает, молча пропускаются и перечислены в `result.notes` (например,
  `top_k` у OpenAI или `reasoning` у не-reasoning моделей). Размышления:
  `thinking=True/False/"low"/"medium"/"high"` или число токенов. Google Search —
  `search=True`. JSON — `json_mode=True` или `json_schema=<pydantic.BaseModel | dict>`.
- **Паузы общие.** Квоты Gemini (минута/день/ноль) живут в `gemini_budget` для пары
  (ключ, модель) и видны всем проектам пользователя. Остальные паузы (отклонённые ключи,
  «нет кредитов» OpenAI, пер-модельные OpenAI-кулдауны) — в `~/.ai_router/state.json`.
- **Бесплатный пул OpenAI.** Для модели из бесплатной группы (`openai_budget.group_for`)
  `generate()` сам откажет запросу, который вылезет в платный тариф. `allow_paid=True`
  в вызове отключает защиту (не глобально). С `allow_paid=True` включается веб-поиск
  OpenAI; иначе `search=True` просто пропускается с заметкой.
- **Повторов SDK нет.** У OpenAI `max_retries=0`, у Gemini `attempts=1`, иначе SDK
  спрятал бы от пула 429/5xx.

## Статус

```python
status = ai.status()                                     # ключи, паузы, заблокированные модели
status = ai.status(provider="gemini")                    # только Gemini
ai.reset_pauses()                                        # после «починки» биллинга
ai.reset_pauses(key_id, model="gemini-3.8-flash")        # одна строка
ai.test_key(key_id, model="gpt-5.4-mini")                # явная проверка
```

В строке статуса есть и `quota` (снимок бесплатных моделей Gemini), и `cooldowns`;
`blocked_models` — модели, недоступные одновременно на всех рабочих ключах их
провайдера. `status()` сетевых вызовов не делает — это снимок.

## Модели

```python
ai.models()                                 # каталог Gemini + OpenAI
ai.models(provider="gemini", task="chat")   # task: 'chat' | 'tts' | 'transcription' | …
ai.model_info("gpt-5.4-mini")               # строка каталога одной модели
ai.is_text_model(name)
ai.is_retired("gemini-2.0-flash")           # текст объяснения, если Google её выключил
ai.live_models("gemini")                    # id моделей, доступных первому включённому ключу
ai.default_model("openai")                  # дефолты по провайдерам
ai.light_model("gemini")                    # дешёвая модель того же провайдера для памяти
```

## Live-модели

`gemini-3.8-live`, `gemini-3.1-flash-live-preview` и `gemini-2.5-flash-native-audio-*`
работают через Live API (websocket-сессия), но отвечают в `generate()`/`stream()` как обычные
модели, в цепочке тоже. На бесплатном тарифе у них считается только TPM (65K у 3.8), без
лимитов RPM/RPD.

```python
result = ai.generate("Привет!", model="gemini-3.8-live")
for event in ai.stream("Расскажи сказку", model="gemini-3.8-live"):
    ...
speech = ai.generate("Привет!", model="gemini-3.8-live", audio=True, voice="Kore")
speech.text, speech.save("answer.wav")                   # SpeechResult: текст и его озвучка
ai.is_live_model("gemini-3.8-live")                       # True
```

- Live-модель всегда говорит; текст — расшифровка её речи и приходит со скоростью речи
  (ответ на 300 слов идёт около 100 с). Код и JSON приходят текстом без озвучки.
- Каждый вызов — одна сессия: вся переписка (с картинками) уходит одним ходом.
- Применяются `system`, `temperature`, `top_p`, `top_k`, `seed`, `voice` (30 базовых голосов);
  `max_output_tokens`, `stop`, штрафы, `safety`, `search`, `thinking` отбрасываются в
  `result.notes`. `json_mode`/`json_schema` дают `RequestError` — цепочка идёт дальше.
- С `audio=True` `stream()` отдаёт ещё `{'type': 'delta', 'audio': PCM, 'sample_rate': 24000}`.
- `timeout_s` — самая долгая тишина между сообщениями, а не длина ответа.
- `*-live-translate-*`, `*-transcribe-live` (только звук) и `*-extended-thinking` (не отвечает
  на текст) дают `RequestError`.

## Цепочка моделей по умолчанию

Общая цепочка применяется, когда вызов передал `chain=True` без списка.

```python
ai.default_chain()                                       # ['gemini-3.8-flash', ...]
ai.set_default_chain(["gemini-3.8-flash", "gpt-5.4-mini"])
```

Из командной строки: `ai-router chain gemini-3.8-flash gpt-5.4-mini`.

## Командная строка

```bat
ai-router keys                               :: список ключей (без секретов)
ai-router add gemini AIza…
ai-router add openai sk-proj-…
ai-router enable|disable <key_id>
ai-router remove <key_id>
ai-router reset [<key_id>]                   :: снять паузы после починки биллинга
ai-router status                             :: паузы, счётчики, модели в блоке, бюджеты
ai-router models --provider gemini --free    :: каталог: провайдер / задача / только бесплатные
ai-router chain [MODEL ...]                  :: показать или задать цепочку по умолчанию
ai-router test --key <key_id> [--model …]    :: явная проверка
ai-router ask "привет" [--chain]             :: запрос через роутер
ai-router path                               :: где хранилище и состояние
```

## Изоляция

Все три библиотеки открывают хранилище через `AI_ROUTER_DIR` (по умолчанию
`~/.ai_router`). Та же переменная в тестах указывает на временный каталог. Файл
записывается атомарно с блокировкой ОС, параллельные запуски не могут его повредить.
