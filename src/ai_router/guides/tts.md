# Gemini TTS через ai_router — полный гайд

Как заставить Gemini говорить: какие модели есть, какие голоса, как задать манеру, эмоции,
паузы, диалог, свой голос — и как всё это передаётся через `ai.speak()`. Проверено вживую
2026-10-26 на бесплатных ключах (выводы по русскому языку — в разделе «Русский язык»).

Раздел печатается отдельно: `ai-router tts-guide теги`, `ai-router tts-guide голос`,
список разделов — `ai-router tts-guide --list`. Из Python: `ai.tts_guide('стиль')`.

## Быстрый старт

```python
import ai_router as ai

speech = ai.speak('Привет! <laugh> Наконец-то ты пришёл.', voice='Kore', style='warm, playful')
speech.save('hello.wav')                    # WAV 24 кГц моно 16 бит
print(speech.duration_s, speech.model, speech.key_label, speech.notes)

# лестница моделей: 3.8 → 3.8-lite → 3.1 → 2.5, ключи перебираются на каждой ступени
speech = ai.speak('Текст', voice='Charon', style='calm', chain=True)
```

```bat
ai-router say "Привет! <laugh> Наконец-то ты пришёл." -v Kore -s "warm, playful" -o hello.wav
ai-router say "Текст" --chain
```

## Модели и поколения

| модель | поколение | бесплатно | голоса | манера | теги | поток |
|---|---|---|---|---|---|---|
| `gemini-3.8-flash-tts` | 3.8 | да | все | поле `style` | `<laugh>` | да |
| `gemini-3.8-flash-lite-tts` | 3.8 | да | все | поле `style` | `<laugh>` | да |
| `gemini-3.1-flash-tts-preview` | старое | да | 30 базовых | в тексте `[тег]` | `[laughs]` | да |
| `gemini-2.5-flash-preview-tts` | старое | да | 30 базовых | в тексте | `[laughs]` | нет |
| `gemini-2.5-pro-preview-tts` | старое | **нет** (квота 0) | 30 базовых | в тексте | `[laughs]` | нет |

- **Бесплатная квота** у каждой TTS-модели: 3 запроса в минуту, 10 000 токенов в минуту,
  **10 запросов в сутки** — на проект Google, а не на ключ. Несколько ключей одного проекта
  делят одну квоту; ротация помогает, только если ключи из разных проектов. Сброс суток —
  полночь по Тихоокеанскому времени (09:00/08:00 по Дублину).
- **3.8 Flash** — лучшее качество и игра; **3.8 Flash-Lite** — быстрее и дешевле, Google
  советует её как замену 3.1. **3.1 и 2.5** Google пометил устаревшими, но они работают.
- Поколения говорят на **разных диалектах** (у 3.8 отдельное поле манеры и `<угловые>` теги,
  у 3.1/2.5 всё внутри текста в `[квадратных]`). `ai.speak` принимает форму 3.8 и сам
  переписывает запрос для старой модели, если цепочка до неё дошла (раздел «Лестница»).
- Модель понимает текст **как дословный сценарий**: всё, что в тексте, она пытается
  произнести. Указания режиссёра — в `style`, звуки — тегами.
- Язык определяется сам по тексту, отдельного параметра языка нет. 3.8 Flash знает 130+
  языков, Flash-Lite 100+; русский поддерживают обе.
- Всё аудио Gemini помечено невидимым водяным знаком SynthID.

## Голоса

**30 базовых голосов** работают во всех моделях. Пол и высота — как их помечает сама Google:

| голос | пол | высота | характер | голос | пол | высота | характер |
|---|---|---|---|---|---|---|---|
| Zephyr | ж | высокий | яркий | Puck | м | низкий | бодрый |
| Kore | ж | средний | твёрдый | Charon | м | низкий | информативный |
| Leda | ж | высокий | юный | Fenrir | м | низкий | возбуждённый |
| Aoede | ж | средний | лёгкий | Orus | м | низкий | твёрдый |
| Callirrhoe | ж | средний | непринуждённый | Enceladus | м | низкий | с придыханием |
| Autonoe | ж | средний | яркий | Iapetus | м | низкий | чёткий |
| Despina | ж | средний | гладкий | Umbriel | м | низкий | непринуждённый |
| Erinome | ж | средний | чёткий | Algieba | м | низкий | гладкий |
| Laomedeia | ж | высокий | бодрый | Algenib | м | низкий | с хрипотцой |
| Achernar | ж | высокий | мягкий | Rasalgethi | м | средний | информативный |
| Gacrux | ж | средний | зрелый | Alnilam | м | низкий | твёрдый |
| Vindemiatrix | ж | средний | нежный | Schedar | м | низкий | ровный |
| Sulafat | ж | средний | тёплый | Achird | м | низкий | дружелюбный |
| Pulcherrima | нейтр. | средний | напористый | Zubenelgenubi | м | низкий | разговорный |
| | | | | Sadachbia | м | низкий | живой |
| | | | | Sadaltager | м | средний | знающий |

Полные описания: `ai-router voices --prebuilt`; имена регистронезависимы (`kore` = `Kore`).

**Библиотека** (только 3.8): ~2090 голосов — 1066 женских, 966 мужских, 57 нейтральных;
у каждого язык, акцент, возраст, персона («юрист», «ведущий кулинарного шоу», «комик»),
высота. Это персонажи с готовой манерой. Русских (`ru-RU`) в библиотеке **нет** — по-русски
говорит любой голос, но голос «из Польши» может нести свой акцент (см. «Русский язык»).

```python
ai.voices.list(gender='male', language='pl-PL', pitch='low')
ai.voices.list(search='comedian', limit=20)
ai.voices.list(context='Entertainment & Gaming')
ai.speak('Текст', voice='en-us-zeno')              # id из списка
```

```bat
ai-router voices --gender female --lang en-GB --pitch high
ai-router voices --search narrator --limit 20
```

Фильтры: `language` (BCP-47: `en-US`, `pl-PL`), `region` (`US`, `GB`), `accent`,
`gender` (`female`/`male`/`neutral`), `pitch` (`low`/`medium`/`high`), `persona`, `context`,
`type` (`prebuilt`/`prompted`/`replicated`), `search` (подстрока в имени/описании). Список
значений одного фильтра = «любое из», разные фильтры — «и». `ai.voices.library()` отдаёт
весь каталог (кэш на неделю в `~/.ai_router/voice_library.json`).

**Свои голоса** (только 3.8) — по описанию и по записи, разделы ниже. У них id `voice_…`
(или `voicekey_…`). Такой голос **живёт в проекте Google того ключа, которым создан**:
ключ другого проекта ответит 404. ai_router запоминает владельца в `~/.ai_router/voices.json`
и шлёт запросы с этим голосом через него. Голоса, созданные в AI Studio, он узнаёт командой
`ai-router voices --sync` (или сам — после первого удачного запроса).

## Свой голос по описанию (voice design)

```python
voice = ai.voices.design(
    'Хрипловатый мужчина лет семидесяти, тёплый, неторопливый, с добродушной усмешкой; '
    'говорит по-русски как дедушка, который рассказывает сказку у печки.',
    name='Дед', gender='male', language='ru-RU')
open('preview.wav', 'wb').write(voice['sample_audio'])   # проба голоса от Google
ai.speak('Жили-были дед да баба…', voice=voice['id'])
```

```bat
ai-router voice-design "Хрипловатый мужчина лет семидесяти…" --name Дед --gender male --lang ru-RU -o preview.wav
```

- Описывайте **неизменное**: возраст, пол, тембр, фактуру голоса (хрипотца, придыхание),
  акцент, темп по умолчанию, характер. Коротко и без противоречий: «бодрая спортивная
  комментаторша за тридцать с лёгким акцентом Среднего Запада» лучше абзаца.
- **Настроение сюда не пишите** — оно задаётся в каждом запросе через `style`.
- `gender`: `female` / `male`; `language`: BCP-47 (`ru-RU` принимается); `name` — для себя.
- Создание тратит запрос TTS-квоты 3.8 Flash и идёт через ротацию ключей ai_router.
- До **200 своих голосов на проект**, голос хранится **год с последнего использования**.
- Удалить: `ai.voices.delete(id)` / `ai-router voice-delete ID`. Список: `ai-router voices --custom`.
- В старых моделях (3.1/2.5) своего голоса нет — при откате ai_router подставит базовый
  голос того же пола и высоты и напишет об этом в `result.notes`.

## Клонирование голоса (voice replication)

Копия реального голоса по записи. Только 3.8 Flash / Flash-Lite. **Google разрешает это
только на платном тарифе (Paid Quota Tier 1 и выше)**: с бесплатного ключа запрос
возвращает 403 PERMISSION_DENIED. Включите биллинг хотя бы одному ключу — созданный
`voice_…` потом работает и в бесплатных, и в платных вызовах `speak()`.

**Профиль создаётся один раз.** Google по двум записям строит модель голоса и выдаёт id
(`voice_abcd1234` / `voicekey_…`). Дальше в `ai.speak('любой текст', voice='voice_abcd1234')`
передаётся только этот id — исходные записи больше нигде не нужны. Профилей до 200 на проект.

```python
voice = ai.voices.replicate('source.wav', 'consent.wav', name='Мой голос')
ai.speak('Привет, это я, только синтетический.', voice=voice['id'])
```

```bat
ai-router voice-clone source.wav consent.wav --name "Мой голос"
```

- `source` — **10–30 секунд** чистой естественной речи одного взрослого человека.
- `consent` — тот же человек, тот же микрофон и комната, читает фразу согласия **дословно**.
  По-русски:
  «Я являюсь владельцем этого голоса и даю согласие Google на использование этого голоса
  для создания модели синтетического голоса.»
  По-английски: «I am the owner of this voice and I consent to Google using this voice to
  create a synthetic voice model.» (Google принимает 30 языков, русский среди них.)
- Обе записи — **WAV** (лучше 24 кГц, моно, 16 бит). Из другого формата:
  `ffmpeg -i запись.m4a -ar 24000 -ac 1 -sample_fmt s16 запись.wav`.
- `store=True` (по умолчанию) — постоянный `voice_…` в проекте на год с последнего
  использования; `store=False` (`--no-store`) — самодостаточный `voicekey_…`, который Google
  не хранит, живёт **7 дней**.
- Клонировать можно только свой голос или голос человека, который сам записал согласие.

## Стиль: манера на весь отрывок

У 3.8 манера живёт в отдельном поле (`speech_metadata.style`), `ai.speak(style=…)`:

```python
ai.speak('Ты опять опоздал на целый час!', voice='Charon', style='furious, shouting, exasperated')
ai.speak('Я думала, мы навсегда друзья.', voice='Sulafat', style='sad, quiet, holding back tears, slow')
```

Что писать в `style`:
- **эмоция**: `angry`, `sad`, `excited`, `scared`, `sarcastic`, `tender`, `bored`, `nervous`;
- **подача**: `whispering`, `shouting`, `out of breath`, `monotone and flat`, `like a sports commentator`;
- **темп и высота**: `speaking rapidly`, `speaking slowly`, `high pitch, cheerful inflection`;
- коротко, 2–6 слов через запятую. Сочетайте: `whispering urgently, scared`.

Чего не писать:
- **неизменное** (возраст, пол, акцент) — это выбор голоса или свой голос по описанию;
- длинные «Audio Profile», «Director's notes», «сохраняй тот же тембр» — от них голос
  «плывёт» и 3.8 может начать читать это вслух;
- если эмоция меняется по ходу текста — режьте на реплики, у каждой свой `style`:

```python
ai.speak(voice='Puck', turns=[
    {'text': 'Мы выиграли! Я не могу поверить!', 'style': 'excited, joyful'},
    {'text': 'Подожди… а где билет?', 'style': 'confused, slowing down'},
    {'text': 'Где билет?! Он же был здесь!', 'style': 'panicked, fast'},
])
```

У 3.1/2.5 отдельного поля нет (запрос с ним — ошибка 400 «Speech metadata is not supported
for this model»), поэтому при откате ai_router ставит стиль **тегом в начало реплики**:
`[furious, shouting] Ты опять опоздал…` — так 3.1 его понимает.

## Теги в тексте

### 3.8: угловые теги — мгновенные звуки

Тег ставится ровно туда, где должен прозвучать звук. Пишутся **по-английски**, даже в
русском тексте. Это короткие события, а не манера на весь текст (манера — `style`).

| тег | что это | тег | что это |
|---|---|---|---|
| `<laugh>` / `<laughter>` | смех | `<chuckle>` | смешок |
| `<giggle>` | хихиканье | `<cackle>` | злорадный хохот |
| `<snicker>` | ехидный смешок | `<sigh>` / `<sighs>` | вздох |
| `<gasp>` | ахнуть | `<phew>` | «фух» облегчения |
| `<breath>` | вдох | `<heavy breath>` | тяжёлое дыхание |
| `<exhales>` | выдох | `<pant>` | одышка |
| `<cough>` | кашель | `<throat-clearing>` | прочистить горло |
| `<sneeze>` | чих | `<yawn>` | зевок |
| `<sob>` | всхлип | `<cry>` | плач |
| `<whimper>` | хныканье | `<groan>` | стон досады |
| `<moan>` | стон | `<grunt>` | кряхтение |
| `<growl>` / `<grr>` | рык | `<argh>` | «аргх» |
| `<hiss>` | шипение | `<snort>` | фырканье |
| `<tsk>` | цоканье | `<pff>` | «пфф» |
| `<cheer>` | радостный возглас | `<shout>` | выкрик |
| `<scream>` | крик | `<shriek>` | визг |
| `<whispers>` / `<whispering>` | переход на шёпот | `<short pause>` / `<long pause>` | пауза |

```python
ai.speak('<whispers> Тише… они рядом. <short pause> Слышишь? <gasp> Нет, это кот. <laugh> Ну ты и трус!')
```

### 3.1 / 2.5: квадратные теги — эмоции, темп, звуки

У 3.1 «200+ тегов», закрытого списка нет; работают английские слова и короткие фразы.
Правила Google: теги по-английски, **два тега подряд нельзя** — между ними текст или знак.

- эмоции: `[excitement]`, `[enthusiasm]`, `[curiosity]`, `[confusion]`, `[nervousness]`,
  `[frustration]`, `[annoyance]`, `[anger]`, `[aggression]`, `[tension]`, `[panic]`,
  `[anxiety]`, `[alarm]`, `[relief]`, `[hope]`, `[awe]`, `[admiration]`, `[adoration]`,
  `[amusement]`, `[determination]`, `[positive]`, `[neutral]`, `[negative]`;
- подача: `[whispers]`, `[shouting]`, `[sarcastic]`, `[serious]`, `[tired]`, `[trembling]`,
  `[mischievously]`, `[robotic]`, `[crying]`;
- звуки: `[laughs]`, `[giggles]`, `[sighs]`, `[gasp]`, `[coughs]`, `[snorts]`, `[yawns]`;
- темп: `[slow]`, `[fast]`, `[short pause]`, `[medium pause]`, `[long pause]`;
- свободная форма тоже работает: `[furious, shouting]`, `[cautious]`.

### Автоперевод в ai_router

Пишите в форме 3.8 — ai_router переведёт для старой модели, и наоборот:

| вы пишете | 3.8 получает | 3.1/2.5 получают |
|---|---|---|
| `style='scared'` | поле `style` | `[scared] текст` |
| `<laugh>`, `<sigh>`, `<yawn>` | как есть | `[laughs]`, `[sighs]`, `[yawns]` |
| `<whispers>` + `style` в начале | как есть | `[scared, whispers] …` (слитно — подряд нельзя) |
| `[laughs]`, `[gasp]` (звук) | `<laugh>`, `<gasp>` | как есть |
| `[excitement] А [confusion] Б` (эмоция) | реплики со `style`: excitement / confusion | как есть |
| `\|ага\|` (подхват в диалоге) | как есть | убирается, заметка в `notes` |

## Паузы, темп, ударение

- **Знаки препинания** — главный инструмент: запятая, тире, многоточие дают естественную
  заминку; `?!` и `!!!` поднимают напор.
- **Паузы**: `<short pause>`, `<long pause>` (3.8); `[short pause]`, `[long pause]` (3.1).
- **Темп** всего отрывка — в `style`: `speaking slowly` / `speaking rapidly`; у 3.1 — `[slow]` / `[fast]`.
- **Ударение на слове**: КАПСОМ — «Это ОЧЕНЬ важно». Для русского ударения в слове
  надёжного способа нет; помогает переписать слово так, как оно звучит, или поставить
  знак ударения (а́) — проверяйте на слух.
- **Живость**: модель хорошо отыгрывает разговорные «ну», «э-э», «хм», повторы и обрывы фраз.

## Диалоги: несколько дикторов

```python
ai.speak(speakers={'Anya': 'Kore', 'Boris': 'Charon'}, turns=[
    {'speaker': 'Anya', 'text': 'Борь, вставай! Мы выиграли в лотерею!', 'style': 'excited, fast'},
    {'speaker': 'Boris', 'text': '<yawn> Ань, сейчас шесть утра… Какую ещё лотерею?', 'style': 'sleepy, grumpy'},
    {'speaker': 'Anya', 'text': 'Ту самую! Миллион! <laugh>'},
])
```

```bat
ai-router say "Anya: Борь, вставай!
Boris: <yawn> Шесть утра…" --speaker Anya=Kore --speaker Boris=Charon -o dialog.wav
```

- В одном запросе — **до 2 дикторов** с базовыми или библиотечными голосами.
- Имя диктора — буквы и цифры без пробелов; у каждой реплики `speaker` из `speakers`.
- **Подхваты** (3.8): реплика собеседника внутри чужой фразы в `|…|` —
  `'Запуск в четверг |ого| мы точно готовы?'`. Перекрывающаяся речь: `'раз. два. три. |с днём| с днём |рождения| рождения!'`.
- Google с 2026-10-23 отвергает `"mode": "conversational"` (ошибка 400) — ai_router его не шлёт.
- **3 и больше дикторов или свои голоса** в диалоге: ai_router озвучивает каждую реплику
  отдельным запросом и склеивает с паузой 0,25 с. Это N запросов квоты вместо одного.
- У 3.1/2.5 диалог уходит текстом `Anya: …` построчно с `multiSpeakerVoiceConfig`.

## Формат аудио и поток

| `format` | что в `result.audio` | частота |
|---|---|---|
| `wav` (по умолчанию) | WAV, 16 бит, моно | 24 000 |
| `pcm` | сырой 16-бит little-endian PCM без заголовка | 24 000 |
| `mulaw`, `alaw` | G.711, 8 бит (телефония) | 8 000 обычно |

- `sample_rate=24000 | 16000 | 8000` и форматы `mulaw/alaw` умеют только модели 3.8; старые
  отдают 24 кГц PCM. С такими настройками ai_router сам выкидывает старые модели из цепочки.
- `result.pcm` — сэмплы без заголовка, `result.duration_s`, `result.save(path)`.
- В запросе Google формат задаётся перечислением (`AUDIO_L16`, `AUDIO_MULAW`…), MIME-строка
  вроде `audio/l16` — ошибка 400. ai_router делает это сам.

**Поток** — звук приходит кусками по мере генерации (для голосовых ответов в реальном времени):

```python
for event in ai.speak_stream('Длинный текст…', voice='Kore', chain=True):
    if event['type'] == 'delta':
        player.write(event['audio'])          # PCM 16 бит, 24 кГц, моно
    elif event['type'] == 'done':
        event['result'].save('all.wav')
```

Ключ и модель меняются только до первого куска. 2.5 не стримит — отдаёт всё одним куском.

## Лестница моделей и ключи

- `model='…'` — одна модель; `chain=[…]` — своя лестница; `chain=True` — общая TTS-цепочка
  (`ai-router tts-chain` показать/задать; по умолчанию 3.8 Flash → 3.8 Flash-Lite → 3.1 → 2.5).
- На каждой ступени ai_router перебирает ключи: 429 → пауза этой пары ключ+модель
  (минута / до полуночи по Тихоокеанскому / 30 мин для квоты 0), следующий ключ сразу;
  5xx → два повтора с паузой; все ключи не смогли → следующая модель.
- Иногда TTS вместо звука генерирует текст, и сервер отвечает 500 — это лечится повтором,
  роутер делает его сам.
- Учёт бесплатной квоты — общий для всех программ (`gemini_budget`): если у ключа исчерпаны
  10 запросов в сутки, его даже не пытаются.
- Свой голос — запрос идёт ключом-владельцем. Неизвестный владелец — перебираются все ключи,
  первый удачный запоминается.
- При откате на 3.1/2.5 всё, что модель не умеет, переводится или заменяется, а что
  заменено — написано в `result.notes`.

## ai.speak() целиком

```python
ai.speak(
    text=None,                 # или turns=[{'text', 'style'?, 'speaker'?}, …]
    voice='Kore',              # базовый, из библиотеки, voice_…/voicekey_…
    style=None,                # манера по умолчанию для всех реплик
    turns=None,
    speakers=None,             # {'Имя': 'Голос'} — диалог
    model=None, chain=None,    # одна модель / лестница / True — общая
    format='wav', sample_rate=None,
    extra=None,                # dict, вливается в тело generateContent (новые поля API)
    on_event=None,             # прогресс: attempt / failure / retry / next_model / success
    key_id=None, prefer_key=None, try_all_keys=False, stop_event=None, timeout_s=300,
)
```

`SpeechResult`: `audio`, `format`, `mime_type`, `sample_rate`, `pcm`, `duration_s`,
`save(path)`, `voices` (`{'': 'Kore'}` или `{'Anya': 'Kore', …}` — какие голоса реально
звучали), `model`, `key_id`, `key_label`, `position`/`fallback_used`, `usage`, `notes`,
`attempts`, `finish_reason`, `response` (ответ Google без base64). `to_dict()` — без аудио.

Ошибки — те же, что у `generate()`: `KeysBusyError` (все ключи на паузе, есть `retry_at`),
`AllFailedError`, `RequestError` (кривой запрос), `SafetyError`, `EmptyResponseError` (модель
не вернула звук; причина в тексте), `NoKeysError`.

`ai.speak_stream(...)` — те же параметры, кроме `sample_rate`; `format` только `wav`/`pcm`.

Голоса: `ai.voices.prebuilt()`, `.list(...)`, `.library()`, `.custom()`, `.sync()`,
`.design(...)`, `.replicate(...)`, `.get(id)`, `.delete(id)`, `.describe(id)`,
`.CONSENT_PHRASES`.

## Командная строка

```bat
ai-router say "текст" [-v ГОЛОС] [-s СТИЛЬ] [-m МОДЕЛЬ | --chain] [-o out.wav]
              [--format wav|pcm|mulaw|alaw] [--rate 24000|16000|8000] [--stream]
              [--speaker Имя=Голос …]          :: с --speaker текст — строки «Имя: реплика»
ai-router voices [--prebuilt | --custom | --sync] [--lang …] [--gender …] [--pitch …]
                 [--accent …] [--persona …] [--context …] [--type …] [--search …] [--limit N] [--json]
ai-router voice-design "описание" [--name] [--gender] [--lang ru-RU] [-o preview.wav]
ai-router voice-clone source.wav consent.wav [--name] [--no-store]
ai-router voice-delete voice_…
ai-router tts-chain [МОДЕЛЬ …]                 :: показать / задать общую TTS-цепочку
ai-router tts-guide [раздел] [--list]          :: этот гайд
```

## Ошибки и подводные камни

- **400 «Speech metadata is not supported for this model»** — `style` отдельным полем у
  3.1/2.5. ai_router переводит стиль в тег сам.
- **400 «Unknown parameter 'mode'»** — `mode: conversational` убран Google 2026-10-23.
- **400 про `response_format.audio.mime_type`** — нужен `AUDIO_L16`, а не `audio/l16`.
- **Ответ без звука, `finishReason: OTHER`** — текст не похож на реплику: одни теги, служебный
  текст, теги на русском. Перепишите как живую речь.
- **`SAFETY` на безобидном тексте** у 3.1 — бывает от русских слов в квадратных скобках
  («[злобно] Убирайся!»). Теги — только по-английски.
- **Модель читает указания вслух** — у 3.1/2.5 при длинных/расплывчатых указаниях в тексте;
  держите стиль коротким тегом в начале. У 3.8 — если писать указания в сам текст, а не в `style`.
- **500 «model returned text instead of audio»** — редкий сбой, лечится повтором.
- **404 на свой голос** — запрос ушёл ключом другого проекта. `ai-router voices --sync`
  запомнит владельцев.
- **10 запросов в сутки на модель и проект** — для серьёзной озвучки нужны ключи из
  разных проектов или платный тариф; `ai-router status` покажет, что на паузе.
- Длинный текст: TPM 10 000 на модель — один запрос примерно до 30 000 символов русского
  текста, но качество и стабильность лучше на кусках по абзацу.

## Источники

- Gemini API: Text-to-speech — https://ai.google.dev/gemini-api/docs/speech-generation
- generateContent TTS (legacy) — https://ai.google.dev/gemini-api/docs/generate-content/speech-generation
- Voice design — https://ai.google.dev/gemini-api/docs/voice-design
- Voice replication — https://ai.google.dev/gemini-api/docs/voice-replication
- Voices API — https://ai.google.dev/api/voices
- Теги 3.1 — https://cloud.google.com/blog/products/ai-machine-learning/gemini-3-1-flash-tts-on-google-cloud
- Cloud Gemini-TTS (пол голосов, языки) — https://docs.cloud.google.com/text-to-speech/docs/gemini-tts
- Форум: mode conversational → 400 — https://discuss.ai.google.dev/t/195156
