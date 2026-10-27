"""Speech synthesis with Gemini TTS through the same attempt loop as generate().

Two generations of TTS models speak different dialects:

- 3.8+ (gemini-3.8-flash-tts, gemini-3.8-flash-lite-tts): the text is a verbatim
  transcript, sustained delivery goes to speech_metadata.style, momentary sounds are
  `<laugh>`-style tags; any voice works (prebuilt, library, designed, replicated).
- legacy (gemini-3.1-flash-tts-preview, gemini-2.5-*-tts): direction lives inside the
  text as `[tags]` and only the 30 prebuilt voices exist.

speak() takes the 3.8 form (and understands 3.1-style `[tags]` too) and rewrites it for
whichever model of the chain answers.
"""

import base64
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from . import catalog, voices
from .errors import EmptyResponseError, SafetyError
from .request import SAFETY_FINISH_REASONS
from .router import Result, _drive, _execute, _routing, _stream

DEFAULT_TTS_MODEL = 'gemini-3.8-flash-tts'
SAMPLE_RATE = 24000
FORMATS = {'wav': 'audio/wav', 'pcm': 'audio/l16', 'mulaw': 'audio/mulaw', 'alaw': 'audio/alaw'}
# generateContent takes the enum name, not the MIME type ("audio/l16" is a 400).
REQUEST_FORMATS = {'wav': 'AUDIO_WAV', 'pcm': 'AUDIO_L16', 'mulaw': 'AUDIO_MULAW', 'alaw': 'AUDIO_ALAW'}
SAMPLE_RATES = (24000, 16000, 8000)
MAX_SPEAKERS = 2
TURN_GAP_S = 0.25

# 3.8 inline vocal tags → the closest 3.1 audio tag.
TO_LEGACY = {
    'laugh': 'laughs', 'laughter': 'laughs', 'chuckle': 'chuckles', 'giggle': 'giggles', 'cackle': 'cackles',
    'snicker': 'snickers', 'sigh': 'sighs', 'sighs': 'sighs', 'gasp': 'gasp', 'cough': 'coughs',
    'yawn': 'yawns', 'sob': 'sobbing', 'cry': 'crying', 'whimper': 'whimpering', 'scream': 'screaming',
    'shout': 'shouting', 'shriek': 'shrieking', 'whispers': 'whispers', 'whispering': 'whispers',
    'breath': 'breathes in', 'heavy breath': 'heavy breathing', 'exhales': 'exhales', 'pant': 'panting',
    'groan': 'groans', 'growl': 'growls', 'grr': 'growls', 'grunt': 'grunts', 'hiss': 'hisses',
    'moan': 'moans', 'snort': 'snorts', 'sneeze': 'sneezes', 'throat-clearing': 'clears throat',
    'tsk': 'tsk', 'pff': 'scoffs', 'phew': 'sighs with relief', 'argh': 'frustrated groan',
    'cheer': 'cheering', 'short pause': 'short pause', 'long pause': 'long pause',
}
# 3.1 tags that are point-in-time sounds in 3.8 too; every other [tag] is a mood → style.
TO_MODERN = {
    'laughs': 'laugh', 'laughing': 'laugh', 'chuckles': 'chuckle', 'giggles': 'giggle', 'sighs': 'sigh',
    'gasp': 'gasp', 'gasps': 'gasp', 'coughs': 'cough', 'yawns': 'yawn', 'sobbing': 'sob', 'sobs': 'sob',
    'crying': 'cry', 'whispers': 'whispers', 'whispering': 'whispers', 'screaming': 'scream',
    'screams': 'scream', 'shouting': 'shout', 'shouts': 'shout', 'groans': 'groan', 'snorts': 'snort',
    'sneezes': 'sneeze', 'clears throat': 'throat-clearing', 'short pause': 'short pause',
    'medium pause': 'short pause', 'long pause': 'long pause', 'pause': 'short pause', 'tsk': 'tsk',
}
_ANGLE = re.compile(r'<\s*([A-Za-z][A-Za-z \-]*?)\s*>')
_SQUARE = re.compile(r'\[\s*([^\[\]\n]+?)\s*\]')
_PIPES = re.compile(r'\s*\|[^|\n]*\|')


def is_modern(model):
    """3.8 and newer: speech_metadata, angle tags, every kind of voice."""
    match = re.match(r'gemini-(\d+)(?:\.(\d+))?', catalog.canonical(model))
    return bool(match) and (int(match[1]), int(match[2] or 0)) >= (3, 8)


def can_stream(model):
    match = re.match(r'gemini-(\d+)(?:\.(\d+))?', catalog.canonical(model))
    return bool(match) and (int(match[1]), int(match[2] or 0)) >= (3, 1)


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------

def wav(pcm, sample_rate=SAMPLE_RATE):
    """Mono 16-bit PCM → WAV bytes."""
    return (b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVEfmt '
            + struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b'data' + struct.pack('<I', len(pcm)) + pcm)


def wav_parts(data):
    """(sample_rate, payload) of a RIFF file; the payload of a non-RIFF blob is the blob itself."""
    if data[:4] != b'RIFF':
        return None, data
    rate, offset = None, 12
    while offset + 8 <= len(data):
        chunk, size = data[offset:offset + 4], struct.unpack('<I', data[offset + 4:offset + 8])[0]
        body = data[offset + 8:offset + 8 + size]
        if chunk == b'fmt ':
            rate = struct.unpack('<I', body[4:8])[0]
        elif chunk == b'data':
            return rate, body
        offset += 8 + size + (size & 1)
    return rate, b''


def _mime_rate(mime):
    match = re.search(r'rate=(\d+)', mime or '')
    return int(match.group(1)) if match else None


def _silence(fmt, sample_rate, seconds):
    count = int(sample_rate * seconds)
    return {'mulaw': b'\xff' * count, 'alaw': b'\xd5' * count}.get(fmt, b'\x00\x00' * count)


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass
class SpeechResult(Result):
    """Result of speak(): audio in the requested format plus the usual routing details."""
    audio: bytes = b''
    format: str = 'wav'
    mime_type: str = 'audio/wav'
    sample_rate: int = SAMPLE_RATE
    voices: dict = field(default_factory=dict)

    @property
    def pcm(self):
        """Raw samples without a container (bytes per sample: 2 for wav/pcm, 1 for mulaw/alaw)."""
        return wav_parts(self.audio)[1] if self.format == 'wav' else self.audio

    @property
    def duration_s(self):
        width = 2 if self.format in ('wav', 'pcm') else 1
        return round(len(self.pcm) / (self.sample_rate * width), 2) if self.sample_rate else 0.0

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.audio)
        return path

    def to_dict(self):
        row = super().to_dict()
        row.pop('audio', None)
        return row | {'bytes': len(self.audio), 'duration_s': self.duration_s}


def _speech_result(result, audio, fmt, sample_rate, used):
    fields = {name: value for name, value in result.__dict__.items()}
    fields.update(audio=audio, data=None, format=fmt, sample_rate=sample_rate, voices=used,
                  mime_type=FORMATS[fmt] if fmt != 'pcm' else f'audio/l16;rate={sample_rate}')
    return SpeechResult(**fields)


# --------------------------------------------------------------------------
# Input → plan
# --------------------------------------------------------------------------

def _plan(text, turns, voice, style, speakers):
    """(rows, speakers): rows are {'text', 'style', 'speaker'}; speakers is {name: voice} or None."""
    if (text is None) == (turns is None):
        raise ValueError('Передайте либо text, либо turns.')
    raw = [{'text': text}] if turns is None else [{'text': t} if isinstance(t, str) else dict(t) for t in turns]
    rows = []
    for turn in raw:
        spoken = str(turn.get('text') or '').strip()
        if spoken:
            rows.append({'text': spoken, 'style': turn.get('style', style) or None, 'speaker': turn.get('speaker')})
    if not rows:
        raise ValueError('Пустой текст для озвучки.')
    if speakers:
        if voice:
            raise ValueError('voice задаёт голос одного диктора; с speakers голоса берутся из словаря.')
        speakers = {str(name): voices.canonical(value) for name, value in dict(speakers).items()}
        bad = [name for name in speakers if not name.isalnum()]
        if bad:
            raise ValueError(f'Имя диктора — только буквы и цифры, без пробелов: {", ".join(bad)}.')
        unknown = sorted({str(row['speaker']) for row in rows} - set(speakers))
        if unknown:
            raise ValueError(f'У каждой реплики нужен speaker из speakers; не найдены: {", ".join(unknown)}.')
    elif any(row['speaker'] for row in rows):
        raise ValueError('У реплик указан speaker — передайте speakers={"Имя": "Голос", …}.')
    return rows, speakers or None


def _modern_rows(rows):
    """[mood] tags of 3.1 become style segments, [sound] tags become <sound>."""
    result = []
    for row in rows:
        segments, mood, cursor, text = [], None, 0, row['text']
        for match in _SQUARE.finditer(text):
            name = match.group(1).strip().lower()
            if name in TO_MODERN:
                continue
            segments.append((mood, text[cursor:match.start()]))
            mood, cursor = match.group(1).strip(), match.end()
        segments.append((mood, text[cursor:]))
        for mood, chunk in segments:
            chunk = _SQUARE.sub(lambda m: f'<{TO_MODERN[m.group(1).strip().lower()]}>'
                                if m.group(1).strip().lower() in TO_MODERN else m.group(0), chunk).strip()
            if not chunk:
                continue
            style = ', '.join(part for part in (row['style'], mood) if part) or None
            result.append({**row, 'text': chunk, 'style': style})
    return result


def _legacy_text(row):
    text = _ANGLE.sub(lambda m: f'[{TO_LEGACY.get(m.group(1).strip().lower(), m.group(1).strip())}]', row['text'])
    text = _PIPES.sub('', text).strip()
    if row['style']:
        # 3.1 rejects two tags side by side, so a leading tag merges into the style.
        lead = _SQUARE.match(text)
        if lead:
            text = f"[{row['style']}, {lead.group(1)}]{text[lead.end():]}"
        else:
            text = f"[{row['style']}] {text}"
    return text


def build_request(model, rows, voice, speakers, fmt='wav', sample_rate=None):
    """(generateContent body, {speaker or '': voice actually used}, notes) for this model."""
    notes = []
    if is_modern(model):
        rows = _modern_rows(rows)
        parts = []
        for row in rows:
            part = {'text': row['text']}
            meta = {name: row[name] for name in ('style', 'speaker') if row.get(name)}
            if meta:
                part['speech_metadata'] = meta
            parts.append(part)
        if speakers:
            speech = {'multiSpeakerVoiceConfig': {'speakerVoiceConfigs': [
                {'speaker': name, 'voiceConfig': {'voice': value}} for name, value in speakers.items()]}}
            used = dict(speakers)
        else:
            speech = {'voiceConfig': {'voice': voice}}
            used = {'': voice}
        config = {'responseModalities': ['AUDIO'], 'speechConfig': speech}
        if fmt != 'wav' or sample_rate:
            audio = {'mimeType': REQUEST_FORMATS[fmt]}
            if sample_rate:
                audio['sampleRate'] = sample_rate
            config['responseFormat'] = {'audio': audio}
        return {'contents': [{'role': 'user', 'parts': parts}], 'generationConfig': config}, used, notes

    def legacy_voice(value):
        if voices.is_prebuilt(value):
            return voices.canonical(value)
        substitute, why = voices.substitute(value)
        notes.append(f'{model} знает только 30 базовых голосов — вместо «{value}» звучит {substitute} ({why}).')
        return substitute

    if any('|' in row['text'] for row in rows):
        notes.append(f'{model} не умеет реплики-подхваты |…| — они убраны.')
    if speakers:
        used = {name: legacy_voice(value) for name, value in speakers.items()}
        text = '\n'.join(f"{row['speaker']}: {_legacy_text(row)}" for row in rows)
        speech = {'multiSpeakerVoiceConfig': {'speakerVoiceConfigs': [
            {'speaker': name, 'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': value}}}
            for name, value in used.items()]}}
    else:
        used = {'': legacy_voice(voice)}
        text = ' '.join(_legacy_text(row) for row in rows)
        speech = {'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': used['']}}}
    return {'contents': [{'role': 'user', 'parts': [{'text': text}]}],
            'generationConfig': {'responseModalities': ['AUDIO'], 'speechConfig': speech}}, used, notes


# --------------------------------------------------------------------------
# Response
# --------------------------------------------------------------------------

def _usage(payload):
    usage = payload.get('usageMetadata') or payload.get('usage_metadata') or {}
    return {'input_tokens': usage.get('promptTokenCount'), 'output_tokens': usage.get('candidatesTokenCount'),
            'thinking_tokens': usage.get('thoughtsTokenCount'), 'total_tokens': usage.get('totalTokenCount')}


def parse_response(payload, model):
    """(audio bytes, mime type, finish reason) of one generateContent answer or stream chunk."""
    candidates = payload.get('candidates') or []
    feedback = payload.get('promptFeedback') or {}
    if not candidates:
        if feedback.get('blockReason'):
            raise SafetyError(f"Заблокировано Gemini ({feedback['blockReason']}).", model=model, provider='gemini')
        return b'', None, ''
    candidate = candidates[0]
    finish = candidate.get('finishReason') or ''
    if finish in SAFETY_FINISH_REASONS:
        raise SafetyError(f'Озвучка остановлена фильтром безопасности Gemini ({finish}).', model=model,
                          provider='gemini')
    audio, mime = b'', None
    for part in (candidate.get('content') or {}).get('parts') or []:
        inline = part.get('inlineData') or part.get('inline_data') or {}
        if inline.get('data'):
            audio += base64.b64decode(inline['data'])
            mime = inline.get('mimeType') or inline.get('mime_type') or mime
    return audio, mime, finish


def _empty(model, finish):
    reason = f'{model} не вернула аудио.'
    if finish:
        reason += f' Причина завершения: {finish}.'
    if finish == 'OTHER':
        reason += ' Так бывает, когда текст не похож на реплику (например, только служебные теги) ' \
                  'или в тексте теги на русском — теги пишутся по-английски.'
    return EmptyResponseError(reason, model=model, provider='gemini')


def _convert(audio, mime, fmt, sample_rate):
    """Model output → requested format; (bytes, sample rate)."""
    if fmt in ('mulaw', 'alaw'):
        return audio, _mime_rate(mime) or sample_rate or 8000
    rate, payload = wav_parts(audio)
    rate = rate or _mime_rate(mime) or sample_rate or SAMPLE_RATE
    return (wav(payload, rate) if fmt == 'wav' else payload), rate


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def _models(model, chain, fmt, sample_rate, streaming=False):
    if chain is True:
        models = catalog.default_tts_chain()
    elif chain:
        models = [catalog.canonical(m) for m in ([chain] if isinstance(chain, str) else chain) if catalog.canonical(m)]
    else:
        models = [catalog.canonical(model) or DEFAULT_TTS_MODEL]
    if fmt in ('mulaw', 'alaw') or (sample_rate and sample_rate != SAMPLE_RATE):
        # Older models return 24 kHz PCM only.
        kept = [m for m in models if is_modern(m)]
        if not kept:
            raise ValueError(f'Формат {fmt}/{sample_rate or SAMPLE_RATE} Гц умеют только модели 3.8+; '
                             f'в цепочке: {", ".join(models)}.')
        models = kept
    if not models:
        raise ValueError('Цепочка моделей пуста.')
    return models


def _check_format(fmt, sample_rate):
    if fmt not in FORMATS:
        raise ValueError(f'format: одно из {", ".join(FORMATS)}.')
    if sample_rate is not None and sample_rate not in SAMPLE_RATES:
        raise ValueError(f'sample_rate: одно из {", ".join(map(str, SAMPLE_RATES))}.')


def _needs_split(speakers):
    return bool(speakers) and (len(speakers) > MAX_SPEAKERS or any(voices.is_custom(v) for v in speakers.values()))


def _prepare_run(routing, models, rows, voice, speakers):
    run = _routing(routing)
    if routing:
        raise TypeError(f'Неизвестные параметры: {", ".join(routing)}')
    if run.input_tokens is None:
        run.input_tokens = sum(len(row['text'].encode('utf-8')) for row in rows) // 3 + 20
    custom = [v for v in ([voice] if voice else []) + list((speakers or {}).values()) if voices.is_custom(v)]
    owners = {voices.owner(v) for v in custom if not v.startswith('voicekey_')}
    if owners and None not in owners and len(owners) == 1 and not run.key_id:
        run.pin_keys = {m: next(iter(owners)) for m in models if is_modern(m)}
    elif custom:
        # Owner unknown: a key of another project answers 404, so walk all keys.
        run.try_all_keys = True
    return run


def _learn_owner(result, voice, speakers):
    for value in ([voice] if voice else []) + list((speakers or {}).values()):
        if voices.is_custom(value) and not value.startswith('voicekey_') and is_modern(result.model) \
                and voices.owner(value) is None:
            voices.remember(value, result.key_id)


def speak(text=None, *, voice=None, style=None, turns=None, speakers=None, model=None, chain=None,
          format='wav', sample_rate=None, extra=None, on_event=None, **routing):
    """Text → speech. Returns SpeechResult (.audio, .save(path), .duration_s, .voices, .notes).

    text + voice + style    one speaker; style is the delivery ("angry, shouting").
    turns=[{text, style?, speaker?}]   several lines; a style per line.
    speakers={'Anya': 'Kore', 'Boris': 'Charon'}   dialogue; every turn names its speaker.
    model= one model, chain=[…] or chain=True (default TTS chain) — keys rotate per model.
    format 'wav' | 'pcm' | 'mulaw' | 'alaw'; sample_rate 24000 | 16000 | 8000 (3.8+ only).
    extra: dict deep-merged into the generateContent body (fields this API has not mapped yet).
    Routing options are the same as generate(): key_id, prefer_key, try_all_keys, stop_event,
    timeout_s, …. See `ai-router tts-guide` for voices, tags and style recipes.
    """
    _check_format(format, sample_rate)
    rows, speakers = _plan(text, turns, voice, style, speakers)
    voice = voices.canonical(voice) if voice else (None if speakers else voices.DEFAULT_VOICE)
    if _needs_split(speakers):
        return _speak_per_turn(rows, speakers, model, chain, format, sample_rate, extra, on_event, routing)
    models = _models(model, chain, format, sample_rate)
    run = _prepare_run(routing, models, rows, voice, speakers)
    notes, used = [], {}

    def perform(client, ctx):
        body, used_now, attempt_notes = build_request(ctx.model, rows, voice, speakers, format, sample_rate)
        if extra:
            _merge(body, extra)
        payload = voices.raw_request(client, 'post', f'models/{ctx.model}:generateContent', body)
        audio, mime, finish = parse_response(payload, ctx.model)
        if not audio:
            raise _empty(ctx.model, finish)
        audio, rate = _convert(audio, mime, format, sample_rate)
        notes.extend(attempt_notes)
        used.clear()
        used.update(used_now, _rate=rate)
        yield {'type': '_final', 'response': _strip_audio(payload), 'parsed': {
            'text': '', 'data': audio, 'finish_reason': finish, 'usage': _usage(payload)}}

    result = _drive(_execute(models, perform, run, notes), on_event)
    rate = used.pop('_rate', SAMPLE_RATE)
    _learn_owner(result, voice, speakers)
    return _speech_result(result, result.data, format, rate, used)


def speak_stream(text=None, *, voice=None, style=None, turns=None, speakers=None, model=None, chain=None,
                 format='wav', extra=None, on_event=None, **routing):
    """Streaming speak(): yields {'type': 'delta', 'audio': PCM bytes, 'sample_rate'} as it is
    generated, progress events, and finally {'type': 'done', 'result': SpeechResult}.

    Deltas are always raw 16-bit mono PCM; the final result is in `format` ('wav' or 'pcm').
    Keys and models switch only before the first chunk. 2.5 models do not stream — they
    answer in one chunk.
    """
    if format not in ('wav', 'pcm'):
        raise ValueError('В потоке отдаётся только PCM: format = wav или pcm.')
    rows, speakers = _plan(text, turns, voice, style, speakers)
    voice = voices.canonical(voice) if voice else (None if speakers else voices.DEFAULT_VOICE)
    if _needs_split(speakers):
        yield from _stream_per_turn(rows, speakers, model, chain, format, extra, on_event, routing)
        return
    models = _models(model, chain, 'pcm', None)
    run = _prepare_run(routing, models, rows, voice, speakers)
    run.streaming = True
    notes, used = [], {}

    def perform(client, ctx):
        # Streams are headerless PCM by default, so no responseFormat is needed.
        body, used_now, attempt_notes = build_request(ctx.model, rows, voice, speakers)
        if extra:
            _merge(body, extra)
        if can_stream(ctx.model):
            chunks = voices.raw_request(client, 'post', f'models/{ctx.model}:streamGenerateContent?alt=sse', body,
                                        stream=True)
        else:
            chunks = [voices.raw_request(client, 'post', f'models/{ctx.model}:generateContent', body)]
        pcm, rate, finish, usage = [], SAMPLE_RATE, '', {}
        for payload in chunks:
            audio, mime, chunk_finish = parse_response(payload, ctx.model)
            finish, usage = chunk_finish or finish, _usage(payload) if payload.get('usageMetadata') else usage
            if not audio:
                continue
            chunk_rate, audio = wav_parts(audio)
            rate = chunk_rate or _mime_rate(mime) or rate
            pcm.append(audio)
            yield {'type': 'delta', 'audio': audio, 'sample_rate': rate, 'model': ctx.model}
        if not pcm:
            raise _empty(ctx.model, finish)
        notes.extend(attempt_notes)
        used.clear()
        used.update(used_now, _rate=rate)
        joined = b''.join(pcm)
        yield {'type': '_final', 'response': None, 'parsed': {
            'text': '', 'data': wav(joined, rate) if format == 'wav' else joined, 'finish_reason': finish,
            'usage': usage}}

    for event in _stream(_execute(models, perform, run, notes), on_event):
        if event['type'] == 'done':
            result = event['result']
            rate = used.pop('_rate', SAMPLE_RATE)
            _learn_owner(result, voice, speakers)
            event = {'type': 'done', 'result': _speech_result(result, result.data, format, rate, used)}
        yield event


def _speak_per_turn(rows, speakers, model, chain, fmt, sample_rate, extra, on_event, routing):
    """More than two speakers or custom voices in a dialogue: one request per line, glued together."""
    pieces, results = [], []
    raw_fmt = 'pcm' if fmt == 'wav' else fmt
    for row in rows:
        result = speak(turns=[{'text': row['text'], 'style': row['style']}], voice=speakers[row['speaker']],
                       model=model, chain=chain, format=raw_fmt, sample_rate=sample_rate, extra=extra,
                       on_event=on_event, **dict(routing))
        results.append((row['speaker'], result))
    rate = results[0][1].sample_rate
    for index, (_, result) in enumerate(results):
        if index:
            pieces.append(_silence(raw_fmt, rate, TURN_GAP_S))
        pieces.append(result.audio)
    return _merged(results, b''.join(pieces), fmt, rate)


def _stream_per_turn(rows, speakers, model, chain, fmt, extra, on_event, routing):
    results, pieces, rate = [], [], SAMPLE_RATE
    for index, row in enumerate(rows):
        if index:
            gap = _silence('pcm', rate, TURN_GAP_S)
            pieces.append(gap)
            yield {'type': 'delta', 'audio': gap, 'sample_rate': rate}
        for event in speak_stream(turns=[{'text': row['text'], 'style': row['style']}],
                                  voice=speakers[row['speaker']], model=model, chain=chain, format='pcm',
                                  extra=extra, on_event=on_event, **dict(routing)):
            if event['type'] == 'done':
                results.append((row['speaker'], event['result']))
                rate = event['result'].sample_rate
                continue
            if event['type'] == 'delta':
                pieces.append(event['audio'])
            yield event
    yield {'type': 'done', 'result': _merged(results, b''.join(pieces), fmt, rate)}


def _merged(results, raw, fmt, rate):
    first = results[0][1]
    merged = SpeechResult(**{name: value for name, value in first.__dict__.items()})
    merged.model = ' + '.join(dict.fromkeys(result.model for _, result in results))
    merged.attempts = [attempt for _, result in results for attempt in result.attempts]
    merged.notes = [note for _, result in results for note in result.notes]
    merged.position = max(result.position for _, result in results)
    merged.usage = {name: sum((result.usage.get(name) or 0) for _, result in results)
                    for name in ('input_tokens', 'output_tokens', 'total_tokens')}
    merged.voices = {speaker: next(iter(result.voices.values())) for speaker, result in results}
    merged.audio = wav(raw, rate) if fmt == 'wav' else raw
    merged.format, merged.sample_rate = fmt, rate
    merged.mime_type = FORMATS[fmt] if fmt != 'pcm' else f'audio/l16;rate={rate}'
    return merged


def _merge(target, extra):
    for name, value in extra.items():
        if isinstance(value, dict) and isinstance(target.get(name), dict):
            _merge(target[name], value)
        else:
            target[name] = value


def _strip_audio(payload):
    """The raw answer without the base64 audio, for logs and result.response."""
    def walk(value):
        if isinstance(value, dict):
            return {k: (f'<{len(v)} символов base64>' if k == 'data' and isinstance(v, str) else walk(v))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value
    return walk(payload)


def guide(section=None):
    """The TTS guide (Russian); a section is picked by a word of its heading."""
    text = (Path(__file__).parent / 'guides' / 'tts.md').read_text(encoding='utf-8')
    if not section:
        return text
    chunks = re.split(r'(?m)^(?=## )', text)
    wanted = str(section).lower()
    picked = [chunk for chunk in chunks if chunk.startswith('## ') and wanted in chunk.splitlines()[0].lower()]
    if not picked:
        titles = [chunk.splitlines()[0][3:] for chunk in chunks if chunk.startswith('## ')]
        raise ValueError(f'Раздел «{section}» не найден. Разделы: ' + '; '.join(titles))
    return '\n'.join(picked)


def sections():
    text = (Path(__file__).parent / 'guides' / 'tts.md').read_text(encoding='utf-8')
    return [line[3:].strip() for line in text.splitlines() if line.startswith('## ')]
