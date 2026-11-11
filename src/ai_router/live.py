"""Gemini Live models (bidiGenerateContent) as ordinary text models of generate()/stream().

A Live model always answers with speech: the text is the transcription of that speech and
arrives at speaking pace, so a long answer takes as long as reading it aloud. The free
tier counts only TPM for these models, without RPM/RPD.

Two usage shapes:
  * generate()/stream() — one short session per call: send one turn, collect the answer,
    close the socket. Implemented by perform().
  * StreamSession — a long-lived bidirectional WebSocket for real-time voice: push PCM
    audio in with send_audio(), consume audio/text out with events(). Phone-call style.
"""

import asyncio
import queue
import re
import threading
import time

import gemini_budget as gb

from . import catalog, request as rq, voices
from .errors import EmptyResponseError, RequestError, StoppedError

SAMPLE_RATE = 24000
_POLL_S = 0.5
_DROPPED = {
    'max_output_tokens': 'max_output_tokens (Live-сессия с ним зависает)',
    'stop': 'stop', 'presence_penalty': 'presence_penalty', 'frequency_penalty': 'frequency_penalty',
    'safety': 'safety', 'include_thoughts': 'include_thoughts',
    'search': 'search (на бесплатном ключе Live отвечает ошибкой квоты)',
}


def is_live_model(model):
    """True for models served only through the Live API (gemini-3.8-live, *-native-audio-*, …)."""
    name = catalog.canonical(model).lower()
    info = gb.model_info(name)
    return (info is not None and info['task'] == 'live') or bool(re.search(r'-live\b|native-audio', name))


def _check(model, options):
    name = model.lower()
    if 'translate' in name or 'transcribe' in name:
        raise RequestError(f'{model} работает только со звуком (перевод или распознавание речи) — '
                           'текстовый запрос ей не отправить.', model=model, provider='gemini')
    if 'extended-thinking' in name:
        raise RequestError(f'{model} думает асинхронно и на текстовый запрос не отвечает — '
                           'возьмите gemini-3.8-live.', model=model, provider='gemini')
    if options.json_schema is not None or options.json_mode:
        raise RequestError(f'{model}: Live-модели не умеют JSON-режим (json_mode/json_schema).', model=model,
                           provider='gemini')


def connect_config(model, options, notes):
    """LiveConnectConfig of a text request; options Live rejects are dropped into notes."""
    from google.genai import types
    kw = {'response_modalities': ['AUDIO'], 'output_audio_transcription': types.AudioTranscriptionConfig()}
    if options.system:
        kw['system_instruction'] = options.system
    for name in ('temperature', 'top_p', 'top_k', 'seed'):
        if getattr(options, name) is not None:
            kw[name] = getattr(options, name)
    dropped = [label for name, label in _DROPPED.items() if getattr(options, name) not in (None, False)]
    if options.thinking not in (None, False, 'off', 'none', 0):
        dropped.append('thinking')
    if options.voice:
        if voices.is_prebuilt(options.voice):
            kw['speech_config'] = types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voices.canonical(options.voice))))
        else:
            notes.append(f'Live-модели знают только 30 базовых голосов — «{options.voice}» пропущен.')
    extra = options.gemini or {}
    unknown = [name for name in extra if name not in types.LiveConnectConfig.model_fields]
    kw.update({name: value for name, value in extra.items() if name not in unknown})
    dropped += unknown
    if dropped:
        notes.append(f'{model} (Live API) не принимает: {", ".join(dropped)} — пропущено.')
    return types.LiveConnectConfig(**kw)


def _messages(client, model, config, contents, idle_s, stop_event):
    """LiveServerMessages of one turn; the async session runs on a private loop in a thread,
    so this works from synchronous code and from inside someone else's event loop alike."""
    out = queue.Queue()
    done = object()
    loop = asyncio.new_event_loop()

    async def session():
        async with client.aio.live.connect(model=model, config=config) as live:
            await live.send_client_content(turns=contents, turn_complete=True)
            async for message in live.receive():
                out.put(message)

    task = loop.create_task(session())

    def worker():
        try:
            loop.run_until_complete(task)
            out.put(done)
        except asyncio.CancelledError:
            out.put(done)
        except BaseException as exc:
            out.put(exc)
        finally:
            loop.close()

    thread = threading.Thread(target=worker, name=f'ai_router-live-{model}', daemon=True)
    thread.start()
    try:
        waited = 0.0
        while True:
            if stop_event is not None and stop_event.is_set():
                raise StoppedError('Генерация остановлена.', model=model, provider='gemini')
            try:
                item = out.get(timeout=_POLL_S)
            except queue.Empty:
                waited += _POLL_S
                if waited >= idle_s:
                    raise TimeoutError(f'Live API молчит дольше {int(idle_s)} с.')
                continue
            waited = 0.0
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        if thread.is_alive():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:  # the loop closed between is_alive() and here
                pass
            thread.join(timeout=5)


def _usage(usage):
    if usage is None:
        return {}
    return {'input_tokens': usage.prompt_token_count, 'output_tokens': usage.response_token_count,
            'thinking_tokens': usage.thoughts_token_count, 'total_tokens': usage.total_token_count}


def perform(client, ctx, messages, options, notes, run, streaming):
    """The attempt of one key, in the event shape of router perform(): text deltas (and PCM
    deltas with options.audio) when streaming, then '_final' with the audio in parsed['data']."""
    _check(ctx.model, options)
    attempt_notes = []
    config = connect_config(ctx.model, options, attempt_notes)
    text, audio, usage, finish = [], [], None, ''
    for message in _messages(client, ctx.model, config, rq.gemini_contents(messages), run.timeout_s,
                             run.stop_event):
        usage = message.usage_metadata or usage
        content = message.server_content
        if content is None:
            continue
        piece = content.output_transcription.text if content.output_transcription else None
        if piece:
            text.append(piece)
            if streaming:
                yield {'type': 'delta', 'text': piece}
        if options.audio and content.model_turn:
            for part in content.model_turn.parts or []:
                if part.inline_data and part.inline_data.data:
                    audio.append(part.inline_data.data)
                    if streaming:
                        yield {'type': 'delta', 'audio': part.inline_data.data, 'sample_rate': SAMPLE_RATE}
        if content.turn_complete:
            finish = rq._name(content.turn_complete_reason) or 'STOP'
    joined = ''.join(text).strip()
    if not joined:
        raise EmptyResponseError(f'Модель {ctx.model} не вернула текста.'
                                 + (f' Причина завершения: {finish}.' if finish else ''), model=ctx.model,
                                 provider='gemini')
    for note in attempt_notes:
        if note not in notes:
            notes.append(note)
    yield {'type': '_final', 'response': None, 'parsed': {
        'text': joined, 'data': b''.join(audio) if options.audio else None, 'finish_reason': finish,
        'truncated': False, 'usage': _usage(usage)}}


def stream_config(model, *, system=None, voice=None, temperature=None, extra=None,
                  user_transcription=True, assistant_transcription=True):
    """LiveConnectConfig for a long-lived voice session: AUDIO out + both-side transcriptions."""
    from google.genai import types
    kw = {'response_modalities': ['AUDIO']}
    if assistant_transcription:
        kw['output_audio_transcription'] = types.AudioTranscriptionConfig()
    if user_transcription:
        kw['input_audio_transcription'] = types.AudioTranscriptionConfig()
    if system:
        kw['system_instruction'] = system
    if temperature is not None:
        kw['temperature'] = temperature
    if voice:
        if voices.is_prebuilt(voice):
            kw['speech_config'] = types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voices.canonical(voice))))
    for name, value in (extra or {}).items():
        if name in types.LiveConnectConfig.model_fields:
            kw[name] = value
    return types.LiveConnectConfig(**kw)


class StreamSession:
    """Persistent bidirectional Gemini Live session for real-time voice.

    Lifecycle: start() → send_audio() / events() → close(). Thread-safe: push audio from one
    thread, drain events from another; the WebSocket runs on a private asyncio loop in a
    daemon thread, so this works from synchronous code and from inside someone else's loop
    alike.

    send_audio(pcm, sample_rate=16000) — 16-bit little-endian mono PCM, any multiple of 20ms
      is fine; Gemini wants 16kHz but will accept the mime-type hint. VAD runs server-side.
    events() — a blocking generator yielding dicts:
      {'type': 'audio', 'data': bytes, 'sample_rate': 24000}
      {'type': 'text', 'role': 'assistant'|'user', 'text': str}
      {'type': 'turn_complete'}
      {'type': 'interrupted'}             — user barged in while model was speaking
      {'type': 'error', 'error': Exception}
      {'type': 'closed'}                  — final event, iteration stops after
    """

    def __init__(self, client, model, config, *, idle_s=120):
        self._client = client
        self._model = catalog.canonical(model)
        self._config = config
        self._idle_s = float(idle_s)
        self._out = queue.Queue()
        self._in = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._loop = None
        self._started_at = None
        self._closed = False
        self._done_sentinel = object()
        self._input_end_sentinel = object()

    def start(self):
        """Open the WebSocket and begin receiving. Returns immediately; events() blocks."""
        if self._thread is not None:
            raise RuntimeError('StreamSession already started')
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._thread_main, name=f'ai_router-live-stream-{self._model}', daemon=True)
        self._thread.start()

    def send_audio(self, pcm_bytes, sample_rate=16000):
        """Push a PCM chunk towards the model. Non-blocking."""
        if self._closed or self._stop.is_set():
            return
        self._in.put(('audio', pcm_bytes, int(sample_rate)))

    def send_text(self, text, *, end_of_turn=True):
        """Inject a text turn mid-session (rare — most of the time audio is enough)."""
        if self._closed or self._stop.is_set():
            return
        self._in.put(('text', text, bool(end_of_turn)))

    def end_input(self):
        """Mark end of user audio stream; model finishes its reply and the socket closes."""
        if self._closed:
            return
        self._in.put(self._input_end_sentinel)

    def events(self):
        """Drain events until the session closes. Yields dicts; see class doc for shape."""
        waited = 0.0
        while True:
            if self._stop.is_set() and self._out.empty():
                yield {'type': 'closed'}
                return
            try:
                item = self._out.get(timeout=0.5)
            except queue.Empty:
                waited += 0.5
                if waited >= self._idle_s:
                    yield {'type': 'error', 'error': TimeoutError(f'Live stream idle for {int(self._idle_s)} s.')}
                    yield {'type': 'closed'}
                    return
                continue
            waited = 0.0
            if item is self._done_sentinel:
                yield {'type': 'closed'}
                return
            yield item

    def close(self):
        """Cancel the session; blocks briefly for a clean shutdown."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._in.put(self._input_end_sentinel)
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)
            except RuntimeError:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_until_complete(self._run())
        except BaseException as exc:
            self._out.put({'type': 'error', 'error': exc})
        finally:
            self._out.put(self._done_sentinel)
            try:
                self._loop.close()
            except Exception:
                pass

    async def _run(self):
        async with self._client.aio.live.connect(model=self._model, config=self._config) as live:
            sender = asyncio.create_task(self._sender(live))
            receiver = asyncio.create_task(self._receiver(live))
            done, pending = await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                if task.exception():
                    raise task.exception()

    async def _sender(self, live):
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            try:
                item = await loop.run_in_executor(None, self._in.get, True, 0.3)
            except queue.Empty:
                continue
            if item is self._input_end_sentinel:
                try:
                    await live.send_realtime_input(audio_stream_end=True)
                except Exception:
                    pass
                return
            kind = item[0]
            if kind == 'audio':
                _, pcm, rate = item
                mime = f'audio/pcm;rate={rate}'
                await live.send_realtime_input(audio={'data': pcm, 'mime_type': mime})
            elif kind == 'text':
                _, text, end_of_turn = item
                await live.send_client_content(turns=[{'role': 'user', 'parts': [{'text': text}]}],
                                               turn_complete=end_of_turn)

    async def _receiver(self, live):
        # google.genai receive() ends after every turn_complete; the socket stays open.
        while not self._stop.is_set():
            async for message in live.receive():
                if self._stop.is_set():
                    return
                self._put_message(message)

    def _put_message(self, message):
        content = message.server_content
        if content is None:
            return
        if content.model_turn:
            for part in content.model_turn.parts or []:
                if part.inline_data and part.inline_data.data:
                    self._out.put({'type': 'audio', 'data': part.inline_data.data, 'sample_rate': SAMPLE_RATE})
        if content.output_transcription and content.output_transcription.text:
            self._out.put({'type': 'text', 'role': 'assistant', 'text': content.output_transcription.text})
        if content.input_transcription and content.input_transcription.text:
            self._out.put({'type': 'text', 'role': 'user', 'text': content.input_transcription.text})
        if content.interrupted:
            self._out.put({'type': 'interrupted'})
        if content.turn_complete:
            self._out.put({'type': 'turn_complete'})
