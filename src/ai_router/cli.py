"""Command line: manage the shared key vault, check status, run a test request, speak."""

import argparse
import json
import re
import sys

import ai_router as ai
from ai_router import keys
from ai_router._storage import state_dir


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='ai-router', description='Общие ключи и запросы к Gemini и OpenAI')
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('keys', help='список ключей (без секретов)')
    add = sub.add_parser('add', help='добавить ключ')
    add.add_argument('provider', choices=keys.PROVIDERS)
    add.add_argument('key')
    add.add_argument('--label', default='')
    remove = sub.add_parser('remove', help='удалить ключ')
    remove.add_argument('key_id')
    for name in ('enable', 'disable'):
        toggle = sub.add_parser(name, help=f'{"включить" if name == "enable" else "выключить"} ключ')
        toggle.add_argument('key_id')
    reset = sub.add_parser('reset', help='снять паузы (всех ключей или одного)')
    reset.add_argument('key_id', nargs='?')
    sub.add_parser('status', help='паузы, счётчики, квоты')
    models = sub.add_parser('models', help='каталог моделей')
    models.add_argument('--provider', choices=keys.PROVIDERS)
    models.add_argument('--task', default='chat')
    models.add_argument('--free', action='store_true', help='только с бесплатной квотой')
    chain = sub.add_parser('chain', help='показать или задать цепочку моделей по умолчанию')
    chain.add_argument('models', nargs='*')
    test = sub.add_parser('test', help='проверить ключ или модель коротким запросом')
    test.add_argument('--key', dest='key_id')
    test.add_argument('--model')
    ask = sub.add_parser('ask', help='задать вопрос через роутер')
    ask.add_argument('prompt')
    ask.add_argument('--model')
    ask.add_argument('--chain', action='store_true', help='цепочка по умолчанию')
    sub.add_parser('path', help='где лежат ключи и состояние')

    say = sub.add_parser('say', help='озвучить текст (Gemini TTS)')
    say.add_argument('text', help='текст; для диалога строки «Имя: реплика» и --speaker Имя=Голос')
    say.add_argument('-v', '--voice', help='голос: Kore, pl-pl-storyteller-1, voice_… (по умолчанию Kore)')
    say.add_argument('-s', '--style', help='манера: "angry, shouting", "грустно, тихо"…')
    say.add_argument('-m', '--model', help='модель (по умолчанию gemini-3.8-flash-tts)')
    say.add_argument('--chain', action='store_true', help='TTS-цепочка по умолчанию (лестница моделей)')
    say.add_argument('--speaker', action='append', default=[], metavar='ИМЯ=ГОЛОС', help='диктор диалога')
    say.add_argument('-o', '--out', default='speech.wav', help='куда сохранить (speech.wav)')
    say.add_argument('--format', default='wav', choices=['wav', 'pcm', 'mulaw', 'alaw'])
    say.add_argument('--rate', type=int, choices=[24000, 16000, 8000], help='частота (только 3.8+)')
    say.add_argument('--stream', action='store_true', help='потоковая генерация (печатает приход чанков)')
    voice_list = sub.add_parser('voices', help='голоса: базовые, библиотека, свои')
    voice_list.add_argument('--prebuilt', action='store_true', help='30 базовых голосов (без сети)')
    voice_list.add_argument('--custom', action='store_true', help='свои голоса, известные ai_router')
    voice_list.add_argument('--sync', action='store_true', help='спросить у Google свои голоса всех проектов')
    for flag in ('lang', 'gender', 'pitch', 'accent', 'persona', 'context', 'type'):
        voice_list.add_argument(f'--{flag}', action='append')
    voice_list.add_argument('--search')
    voice_list.add_argument('--limit', type=int, default=50)
    voice_list.add_argument('--json', action='store_true')
    design = sub.add_parser('voice-design', help='создать голос по описанию')
    design.add_argument('description')
    design.add_argument('--name')
    design.add_argument('--gender', choices=['female', 'male', 'neutral'])
    design.add_argument('--lang', help='BCP-47, например ru-RU')
    design.add_argument('-o', '--out', default='voice_preview.wav', help='куда сохранить пробу голоса')
    clone = sub.add_parser('voice-clone', help='клонировать голос по записи (нужна запись согласия)')
    clone.add_argument('source', help='WAV 10–30 с чистой речи')
    clone.add_argument('consent', help='WAV с фразой согласия (ai-router tts-guide клон)')
    clone.add_argument('--name')
    clone.add_argument('--no-store', action='store_true', help='voicekey_… на 7 дней вместо voice_… на год')
    delete = sub.add_parser('voice-delete', help='удалить свой голос')
    delete.add_argument('voice_id')
    tts_chain = sub.add_parser('tts-chain', help='показать или задать TTS-цепочку по умолчанию')
    tts_chain.add_argument('models', nargs='*')
    guide = sub.add_parser('tts-guide', help='подробный гайд по голосу, стилю, тегам')
    guide.add_argument('section', nargs='?', help='раздел: слово из заголовка (голоса, теги, русский…)')
    guide.add_argument('--list', action='store_true', help='только список разделов')
    args = parser.parse_args(argv)

    try:
        if args.command in (None, 'keys'):
            _print(keys.list_keys())
        elif args.command == 'add':
            _print(keys.add(args.provider, args.key, label=args.label, source='cli'))
        elif args.command == 'remove':
            print('удалён' if keys.remove(args.key_id) else 'не найден')
        elif args.command in ('enable', 'disable'):
            _print(keys.update(args.key_id, enabled=args.command == 'enable'))
            if args.command == 'enable':
                ai.reset_pauses(args.key_id)
        elif args.command == 'reset':
            print(f'Сброшено ключей: {ai.reset_pauses(args.key_id)}')
        elif args.command == 'status':
            _print(ai.status())
        elif args.command == 'models':
            rows = ai.models(args.provider, task=args.task)
            _print([row for row in rows if row['free_tier']] if args.free else rows)
        elif args.command == 'chain':
            _print(ai.set_default_chain(args.models) if args.models else ai.default_chain())
        elif args.command == 'test':
            if args.key_id:
                _print(ai.test_key(args.key_id, args.model))
            else:
                result = ai.generate('Ответь одним словом: OK', model=args.model, thinking='off',
                                     on_event=lambda e: print('·', e['text'], file=sys.stderr))
                _print(result.to_dict())
        elif args.command == 'ask':
            result = ai.generate(args.prompt, model=args.model, chain=True if args.chain else None,
                                 on_event=lambda e: print('·', e['text'], file=sys.stderr))
            print(result.text)
            print(f'— {result.model}, ключ «{result.key_label}»', file=sys.stderr)
        elif args.command == 'path':
            print(state_dir())
        elif args.command == 'say':
            _say(args)
        elif args.command == 'voices':
            _voices(args)
        elif args.command == 'voice-design':
            voice = ai.voices.design(args.description, name=args.name, gender=args.gender, language=args.lang,
                                     on_event=_progress)
            if voice.get('sample_audio'):
                with open(args.out, 'wb') as file:
                    file.write(voice['sample_audio'])
                print(f'Проба голоса: {args.out}', file=sys.stderr)
            _print({k: v for k, v in voice.items() if k not in ('sample_audio', 'attempts')})
        elif args.command == 'voice-clone':
            voice = ai.voices.replicate(args.source, args.consent, name=args.name, store=not args.no_store,
                                        on_event=_progress)
            _print({k: v for k, v in voice.items() if k not in ('sample_audio', 'attempts')})
        elif args.command == 'voice-delete':
            ai.voices.delete(args.voice_id)
            print('удалён')
        elif args.command == 'tts-chain':
            _print(ai.set_default_tts_chain(args.models) if args.models else ai.default_tts_chain())
        elif args.command == 'tts-guide':
            from ai_router import tts
            print('\n'.join(tts.sections()) if args.list else ai.tts_guide(args.section))
    except (ai.RouterError, KeyError, ValueError) as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        return 1
    return 0


def _progress(event):
    print('·', event['text'], file=sys.stderr)


def _say(args):
    speakers = dict(item.split('=', 1) for item in args.speaker)
    request = {'style': args.style, 'model': args.model, 'chain': True if args.chain else None,
               'on_event': _progress}
    if speakers:
        turns = []
        for line in args.text.splitlines():
            match = re.match(r'\s*([^:]+?)\s*:\s*(.+)', line)
            if match:
                turns.append({'speaker': match.group(1), 'text': match.group(2)})
            elif line.strip() and turns:
                turns[-1]['text'] += ' ' + line.strip()
        request.update(turns=turns, speakers=speakers)
    else:
        request.update(text=args.text, voice=args.voice)
    if args.stream:
        result, received = None, 0
        for event in ai.speak_stream(format=args.format, **request):
            if event['type'] == 'delta':
                received += len(event['audio'])
                print(f'  чанк {len(event["audio"])} байт, всего {received}', file=sys.stderr)
            elif event['type'] == 'done':
                result = event['result']
    else:
        result = ai.speak(format=args.format, sample_rate=args.rate, **request)
    result.save(args.out)
    print(f'{args.out}: {result.duration_s} с, {result.model}, ключ «{result.key_label}», '
          f'голоса {result.voices}', file=sys.stderr)
    for note in result.notes:
        print('! ' + note, file=sys.stderr)


def _voices(args):
    if args.prebuilt:
        rows = ai.voices.prebuilt()
    elif args.sync:
        rows = ai.voices.sync()
    elif args.custom:
        rows = ai.voices.custom()
    else:
        rows = ai.voices.list(language=args.lang, gender=args.gender, pitch=args.pitch, accent=args.accent,
                              persona=args.persona, context=args.context, type=args.type, search=args.search,
                              limit=args.limit)
    if args.json:
        _print(rows)
        return
    for row in rows:
        about = row.get('description') or row.get('descriptor') or row.get('prompt') or ''
        name = row.get('display_name') or row.get('name') or ''
        print(f"{row['id']:<28} {row.get('gender') or '':<8} {row.get('pitch') or '':<7} "
              f"{row.get('language_code') or '':<7} {name[:24]:<24} {about[:90]}")


if __name__ == '__main__':
    sys.exit(main())
