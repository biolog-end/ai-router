"""Command line: manage the shared key vault, check status, run a test request."""

import argparse
import json
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
    except (ai.RouterError, KeyError, ValueError) as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
