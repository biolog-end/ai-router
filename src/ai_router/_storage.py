"""Shared state directory, cross-process file locks and atomic JSON writes."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import threading
import time

_guard = threading.Lock()
_locks = {}
_held = threading.local()


def state_dir():
    """~/.ai_router or AI_ROUTER_DIR: one directory for every project of this user."""
    directory = os.getenv('AI_ROUTER_DIR')
    path = Path(directory).expanduser().resolve() if directory else Path.home() / '.ai_router'
    path.mkdir(parents=True, exist_ok=True)
    return path


class _FileLease:
    """An OS byte-range lock; the OS drops it when the process dies, even after a crash."""

    def __init__(self, path, timeout=10):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, 'a+b')
        self.file.seek(0, os.SEEK_END)
        if not self.file.tell():
            self.file.write(b'0')
            self.file.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.file.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    self.file.close()
                    raise TimeoutError('Файл ai_router занят другим процессом.')
                time.sleep(.05)

    def close(self):
        if self.file.closed:
            return
        try:
            self.file.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        finally:
            self.file.close()


@contextmanager
def file_lock(path):
    """Re-entrant within a thread, exclusive across threads and processes."""
    key = os.path.normcase(str(Path(path).resolve()))
    with _guard:
        lock = _locks.setdefault(key, threading.RLock())
    with lock:
        held = getattr(_held, 'paths', None)
        if held is None:
            held = _held.paths = set()
        if key in held:
            yield
            return
        lease = _FileLease(key + '.lock')
        held.add(key)
        try:
            yield
        finally:
            held.remove(key)
            lease.close()


def read_json(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as file:
                json.dump(value, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
