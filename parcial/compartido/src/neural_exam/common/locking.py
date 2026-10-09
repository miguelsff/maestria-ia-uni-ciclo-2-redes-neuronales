"""A process lock keeps runs sharing an output directory sequential."""

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def execution_lock(root: Path):
    """Acquire an OS-released lock so interrupted runs do not leave stale PID files."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".execution.lock").open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b" ")
            stream.flush()
        stream.seek(0)
        try:
            if __import__("os").name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another experiment is using this output directory.") from error
        try:
            yield
        finally:
            stream.seek(0)
            if __import__("os").name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
