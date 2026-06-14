"""Per-routine lockfile that coalesces overlapping fires.

If a routine's run is still in flight when the next tick fires, the new fire is
skipped rather than double-run: re-running the same literature window adds no value.
Stale locks (dead PID) are reclaimed on the next acquire.
"""

from __future__ import annotations

import os
from pathlib import Path

LOCK_DIR = Path.home() / ".litstream" / "locks"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RoutineLock:
    """Context manager. `acquired` is False if another live run holds the lock."""

    def __init__(self, routine: str, lock_dir: Path = LOCK_DIR):
        lock_dir.mkdir(parents=True, exist_ok=True)
        self.path = lock_dir / f"{routine}.lock"
        self.acquired = False

    def __enter__(self) -> "RoutineLock":
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip() or "0")
            except ValueError:
                pid = 0
            if pid and _pid_alive(pid):
                self.acquired = False
                return self
            self.path.unlink(missing_ok=True)   # stale → reclaim
        self.path.write_text(str(os.getpid()))
        self.acquired = True
        return self

    def __exit__(self, *exc) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
