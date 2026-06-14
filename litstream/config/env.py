"""Stdlib .env loader so the harness picks up API keys uniformly.

A cron/launchd daemon runs non-interactively and does not source ~/.zshrc, so
shell exports are invisible to a scheduled run. A project .env read explicitly
works for both interactive use and the daemon.

A variable already set in the real environment takes precedence over .env, so
any value can be overridden per-invocation (e.g. `DEEPSEEK_API_KEY=… python -m …`).

    from litstream.config.env import load_env
    load_env()                       # finds the project .env, or pass a path
    key = os.environ["DEEPSEEK_API_KEY"]
"""

from __future__ import annotations

import os
from pathlib import Path


def find_dotenv(start: Path | None = None) -> Path | None:
    """Walk up from `start` (default: this file) looking for a .env."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE lines from a .env into os.environ. Returns what was loaded.

    Skips blank lines and `#` comments; strips surrounding quotes; existing
    environment variables are preserved unless override=True.
    """
    dotenv = Path(path) if path else find_dotenv()
    loaded: dict[str, str] = {}
    if not dotenv or not dotenv.is_file():
        return loaded
    for raw in dotenv.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = val
        loaded[key] = val
    return loaded
