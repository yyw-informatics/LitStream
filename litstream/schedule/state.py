"""Per-routine scheduler state — when each routine last fired, so catch-up can
detect ticks missed while the machine was asleep/off. One small JSON file per
routine under ~/.litstream/state/.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path.home() / ".litstream" / "state"


def get_last_fired(routine: str, state_dir: Path = STATE_DIR) -> datetime | None:
    p = state_dir / f"{routine}.json"
    if not p.exists():
        return None
    try:
        return datetime.fromisoformat(json.loads(p.read_text())["last_fired"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def set_last_fired(routine: str, when: datetime, state_dir: Path = STATE_DIR) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    when = when.astimezone(timezone.utc)
    (state_dir / f"{routine}.json").write_text(json.dumps({"last_fired": when.isoformat()}))
