"""`litstream tick` — fire any routine whose cron matches now.

A single scheduled job (launchd/cron, every minute) calls this. For each routine
YAML whose `schedule` matches the current minute in its timezone, run it end-to-end
— unless its lock is held (a prior run still in flight → coalesce/skip).
"""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from .cron import most_recent_tick
from .lock import RoutineLock
from .state import get_last_fired, set_last_fired

ROOT = Path(__file__).resolve().parents[2]
ROUTINES_DIR = ROOT / "litstream" / "config" / "routines"
DEFAULT_PROJECT_DIR = ROOT / "kb-skills-bioinformatics"
DEFAULT_SKILLS_DIR = DEFAULT_PROJECT_DIR / "skills"
DB_PATH = str(ROOT / "litstream.db")
LIBRARY_DIR = str(ROOT / "library")


def compute_due(now_utc: dt.datetime, routines_dir: Path = ROUTINES_DIR) -> list[dict]:
    """Decide which routines should fire, with catch-up. A routine is due if its
    most-recent scheduled tick is later than its last_fired (so a fire missed while
    the machine slept is recovered on the next tick), capped at one make-up run.

    Side effects: first-seen routines are initialized to now without firing (no
    surprise run on install); firing advances last_fired so the same occurrence
    isn't repeated.
    """
    due = []
    for yf in sorted(routines_dir.glob("*.yaml")):
        cfg = yaml.safe_load(yf.read_text())
        sched = cfg.get("schedule")
        if not sched or not cfg.get("enabled", True):   # paused routines never auto-fire
            continue
        tz = ZoneInfo(cfg.get("timezone", "UTC"))
        recent = most_recent_tick(sched, now_utc.astimezone(tz))
        if recent is None:
            continue
        last = get_last_fired(cfg["name"])
        if last is None:                       # first install: start the clock, don't fire
            set_last_fired(cfg["name"], now_utc)
            continue
        if recent.astimezone(dt.timezone.utc) > last:
            set_last_fired(cfg["name"], now_utc)   # acknowledge this occurrence
            due.append(cfg)
    return due


async def tick(now_utc: dt.datetime | None = None, *, project_dir: Path = DEFAULT_PROJECT_DIR,
               skills_dir: Path = DEFAULT_SKILLS_DIR) -> list[tuple[str, str]]:
    from litstream_lg.run import run_routine_cfg
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    fired: list[tuple[str, str]] = []
    for cfg in compute_due(now_utc):
        with RoutineLock(cfg["name"]) as lk:
            if not lk.acquired:
                fired.append((cfg["name"], "coalesced (already running)"))
                continue
            res = await asyncio.to_thread(
                run_routine_cfg, cfg, str(project_dir),
                db_path=DB_PATH, library_dir=LIBRARY_DIR, skills_dir=str(skills_dir))
            fired.append((cfg["name"], res.get("status", "?")))
    return fired


def main() -> None:
    results = asyncio.run(tick())
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not results:
        print(f"{stamp} tick: no routines due")
    for name, status in results:
        print(f"{stamp} tick: {name} → {status}")


if __name__ == "__main__":
    main()
