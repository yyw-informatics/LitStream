"""Minimal cron-expression matching — does a 5-field cron expr fire at a given
(timezone-aware) minute? Used by `litstream tick`. Supports *, lists, ranges,
steps, and named months/weekdays (cron convention: Sunday=0).
"""

from __future__ import annotations

from datetime import datetime, timedelta

_MONTHS = {m: i for i, m in enumerate(
    "jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1)}
_DOWS = {d: i for i, d in enumerate("sun mon tue wed thu fri sat".split())}


def _norm(token: str, names: dict) -> str:
    return str(names.get(token.lower(), token))


def _field_matches(field: str, value: int, lo: int, hi: int, names: dict | None = None) -> bool:
    if field == "*":
        return True
    for part in field.split(","):
        part = part.strip()
        step = 1
        if "/" in part:
            part, s = part.split("/")
            step = int(s)
        if part in ("*", ""):
            start, end = lo, hi
        elif "-" in part:
            a, b = part.split("-")
            start, end = int(_norm(a, names or {})), int(_norm(b, names or {}))
        else:
            v = int(_norm(part, names or {}))
            start = end = v
        if start <= value <= end and (value - start) % step == 0:
            return True
    return False


def cron_matches(expr: str, dt: datetime) -> bool:
    """expr = '<min> <hour> <dom> <month> <dow>'. dt should be timezone-aware in the
    routine's own timezone. Sunday=0 for the day-of-week field."""
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(f"cron expr must have 5 fields: {expr!r}")
    minute, hour, dom, month, dow = parts
    cron_dow = (dt.weekday() + 1) % 7   # python Mon=0..Sun=6 → cron Sun=0..Sat=6
    return (
        _field_matches(minute, dt.minute, 0, 59)
        and _field_matches(hour, dt.hour, 0, 23)
        and _field_matches(dom, dt.day, 1, 31)
        and _field_matches(month, dt.month, 1, 12, _MONTHS)
        and _field_matches(dow, cron_dow, 0, 6, _DOWS)
    )


def most_recent_tick(expr: str, now: datetime, max_lookback_days: int = 14) -> datetime | None:
    """Most recent minute at or before `now` (tz-aware) that the cron expr fires.

    Lookback is bounded so a long-idle routine doesn't scan indefinitely; returns
    None if no tick falls in the window. Catch-up uses this to detect a fire missed
    while the machine was asleep.
    """
    cur = now.replace(second=0, microsecond=0)
    for _ in range(max_lookback_days * 24 * 60 + 1):
        if cron_matches(expr, cur):
            return cur
        cur -= timedelta(minutes=1)
    return None
