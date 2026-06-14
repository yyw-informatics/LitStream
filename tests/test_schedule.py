"""Tests for the scheduler modules: cron matching, lockfile coalescing,
last-run state persistence, and the catch-up/missed-run decision.

Scope: pure logic only. We never run a real routine, hit the network, an LLM, or
launchd. The live `tick.tick()` coroutine (which imports and runs routines) is NOT
exercised — only its pure decision helper `compute_due`, with state redirected to a
temp dir and routine YAMLs written under `tmp_path`.

NOTE on datetimes: every datetime here is explicit and timezone-aware
(`tzinfo=timezone.utc`). We deliberately never call an argless `now()`.
Known calendar anchors used below:
  - 2024-01-15 is a Monday  -> cron weekday 1
  - 2024-01-07 is a Sunday  -> cron weekday 0
  - 2024-01-20 is a Saturday -> cron weekday 6
"""

from __future__ import annotations

import functools
import os
from datetime import datetime, timedelta, timezone

import pytest

from litstream.schedule import state as state_mod
from litstream.schedule import tick as tick_mod
from litstream.schedule.cron import cron_matches, most_recent_tick
from litstream.schedule.lock import RoutineLock
from litstream.schedule.state import get_last_fired, set_last_fired


def _dt(y, mo, d, h=0, mi=0):
    """Helper: build a tz-aware UTC datetime (no argless now() anywhere)."""
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


# Calendar anchors (verified against the stdlib calendar).
MONDAY = _dt(2024, 1, 15)      # cron dow 1
SUNDAY = _dt(2024, 1, 7)       # cron dow 0
SATURDAY = _dt(2024, 1, 20)    # cron dow 6


# ---------------------------------------------------------------------------
# cron_matches: wildcard, lists, ranges, steps, named fields, validation
# ---------------------------------------------------------------------------

class TestCronWildcard:
    def test_all_wildcards_always_match(self):
        assert cron_matches("* * * * *", _dt(2024, 6, 9, 13, 37))

    def test_exact_minute_hour(self):
        assert cron_matches("30 13 * * *", _dt(2024, 6, 9, 13, 30))
        assert not cron_matches("30 13 * * *", _dt(2024, 6, 9, 13, 31))
        assert not cron_matches("30 13 * * *", _dt(2024, 6, 9, 14, 30))


class TestCronLists:
    def test_day_of_month_list(self):
        assert cron_matches("0 0 1,15 * *", _dt(2024, 1, 1))
        assert cron_matches("0 0 1,15 * *", _dt(2024, 1, 15))
        assert not cron_matches("0 0 1,15 * *", _dt(2024, 1, 2))

    def test_minute_list_with_spaces_tolerated(self):
        # field_matches strips parts, so "0, 30" style lists still work
        assert cron_matches("0,30 * * * *", _dt(2024, 1, 1, 9, 0))
        assert cron_matches("0,30 * * * *", _dt(2024, 1, 1, 9, 30))
        assert not cron_matches("0,30 * * * *", _dt(2024, 1, 1, 9, 15))


class TestCronRanges:
    def test_hour_range(self):
        for h in (9, 12, 17):
            assert cron_matches("0 9-17 * * *", _dt(2024, 1, 1, h, 0))
        assert not cron_matches("0 9-17 * * *", _dt(2024, 1, 1, 8, 0))
        assert not cron_matches("0 9-17 * * *", _dt(2024, 1, 1, 18, 0))

    def test_weekday_numeric_range_mon_to_fri(self):
        # 1-5 == Mon..Fri
        assert cron_matches("0 0 * * 1-5", MONDAY)
        assert not cron_matches("0 0 * * 1-5", SATURDAY)
        assert not cron_matches("0 0 * * 1-5", SUNDAY)


class TestCronSteps:
    def test_every_15_minutes(self):
        for m in (0, 15, 30, 45):
            assert cron_matches("*/15 * * * *", _dt(2024, 1, 1, 6, m)), m
        for m in (1, 7, 14, 16, 31):
            assert not cron_matches("*/15 * * * *", _dt(2024, 1, 1, 6, m)), m

    def test_step_over_explicit_range(self):
        # 0-30/10 -> minutes 0, 10, 20, 30
        for m in (0, 10, 20, 30):
            assert cron_matches("0-30/10 * * * *", _dt(2024, 1, 1, 0, m)), m
        for m in (5, 15, 40, 50):
            assert not cron_matches("0-30/10 * * * *", _dt(2024, 1, 1, 0, m)), m


class TestCronNamedFields:
    def test_named_month(self):
        assert cron_matches("0 0 1 JAN *", _dt(2024, 1, 1))
        assert not cron_matches("0 0 1 JAN *", _dt(2024, 2, 1))
        # case-insensitive
        assert cron_matches("0 0 1 jan *", _dt(2024, 1, 1))
        assert cron_matches("0 0 25 DEC *", _dt(2024, 12, 25))

    def test_named_month_range(self):
        # MAR-MAY == months 3..5
        for mo in (3, 4, 5):
            assert cron_matches("0 0 1 MAR-MAY *", _dt(2024, mo, 1)), mo
        assert not cron_matches("0 0 1 MAR-MAY *", _dt(2024, 2, 1))
        assert not cron_matches("0 0 1 MAR-MAY *", _dt(2024, 6, 1))

    def test_named_weekday(self):
        assert cron_matches("0 6 * * MON", MONDAY.replace(hour=6))
        assert not cron_matches("0 6 * * SUN", MONDAY.replace(hour=6))
        assert cron_matches("0 6 * * SUN", SUNDAY.replace(hour=6))

    def test_named_weekday_range(self):
        # MON-FRI matches Mon, excludes Sat/Sun
        assert cron_matches("0 0 * * MON-FRI", MONDAY)
        assert not cron_matches("0 0 * * MON-FRI", SATURDAY)
        assert not cron_matches("0 0 * * MON-FRI", SUNDAY)


class TestCronWeekdayConversion:
    """The load-bearing Mon=0 (python) -> Sun=0 (cron) conversion."""

    def test_monday_is_cron_weekday_1(self):
        # 2024-01-15 is a known Monday; "0 6 * * MON" must fire at 06:00.
        assert MONDAY.weekday() == 0  # python: Monday is 0 (sanity anchor)
        assert cron_matches("0 6 * * MON", _dt(2024, 1, 15, 6, 0))
        assert cron_matches("0 6 * * 1", _dt(2024, 1, 15, 6, 0))

    def test_sunday_is_cron_weekday_0(self):
        # 2024-01-07 is a known Sunday; cron Sunday is 0.
        assert SUNDAY.weekday() == 6  # python: Sunday is 6 (sanity anchor)
        assert cron_matches("0 6 * * 0", _dt(2024, 1, 7, 6, 0))
        assert cron_matches("0 6 * * SUN", _dt(2024, 1, 7, 6, 0))

    def test_saturday_is_cron_weekday_6(self):
        assert SATURDAY.weekday() == 5  # python: Saturday is 5
        assert cron_matches("0 6 * * 6", _dt(2024, 1, 20, 6, 0))
        assert cron_matches("0 6 * * SAT", _dt(2024, 1, 20, 6, 0))


class TestCronValidation:
    def test_four_field_expr_raises(self):
        with pytest.raises(ValueError):
            cron_matches("* * * *", _dt(2024, 1, 1))

    def test_six_field_expr_raises(self):
        with pytest.raises(ValueError):
            cron_matches("* * * * * *", _dt(2024, 1, 1))

    def test_empty_expr_raises(self):
        with pytest.raises(ValueError):
            cron_matches("", _dt(2024, 1, 1))


# ---------------------------------------------------------------------------
# most_recent_tick: the missed-run catch-up detector
# ---------------------------------------------------------------------------

class TestMostRecentTick:
    def test_returns_current_minute_when_it_fires(self):
        now = _dt(2024, 1, 15, 6, 0)
        assert most_recent_tick("0 6 * * *", now) == now

    def test_returns_most_recent_past_firing(self):
        # daily 06:00; at 06:30 the most recent firing is 06:00 same day.
        now = _dt(2024, 1, 15, 6, 30)
        assert most_recent_tick("0 6 * * *", now) == _dt(2024, 1, 15, 6, 0)

    def test_looks_back_across_days(self):
        # daily 06:00; at 05:00 the most recent firing is yesterday 06:00.
        now = _dt(2024, 1, 15, 5, 0)
        assert most_recent_tick("0 6 * * *", now) == _dt(2024, 1, 14, 6, 0)

    def test_strips_seconds_and_microseconds(self):
        now = datetime(2024, 1, 15, 6, 0, 45, 123456, tzinfo=timezone.utc)
        got = most_recent_tick("0 6 * * *", now)
        assert got == _dt(2024, 1, 15, 6, 0)
        assert got.second == 0 and got.microsecond == 0

    def test_none_when_nothing_in_lookback_window(self):
        # Fires only on Feb 29; from mid-January with a 1-day lookback, nothing.
        now = _dt(2024, 1, 15, 6, 30)
        assert most_recent_tick("0 6 29 2 *", now, max_lookback_days=1) is None

    def test_respects_max_lookback_days_boundary(self):
        # weekly Monday 06:00. From Sunday 2024-01-14 05:00, the previous Monday
        # firing is 2024-01-08 06:00 — ~6 days back.
        now = _dt(2024, 1, 14, 5, 0)
        expected = _dt(2024, 1, 8, 6, 0)
        # Too-short window cannot reach it:
        assert most_recent_tick("0 6 * * MON", now, max_lookback_days=2) is None
        # Wide enough window finds it:
        assert most_recent_tick("0 6 * * MON", now, max_lookback_days=14) == expected


# ---------------------------------------------------------------------------
# lock.py: RoutineLock acquire/release coalescing
# ---------------------------------------------------------------------------

class TestRoutineLock:
    def test_acquire_writes_lockfile_with_pid(self, tmp_path):
        with RoutineLock("r", lock_dir=tmp_path) as lk:
            assert lk.acquired is True
            assert lk.path.exists()
            assert lk.path.read_text().strip() == str(os.getpid())
        # released on exit
        assert not lk.path.exists()

    def test_second_concurrent_acquire_coalesces(self, tmp_path):
        # Holding the lock (live PID), a second acquire must NOT acquire.
        with RoutineLock("r", lock_dir=tmp_path) as first:
            assert first.acquired is True
            with RoutineLock("r", lock_dir=tmp_path) as second:
                assert second.acquired is False
            # the coalesced (non-owning) context must not delete the live lock
            assert first.path.exists()

    def test_lock_reacquirable_after_release(self, tmp_path):
        with RoutineLock("r", lock_dir=tmp_path) as a:
            assert a.acquired is True
        with RoutineLock("r", lock_dir=tmp_path) as b:
            assert b.acquired is True

    def test_stale_lock_from_dead_pid_is_reclaimed(self, tmp_path):
        lock_path = tmp_path / "r.lock"
        lock_path.write_text("999999")  # a PID that is essentially never alive
        with RoutineLock("r", lock_dir=tmp_path) as lk:
            assert lk.acquired is True
            assert lk.path.read_text().strip() == str(os.getpid())

    def test_garbage_lockfile_contents_are_reclaimed(self, tmp_path):
        # Non-integer contents -> pid parses to 0 -> treated as reclaimable.
        (tmp_path / "r.lock").write_text("not-a-pid")
        with RoutineLock("r", lock_dir=tmp_path) as lk:
            assert lk.acquired is True

    def test_distinct_routines_do_not_collide(self, tmp_path):
        with RoutineLock("alpha", lock_dir=tmp_path) as a:
            with RoutineLock("beta", lock_dir=tmp_path) as b:
                assert a.acquired is True
                assert b.acquired is True

    def test_coalesced_acquire_leaves_owner_lock_intact_after_both_exit(self, tmp_path):
        outer = RoutineLock("r", lock_dir=tmp_path)
        outer.__enter__()
        assert outer.acquired is True
        inner = RoutineLock("r", lock_dir=tmp_path)
        inner.__enter__()
        assert inner.acquired is False
        inner.__exit__(None, None, None)
        assert outer.path.exists()  # still held by outer
        outer.__exit__(None, None, None)
        assert not outer.path.exists()


# ---------------------------------------------------------------------------
# state.py: last-run persistence
# ---------------------------------------------------------------------------

class TestState:
    def test_missing_state_returns_none(self, tmp_path):
        assert get_last_fired("never_seen", state_dir=tmp_path) is None

    def test_roundtrip(self, tmp_path):
        when = _dt(2024, 1, 15, 6, 0)
        set_last_fired("r", when, state_dir=tmp_path)
        assert get_last_fired("r", state_dir=tmp_path) == when

    def test_set_creates_state_dir(self, tmp_path):
        nested = tmp_path / "deep" / "state"
        assert not nested.exists()
        set_last_fired("r", _dt(2024, 1, 15, 6, 0), state_dir=nested)
        assert nested.exists()
        assert get_last_fired("r", state_dir=nested) is not None

    def test_naive_datetime_is_stored_as_utc_aware(self, tmp_path):
        # set_last_fired calls .astimezone(utc); a naive value is interpreted in
        # local time then normalized. The result must come back tz-aware.
        naive = datetime(2024, 1, 15, 6, 0)
        set_last_fired("r", naive, state_dir=tmp_path)
        got = get_last_fired("r", state_dir=tmp_path)
        assert got is not None
        assert got.tzinfo is not None

    def test_aware_datetime_normalized_to_utc(self, tmp_path):
        # A non-UTC tz-aware value is stored normalized to UTC; the instant is preserved.
        from datetime import timezone as _tz
        plus5 = _tz(timedelta(hours=5))
        when = datetime(2024, 1, 15, 11, 0, tzinfo=plus5)  # == 06:00 UTC
        set_last_fired("r", when, state_dir=tmp_path)
        got = get_last_fired("r", state_dir=tmp_path)
        assert got == _dt(2024, 1, 15, 6, 0)

    def test_corrupt_state_file_returns_none(self, tmp_path):
        (tmp_path / "r.json").write_text("{ not valid json")
        assert get_last_fired("r", state_dir=tmp_path) is None

    def test_state_file_missing_key_returns_none(self, tmp_path):
        (tmp_path / "r.json").write_text('{"other": 1}')
        assert get_last_fired("r", state_dir=tmp_path) is None


# ---------------------------------------------------------------------------
# compute_due: the catch-up / missed-run decision (cron + state composed)
#
# This is the pure decision helper from tick.py. It calls the bare
# get_last_fired/set_last_fired (default STATE_DIR), so we redirect those to a
# tmp dir via monkeypatch on the tick module's references. We never call the
# live tick() coroutine that would actually run a routine.
# ---------------------------------------------------------------------------

ROUTINE_YAML = """\
name: daily_am
schedule: '0 6 * * *'
timezone: UTC
enabled: true
"""


@pytest.fixture
def redirected_state(tmp_path, monkeypatch):
    """Point compute_due's state read/write at an isolated tmp dir."""
    sdir = tmp_path / "state"
    monkeypatch.setattr(
        tick_mod, "get_last_fired",
        functools.partial(state_mod.get_last_fired, state_dir=sdir),
    )
    monkeypatch.setattr(
        tick_mod, "set_last_fired",
        functools.partial(state_mod.set_last_fired, state_dir=sdir),
    )
    return sdir


def _write_routine(routines_dir, yaml_text=ROUTINE_YAML, name="daily.yaml"):
    routines_dir.mkdir(parents=True, exist_ok=True)
    (routines_dir / name).write_text(yaml_text)
    return routines_dir


class TestComputeDue:
    def test_first_sight_initializes_without_firing(self, tmp_path, redirected_state):
        rdir = _write_routine(tmp_path / "routines")
        now = _dt(2024, 1, 15, 6, 30)
        due = tick_mod.compute_due(now, routines_dir=rdir)
        assert due == []  # no surprise fire on install
        # but the clock is now started
        assert state_mod.get_last_fired("daily_am", state_dir=redirected_state) == now

    def test_missed_tick_while_idle_is_detected(self, tmp_path, redirected_state):
        rdir = _write_routine(tmp_path / "routines")
        # Install at day 15, 06:30 (just past today's fire) -> initialized, not due.
        tick_mod.compute_due(_dt(2024, 1, 15, 6, 30), routines_dir=rdir)
        # Machine slept; next tick is the following day at 06:30. The 06:00 fire on
        # day 16 was missed and must now be caught up.
        due = tick_mod.compute_due(_dt(2024, 1, 16, 6, 30), routines_dir=rdir)
        assert [c["name"] for c in due] == ["daily_am"]

    def test_already_fired_tick_is_not_refired(self, tmp_path, redirected_state):
        rdir = _write_routine(tmp_path / "routines")
        tick_mod.compute_due(_dt(2024, 1, 15, 6, 30), routines_dir=rdir)
        # First tick of day 16 fires (catch-up); a second tick in the same minute
        # must NOT re-fire the same occurrence.
        first = tick_mod.compute_due(_dt(2024, 1, 16, 6, 30), routines_dir=rdir)
        second = tick_mod.compute_due(_dt(2024, 1, 16, 6, 30), routines_dir=rdir)
        assert [c["name"] for c in first] == ["daily_am"]
        assert second == []

    def test_disabled_routine_never_fires(self, tmp_path, redirected_state):
        rdir = _write_routine(
            tmp_path / "routines",
            yaml_text=ROUTINE_YAML.replace("enabled: true", "enabled: false"),
        )
        # Even long after a scheduled tick, a disabled routine is skipped, and its
        # state is never initialized.
        due = tick_mod.compute_due(_dt(2024, 1, 16, 6, 30), routines_dir=rdir)
        assert due == []
        assert state_mod.get_last_fired("daily_am", state_dir=redirected_state) is None

    def test_routine_without_schedule_is_skipped(self, tmp_path, redirected_state):
        rdir = _write_routine(
            tmp_path / "routines",
            yaml_text="name: no_sched\ntimezone: UTC\nenabled: true\n",
        )
        assert tick_mod.compute_due(_dt(2024, 1, 16, 6, 30), routines_dir=rdir) == []

    def test_no_catchup_when_no_tick_since_last_fired(self, tmp_path, redirected_state):
        rdir = _write_routine(tmp_path / "routines")
        # Initialize at 06:30 on day 15 (most recent tick 06:00 acknowledged via
        # last_fired=06:30). A tick at 06:45 the SAME day has no newer firing, so
        # nothing is due.
        tick_mod.compute_due(_dt(2024, 1, 15, 6, 30), routines_dir=rdir)
        assert tick_mod.compute_due(_dt(2024, 1, 15, 6, 45), routines_dir=rdir) == []
