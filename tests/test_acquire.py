"""Tests for the acquire/triage layer.

Covers two modules:

  * ``litstream.acquire.source_policy.SourcePolicy`` — the per-(routine, source)
    auto-mute / reprobe state machine. We drive ``update(...)`` with seen/kept
    counts across several cycles and assert ``decide(...)`` returns the expected
    'run' / 'muted_skip' / 'reprobe' decisions, exercising the threshold
    boundaries directly.

  * ``litstream.acquire.triage.triage_project`` — relevance triage. A FAKE model
    (no network, no real LLM) is injected via the ``model=`` argument the code
    already takes; it returns canned scored output so we can assert score
    extraction, the keep/drop threshold, graceful handling of malformed output,
    and that the real ``CostLedger`` is debited.

Everything runs against a temp sqlite DB (``tmp_path``); no network, no API keys.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from litstream.acquire.source_policy import SourcePolicy
from litstream.acquire import triage as triage_mod
from litstream.acquire.triage import parse_score, triage_project
from litstream.ledger.cost import CostLedger
from litstream.library.store import PaperStore, PaperRecord


# --------------------------------------------------------------------------- #
# Fixtures / helpers                                                          #
# --------------------------------------------------------------------------- #

@pytest.fixture
def conn():
    """An in-memory sqlite connection with the row factory the policy expects."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    yield c
    c.close()


@pytest.fixture
def store(tmp_path):
    s = PaperStore(tmp_path / "lib.db", library_dir=tmp_path / "library")
    yield s
    s.close()


@pytest.fixture
def ledger(tmp_path):
    """A REAL CostLedger on a temp DB (separate file from the store)."""
    lg = CostLedger(tmp_path / "ledger.db")
    yield lg
    lg.close()


class FakeResult:
    """Mimics tasks.models.TaskResult — only the attributes triage reads."""

    def __init__(self, text, *, model="fake-model", input_tokens=10,
                 output_tokens=4, cached_input_tokens=0):
        self.text = text
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens


class FakeModel:
    """Injected in place of the DeepSeek client. ``complete`` returns canned text.

    Configure with ``responses`` (a list, consumed in order) or ``default`` (used
    when the list is exhausted). Records every prompt for inspection.
    """

    def __init__(self, responses=None, default="RELEVANT 0.9", model="fake-model"):
        self._responses = list(responses or [])
        self._default = default
        self.model = model
        self.calls = []

    def complete(self, prompt, *, system=None, max_tokens=1024, temperature=0.0):
        self.calls.append({"prompt": prompt, "system": system,
                           "max_tokens": max_tokens})
        if self._responses:
            spec = self._responses.pop(0)
        else:
            spec = self._default
        if isinstance(spec, FakeResult):
            return spec
        return FakeResult(spec, model=self.model)


def _add_paper(store, project, *, paper_id_seed, title, abstract, source="europepmc"):
    """Insert a paper + link to a project with triage_label NULL (un-triaged).

    Returns the canonical paper_id chosen by the store.
    """
    rec = PaperRecord(title=title, abstract=abstract,
                      doi=f"10.1234/{paper_id_seed}", source=source)
    pid = store.upsert(rec)
    linked = store.add_to_project(project, pid)
    assert linked, "paper should be newly linked to the project"
    return pid


def _project_row(store, project, pid):
    return store._conn.execute(
        "SELECT triage_label, triage_score, triage_model, status "
        "FROM project_papers WHERE project=? AND paper_id=?",
        (project, pid)).fetchone()


def _new_policy(conn, **kw):
    """Small, easy-to-reason-about defaults for boundary tests."""
    defaults = dict(min_seen=10, keep_threshold=0.20, reprobe_every=3)
    defaults.update(kw)
    return SourcePolicy(conn, **defaults)


# --------------------------------------------------------------------------- #
# SourcePolicy — fresh sources                                               #
# --------------------------------------------------------------------------- #

def test_decide_unknown_source_runs(conn):
    """A source never seen before is always 'run'."""
    pol = _new_policy(conn)
    assert pol.decide("r", ["arxiv", "europepmc"]) == {
        "arxiv": "run", "europepmc": "run"}


def test_productive_source_keeps_running(conn):
    """A source whose keep-rate stays well above threshold is never muted."""
    pol = _new_policy(conn)  # min_seen=10, keep_threshold=0.20
    for _ in range(5):
        d = pol.decide("r", ["good"])
        assert d == {"good": "run"}
        # 6 seen / 3 kept = 50% keep-rate each cycle — far above 20%.
        pol.update("r", d, seen={"good": 6}, kept={"good": 3})
    row = pol.status("r")[0]
    assert row["muted"] == 0
    assert row["seen"] == 30 and row["kept"] == 15
    assert row["runs"] == 5
    assert pol.decide("r", ["good"]) == {"good": "run"}


# --------------------------------------------------------------------------- #
# SourcePolicy — mute threshold boundaries                                   #
# --------------------------------------------------------------------------- #

def test_not_muted_below_min_seen_even_with_zero_keeps(conn):
    """Below min_seen the source is given the benefit of the doubt (no mute)."""
    pol = _new_policy(conn)  # min_seen=10
    d = pol.decide("r", ["x"])
    pol.update("r", d, seen={"x": 9}, kept={"x": 0})  # 9 < 10 → not eligible
    assert pol.status("r")[0]["muted"] == 0
    assert pol.decide("r", ["x"]) == {"x": "run"}


def test_mute_at_min_seen_boundary_below_threshold(conn):
    """At exactly min_seen with keep-rate below threshold → MUTED."""
    pol = _new_policy(conn)  # min_seen=10, keep_threshold=0.20
    d = pol.decide("r", ["x"])
    # 10 seen, 1 kept → rate 0.10 < 0.20 → mute fires.
    pol.update("r", d, seen={"x": 10}, kept={"x": 1})
    assert pol.status("r")[0]["muted"] == 1
    assert pol.decide("r", ["x"]) == {"x": "muted_skip"}


def test_not_muted_when_rate_equals_threshold(conn):
    """Rate == threshold is NOT below threshold → stays running (strict '<')."""
    pol = _new_policy(conn)  # min_seen=10, keep_threshold=0.20
    d = pol.decide("r", ["x"])
    # 10 seen, 2 kept → rate exactly 0.20, condition is rate < 0.20 → no mute.
    pol.update("r", d, seen={"x": 10}, kept={"x": 2})
    assert pol.status("r")[0]["muted"] == 0
    assert pol.decide("r", ["x"]) == {"x": "run"}


def test_mute_accumulates_across_runs(conn):
    """Mute uses CUMULATIVE seen/kept, not a single run's counts."""
    pol = _new_policy(conn)  # min_seen=10, keep_threshold=0.20
    # Three runs of 4 seen / 0 kept = 12 seen / 0 kept cumulatively.
    for i in range(3):
        d = pol.decide("r", ["x"])
        assert d == {"x": "run"}, f"still running before mute, cycle {i}"
        pol.update("r", d, seen={"x": 4}, kept={"x": 0})
    row = pol.status("r")[0]
    assert row["seen"] == 12 and row["kept"] == 0
    assert row["muted"] == 1
    assert pol.decide("r", ["x"]) == {"x": "muted_skip"}


# --------------------------------------------------------------------------- #
# SourcePolicy — reprobe cycle                                               #
# --------------------------------------------------------------------------- #

def _mute_source(pol, routine="r", source="x"):
    """Drive a source into the muted state and return it muted."""
    d = pol.decide(routine, [source])
    pol.update(routine, d, seen={source: 10}, kept={source: 0})
    assert pol.status(routine)[0]["muted"] == 1
    return source


def test_muted_source_skips_then_reprobes(conn):
    """A muted source is skipped for reprobe_every-1 runs, then reprobed."""
    pol = _new_policy(conn)  # reprobe_every=3
    _mute_source(pol)

    # skips_since_probe starts at 0. decide() looks at skips+1 >= reprobe_every(3).
    # skips=0 -> 0+1=1  >=3? no  -> muted_skip ; update bumps skips to 1
    d = pol.decide("r", ["x"])
    assert d == {"x": "muted_skip"}
    pol.update("r", d, seen={}, kept={})
    assert pol.status("r")[0]["skips_since_probe"] == 1

    # skips=1 -> 1+1=2  >=3? no  -> muted_skip ; update bumps skips to 2
    d = pol.decide("r", ["x"])
    assert d == {"x": "muted_skip"}
    pol.update("r", d, seen={}, kept={})
    assert pol.status("r")[0]["skips_since_probe"] == 2

    # skips=2 -> 2+1=3  >=3? YES -> reprobe
    d = pol.decide("r", ["x"])
    assert d == {"x": "reprobe"}


def test_reprobe_that_stays_unproductive_remutes_and_resets_skips(conn):
    """A reprobe run is treated like a run: it re-evaluates the mute threshold and
    resets skips_since_probe to 0, restarting the skip countdown."""
    pol = _new_policy(conn)  # min_seen=10, keep_threshold=0.20, reprobe_every=3
    _mute_source(pol)
    # advance to the reprobe decision
    for _ in range(2):
        d = pol.decide("r", ["x"])
        pol.update("r", d, seen={}, kept={})
    d = pol.decide("r", ["x"])
    assert d == {"x": "reprobe"}
    # reprobe fetches but the source is still junk (0 kept) → stays muted, skips reset.
    pol.update("r", d, seen={"x": 5}, kept={"x": 0})
    row = pol.status("r")[0]
    assert row["muted"] == 1
    assert row["skips_since_probe"] == 0
    assert row["seen"] == 15 and row["kept"] == 0
    # next decide is a skip again (countdown restarted)
    assert pol.decide("r", ["x"]) == {"x": "muted_skip"}


def test_reprobe_that_becomes_productive_unmutes(conn):
    """If a reprobe run now keeps enough papers, the source un-mutes and runs."""
    pol = _new_policy(conn)  # min_seen=10, keep_threshold=0.20, reprobe_every=3
    _mute_source(pol)  # 10 seen / 0 kept, muted
    for _ in range(2):
        d = pol.decide("r", ["x"])
        pol.update("r", d, seen={}, kept={})
    d = pol.decide("r", ["x"])
    assert d == {"x": "reprobe"}
    # Reprobe brings 10 new, 8 kept → cumulative 20 seen / 8 kept = 40% > 20%.
    pol.update("r", d, seen={"x": 10}, kept={"x": 8})
    row = pol.status("r")[0]
    assert row["muted"] == 0, "keep-rate recovered above threshold → un-muted"
    assert row["seen"] == 20 and row["kept"] == 8
    assert pol.decide("r", ["x"]) == {"x": "run"}


def test_decide_is_side_effect_free(conn):
    """decide() must not mutate stored stats (docstring promises side-effect free)."""
    pol = _new_policy(conn)
    _mute_source(pol)
    before = dict(pol.status("r")[0])
    for _ in range(5):
        pol.decide("r", ["x"])
    after = dict(pol.status("r")[0])
    assert before == after


def test_multiple_sources_independent(conn):
    """Per-source state does not bleed across sources in the same routine."""
    pol = _new_policy(conn)
    d = pol.decide("r", ["good", "bad"])
    assert d == {"good": "run", "bad": "run"}
    # good keeps lots, bad keeps none — both cross min_seen this run.
    pol.update("r", d, seen={"good": 10, "bad": 10}, kept={"good": 9, "bad": 0})
    d2 = pol.decide("r", ["good", "bad"])
    assert d2 == {"good": "run", "bad": "muted_skip"}


# --------------------------------------------------------------------------- #
# triage.parse_score — unit                                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,label,expected", [
    ("RELEVANT 0.85", "RELEVANT", 0.85),
    ("BORDERLINE 0.5", "BORDERLINE", 0.5),
    ("NOT_RELEVANT 0.1", "NOT_RELEVANT", 0.1),
    ("RELEVANT 1.0", "RELEVANT", 1.0),
    ("score: 0", "NOT_RELEVANT", 0.0),
    ("definitely 1", "RELEVANT", 1.0),
])
def test_parse_score_extracts_value(text, label, expected):
    assert parse_score(text, label) == pytest.approx(expected)


def test_parse_score_out_of_range_falls_back_to_label_default():
    # 9.9 is out of [0,1]; the regex won't match it as a 0/1 score → label default.
    assert parse_score("RELEVANT 9.9", "RELEVANT") == pytest.approx(0.8)


def test_parse_score_no_number_uses_label_default():
    assert parse_score("RELEVANT, clearly on-topic", "RELEVANT") == pytest.approx(0.8)
    assert parse_score("garbage", "BORDERLINE") == pytest.approx(0.5)
    assert parse_score("garbage", "NOT_RELEVANT") == pytest.approx(0.1)
    assert parse_score("garbage", "UNKNOWN") == pytest.approx(0.3)


def _start_run(ledger, project="proj"):
    ledger.set_policy(cap_usd=50.0)
    ledger.seed_pricing("fake-model", 1.0, 2.0, 0.5)  # so cost is non-zero
    return ledger.start_run(project=project, routine="r", invocation="manual")


def test_triage_keep_drop_threshold_and_parsing(store, ledger):
    project = "proj"
    keep_pid = _add_paper(store, project, paper_id_seed="keep",
                          title="A method", abstract="single cell rna seq method")
    drop_pid = _add_paper(store, project, paper_id_seed="drop",
                          title="Off topic", abstract="quantum chromodynamics review")

    # First un-triaged row processed gets the first response, etc. Order is by the
    # SELECT; make BOTH responses unambiguous and assert per-paper via the store.
    model = FakeModel(responses=["RELEVANT 0.90", "NOT_RELEVANT 0.10"])
    run_id = _start_run(ledger, project)

    results = triage_project(store, project=project, focus="single cell methods",
                             model=model, ledger=ledger, run_id=run_id,
                             keep_threshold=0.5)

    assert len(results) == 2
    by_id = {r["paper_id"]: r for r in results}
    assert by_id[keep_pid]["score"] == pytest.approx(0.90)
    assert by_id[keep_pid]["label"] == "RELEVANT"
    assert by_id[drop_pid]["score"] == pytest.approx(0.10)
    assert by_id[drop_pid]["label"] == "NOT_RELEVANT"

    # Persisted status reflects the keep/drop threshold.
    assert _project_row(store, project, keep_pid)["status"] == "kept"
    assert _project_row(store, project, drop_pid)["status"] == "dropped"
    # sources round-trips as a parsed list.
    assert by_id[keep_pid]["sources"] == ["europepmc"]


def test_triage_threshold_boundary_is_inclusive(store, ledger):
    """status == 'kept' uses score >= keep_threshold (boundary is KEPT)."""
    project = "proj"
    pid = _add_paper(store, project, paper_id_seed="edge",
                     title="t", abstract="some abstract here")
    model = FakeModel(responses=["BORDERLINE 0.50"])
    run_id = _start_run(ledger, project)
    triage_project(store, project=project, focus="f", model=model,
                   ledger=ledger, run_id=run_id, keep_threshold=0.5)
    assert _project_row(store, project, pid)["status"] == "kept"


def test_triage_debits_the_ledger(store, ledger):
    project = "proj"
    for i in range(3):
        _add_paper(store, project, paper_id_seed=f"p{i}",
                   title=f"t{i}", abstract=f"abstract number {i}")
    model = FakeModel(default="RELEVANT 0.8")
    run_id = _start_run(ledger, project)

    cost_before = ledger.run_cost_cents(run_id)
    triage_project(store, project=project, focus="f", model=model,
                   ledger=ledger, run_id=run_id)
    cost_after = ledger.run_cost_cents(run_id)

    # One cost_event per paper that called the model.
    n_events = ledger._conn.execute(
        "SELECT COUNT(*) c FROM cost_events WHERE run_id=? AND phase='triage'",
        (run_id,)).fetchone()["c"]
    assert n_events == 3
    # Pricing was seeded, tokens > 0 → strictly positive debit.
    assert cost_after > cost_before


def test_triage_malformed_output_does_not_crash(store, ledger):
    """Garbage / empty / non-conforming model output must be handled gracefully:
    no exception, every paper still gets a label+score+status."""
    project = "proj"
    pids = [
        _add_paper(store, project, paper_id_seed="g0", title="t0",
                   abstract="abstract zero"),
        _add_paper(store, project, paper_id_seed="g1", title="t1",
                   abstract="abstract one"),
        _add_paper(store, project, paper_id_seed="g2", title="t2",
                   abstract="abstract two"),
    ]
    model = FakeModel(responses=[
        "",                       # empty
        "###?!! no label no score",  # pure junk
        "RELEVANT",               # label but no score → label default applies
    ])
    run_id = _start_run(ledger, project)

    results = triage_project(store, project=project, focus="f", model=model,
                             ledger=ledger, run_id=run_id)
    assert len(results) == 3
    for pid in pids:
        row = _project_row(store, project, pid)
        assert row["triage_label"] in ("RELEVANT", "BORDERLINE",
                                       "NOT_RELEVANT", "UNKNOWN")
        assert row["triage_score"] is not None
        assert row["status"] in ("kept", "dropped")


def test_triage_skips_papers_without_abstract_no_model_call(store, ledger):
    """A paper with a blank abstract is parked as BORDERLINE 0.3 WITHOUT spending
    a model call (and dropped)."""
    project = "proj"
    pid = _add_paper(store, project, paper_id_seed="noabs",
                     title="No abstract", abstract="   ")
    model = FakeModel(default="RELEVANT 0.99")
    run_id = _start_run(ledger, project)

    results = triage_project(store, project=project, focus="f", model=model,
                             ledger=ledger, run_id=run_id)
    assert model.calls == [], "no LLM call should be made for an empty abstract"
    assert len(results) == 1
    assert results[0]["label"] == "BORDERLINE"
    assert results[0]["score"] == pytest.approx(0.3)
    row = _project_row(store, project, pid)
    assert row["status"] == "dropped"
    assert row["triage_model"] == "(no-abstract)"
    # nothing debited
    assert ledger.run_cost_cents(run_id) == 0


def test_triage_only_processes_untriaged(store, ledger):
    """Papers already triaged (triage_label NOT NULL) are skipped on a re-run."""
    project = "proj"
    pid_done = _add_paper(store, project, paper_id_seed="done",
                          title="done", abstract="already triaged abstract")
    pid_new = _add_paper(store, project, paper_id_seed="new",
                         title="new", abstract="fresh abstract to triage")
    # Pre-mark one as triaged.
    store.set_triage(project, pid_done, "RELEVANT", score=0.9,
                     model="prev", status="kept")

    model = FakeModel(default="NOT_RELEVANT 0.1")
    run_id = _start_run(ledger, project)
    results = triage_project(store, project=project, focus="f", model=model,
                             ledger=ledger, run_id=run_id)

    assert len(results) == 1
    assert results[0]["paper_id"] == pid_new
    assert len(model.calls) == 1
    # The already-triaged paper kept its original decision.
    assert _project_row(store, project, pid_done)["triage_score"] == pytest.approx(0.9)


def test_triage_prompt_is_built_with_focus_and_paper(store, ledger):
    """Sanity: the prompt handed to the model carries the focus, title, abstract,
    and the system prompt + max_tokens the code pins."""
    project = "proj"
    _add_paper(store, project, paper_id_seed="px",
               title="My Title", abstract="My distinctive abstract body")
    model = FakeModel(default="RELEVANT 0.8")
    run_id = _start_run(ledger, project)
    triage_project(store, project=project, focus="MY FOCUS STRING",
                   model=model, ledger=ledger, run_id=run_id)

    assert len(model.calls) == 1
    call = model.calls[0]
    assert "MY FOCUS STRING" in call["prompt"]
    assert "My Title" in call["prompt"]
    assert "My distinctive abstract body" in call["prompt"]
    assert call["system"] == triage_mod.SYSTEM
    assert call["max_tokens"] == 24
