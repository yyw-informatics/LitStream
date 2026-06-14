"""Deterministic numeric tests for LitStream's evaluation metrics.

Covers the pure (no network, no LLM) logic in:
  - litstream/eval/triage_eval.py  — Score.acc3 / keep_prec / keep_rec / keep_f1,
    parse_label, _keep, and the metric-assembly path of evaluate_groups (with a
    stubbed model + fake ledger so scoring is fully deterministic).
  - litstream/eval/stats_analysis.py — topic-clustered bootstrap CI (boot_ci),
    paired bootstrap (paired), and the keep-all baseline F1 formula.

All randomness is made deterministic by seeding the module-global `random`
immediately before each call (the bootstrap helpers draw from `random.choices`
on the module-level RNG; they take no seed argument, so we control the global
seed and assert reproducibility across two identically-seeded calls).
"""

from __future__ import annotations

import math
import random
import statistics as st

import pytest

from litstream.eval import triage_eval as te
from litstream.eval.triage_eval import Score, parse_label, _keep, evaluate_groups
from litstream.eval import stats_analysis as sa
from litstream.tasks.models import TaskResult


# ---------------------------------------------------------------------------
# Helpers: build a Score with a known confusion matrix directly.
# ---------------------------------------------------------------------------

def make_score(tp=0, fp=0, fn=0, tn=0, *, correct3=None, n=None):
    """Construct a Score with an explicit KEEP confusion matrix.

    n defaults to tp+fp+fn+tn and correct3 defaults to n (all 3-class correct)
    unless overridden.
    """
    if n is None:
        n = tp + fp + fn + tn
    if correct3 is None:
        correct3 = n
    return Score(backend="b", model="m", n=n, correct3=correct3,
                 tp=tp, fp=fp, fn=fn, tn=tn)


# ===========================================================================
# Score: keep-precision / keep-recall / keep-F1 with KNOWN confusion matrices
# ===========================================================================

def test_keep_precision_recall_f1_basic():
    # TP=6, FP=2, FN=3, TN=1
    s = make_score(tp=6, fp=2, fn=3, tn=1)
    # precision = TP/(TP+FP) = 6/8
    assert s.keep_prec == pytest.approx(6 / 8)
    # recall = TP/(TP+FN) = 6/9
    assert s.keep_rec == pytest.approx(6 / 9)
    # F1 = 2PR/(P+R)
    p, r = 6 / 8, 6 / 9
    assert s.keep_f1 == pytest.approx(2 * p * r / (p + r))


def test_keep_f1_equals_2tp_over_2tp_plus_fp_plus_fn():
    # The 2PR/(P+R) identity must equal 2TP / (2TP + FP + FN).
    s = make_score(tp=6, fp=2, fn=3, tn=1)
    expected = 2 * 6 / (2 * 6 + 2 + 3)
    assert s.keep_f1 == pytest.approx(expected)


def test_perfect_classifier():
    # No FP, no FN -> P=R=F1=1.0
    s = make_score(tp=5, fp=0, fn=0, tn=5)
    assert s.keep_prec == 1.0
    assert s.keep_rec == 1.0
    assert s.keep_f1 == 1.0


def test_all_keep_predicted_everything_positive():
    # Predict KEEP for everything: no TN, no FN. Recall is perfect; precision is
    # the prevalence of true-keeps among all predicted-keeps.
    s = make_score(tp=7, fp=3, fn=0, tn=0)
    assert s.keep_rec == 1.0
    assert s.keep_prec == pytest.approx(7 / 10)
    assert s.keep_f1 == pytest.approx(2 * 7 / (2 * 7 + 3 + 0))


def test_all_drop_predicted_everything_negative():
    # Predict DROP for everything: no TP, no FP. Precision and recall both 0,
    # and F1 must be 0 (NOT a ZeroDivisionError).
    s = make_score(tp=0, fp=0, fn=4, tn=6)
    assert s.keep_prec == 0.0   # 0/(0+0) guarded -> 0.0
    assert s.keep_rec == 0.0    # 0/(0+4) = 0.0
    assert s.keep_f1 == 0.0


def test_empty_score_no_items():
    # n=0, all-zero matrix: every metric is 0.0, no exceptions.
    s = make_score()
    assert s.n == 0
    assert s.acc3 == 0.0
    assert s.keep_prec == 0.0
    assert s.keep_rec == 0.0
    assert s.keep_f1 == 0.0
    assert s.avg_ms == 0


def test_zero_tp_with_fp_and_fn_f1_zero_no_div_error():
    # TP=0 but FP>0 and FN>0: precision=0, recall=0, so P+R=0 -> F1 guard returns 0.
    s = make_score(tp=0, fp=3, fn=2, tn=1)
    assert s.keep_prec == 0.0
    assert s.keep_rec == 0.0
    # Must not raise ZeroDivisionError; must be exactly 0.0.
    assert s.keep_f1 == 0.0


def test_keep_prec_zero_division_guard_only_fn():
    # No predicted-keeps at all (TP+FP=0) -> precision guard returns 0.0.
    s = make_score(tp=0, fp=0, fn=5, tn=0)
    assert s.keep_prec == 0.0


def test_keep_rec_zero_division_guard_only_fp():
    # No true-keeps at all (TP+FN=0) -> recall guard returns 0.0.
    s = make_score(tp=0, fp=5, fn=0, tn=0)
    assert s.keep_rec == 0.0


# ===========================================================================
# Score.acc3 : 3-way accuracy
# ===========================================================================

def test_acc3_basic():
    s = make_score(tp=4, fp=1, fn=2, tn=3, correct3=7, n=10)
    assert s.acc3 == pytest.approx(7 / 10)


def test_acc3_all_correct():
    s = make_score(tp=2, fp=0, fn=0, tn=2, correct3=4, n=4)
    assert s.acc3 == 1.0


def test_acc3_none_correct():
    s = make_score(tp=0, fp=2, fn=2, tn=0, correct3=0, n=4)
    assert s.acc3 == 0.0


def test_acc3_zero_n_guard():
    s = Score(backend="b", model="m", n=0, correct3=0)
    assert s.acc3 == 0.0  # no ZeroDivisionError


# ===========================================================================
# Score.avg_ms : integer division guard
# ===========================================================================

def test_avg_ms_integer_division():
    s = make_score(tp=2, fp=0, fn=0, tn=0, n=4)
    s.latency_ms = 1000
    assert s.avg_ms == 250
    assert isinstance(s.avg_ms, int)


def test_avg_ms_zero_n_guard():
    s = Score(backend="b", model="m", n=0)
    s.latency_ms = 999
    assert s.avg_ms == 0


# ===========================================================================
# _keep : KEEP = (not NOT_RELEVANT)
# ===========================================================================

@pytest.mark.parametrize("label,expected", [
    ("RELEVANT", True),
    ("BORDERLINE", True),
    ("NOT_RELEVANT", False),
    ("UNKNOWN", True),  # _keep only special-cases NOT_RELEVANT
])
def test_keep_predicate(label, expected):
    assert _keep(label) is expected


# ===========================================================================
# parse_label : deterministic label extraction
# ===========================================================================

@pytest.mark.parametrize("text,expected", [
    ("RELEVANT. The paper does not discuss X.", "RELEVANT"),
    ("NOT_RELEVANT — off topic", "NOT_RELEVANT"),
    ("NOT RELEVANT, wrong domain", "NOT_RELEVANT"),
    ("BORDERLINE: could go either way", "BORDERLINE"),
    ("relevant", "RELEVANT"),          # case-insensitive
    ("   RELEVANT   ", "RELEVANT"),    # leading/trailing whitespace
])
def test_parse_label_leading(text, expected):
    assert parse_label(text) == expected


def test_parse_label_not_overrides_trailing_relevant_word():
    # "RELEVANT. The paper does not ..." must stay RELEVANT — a trailing 'not'
    # in the justification must not flip it.
    assert parse_label("RELEVANT. This does not change the result.") == "RELEVANT"


def test_parse_label_not_at_start_wins():
    # 'NOT' appearing in the first 4 tokens => NOT_RELEVANT.
    assert parse_label("NOT relevant to the project") == "NOT_RELEVANT"


def test_parse_label_unknown_when_no_label():
    # Free text with no label token anywhere returns UNKNOWN.
    assert parse_label("totally unrelated free text") == "UNKNOWN"


def test_parse_label_stray_not_in_first_tokens_triggers_not_relevant():
    # 'not' within the first 4 tokens is treated as NOT_RELEVANT by design —
    # this documents the parser's leading-token heuristic.
    assert parse_label("I am not sure what to say.") == "NOT_RELEVANT"


def test_parse_label_empty_string_unknown():
    assert parse_label("") == "UNKNOWN"


def test_parse_label_fallback_whole_text_scan():
    assert parse_label("My considered answer is BORDERLINE here") == "BORDERLINE"


class _StubModel:
    """Returns a canned label per item id; records token usage so the ledger
    path is exercised. No network."""

    def __init__(self, name, model, responses):
        self.name = name
        self.model = model
        self._responses = responses

    def complete(self, prompt, *, system=None, max_tokens=1024, temperature=0.0):
        # build_prompt embeds title + abstract (not id), so we key the canned
        # answer off a unique marker we place in the item's title/abstract.
        for marker, label in self._responses.items():
            if marker in prompt:
                return TaskResult(text=label, model=self.model,
                                  input_tokens=10, output_tokens=5)
        return TaskResult(text="UNKNOWN", model=self.model,
                          input_tokens=10, output_tokens=5)


class _FakeLedger:
    """Minimal ledger: records calls, charges a flat 1 cent each (deterministic)."""

    def __init__(self):
        self.records = []
        self.seeded = []

    def seed_pricing(self, model, inp, out, cached):
        self.seeded.append((model, inp, out, cached))

    def record(self, run_id, model, *, phase=None, role=None, input_tokens=0,
               output_tokens=0, cached_input_tokens=0, cache_creation_tokens=0,
               total_cost_usd=None):
        self.records.append((run_id, model, input_tokens, output_tokens))
        return 1  # flat 1 cent per call -> deterministic cost


@pytest.fixture
def stub_specs(monkeypatch, tmp_path):
    """Write a one-backend config and monkeypatch build_model to return a stub
    whose answers we dictate. Returns (config_path, install(responses))."""
    cfg = tmp_path / "task_models.yaml"
    cfg.write_text(
        "task_models:\n"
        "  - name: stub\n"
        "    kind: openai_compat\n"
        "    model: stub-model\n"
        "    base_url: http://localhost/v1\n"
        "    enabled: true\n"
        "    price_usd_per_mtok:\n"
        "      input: 1.0\n"
        "      output: 2.0\n"
        "      cached_input: 0.5\n"
    )

    def install(responses):
        monkeypatch.setattr(
            te, "build_model",
            lambda spec, resolve_key: _StubModel(spec["name"], spec["model"], responses),
        )
    return cfg, install


def test_evaluate_groups_perfect_predictions(stub_specs):
    cfg, install = stub_specs
    items = [
        {"id": "a1", "gold": "RELEVANT", "title": "MARK_a1", "abstract": "x"},
        {"id": "a2", "gold": "NOT_RELEVANT", "title": "MARK_a2", "abstract": "x"},
        {"id": "a3", "gold": "BORDERLINE", "title": "MARK_a3", "abstract": "x"},
    ]
    install({"MARK_a1": "RELEVANT", "MARK_a2": "NOT_RELEVANT", "MARK_a3": "BORDERLINE"})
    ledger = _FakeLedger()
    scores = evaluate_groups([("ctx", items)], config_path=cfg, ledger=ledger,
                             run_id="run1")
    assert len(scores) == 1
    s = scores[0]
    assert s.n == 3
    assert s.acc3 == 1.0                 # all 3-class correct
    # KEEP gold = {a1, a3}; predicted-keep = {a1, a3} -> TP=2, FP=0, FN=0, TN=1
    assert (s.tp, s.fp, s.fn, s.tn) == (2, 0, 0, 1)
    assert s.keep_prec == 1.0
    assert s.keep_rec == 1.0
    assert s.keep_f1 == 1.0
    assert len(s.mistakes) == 0
    # Ledger was seeded once and recorded once per item.
    assert ledger.seeded == [("stub-model", 1.0, 2.0, 0.5)]
    assert len(ledger.records) == 3
    assert s.cost_cents == 3  # flat 1 cent x 3 items


def test_evaluate_groups_confusion_matrix(stub_specs):
    cfg, install = stub_specs
    # Construct a KNOWN confusion matrix on the KEEP class.
    # gold keep: g1,g2,g3 (RELEVANT/BORDERLINE); gold drop: d1,d2
    items = [
        {"id": "g1", "gold": "RELEVANT", "title": "MARK_g1", "abstract": "x"},
        {"id": "g2", "gold": "BORDERLINE", "title": "MARK_g2", "abstract": "x"},
        {"id": "g3", "gold": "RELEVANT", "title": "MARK_g3", "abstract": "x"},
        {"id": "d1", "gold": "NOT_RELEVANT", "title": "MARK_d1", "abstract": "x"},
        {"id": "d2", "gold": "NOT_RELEVANT", "title": "MARK_d2", "abstract": "x"},
    ]
    # Predictions: g1->keep(TP), g2->keep(TP), g3->drop(FN), d1->keep(FP), d2->drop(TN)
    install({
        "MARK_g1": "RELEVANT",        # TP
        "MARK_g2": "BORDERLINE",      # TP
        "MARK_g3": "NOT_RELEVANT",    # FN (gold keep, predicted drop)
        "MARK_d1": "RELEVANT",        # FP (gold drop, predicted keep)
        "MARK_d2": "NOT_RELEVANT",    # TN
    })
    ledger = _FakeLedger()
    s = evaluate_groups([("ctx", items)], config_path=cfg, ledger=ledger,
                        run_id="r")[0]
    assert (s.tp, s.fp, s.fn, s.tn) == (2, 1, 1, 1)
    assert s.keep_prec == pytest.approx(2 / 3)   # TP/(TP+FP)
    assert s.keep_rec == pytest.approx(2 / 3)    # TP/(TP+FN)
    assert s.keep_f1 == pytest.approx(2 / 3)     # P==R -> F1==P
    # acc3: g1,g2,d2 exact-match (3 of 5); g3 and d1 are 3-class wrong.
    assert s.correct3 == 3
    assert s.acc3 == pytest.approx(3 / 5)


def test_evaluate_groups_only_selector(stub_specs):
    cfg, install = stub_specs
    install({"MARK_a1": "RELEVANT"})
    ledger = _FakeLedger()
    items = [{"id": "a1", "gold": "RELEVANT", "title": "MARK_a1", "abstract": "x"}]
    # `only` selects by name regardless of enabled flag.
    scores = evaluate_groups([("ctx", items)], config_path=cfg, ledger=ledger,
                             run_id="r", only=["stub"])
    assert len(scores) == 1 and scores[0].backend == "stub"
    # A name not present in the config selects nothing.
    scores2 = evaluate_groups([("ctx", items)], config_path=cfg, ledger=ledger,
                              run_id="r", only=["nope"])
    assert scores2 == []


def test_evaluate_groups_pools_across_groups(stub_specs):
    cfg, install = stub_specs
    install({"MARK_a1": "RELEVANT", "MARK_b1": "RELEVANT"})
    ledger = _FakeLedger()
    g1 = ("ctxA", [{"id": "a1", "gold": "RELEVANT", "title": "MARK_a1", "abstract": "x"}])
    g2 = ("ctxB", [{"id": "b1", "gold": "RELEVANT", "title": "MARK_b1", "abstract": "x"}])
    s = evaluate_groups([g1, g2], config_path=cfg, ledger=ledger, run_id="r")[0]
    # Results pool across groups: n=2, both TP.
    assert s.n == 2
    assert s.tp == 2


# ===========================================================================
# stats_analysis: keep-all baseline F1 formula
# ===========================================================================

def test_keep_all_baseline_f1_formula():
    # keep-all labels everything KEEP -> recall=1 always, precision=prevalence.
    # F1 = 2*prev/(prev+1) where prev = pos/(pos+neg).
    # pos=3, neg=1 -> prev=0.75 -> F1 = 2*0.75/1.75
    pos, neg = 3, 1
    prev = pos / (pos + neg)
    expected = 2 * prev / (prev + 1)
    assert expected == pytest.approx(0.857142857, abs=1e-6)

    # Cross-check against the precision/recall identity: keep-all gives
    # TP=pos, FP=neg, FN=0 -> precision=pos/(pos+neg)=prev, recall=1.
    p, r = prev, 1.0
    f1_from_pr = 2 * p * r / (p + r)
    assert f1_from_pr == pytest.approx(expected)


def test_keep_all_baseline_matches_score_keep_f1():
    # The closed-form keep-all F1 must equal a Score with TP=pos, FP=neg, FN=0.
    pos, neg = 9, 5
    prev = pos / (pos + neg)
    closed_form = 2 * prev / (prev + 1)
    s = make_score(tp=pos, fp=neg, fn=0, tn=0)
    assert s.keep_f1 == pytest.approx(closed_form)


def test_keep_all_baseline_all_positive():
    # neg=0 -> prev=1 -> F1 = 2/2 = 1.0
    pos, neg = 4, 0
    prev = pos / (pos + neg)
    assert 2 * prev / (prev + 1) == pytest.approx(1.0)


# ===========================================================================
# stats_analysis: topic-clustered bootstrap CI (boot_ci) — seeded/deterministic
# ===========================================================================

def test_boot_ci_brackets_point_estimate():
    vals = [0.10, 0.50, 0.90, 0.30, 0.70]
    point = st.mean(vals)
    random.seed(2024)
    lo, hi = sa.boot_ci(vals)
    assert lo <= point <= hi
    # A bootstrap of means can never exceed the data range.
    assert lo >= min(vals)
    assert hi <= max(vals)
    assert lo < hi


def test_boot_ci_reproducible_across_two_seeded_calls():
    vals = [0.2, 0.4, 0.6, 0.8, 0.5]
    random.seed(99)
    first = sa.boot_ci(vals)
    random.seed(99)
    second = sa.boot_ci(vals)
    assert first == second


def test_boot_ci_constant_vals_degenerate():
    # If every value is identical, every resample mean equals that value,
    # so the CI collapses to a point.
    vals = [0.42, 0.42, 0.42, 0.42, 0.42]
    random.seed(7)
    lo, hi = sa.boot_ci(vals)
    assert lo == pytest.approx(0.42)
    assert hi == pytest.approx(0.42)


# ===========================================================================
# stats_analysis: paired bootstrap — seeded/deterministic
# ===========================================================================

def test_paired_mean_diff_is_exact():
    a = [0.9, 0.8, 0.7, 0.6, 0.5]
    b = [0.4, 0.3, 0.2, 0.1, 0.0]
    random.seed(1)
    md, lo, hi, p_gt = sa.paired(a, b)
    # mean diff is deterministic (not bootstrapped): each a-b = 0.5 -> mean 0.5
    assert md == pytest.approx(0.5)
    # Strictly-dominant A: every bootstrap diff > 0, so P(A>B)=1.0 and CI>0.
    assert p_gt == 1.0
    assert lo > 0


def test_paired_reproducible_across_two_seeded_calls():
    a = [0.5, 0.6, 0.55, 0.62, 0.48]
    b = [0.5, 0.59, 0.57, 0.6, 0.5]
    random.seed(314)
    first = sa.paired(a, b)
    random.seed(314)
    second = sa.paired(a, b)
    assert first == second


def test_paired_ci_brackets_mean_diff():
    a = [0.7, 0.6, 0.8, 0.65, 0.75]
    b = [0.5, 0.55, 0.6, 0.5, 0.58]
    md = st.mean([x - y for x, y in zip(a, b)])
    random.seed(2025)
    md_out, lo, hi, p_gt = sa.paired(a, b)
    assert md_out == pytest.approx(md)
    assert lo <= md <= hi
    assert 0.0 <= p_gt <= 1.0


def test_paired_identical_inputs_zero_diff():
    a = [0.3, 0.5, 0.7, 0.2, 0.9]
    random.seed(11)
    md, lo, hi, p_gt = sa.paired(a, list(a))
    assert md == pytest.approx(0.0)
    # All diffs are exactly 0 -> CI collapses to 0 and P(A>B)=0 (no strict >0).
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)
    assert p_gt == 0.0


# ===========================================================================
# Bootstrap percentile-index sanity (matches the code's index arithmetic)
# ===========================================================================

def test_boot_ci_index_math():
    # The code uses means[int(.025*B)] and means[int(.975*B)] with B=20000.
    assert int(.025 * sa.B) == 500
    assert int(.975 * sa.B) == 19500
    assert sa.B == 20000
