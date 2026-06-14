"""Integrity guard for the saved claim-entailment battery asset (fast — no model/network)."""

from collections import Counter

from litstream_evidence.claim_battery import battery_meta, load_battery

_MODES = {"faithful_positive", "faithful_paraphrase", "faithful_synonym", "distribution_swap",
          "direction_reversal", "explicit_negation", "contradiction", "quantifier_scope",
          "world_knowledge_bait"}


def test_battery_shape_and_balance():
    cases = load_battery()
    assert len(cases) == 41
    gold = Counter(c["gold"] for c in cases)
    assert gold[True] == 14 and gold[False] == 27
    assert set(c["mode"] for c in cases) == _MODES


def test_every_case_well_formed():
    for c in load_battery():
        assert c["passage"] and c["claim"] and isinstance(c["gold"], bool)
        assert c["mode"] in _MODES and c["rationale"]
        assert {"n_agree", "n_total", "category"} <= set(c["audit"])


def test_audit_quality_held():
    # the 5-stance blind audit found >=40/41 unanimous-with-gold and zero suspect cases
    cat = Counter(c["audit"]["category"] for c in load_battery())
    assert cat["rock_solid"] >= 40
    assert cat.get("suspect", 0) == 0
    assert battery_meta()["n"] == 41
