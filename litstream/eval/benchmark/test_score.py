"""Scorer correctness tests. Run: python3 -m litstream.eval.benchmark.test_score
(also pytest-compatible)."""

from .score import (count, entity_relaxed, entity_strict, number_relaxed, prf)


def test_recall_not_inflated_by_duplicate_predictions():
    # 3 near-duplicate preds all match ONE gold; another gold is missed.
    # One-to-one matching credits 1 of 2 gold (recall 0.5), not 3/4.
    b = count({"cd8", "tp53"}, {"cd8", "cd8a", "cd8b"}, entity_relaxed)
    assert b == {"tp": 1, "fp": 2, "fn": 1}, b
    p, r, _ = prf(b)
    assert abs(p - 1 / 3) < 1e-9 and abs(r - 1 / 2) < 1e-9, (p, r)


def test_precision_not_inflated_by_redundant_predictions():
    # The unit-bearing pred matches gold; the unit-dropped one does not.
    b = count({"2617.4 m"}, {"2617.4 m", "2617.4"}, number_relaxed)
    assert b == {"tp": 1, "fp": 1, "fn": 0}, b


def test_max_matching_beats_greedy():
    # JNLPBA-style case: a flexible pred must yield its gold to a rigid pred.
    # Greedy would yield tp=3; the maximum matching yields tp=4.
    gold = {"lymphocytes", "peripheral blood leukocytes", "peripheral blood lymphocyte",
            "peripheral blood polymorphonuclear leukocyte", "polymorphonuclear leukocytes"}
    pred = {"leukocyte", "leukocytes", "lymphocyte", "lymphocytes"}
    assert count(gold, pred, entity_relaxed)["tp"] == 4


def test_number_relaxed_rejects_distinct_numbers():
    assert not number_relaxed("0.36 m s-1", "0.86 m s-1")
    assert not number_relaxed("4.37 twh", "4.99 twh")
    assert not number_relaxed("5%", "15%")
    # multi-value: EVERY number must match, not just the leading one
    assert not number_relaxed("10-5 mbar", "10-7 mbar")
    assert not number_relaxed("5 to 10 mm", "5 to 80 mm")
    assert not number_relaxed("2619.6 and 2614.7 m", "2619.6 and 9999.9 m")


def test_number_relaxed_requires_units():
    assert not number_relaxed("1", "1 m")            # bare != unit-bearing
    assert not number_relaxed("5%", "5")             # percent != bare
    assert not number_relaxed("5%", "5 m")           # percent != metre
    assert not number_relaxed("2617.4 m", "2617.4")  # don't credit a dropped unit
    assert not number_relaxed("30 m", "30 s")        # different units


def test_number_relaxed_accepts_intended():
    assert number_relaxed("30 c", "30 °c")           # degree symbol is decorative
    assert number_relaxed("approximately 100", "100")  # qualifier word ignored
    assert number_relaxed("5 %", "5%")               # spacing
    assert number_relaxed("5 mm", "mm 5")            # word order


def test_number_relaxed_respects_comparators():
    assert not number_relaxed(">10%", "below 10%")        # opposite directions
    assert not number_relaxed("10 nm", "less than 10 nm")  # bare vs upper-bounded
    assert not number_relaxed("0.05", "< 0.05")
    assert not number_relaxed("more than 10%", "10%")
    assert not number_relaxed("greater than 10%", "10%")   # consistent with 'more than'
    assert number_relaxed("< 0.05", "<0.05")              # same direction, spacing only
    assert number_relaxed("up to 5 m", "at most 5 m")     # synonymous upper bounds


def test_strict_is_one_to_one_exact():
    b = count({"cd8", "tp53"}, {"cd8", "foxp3"}, entity_strict)
    assert b == {"tp": 1, "fp": 1, "fn": 1}, b


def test_alias_group_counts_once_matched_by_any_spelling():
    # BioRED lists a gene as symbol AND full name (one concept). MINE returns only the
    # symbol -> it should score the concept once, with no phantom "miss" of the full name.
    gold = [frozenset({"adcy5", "adenylate cyclase 5"})]
    assert count(gold, {"adcy5"}, entity_relaxed) == {"tp": 1, "fp": 0, "fn": 0}
    # species: person-words + 'human' are one concept; MINE's 'human' matches it.
    species = [frozenset({"human", "patient", "patients", "men", "women"})]
    assert count(species, {"human"}, entity_relaxed) == {"tp": 1, "fp": 0, "fn": 0}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
