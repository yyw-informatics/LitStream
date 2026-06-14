"""The shared grader: maximum one-to-one match of pulled items to gold, then P/R/F1.

MATCHING IS MAXIMUM ONE-TO-ONE. Each pulled item can claim at most one gold answer and
each gold answer can be claimed once; we take the assignment that pairs the MOST items
(maximum bipartite matching, Kuhn's algorithm). This keeps precision and recall coherent
and correct under relaxed matching — a plain greedy pass would leave some pairings on the
table and under-count, and naive any-match counting would over-count near-duplicates.

Matchers, by field family and mode:
  entity  strict  -> exact normalized string
  entity  relaxed -> substring either way (CD8 counts as CD8A)
  number  strict  -> exact normalized string
  number  relaxed -> SAME multiset of numbers AND same unit signature. Keyed on the actual
                     numbers + units, so 0.36 != 0.86, '10-5 mbar' != '10-7 mbar', and a
                     bare '5' != '5 m' != '5%'. Qualitative gates ('high') fall back to tokens.
"""

from __future__ import annotations

import re
from typing import Callable

Matcher = Callable[[str, str], bool]


# ---- entity matchers -----------------------------------------------------------

def entity_strict(a: str, b: str) -> bool:
    return a == b


def entity_relaxed(a: str, b: str) -> bool:
    return a in b or b in a


# ---- number matchers -----------------------------------------------------------

# words that decorate a quantity without being a unit (so they don't affect the unit set).
# comparator words are here too — their DIRECTION is captured separately by _direction().
_QUAL = {"approximately", "approx", "about", "around", "roughly", "nearly", "circa", "ca",
         "between", "to", "and", "or", "x", "over", "under", "up", "down", "greater",
         "less", "than", "above", "below", "more", "fewer", "exceeding", "most", "least",
         "at", "no", "not"}
_UNIT_SYM = re.compile(r"[%‰:/]")          # symbol units (percent, permille, ratio, per)
_CMP_SYMS = (("≤", "le"), ("⩽", "le"), ("≥", "ge"), ("⩾", "ge"), ("<", "lt"), (">", "gt"))


def _nums(s: str) -> list[str]:
    return sorted(re.findall(r"\d+(?:\.\d+)?", s))


def _units(s: str) -> frozenset[str]:
    s = s.lower()
    words = {t for t in re.findall(r"[a-zµμ]+", s) if t not in _QUAL}
    return frozenset(words | set(_UNIT_SYM.findall(s)))   # '°' is decorative, left out


def _direction(s: str) -> str:
    """Comparator direction class: '' (none) / lt / gt / le / ge. A lower bound, an upper
    bound, and a point value are different claims, so this must match too."""
    low = s.lower()
    for sym, d in _CMP_SYMS:
        if sym in s:
            return d
    if "up to" in low or "at most" in low or "no more than" in low or "not more than" in low:
        return "le"
    if "at least" in low or "no less than" in low:
        return "ge"
    words = set(re.findall(r"[a-z]+", low))
    if words & {"below", "under", "less", "fewer"}:
        return "lt"
    if words & {"above", "over", "greater", "more", "exceeding"}:
        return "gt"
    return ""


def _token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0


def number_strict(a: str, b: str) -> bool:
    return a == b


def number_relaxed(a: str, b: str) -> bool:
    na, nb = _nums(a), _nums(b)
    if not na and not nb:                  # qualitative gate, e.g. "high" / "low"
        return _token_overlap(a, b) >= 0.5
    if na != nb:                            # every number must match (handles ranges/lists)
        return False
    if _units(a) != _units(b):              # '5' != '5 m' != '5%'; '30 c' == '30 °c'
        return False
    return _direction(a) == _direction(b)   # '>10%' != 'below 10%' != '10%'


MATCHERS: dict[tuple[str, str], Matcher] = {
    ("entity", "strict"): entity_strict,
    ("entity", "relaxed"): entity_relaxed,
    ("number", "strict"): number_strict,
    ("number", "relaxed"): number_relaxed,
}


# ---- counting (maximum one-to-one matching) ------------------------------------

def count(gold, preds: set[str], matcher: Matcher) -> dict[str, int]:
    """Maximum bipartite matching (Kuhn's). Sorted inputs -> deterministic.

    Each gold item is a CONCEPT = one or more acceptable spellings (aliases). A plain
    string is treated as a one-spelling concept, so callers passing a set[str] match on
    that single string; BioRED passes alias-sets (symbol + full name share one concept) so
    a prediction matching ANY spelling scores the concept once.

    TP = size of the maximum matching · FP = unmatched preds · FN = unmatched concepts."""
    groups = [frozenset(g) if isinstance(g, (set, frozenset)) else frozenset({g}) for g in gold]
    groups.sort(key=lambda grp: tuple(sorted(grp)))      # determinism
    plist = sorted(preds)
    adj = [[j for j, grp in enumerate(groups) if any(matcher(p, a) for a in grp)]
           for p in plist]
    match_g = [-1] * len(groups)            # concept index -> pred index, or -1

    def augment(i: int, seen: list[bool]) -> bool:
        for j in adj[i]:
            if not seen[j]:
                seen[j] = True
                if match_g[j] == -1 or augment(match_g[j], seen):
                    match_g[j] = i
                    return True
        return False

    tp = sum(augment(i, [False] * len(groups)) for i in range(len(plist)))
    return {"tp": tp, "fp": len(plist) - tp, "fn": len(groups) - tp}


def prf(b: dict) -> tuple[float, float, float]:
    p = b["tp"] / (b["tp"] + b["fp"]) if (b["tp"] + b["fp"]) else 0.0
    r = b["tp"] / (b["tp"] + b["fn"]) if (b["tp"] + b["fn"]) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f
