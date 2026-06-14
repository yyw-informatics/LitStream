"""A1 slot-level extraction P/R/F1 of the live MINE sidecars against a gold key.

Scores the production structurer output (`<stem>_evidence.json`, schema =
`litstream_evidence/evidence_schema.py`) under `projects/<name>/literature/` against a
hand-labeled gold key (`extraction_key.jsonl`, one JSON object per line, mirroring
`triage_set.jsonl`). Per field and overall, reports precision (of what MINE pulled, the
fraction that is a real gold item; hallucination rate) split from recall (of the gold items,
the fraction MINE found; miss rate), micro and macro.

Matching is not reimplemented. The maximum one-to-one matcher, the entity/number matcher
families, and P/R/F1 come from `benchmark/score.py`; the per-field item->string keys and
    surface normalization come from `benchmark/fields.py`. This module adds sidecar/gold-key loading
    and species-aware gene alias expansion for the entity matcher.

Two tiers are reported with their delta:
  - strict   : exact normalized string (the floor)
  - normalized: relaxed matcher (substring for entities, value+unit for numbers) plus the
               species-aware gene alias expansion below

    python -m litstream.eval.extraction_score --project lg_smoke \
        --project-dir /tmp/lg_smoke_proj
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .benchmark.fields import SCORED
from .benchmark.schema import normalize
from .benchmark.score import MATCHERS, count, prf

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DEFAULT_KEY = Path(__file__).resolve().parent / "extraction_key.jsonl"

# Every field with a defined key/matcher in fields.py. The gold key may label any subset.
SCORED_FIELDS = list(SCORED)


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


# Mouse uses MGI, human uses HGNC; the structurer emits HUGO (human) / MGI (mouse) casing, so
# genes at the same locus differ only by case across species (human IL2RA vs mouse Il2ra).
# Route by the record's `species` and fold case so a cross-spelling still matches, then expand
# a handful of known symbol aliases.

_HUMAN = "human"
_MOUSE = "mouse"

# Equivalence groups: every spelling in a group grounds the others, since a synthesis may write
# a surface marker (CD45RA) while the cited paper writes the gene (PTPRC), or vice-versa. Kept
# small and auditable.
_ALIAS_GROUPS: list[set[str]] = [
    {"cd25", "il2ra"}, {"cd279", "pdcd1", "pd-1", "pd1"}, {"pd-l1", "cd274"},
    {"cd20", "ms4a1"}, {"foxp3", "scurfin"},
    {"cd45", "cd45ra", "cd45ro", "cd45rb", "ptprc"}, {"cd8", "cd8a", "cd8b"},
    {"cd3", "cd3d", "cd3e", "cd3g"}, {"cd127", "il7r"}, {"cd16", "fcgr3a"},
    {"cd56", "ncam1"}, {"cd169", "siglec1"}, {"cd21", "cr2"}, {"cd11b", "itgam"},
    {"cd11c", "itgax"}, {"cd123", "il3ra"}, {"cd62l", "sell"}, {"ccr7", "cd197"},
    {"ctla4", "cd152"}, {"cd138", "sdc1"}, {"cd117", "kit"},
]
_GENE_ALIASES: dict[str, set[str]] = {}
for _grp in _ALIAS_GROUPS:
    for _sym in _grp:
        _GENE_ALIASES.setdefault(_sym, set()).update(_grp - {_sym})


def species_of(record: dict) -> str:
    """The record's species, lowered to {'human','mouse',''}. Picks the first recognized
    organism in the `species` list; '' (unknown) means no species-specific routing."""
    for s in record.get("species", []) or []:
        t = str(s).casefold()
        if "human" in t or "homo" in t or "patient" in t:
            return _HUMAN
        if "mouse" in t or "mus " in t or t == "mus" or "murine" in t:
            return _MOUSE
    return ""


def gene_aliases(symbol: str, species: str) -> set[str]:
    """Accepted spellings for a gene symbol under a species. Species routing is currently only
    casefold (HGNC human and MGI mouse differ by case); the alias table is shared. When an
    HGNC/MGI table is wired in, branch on `species` here."""
    s = normalize(symbol)
    if not s:
        return set()
    return {s} | _GENE_ALIASES.get(s, set())


# ---- loading sidecars + gold key ------------------------------------------------

def matchable(record: dict, field: str) -> set[str]:
    """Normalized strings for one field of one record (predictions side). Reuses fields.py's
    per-field key + the shared normalizer; identical to the benchmark's own loader."""
    key = SCORED[field]["key"]
    out: set[str] = set()
    for item in record.get(field, []) or []:
        s = normalize(key(item))
        if s:
            out.add(s)
    return out


def load_sidecars(project: str, project_dir: Path) -> dict[str, dict]:
    """{paper_id -> record} from the live `<stem>_evidence.json` sidecars. paper_id is the
    record's own field, falling back to the file stem (matches the structurer's default)."""
    lit = project_dir / f"projects/{project}/literature"
    files = sorted(lit.glob("*_evidence.json")) if lit.is_dir() else []
    recs: dict[str, dict] = {}
    for jf in files:
        stem = jf.name[: -len("_evidence.json")]
        try:
            rec = json.loads(jf.read_text())
        except Exception as exc:
            log(f"{stem}: bad JSON ({exc}) — skipping"); continue
        recs[rec.get("paper_id") or stem] = rec
    return recs


def load_key(path: Path) -> dict[str, dict[str, list[str]]]:
    """Gold key → {paper_id -> {field -> [gold strings]}}. One JSON object per line:
    {"paper_id": ..., "genes": [...], "cell_types": [...]}. Only scored fields are kept."""
    gold: dict[str, dict[str, list[str]]] = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        obj = json.loads(ln)
        pid = obj.get("paper_id")
        if not pid:
            continue
        gold[pid] = {f: list(obj.get(f, []) or []) for f in SCORED_FIELDS if f in obj}
    return gold


def gold_items(values: list[str], field: str, species: str, normalized: bool) -> list[set[str]]:
    """Gold side for count(), as alias-concepts: each item is a set of accepted spellings. In the
    normalized tier, genes expand to their full alias-set (gold 'CD25' also accepts 'IL2RA'); every
    other item is a one-spelling set, which count() matches exactly as before. Empty strings dropped."""
    if field == "genes" and normalized:
        return [a for a in (gene_aliases(v, species) for v in values) if a]
    return [{s} for s in (normalize(v) for v in values) if s]


# ---- scoring --------------------------------------------------------------------

def score(recs: dict[str, dict], gold: dict[str, dict[str, list[str]]], tier: str) -> list[dict]:
    """One row per (paper, field): TP/FP/FN under one tier. Only (paper, field) pairs that the
    gold key labels are scored; an unlabeled field is unmeasured, not a miss.

    In the normalized tier, gold genes carry species-aware alias-sets so a prediction matching
    any spelling of the locus scores it once (count()'s alias-group contract); predictions stay
    bare. Non-gene fields go through the shared fields.py key + matcher unchanged."""
    normalized = tier != "strict"
    mode = "relaxed" if normalized else "strict"
    rows: list[dict] = []
    for pid, fields in sorted(gold.items()):
        rec = recs.get(pid)
        species = species_of(rec) if rec else ""
        for field, values in sorted(fields.items()):
            matcher = MATCHERS[(SCORED[field]["match"], mode)]
            g = gold_items(values, field, species, normalized)
            # Predictions are always bare normalized symbols; the gene alias expansion lives on
            # the gold side (count() credits a pred matching any spelling of a gold concept once,
            # the BioRED alias-group contract), so it can't inflate precision.
            pred = matchable(rec, field) if rec is not None else set()
            b = count(g, pred, matcher)
            rows.append({"paper": pid, "field": field, "tier": tier, **b,
                         "mined": rec is not None})
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Roll per-(paper,field) counts up to per-field, micro (count-weighted), and macro
    (unweighted field-mean) P/R/F1, per tier."""
    out: dict = {}
    for tier in sorted({r["tier"] for r in rows}):
        trows = [r for r in rows if r["tier"] == tier]
        by_field: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for r in trows:
            for k in ("tp", "fp", "fn"):
                by_field[r["field"]][k] += r[k]
        field_prf = {f: prf(b) for f, b in by_field.items()}
        micro_b = {k: sum(b[k] for b in by_field.values()) for k in ("tp", "fp", "fn")}
        macro = (
            tuple(statistics.mean(v[i] for v in field_prf.values()) for i in range(3))
            if field_prf else (0.0, 0.0, 0.0))
        out[tier] = {"by_field": dict(by_field), "field_prf": field_prf,
                     "micro": prf(micro_b), "micro_counts": micro_b, "macro": macro}
    return out


# ---- output ---------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict], agg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "tier", "field", "precision", "recall", "f1", "tp", "fp", "fn"])
        for tier, a in agg.items():
            for field, b in sorted(a["by_field"].items()):
                p, r, fl = a["field_prf"][field]
                w.writerow(["field", tier, field, f"{p:.3f}", f"{r:.3f}", f"{fl:.3f}",
                            b["tp"], b["fp"], b["fn"]])
            mp, mr, mf = a["micro"]; mc = a["micro_counts"]
            w.writerow(["overall-micro", tier, "", f"{mp:.3f}", f"{mr:.3f}", f"{mf:.3f}",
                        mc["tp"], mc["fp"], mc["fn"]])
            xp, xr, xf = a["macro"]
            w.writerow(["overall-macro", tier, "", f"{xp:.3f}", f"{xr:.3f}", f"{xf:.3f}",
                        "", "", ""])
    log(f"wrote {path.name}")


def summarize(agg: dict, n_papers: int, n_mined: int) -> str:
    tiers = list(agg)
    lines = ["# A1 — slot-level extraction P/R/F1 (live MINE sidecars vs gold key)", "",
             f"papers in gold key: **{n_papers}** · with a mined sidecar: **{n_mined}** "
             f"(labeled-but-unmined papers count every gold item as a miss)", ""]
    for tier in tiers:
        a = agg[tier]
        lines += [f"## {tier} tier", "",
                  "| field | precision (1−halluc.) | recall (1−miss) | F1 | TP | FP | FN |",
                  "|---|---|---|---|---|---|---|"]
        for field, b in sorted(a["by_field"].items()):
            p, r, fl = a["field_prf"][field]
            lines.append(f"| {field} | {p:.3f} | {r:.3f} | {fl:.3f} "
                         f"| {b['tp']} | {b['fp']} | {b['fn']} |")
        mp, mr, mf = a["micro"]; mc = a["micro_counts"]; xp, xr, xf = a["macro"]
        lines += [f"| **micro** | {mp:.3f} | {mr:.3f} | {mf:.3f} "
                  f"| {mc['tp']} | {mc['fp']} | {mc['fn']} |",
                  f"| **macro** | {xp:.3f} | {xr:.3f} | {xf:.3f} | — | — | — |", ""]
    if {"strict", "normalized"} <= set(tiers):
        ds = agg["normalized"]["micro"][2] - agg["strict"]["micro"][2]
        dm = agg["normalized"]["macro"][2] - agg["strict"]["macro"][2]
        lines += [f"**Δ F1 (normalized − strict):** micro {ds:+.3f} · macro {dm:+.3f} "
                  "— how much normalization (substring + species-aware gene aliases) recovers "
                  "over exact-string matching.", ""]
    lines += [
        "> **precision** = of the items MINE pulled, the fraction that are real gold items "
        "(low precision = hallucinations). **recall** = of the gold items, the fraction MINE "
        "found (low recall = misses). Matching is maximum one-to-one (`benchmark/score.py`).",
        "> **strict** = exact normalized string (the floor). **normalized** = relaxed matcher "
        "(entity substring, number value+unit) plus species-aware gene aliases (human→HGNC, "
        "mouse→MGI, via a casefold + small alias table seam).",
        "",
        "> ⚠️ **Scope & honesty.** This measures extraction against *OUR* hand-labeled key, "
        "not against truth — the key is itself a fallible artifact. The sample is **small-n** "
        "(a 10–12 paper key), so **read the raw TP/FP/FN counts, not just the rates**, and do "
        "not crown a winner on a few papers. surface_markers and signatures are unscored (no "
        "key field); relevance and tissue are internal.",
    ]
    return "\n".join(lines)


# ---- runner + CLI ---------------------------------------------------------------

def run(project: str, project_dir: Path, key_path: Path) -> tuple[list[dict], dict, int, int]:
    gold = load_key(key_path)
    recs = load_sidecars(project, project_dir)
    rows = score(recs, gold, "strict") + score(recs, gold, "normalized")
    agg = aggregate(rows)
    n_mined = sum(1 for pid in gold if pid in recs)
    return rows, agg, len(gold), n_mined


def main() -> None:
    ap = argparse.ArgumentParser(
        description="A1 slot-level extraction P/R/F1 of MINE sidecars vs a gold key")
    ap.add_argument("--project", required=True)
    ap.add_argument("--project-dir", required=True,
                    help="dir containing projects/<project>/literature/*_evidence.json")
    ap.add_argument("--key", default=str(DEFAULT_KEY),
                    help="gold key JSONL (default: litstream/eval/extraction_key.jsonl)")
    args = ap.parse_args()

    key_path = Path(args.key)
    if not key_path.exists():
        log(f"gold key not found: {key_path} — nothing to score "
            f"(see {DEFAULT_KEY.with_name('extraction_key.example.jsonl').name})")
        return
    rows, agg, n_papers, n_mined = run(args.project, Path(args.project_dir).resolve(), key_path)
    if not rows:
        log("gold key has no scorable (paper, field) pairs — nothing scored"); return
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "extraction_score.csv", rows, agg)
    summary = summarize(agg, n_papers, n_mined)
    (RESULTS / "extraction_score_SUMMARY.md").write_text(summary + "\n")
    log("wrote extraction_score_SUMMARY.md")
    print("\n" + summary + "\n")


if __name__ == "__main__":
    main()
