"""Deterministic, no-API faithfulness scorer for the SYNTHESIZE stage.

Answers the same attribution question as the model-based citation checker
(`citation_check.py`) with pure string rules — no model, no network, no randomness,
no clock — so it reruns identically and serves as a reproducible baseline alongside
the verifier cascade.

Parsing and resolution are reused from `citation_check.py` (claim/citation pairing,
the Appendix-A table -> `*_evidence.md` mapping, frontmatter stripping, External-Review
handling). Per resolved claim/evidence pair, three checks:

  number_grounded  every number (with unit) in the claim appears in the cited evidence;
                   unit-aware, so '5%' is not satisfied by a bare '5'. Catches fabricated stats.
  entity_grounded  every domain entity in the claim (genes, markers, acronym symbols)
                   appears in the evidence by string match.
  lexical_overlap  token-F1 between the claim and its best-matching evidence sentence,
                   thresholded.

A claim passes iff all three hold against the union of its cited evidence. We aggregate a
faithfulness rate, per-check pass-rates, and the flagged claims, plus structural metrics
(uncited-claim rate, dangling-citation rate, coverage, count/set consistency).

Being lexical, it cannot credit a valid paraphrase (false unsupported) nor catch a
negation/direction flip when words overlap (false supported). It is the deterministic
floor, not the ceiling.

    python -m litstream.eval.synthesis_faithfulness --project citeseq_methods \
        --project-dir kb-skills-bioinformatics
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

# Reuse parsing/resolution rather than re-implementing it.
from litstream.eval.citation_check import (
    Claim,
    iter_clauses,
    parse_appendix_a,
    parse_citations,
    parse_claims,
    split_frontmatter,
    _CITE_RE,
    _ANY_APPENDIX_RE,
)
# Benchmark number extractor; see number_grounded for how its semantics are applied.
from litstream.eval.benchmark.score import _nums
from litstream.eval.benchmark.schema import normalize
# Shared surface-marker/gene-symbol aliases, so 'PTPRC' grounds against 'CD45RA'.
from litstream.eval.extraction_score import gene_aliases

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

DEFAULT_OVERLAP_THRESHOLD = 0.3   # token-F1 floor; tuned for short cited clauses vs evidence prose

# The Appendix-A provenance table supplies the citation->file mapping but is not scored as prose;
# we trim it (and anything after) from the body via the shared _ANY_APPENDIX_RE from citation_check.
_MAX_FLAGGED = 25   # cap the summary's flagged list; the CSV always carries the full set


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


# ---- domain-entity extraction --------------------------------------------------

# A gene/marker/protein token, matched by one of the three branches below. Conservative by
# design, so prose words like "Cell" or "Protein" don't qualify and make entity_grounded
# pass on every claim. Examples matched: CD8, FOXP3, totalVI, CITE-seq, IL2RA.
_ENTITY_RE = re.compile(
    r"\b(?:"
    r"CD\d+[A-Za-z]?"                       # CD markers: CD4, CD8A, CD25
    r"|[A-Z][A-Za-z]*\d[A-Za-z0-9\-]*"      # symbol with an embedded digit: FOXP3, IL2RA, totalVI
    r"|[A-Z]{3,}(?:-[A-Za-z]+)?"            # all-caps acronym, optional hyphen tail: ADT, CITE-seq, scRNA
    r")\b"
)
# All-caps tokens that are domain noise or generic method/analysis acronyms, not specific entities;
# requiring them in the evidence would wrongly flag a claim that merely names a common method.
_ENTITY_STOP = {
    "DNA", "RNA", "AND", "THE", "FOR", "WITH", "NK",
    "PCA", "UMAP", "TSNE", "GPU", "CPU", "AUC", "AUROC", "AUPRC", "ROC",
    "FACS", "FDR", "PBMC", "ATAC", "QC", "API", "CSV", "JSON", "HTML",
    # table status / relevance labels, not domain entities
    "CORE", "SUPPORTING", "EXPLORATORY", "YES", "MIXED", "HIGH", "MODERATE", "LOW", "TBD", "NA",
}


def claim_entities(text: str) -> list[str]:
    """Domain-entity tokens in a claim, normalized and de-duplicated (order preserved). Genes,
    markers and acronym-style cell labels; common prose words are excluded by construction."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _ENTITY_RE.finditer(text or ""):
        tok = m.group(0)
        if tok.upper() in _ENTITY_STOP:
            continue
        key = normalize(tok)
        if key and key not in seen:
            seen.add(key)
            out.append(tok)
    return out


# ---- the three deterministic checks --------------------------------------------

# A quantity is a number with an immediately-attached unit symbol (%, ‰), the unit-bearing form that
# must match exactly: '5%' is not the same as a bare '5'.
_QUANTITY_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?\s*[%‰]")


def number_grounded(claim: str, evidence: str) -> bool:
    """Every number in the claim appears as a standalone quantity in the evidence, unit-aware.

    Two requirements:
      - a unit-bearing quantity in the claim ('0.4%', '5.2%') must appear with its number anchored on
        the left, so a fabricated '0.4%' is not satisfied by '5.2%', '15.4%', or a bare '0.4';
      - any remaining bare number must appear as a standalone token (not a digit inside CD8/FOXP3),
        extracted with the benchmark `_nums` helper so the semantics match the extraction scorer.
    A claim with no numbers is trivially number-grounded."""
    ev_low = (evidence or "").lower()
    # '#1' / 'No. 2' are rank markers, not groundable quantities; drop them before extraction.
    claim_low = re.sub(r"#\s*\d+|\bno\.?\s*\d+\b", " ", claim.lower())

    # Unit-bearing quantities first; anchor the number on the left so a claimed '5%' is not satisfied
    # by '15%' or '0.5%' in the evidence.
    for q in _QUANTITY_RE.findall(claim_low):
        num, unit = re.match(r"(\d+(?:\.\d+)?)\s*([%‰])", q).group(1, 2)
        if not re.search(rf"(?<![\d.]){re.escape(num)}\s*{re.escape(unit)}", ev_low):
            return False

    # Bare numbers (not part of a unit-bearing quantity) must appear as standalone tokens. Strip
    # the quantities already checked above so the '5' of a '5%' isn't re-counted as a bare number.
    remainder = " ".join(_QUANTITY_RE.sub(" ", claim_low).split())
    for n in _nums(remainder):
        if not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d.])", ev_low):
            return False
    return True


def entity_grounded(claim: str, evidence: str) -> bool:
    """Every domain entity in the claim appears in the evidence. Surface-marker/gene-symbol
    equivalences are accepted via the shared alias seam, so a claim's gene ('PTPRC') grounds against
    the cited paper's marker ('CD45RA') and vice-versa. A claim naming no entities is trivially
    grounded."""
    ents = claim_entities(claim)
    if not ents:
        return True
    ev_norm = normalize(evidence)
    return all(any(alias in ev_norm for alias in gene_aliases(ent, "")) for ent in ents)


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n")


def lexical_overlap(claim: str, evidence: str) -> float:
    """Token-F1 between the claim and its best-matching evidence sentence (pure string metric).

    Per-sentence token-F1, taking the max, so a short claim is scored against the single sentence
    that backs it rather than diluted across the whole file. Returns 0.0 when there is no evidence
    or no claim tokens."""
    sents = [s for s in _SENT_SPLIT.split(evidence or "") if s.strip()]
    if not sents:
        return 0.0
    best = 0.0
    ct = set(re.findall(r"[a-z0-9]+", claim.lower()))
    if not ct:
        return 0.0
    for s in sents:
        st = set(re.findall(r"[a-z0-9]+", s.lower()))
        if not st:
            continue
        inter = len(ct & st)
        if not inter:
            continue
        prec, rec = inter / len(st), inter / len(ct)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best:
            best = f1
    return round(best, 3)


# ---- per-claim scoring ---------------------------------------------------------

@dataclass
class ClaimScore:
    """The deterministic verdict for one cited claim (a row in the output CSV)."""
    claim: str
    citations: str                       # raw markers joined, e.g. "[Zheng 2025]; [Yin 2025]"
    evidence_files: str                  # resolved files joined; "" when unresolved
    number_grounded: bool
    entity_grounded: bool
    lexical_overlap: float
    passed: bool
    reason: str = ""                     # why it failed (for the flagged list)


def _resolve_evidence(claim: Claim, mapping: dict[str, list[str]],
                      evidence_text: dict[str, str]) -> tuple[list[str], str]:
    """Resolve a claim's non-external citations to evidence text. Returns (files, joined_text) where
    files are the resolvable, non-empty evidence files and joined_text is their concatenation, the
    union the checks run against (any one cited source may carry the support)."""
    files: list[str] = []
    for cite in claim.citations:
        if cite.external or not cite.key:
            continue
        for fn in mapping.get(cite.key, []):
            if fn in files:
                continue
            if evidence_text.get(fn, "").strip():
                files.append(fn)
    joined = "\n".join(evidence_text.get(fn, "") for fn in files)
    return files, joined


def score_claim(claim: Claim, mapping: dict[str, list[str]], evidence_text: dict[str, str],
                threshold: float) -> ClaimScore | None:
    """Score one cited claim, or None if it carries no resolvable (non-external) citation — those are
    structural concerns (dangling/uncited), handled by `structural_metrics`, not faithfulness."""
    cites = "; ".join(c.raw for c in claim.citations)
    files, joined = _resolve_evidence(claim, mapping, evidence_text)
    if not files:
        return None
    num_ok = number_grounded(claim.text, joined)
    ent_ok = entity_grounded(claim.text, joined)
    lex = lexical_overlap(claim.text, joined)
    passed = num_ok and ent_ok and lex >= threshold
    reasons = []
    if not num_ok:
        reasons.append("number not grounded")
    if not ent_ok:
        reasons.append("entity not grounded")
    if lex < threshold:
        reasons.append(f"lexical_overlap {lex} < {threshold}")
    return ClaimScore(claim.text, cites, ", ".join(files), num_ok, ent_ok, lex, passed,
                      "; ".join(reasons))


# ---- structural integrity (no semantics) ---------------------------------------

# A count phrase the synthesis might assert, e.g. "28 clustering methods" or "12 papers". Such
# counts are checked against a co-located list/set when one is detectable in the same clause.
_COUNT_RE = re.compile(r"\b(\d+)\s+(?:[a-z]+\s+){0,3}(methods|papers|studies|markers|genes|datasets)\b",
                       re.IGNORECASE)
def _all_clauses(body: str) -> list[str]:
    """Every declarative clause in the synthesis body (cited or uncited), the denominator for the
    uncited-claim rate. Table-aware via `iter_clauses`; drops headings."""
    return [c for c in (cl.strip() for cl in iter_clauses(body))
            if len(c) >= 8 and not c.startswith("#")]


def structural_metrics(body: str, mapping: dict[str, list[str]],
                       evidence_text: dict[str, str]) -> dict:
    """Structure only, no entailment or semantic similarity. Takes the synthesis body so it can
    count uncited clauses, which `parse_claims` drops.

      uncited_claim_rate   of all declarative clauses, the fraction with no citation marker.
      dangling_rate        of all citation markers on cited claims, the fraction that resolve to
                           nothing (no Appendix-A entry, or an entry whose file is missing/empty).
      coverage             of the union of known evidence files, the fraction cited by >=1 claim.
      count_consistency    where a "N <things>" phrase sits next to a delimited list, whether N
                           matches the list length. Reported only for detectable cases.
    """
    body = _ANY_APPENDIX_RE.split(body, maxsplit=1)[0]   # don't count the provenance table as prose
    clauses = _all_clauses(body)
    n_clauses = len(clauses)
    n_uncited = sum(1 for c in clauses if not parse_citations(c))
    cited = [c for c in parse_claims(body) if any(not ct.external for ct in c.citations)]

    # dangling citations: markers that don't resolve to a present, non-empty evidence file
    markers = dangling = 0
    for c in cited:
        for ct in c.citations:
            if ct.external or not ct.key:
                continue
            markers += 1
            files = mapping.get(ct.key, [])
            if not files or not any(evidence_text.get(fn, "").strip() for fn in files):
                dangling += 1

    # coverage: evidence files cited by >=1 claim vs the union of known files
    used: set[str] = set()
    for c in cited:
        for ct in c.citations:
            if ct.external:
                continue
            used.update(mapping.get(ct.key, []))
    union = set(evidence_text) | {f for fs in mapping.values() for f in fs}
    present = {f for f in union if evidence_text.get(f, "").strip()}
    covered = used & present

    # count/set consistency where detectable (on the raw clauses, citation markers stripped)
    count_checks = _count_consistency(clauses)

    return {
        "n_clauses": n_clauses,
        "n_cited_claims": len(cited),
        "n_uncited_claims": n_uncited,
        "uncited_claim_rate": _rate(n_uncited, n_clauses),
        "n_citation_markers": markers,
        "n_dangling": dangling,
        "dangling_rate": _rate(dangling, markers),
        "coverage": _rate(len(covered), len(present)),
        "evidence_used": len(covered),
        "evidence_total": len(present),
        "evidence_uncited": sorted(present - used),
        "count_checks": count_checks,
        "count_consistent": all(ck["ok"] for ck in count_checks),
    }


# Words that mark a tail item as prose, not a list entry, so a count followed by an ordinary
# sentence ("5 methods were tested, and we report ...") is not mistaken for an enumerated list.
_LIST_NOISE = {"we", "i", "were", "was", "is", "are", "be", "been", "being", "have", "has", "had",
               "the", "a", "an", "that", "this", "these", "those", "which", "it", "they", "found",
               "showed", "report", "reported", "using", "used", "with", "for", "to", "of", "in",
               "on", "as", "by", "such", "including"}


def _looks_like_list(items: list[str]) -> bool:
    """True only if every item reads like a list entry — short and free of prose/verb words."""
    for it in items:
        words = it.split()
        if not (1 <= len(words) <= 4) or {w.lower() for w in words} & _LIST_NOISE:
            return False
    return True


def _count_consistency(clauses: list[str]) -> list[dict]:
    """For each clause asserting 'N <things>' followed by a co-located enumerated list, compare N to
    the list size. The tail after the count phrase is split on commas/semicolons and a trailing
    'and'/'or'; this is syntactic, not semantic, and skips clauses whose tail does not read like a
    short list (see `_looks_like_list`)."""
    out: list[dict] = []
    for raw in clauses:
        text = _CITE_RE.sub("", raw)          # drop [Author Year] markers so they aren't counted
        m = _COUNT_RE.search(text)
        if not m:
            continue
        items = [p.strip() for p in re.split(r"[,;]|\band\b|\bor\b", text[m.end():]) if p.strip()]
        if len(items) < 2 or not _looks_like_list(items):
            continue
        claimed = int(m.group(1))
        out.append({"claim": text.strip(), "claimed": claimed, "listed": len(items),
                    "ok": claimed == len(items)})
    return out


def _rate(num: int, denom: int) -> float | None:
    return None if not denom else round(num / denom, 3)


# ---- aggregation ---------------------------------------------------------------

def faithfulness_metrics(scores: list[ClaimScore], threshold: float) -> dict:
    """Aggregate per-check pass-rates and the overall faithfulness rate over scored (resolvable)
    claims. `flagged` is every claim that failed the conjunction, with its reason."""
    n = len(scores)
    num = sum(s.number_grounded for s in scores)
    ent = sum(s.entity_grounded for s in scores)
    lex = sum(s.lexical_overlap >= threshold for s in scores)
    passed = sum(s.passed for s in scores)
    return {
        "n_scored_claims": n,
        "faithfulness_rate": _rate(passed, n),
        "number_grounded_rate": _rate(num, n),
        "entity_grounded_rate": _rate(ent, n),
        "lexical_overlap_rate": _rate(lex, n),
        "threshold": threshold,
        "flagged": [{"claim": s.claim, "citations": s.citations,
                     "evidence_files": s.evidence_files, "reason": s.reason}
                    for s in scores if not s.passed],
    }


# ---- end-to-end ----------------------------------------------------------------

def score_synthesis(synthesis_md: str, evidence_text: dict[str, str],
                    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
                    mapping: dict[str, list[str]] | None = None
                    ) -> tuple[list[ClaimScore], dict]:
    """End-to-end on in-memory strings (the unit-test seam): parse, resolve, three checks,
    aggregate, plus structural integrity. `mapping` overrides Appendix-A parsing when supplied."""
    _, full_body = split_frontmatter(synthesis_md)
    mapping = mapping if mapping is not None else parse_appendix_a(synthesis_md)
    body = _ANY_APPENDIX_RE.split(full_body, maxsplit=1)[0]   # score the prose, not the provenance table
    claims = parse_claims(body)
    scores = [s for s in (score_claim(c, mapping, evidence_text, threshold) for c in claims)
              if s is not None]
    metrics = {
        "faithfulness": faithfulness_metrics(scores, threshold),
        "structural": structural_metrics(body, mapping, evidence_text),
    }
    return scores, metrics


def run(project: str, project_dir: Path, threshold: float = DEFAULT_OVERLAP_THRESHOLD
        ) -> tuple[list[ClaimScore], dict]:
    lit = project_dir / f"projects/{project}/literature"
    synth = lit / "0_synthesis_literature.md"
    if not synth.exists():
        log(f"no synthesis at {synth} — nothing to score")
        return [], {}
    evidence_text = {ev.name: ev.read_text() for ev in sorted(lit.glob("*_evidence.md"))}
    return score_synthesis(synth.read_text(), evidence_text, threshold)


# ---- IO ------------------------------------------------------------------------

def write_csv(path: Path, scores: list[ClaimScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["claim", "citations", "evidence_files", "number_grounded", "entity_grounded",
            "lexical_overlap", "passed", "reason"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in scores:
            w.writerow({"claim": s.claim, "citations": s.citations,
                        "evidence_files": s.evidence_files,
                        "number_grounded": s.number_grounded, "entity_grounded": s.entity_grounded,
                        "lexical_overlap": s.lexical_overlap, "passed": s.passed, "reason": s.reason})
    log(f"wrote {path.name} ({len(scores)} scored claims)")


def summarize(project: str, m: dict) -> str:
    if not m:
        return "# Synthesis faithfulness (deterministic)\n\nNo synthesis found.\n"
    f, st = m["faithfulness"], m["structural"]
    pct = lambda v: "—" if v is None else f"{v:.1%}"
    lines = [
        f"# Synthesis faithfulness — {project} (deterministic, no-API)", "",
        f"scored cited claims: **{f['n_scored_claims']}** · "
        f"lexical-overlap threshold: **{f['threshold']}**", "",
        "## Faithfulness (a claim PASSES iff number_grounded AND entity_grounded AND "
        "lexical_overlap ≥ threshold)", "",
        "| Check | Pass rate |",
        "|---|---|",
        f"| **Faithfulness (all three)** | **{pct(f['faithfulness_rate'])}** |",
        f"| number_grounded (no fabricated stats) | {pct(f['number_grounded_rate'])} |",
        f"| entity_grounded (genes/markers/cell-types present) | {pct(f['entity_grounded_rate'])} |",
        f"| lexical_overlap ≥ {f['threshold']} | {pct(f['lexical_overlap_rate'])} |", "",
        "## Structural integrity (no semantics)", "",
        "| Metric | Value |",
        "|---|---|",
        f"| Uncited-claim rate | {pct(st['uncited_claim_rate'])} "
        f"({st['n_uncited_claims']}/{st['n_clauses']}) |",
        f"| Dangling-citation rate | {pct(st['dangling_rate'])} "
        f"({st['n_dangling']}/{st['n_citation_markers']}) |",
        f"| Coverage (evidence files cited) | {pct(st['coverage'])} "
        f"({st['evidence_used']}/{st['evidence_total']}) |",
        f"| Count/set consistency | "
        f"{'consistent' if st['count_consistent'] else 'INCONSISTENT'} "
        f"({len(st['count_checks'])} checked) |",
    ]
    if st["evidence_uncited"]:
        lines += ["", "**Evidence files cited by no claim:** "
                  + ", ".join(f"`{x}`" for x in st["evidence_uncited"])]
    if f["flagged"]:
        shown, total = f["flagged"][:_MAX_FLAGGED], len(f["flagged"])
        header = "## Flagged claims (failed at least one check)"
        if total > _MAX_FLAGGED:
            header += f" — first {_MAX_FLAGGED} of {total}; full list in the CSV"
        lines += ["", header, ""]
        for fl in shown:
            claim = fl["claim"] if len(fl["claim"]) <= 160 else fl["claim"][:157] + "..."
            lines.append(f"- _{fl['reason']}_ — {claim}  "
                         f"[{fl['citations']} → {fl['evidence_files'] or 'unresolved'}]")
    lines += [
        "",
        "> **Scope — read before trusting these numbers.**",
        "> - This is a **lexical / rule-based** scorer: deterministic, no model, no network, no "
        "randomness. It reruns identically on the same inputs.",
        "> - It **CANNOT recognize a valid paraphrase** — a correctly-attributed claim worded "
        "differently from its evidence will show as low `lexical_overlap` and be flagged "
        "*unsupported* (a FALSE negative).",
        "> - It **CANNOT catch a negation or direction flip** when the words still overlap — "
        "\"X increases Y\" vs evidence \"X decreases Y\" can read as *supported* (a FALSE positive). "
        "`number_grounded` only checks a quantity is PRESENT, not that the claim uses it correctly.",
        "> - `number_grounded` is the high-value check: it catches **fabricated statistics**, the "
        "dangerous failure mode generalist verifiers miss.",
        "> - This is the **reproducible, no-API baseline** that COMPLEMENTS the model-based citation "
        "check (`citation_check.py`); it does not replace it. Use both: the deterministic floor here, "
        "the entailment verifier for paraphrase/direction.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deterministic (no-API) faithfulness scorer for the SYNTHESIZE stage")
    ap.add_argument("--project", required=True)
    ap.add_argument("--project-dir", required=True,
                    help="dir containing projects/<project>/literature/")
    ap.add_argument("--threshold", type=float, default=DEFAULT_OVERLAP_THRESHOLD,
                    help="lexical_overlap (token-F1) pass threshold")
    args = ap.parse_args()

    scores, m = run(args.project, Path(args.project_dir).resolve(), args.threshold)
    if not m:
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "synthesis_faithfulness.csv", scores)
    summary = summarize(args.project, m)
    (RESULTS / "synthesis_faithfulness_SUMMARY.md").write_text(summary)
    log("wrote synthesis_faithfulness_SUMMARY.md")
    print("\n" + summary + "\n")


if __name__ == "__main__":
    main()
