"""A2 citation/attribution checker for the SYNTHESIZE stage.

The synthesizer writes prose with inline `[Author Year]` citations and an Appendix-A table mapping
each citation to its `*_evidence.md` file(s). Per cited claim, this tool asks whether the cited
source actually supports it: the attribution question, not whether the claim is true in the world.

  parse:   split the synthesis body into claims (each sentence/clause/table-cell carrying a citation
           marker), pairing every claim with its `[Author Year]` citation(s).
  resolve: `[Author Year]` -> full `*_evidence.md` via the Appendix-A table (or a provided mapping).
           `[External Review]` is special-cased: never resolved through Appendix A, never scored.
  verify:  for each claim/evidence pair, an entailment verifier labels the support
           {Supportive, Partial, Contradictory, Irrelevant}.

The verifier is reused from `litstream_evidence/ground_retrieval.py` (`make_verifier('minicheck')`,
`.verify(claim, passage) -> (bool, float)`) and is injected, so tests pass a fake and MiniCheck is
never loaded offline. The four-way label comes from a verifier that exposes `.label(...)`; a plain
boolean verifier collapses to Supportive/Irrelevant.

This measures whether the cited source supports the claim, not world-truth, and the checker itself
is unvalidated until a human attribution key (see `attribution_key.example.jsonl`) measures its own
precision/recall. Both caveats are written into the SUMMARY.

    python -m litstream.eval.citation_check --project citeseq_methods \
        --project-dir kb-skills-bioinformatics --verifier minicheck
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from litstream_evidence.ground_retrieval import Verifier, make_verifier

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Four-way support labels: Supportive = source entails the claim; Partial = supports part of it;
# Contradictory = source states the opposite; Irrelevant = neither supports nor contradicts.
SUPPORTIVE, PARTIAL, CONTRADICTORY, IRRELEVANT = "Supportive", "Partial", "Contradictory", "Irrelevant"
LABELS = [SUPPORTIVE, PARTIAL, CONTRADICTORY, IRRELEVANT]

EXTERNAL_REVIEW = "[External Review]"   # supporting context, no evidence file — never resolved


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


# ---- claim/citation parsing ----------------------------------------------------

# A citation marker: [Author Year] (e.g. [Zheng 2025]), an abbreviated [Author] (e.g. [Zheng]), or
# the literal [External Review]. Captures the inner text; normalization happens separately.
_CITE_RE = re.compile(r"\[([A-Z][^\[\]]*?)\]")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# Prose sentence boundary; tables are handled separately by `iter_clauses`.
_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+")
_TABLE_SEP_RE = re.compile(r"^[\s|:\-]+$")     # a |---|:--| table separator row


@dataclass(frozen=True)
class Citation:
    """One inline citation marker and its resolved key. `external` marks `[External Review]`."""
    raw: str            # as written, e.g. "[Zheng 2025]" or "[Zheng]"
    key: str            # normalized author surname, e.g. "zheng" (resolution key)
    external: bool = False


@dataclass
class Claim:
    """A sentence/clause carrying one or more citations."""
    text: str
    citations: list[Citation] = field(default_factory=list)


def _norm_key(author_year: str) -> str:
    """Resolution key from a citation's inner text: the first author surname, lowercased. Drops the
    year and any 'et al.' so `[Zheng 2025]` and `[Zheng]` resolve identically."""
    txt = _YEAR_RE.sub("", author_year)
    txt = re.sub(r"\bet al\.?", "", txt, flags=re.IGNORECASE)
    first = re.split(r"[\s,;/&]+", txt.strip())[0] if txt.strip() else ""
    return first.lower().strip(" .,")


def _is_citation(inner: str) -> bool:
    """True if a bracketed span looks like a citation: an author name (optionally a year) or External
    Review. Rejects markdown noise like [link] or numeric footnotes [12]."""
    if inner.strip().lower() == "external review":
        return True
    head = inner.strip().split()[0] if inner.strip() else ""
    return bool(re.match(r"^[A-Z][A-Za-z\-']+$", head))


def _strip_code(text: str) -> str:
    """Drop fenced code blocks except the ASCII gating tree (which carries real citations); Python
    cells contain bracketed list literals that would masquerade as citations."""
    def keep(m: re.Match) -> str:
        block = m.group(0)
        return block if "►" in block or "──" in block else " "
    return _FENCE_RE.sub(keep, text)


def _is_table_row(line: str) -> bool:
    """A markdown table row — it starts/ends with a pipe or has at least two internal pipes."""
    return line.startswith("|") or line.count("|") >= 2


def iter_clauses(body: str) -> list[str]:
    """Split the synthesis body into clauses, table-aware: a markdown table data row becomes one
    clause (its non-empty cells joined), separator rows are dropped, and prose lines split on
    sentence boundaries, so a `|` never shreds a table into per-cell fragments."""
    out: list[str] = []
    for line in _strip_code(body).splitlines():
        s = line.strip()
        if not s:
            continue
        if _is_table_row(s):
            if _TABLE_SEP_RE.match(s):
                continue
            cells = [c.strip() for c in s.strip("|").split("|") if c.strip()]
            if cells:
                out.append(" · ".join(cells))
        else:
            out.extend(p.strip() for p in _SENT_SPLIT.split(s) if p.strip())
    return out


def parse_citations(text: str) -> list[Citation]:
    """Every citation marker in a span, in order, de-duplicated by raw form."""
    out: list[Citation] = []
    seen: set[str] = set()
    for m in _CITE_RE.finditer(text):
        inner = m.group(1)
        if not _is_citation(inner) or m.group(0) in seen:
            continue
        seen.add(m.group(0))
        if inner.strip().lower() == "external review":
            out.append(Citation(raw=EXTERNAL_REVIEW, key="", external=True))
        else:
            out.append(Citation(raw=m.group(0), key=_norm_key(inner)))
    return out


def parse_claims(body: str) -> list[Claim]:
    """Split the synthesis body into cited claims: each clause with >=1 citation marker, paired with
    the citation(s) inside it. Clauses with no citation are dropped."""
    claims: list[Claim] = []
    for clause in iter_clauses(body):
        clause = clause.strip()
        if not clause or "[" not in clause:
            continue
        cites = parse_citations(clause)
        if cites:
            # claim text minus the bracketed markers, so the verifier reads prose not provenance
            claim_text = _CITE_RE.sub("", clause).strip(" \t·-—|")
            if claim_text:
                claims.append(Claim(text=claim_text, citations=cites))
    return claims


# ---- Appendix-A resolution -----------------------------------------------------

_EV_FILE_RE = re.compile(r"`([^`]*_evidence\.md)`")

# Appendix-heading patterns, kept distinct on purpose: _APPENDIX_A_RE locates the Appendix-A
# provenance table for citation resolution, while _ANY_APPENDIX_RE cuts the body at the first
# appendix of any kind so no appendix is parsed as a claim. _NEXT_SECTION_RE bounds the table.
_APPENDIX_A_RE = re.compile(r"(?im)^##+\s*Appendix\s+A\b.*$")
_ANY_APPENDIX_RE = re.compile(r"(?im)^##+\s*Appendix\s")
_NEXT_SECTION_RE = re.compile(r"(?im)^##+\s+\S")


def parse_appendix_a(text: str) -> dict[str, list[str]]:
    """Build {citation_key -> [evidence_file, ...]} from the Appendix-A table. Each row is a markdown
    table line whose cells contain a `[Author Year]` citation and one or more `*_evidence.md` names
    in backticks. Tolerates multiple files per paper and abbreviated keys."""
    mapping: dict[str, list[str]] = {}
    section = _APPENDIX_A_RE.split(text, maxsplit=1)
    region = section[1] if len(section) > 1 else text
    # stop at the next top-level section so Appendix B etc. don't leak in
    region = _NEXT_SECTION_RE.split(region, maxsplit=1)[0]
    for line in region.splitlines():
        files = _EV_FILE_RE.findall(line)
        if not files:
            continue
        cites = parse_citations(line)
        for c in cites:
            if c.external or not c.key:
                continue
            mapping.setdefault(c.key, [])
            for fn in files:
                if fn not in mapping[c.key]:
                    mapping[c.key].append(fn)
    return mapping


def split_frontmatter(md_text: str) -> tuple[str, str]:
    """Return (frontmatter, body). Leading `--- ... ---` is stripped so it isn't parsed for claims."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", md_text or "", re.DOTALL)
    return (m.group(1), m.group(2)) if m else ("", md_text or "")


# ---- verification --------------------------------------------------------------

def _four_way(verifier: Verifier, claim: str, passage: str) -> str:
    """Map a verifier verdict to one of the four labels. A verifier may expose a richer
    `.label(claim, passage) -> str` (e.g. a NLI head with a contradiction class); otherwise the
    boolean `.verify(...)` collapses to Supportive/Irrelevant. Partial and Contradictory require a
    `.label`-capable verifier."""
    labeler = getattr(verifier, "label", None)
    if callable(labeler):
        out = labeler(claim, passage)
        if out in LABELS:
            return out
    supported, _ = verifier.verify(claim, passage)
    return SUPPORTIVE if supported else IRRELEVANT


@dataclass
class Pairing:
    """One claim<->cited-source verdict (a row in the output CSV)."""
    claim: str
    citation: str          # raw marker, e.g. "[Zheng 2025]"
    evidence_file: str     # resolved *_evidence.md, or "" when unresolved
    label: str
    note: str = ""


def check_claim(claim: Claim, mapping: dict[str, list[str]], evidence_text: dict[str, str],
                verifier: Verifier) -> list[Pairing]:
    """Resolve and verify one claim, one Pairing per (citation, evidence-file). `[External Review]`
    is recorded but never resolved/scored; an unresolved citation is flagged, not labeled."""
    out: list[Pairing] = []
    for cite in claim.citations:
        if cite.external:
            out.append(Pairing(claim.text, EXTERNAL_REVIEW, "", IRRELEVANT,
                               note="external review — skipped (no evidence file)"))
            continue
        files = mapping.get(cite.key, [])
        if not files:
            out.append(Pairing(claim.text, cite.raw, "", IRRELEVANT,
                               note="unresolved — no Appendix-A entry"))
            continue
        for fn in files:
            passage = evidence_text.get(fn, "")
            if not passage:
                out.append(Pairing(claim.text, cite.raw, fn, IRRELEVANT,
                                   note="evidence file missing/empty"))
                continue
            out.append(Pairing(claim.text, cite.raw, fn,
                               _four_way(verifier, claim.text, passage)))
    return out


# ---- metrics -------------------------------------------------------------------

def _scored(pairings: list[Pairing]) -> list[Pairing]:
    """Pairings that actually exercised the verifier — resolved, non-external, with evidence text."""
    return [p for p in pairings
            if p.citation != EXTERNAL_REVIEW and p.evidence_file and "missing" not in p.note]


def metrics(claims: list[Claim], pairings: list[Pairing], all_evidence: list[str]) -> dict:
    """Citation precision, claim recall, and coverage.

      citation precision = of scored claim/cite pairs, the fraction labeled Supportive.
      claim recall       = of cited claims, the fraction fully supported, i.e. every resolved
                           citation is Supportive (a single contradicted/irrelevant cite fails it).
      coverage           = of the union of evidence files, the fraction cited by >=1 claim.
    """
    scored = _scored(pairings)
    n_pairs = len(scored)
    supportive = sum(p.label == SUPPORTIVE for p in scored)
    precision = supportive / n_pairs if n_pairs else None

    # group scored pairs by claim text; a claim is fully supported if it had >=1 scored pair and all
    # of them are Supportive.
    by_claim: dict[str, list[str]] = defaultdict(list)
    for p in scored:
        by_claim[p.claim].append(p.label)
    cited_claims = [c for c in claims if any(not ct.external for ct in c.citations)]
    fully = sum(1 for c in cited_claims
                if by_claim.get(c.text) and all(l == SUPPORTIVE for l in by_claim[c.text]))
    recall = fully / len(cited_claims) if cited_claims else None

    used = {p.evidence_file for p in scored}
    union = set(all_evidence)
    coverage = len(used & union) / len(union) if union else None

    counts = {lab: sum(p.label == lab for p in scored) for lab in LABELS}
    return {
        "n_claims": len(claims),
        "n_cited_claims": len(cited_claims),
        "n_scored_pairs": n_pairs,
        "citation_precision": None if precision is None else round(precision, 3),
        "claim_recall": None if recall is None else round(recall, 3),
        "coverage": None if coverage is None else round(coverage, 3),
        "label_counts": counts,
        "evidence_used": len(used & union),
        "evidence_total": len(union),
        "evidence_uncited": sorted(union - used),
    }


# ---- IO + runner ---------------------------------------------------------------

def write_csv(path: Path, pairings: list[Pairing]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["claim", "citation", "evidence_file", "label", "note"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in pairings:
            w.writerow({"claim": p.claim, "citation": p.citation,
                        "evidence_file": p.evidence_file, "label": p.label, "note": p.note})
    log(f"wrote {path.name} ({len(pairings)} pairs)")


def check_synthesis(synthesis_md: str, evidence_text: dict[str, str], verifier: Verifier,
                    mapping: dict[str, list[str]] | None = None) -> tuple[list[Pairing], dict]:
    """End-to-end on in-memory strings (the unit-test seam): parse -> resolve -> verify -> score.
    `mapping` overrides Appendix-A parsing when a caller supplies the citation->file dict directly."""
    _, body = split_frontmatter(synthesis_md)
    mapping = mapping if mapping is not None else parse_appendix_a(synthesis_md)
    # parse claims from the prose only; the Appendix-A provenance table is resolution data, not claims
    body = _ANY_APPENDIX_RE.split(body, maxsplit=1)[0]
    claims = parse_claims(body)
    pairings: list[Pairing] = []
    for claim in claims:
        pairings.extend(check_claim(claim, mapping, evidence_text, verifier))
    union = sorted({f for files in mapping.values() for f in files} | set(evidence_text))
    return pairings, metrics(claims, pairings, union)


def run(project: str, project_dir: Path, verifier: Verifier) -> tuple[list[Pairing], dict]:
    lit = project_dir / f"projects/{project}/literature"
    synth = lit / "0_synthesis_literature.md"
    if not synth.exists():
        log(f"no synthesis at {synth} — nothing to check"); return [], {}
    evidence_text = {ev.name: ev.read_text()
                     for ev in sorted(lit.glob("*_evidence.md"))}
    return check_synthesis(synth.read_text(), evidence_text, verifier)


def summarize(project: str, m: dict, verifier_name: str) -> str:
    if not m:
        return "# Citation/attribution check\n\nNo synthesis found.\n"
    pct = lambda v: "—" if v is None else f"{v:.1%}"
    lc = m["label_counts"]
    lines = [
        f"# Citation/attribution check — {project}", "",
        f"claims parsed: **{m['n_claims']}** · cited claims: **{m['n_cited_claims']}** · "
        f"scored claim↔source pairs: **{m['n_scored_pairs']}**", "",
        "| Metric | Value |",
        "|---|---|",
        f"| Citation precision (cited→supported) | {pct(m['citation_precision'])} |",
        f"| Claim recall (claims fully supported) | {pct(m['claim_recall'])} |",
        f"| Coverage (evidence files cited) | {pct(m['coverage'])} "
        f"({m['evidence_used']}/{m['evidence_total']}) |", "",
        "| Label | Pairs |",
        "|---|---|",
    ] + [f"| {lab} | {lc[lab]} |" for lab in LABELS]
    if m["evidence_uncited"]:
        lines += ["", "**Evidence files cited by no claim:** "
                  + ", ".join(f"`{f}`" for f in m["evidence_uncited"])]
    lines += [
        "",
        "> **Scope — read before trusting these numbers.**",
        "> - This measures **\"the cited source supports the claim,\"** NOT whether the claim is "
        "true in the world. A well-attributed claim can still be wrong if the source is wrong.",
        "> - **The checker itself is unvalidated.** Its precision/recall are only meaningful once a "
        "**human attribution key** (~50 hand-labeled claim-citation pairs — see "
        "`litstream/eval/attribution_key.example.jsonl`) measures the checker's OWN precision/"
        "recall/κ against human judgment. Treat these as provisional until then.",
        "> - `[External Review]` citations have no evidence file and are **skipped**, never resolved "
        "through Appendix A.",
        "> - The entailment verifier is weak on numeric/direction/species claims (documented); route "
        "those to dedicated checks rather than trusting the label.",
        "",
        f"*verifier: {verifier_name}*",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="A2 — check synthesis [Author Year] citations against their evidence files")
    ap.add_argument("--project", required=True)
    ap.add_argument("--project-dir", required=True,
                    help="dir containing projects/<project>/literature/")
    ap.add_argument("--verifier", default="minicheck", choices=["overlap", "minicheck"],
                    help="entailment verifier (reused from ground_retrieval)")
    ap.add_argument("--minicheck-model", default="flan-t5-large")
    args = ap.parse_args()

    verifier = make_verifier(args.verifier, args.minicheck_model)
    pairings, m = run(args.project, Path(args.project_dir).resolve(), verifier)
    if not m:
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "citation_check.csv", pairings)
    summary = summarize(args.project, m, args.verifier)
    (RESULTS / "citation_check_SUMMARY.md").write_text(summary)
    log("wrote citation_check_SUMMARY.md")
    print("\n" + summary + "\n")


if __name__ == "__main__":
    main()
