"""Report writers for JSONL, CSV, Markdown, diagnostics, and skipped-findings artifacts.

Every run writes a report, even with zero candidates (an honest empty report). The Markdown leads with
the interpretation note: these are *candidates*, novelty is *local to the corpus*, the hypothesis is
*untested*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import io, visualize
from .config import HypothesisConfig
from .schema import HypothesisCandidate, HypothesisRunResult

CSV_COLUMNS = [
    "rank", "hypothesis_id", "claim", "rank_score", "motif", "predicted_direction",
    "species", "tissue", "disease", "cell_type", "perturbation", "readouts", "support_papers",
    "evidence_modes", "grounding_score", "context_match_score", "measurability_score",
    "evidence_design_score", "local_nonredundancy_score", "specificity_score", "risk_penalty",
    "warnings", "novelty_scope",
]

INTERPRETATION_NOTE = (
    "**These are hypothesis candidates, not validated discoveries.** Novelty is assessed only against "
    "this input corpus (`local_corpus_only`) — not against PubMed or any external database. Each "
    "hypothesis is an untested prediction with an explicit evidence path; treat it as a prioritised "
    "experiment to run, not a finding."
)


def candidate_to_record(c: HypothesisCandidate) -> dict[str, Any]:
    return {
        "hypothesis_id": c.hypothesis_id,
        "claim": c.claim,
        "motif": c.motif,
        "predicted_direction": c.predicted_direction,
        "context": io.to_jsonable(c.context),
        "anchor": io.to_jsonable(c.anchor),
        "mediator": io.to_jsonable(c.mediator) if c.mediator else None,
        "readouts": [io.to_jsonable(r) for r in c.readouts],
        "support": {
            "frame_ids": c.support_frame_ids,
            "edge_ids": c.support_edge_ids,
            "paper_ids": c.support_paper_ids,
        },
        "evidence_modes": c.evidence_mode_summary,
        "scores": c.scores,
        "warnings": c.warnings,
        "test_design": c.test_design,
        "novelty_scope": c.novelty_scope,
    }


def candidate_to_row(c: HypothesisCandidate, rank: int) -> dict[str, Any]:
    ctx = c.context
    s = c.scores
    return {
        "rank": rank, "hypothesis_id": c.hypothesis_id, "claim": c.claim,
        "rank_score": s.get("rank_score", 0.0), "motif": c.motif,
        "predicted_direction": c.predicted_direction,
        "species": list(ctx.species), "tissue": list(ctx.tissue), "disease": list(ctx.disease),
        "cell_type": list(ctx.cell_type), "perturbation": list(ctx.perturbation),
        "readouts": [r.canonical_name for r in c.readouts],
        "support_papers": c.support_paper_ids, "evidence_modes": c.evidence_mode_summary,
        "grounding_score": s.get("grounding_score", 0.0),
        "context_match_score": s.get("context_match_score", 0.0),
        "measurability_score": s.get("measurability_score", 0.0),
        "evidence_design_score": s.get("evidence_design_score", 0.0),
        "local_nonredundancy_score": s.get("local_nonredundancy_score", 0.0),
        "specificity_score": s.get("specificity_score", 0.0),
        "risk_penalty": s.get("risk_penalty", 0.0),
        "warnings": c.warnings, "novelty_scope": c.novelty_scope,
    }


def write_report(result: HypothesisRunResult, out_dir: str | Path,
                 config: HypothesisConfig, synthesis: dict | None = None) -> dict[str, str]:
    out = io.ensure_dir(out_dir)
    paths: dict[str, str] = {}
    cands = result.candidates

    if config.write_jsonl:
        p = io.write_jsonl(out / "hypotheses.jsonl", [candidate_to_record(c) for c in cands])
        paths["jsonl"] = str(p)
    if config.write_csv:
        rows = [candidate_to_row(c, i) for i, c in enumerate(cands, start=1)]
        p = io.write_csv(out / "hypotheses.csv", rows, CSV_COLUMNS)
        paths["csv"] = str(p)
    if config.write_markdown:
        p = (out / "hypotheses.md")
        p.write_text(_render_markdown(result, config))
        paths["markdown"] = str(p)

    io.write_json(out / "diagnostics.json", _diagnostics_summary(result))
    paths["diagnostics"] = str(out / "diagnostics.json")

    io.write_csv(out / "skipped_findings.csv",
                 result.diagnostics.get("skipped_findings", []),
                 ["paper_id", "finding_text", "reason", "source_quote"])
    paths["skipped_findings"] = str(out / "skipped_findings.csv")
    return paths


def _diagnostics_summary(result: HypothesisRunResult) -> dict[str, Any]:
    d = result.diagnostics
    keys = ["records_read", "records_used", "findings_seen", "frames_extracted", "frames_grounded",
            "frames_skipped", "candidates_generated_raw", "candidates_retained",
            "candidates_filtered", "graph_adequacy", "novelty_scope", "warnings"]
    summary = {k: d.get(k) for k in keys}
    summary["filtered_detail"] = d.get("filtered_detail", [])
    summary["grounding"] = d.get("grounding", {})
    return summary


def _render_markdown(result: HypothesisRunResult, config: HypothesisConfig) -> str:
    d = result.diagnostics
    cands = result.candidates
    L: list[str] = []
    L.append("# LitStream Hypothesis Candidate Report\n")

    L.append("## Run summary\n")
    L.append(f"- Evidence records read: {d.get('records_read', 0)}")
    L.append(f"- Records used (relevance ≥ {config.min_paper_relevance}): {d.get('records_used', 0)}")
    L.append(f"- Findings parsed: {d.get('findings_seen', 0)}")
    L.append(f"- Frames extracted: {d.get('frames_extracted', 0)}")
    L.append(f"- Frames grounded: {d.get('frames_grounded', 0)} (grounder: {d.get('grounding', {}).get('grounder', '?')})")
    L.append(f"- Candidates generated (raw): {d.get('candidates_generated_raw', 0)}")
    L.append(f"- Candidates retained: {d.get('candidates_retained', 0)}")
    filtered_total = sum((d.get('candidates_filtered') or {}).values())
    L.append(f"- Candidates filtered: {filtered_total}")
    L.append(f"- Novelty scope: {d.get('novelty_scope', 'local_corpus_only')}\n")

    L.append("## Important interpretation note\n")
    L.append(INTERPRETATION_NOTE + "\n")

    adq = d.get("graph_adequacy", {})
    L.append("## Hypothesis portfolio\n")
    L.append(f"Evidence graph: {adq.get('n_nodes', 0)} entities, {adq.get('n_edges', 0)} edges, "
             f"{adq.get('n_pivot_nodes', 0)} bridgeable pivot(s).\n")
    if not cands:
        L.append("_No hypothesis candidates were generated from this corpus._ This is an honest "
                 "non-result — the evidence graph did not contain a compatible, novel, testable "
                 "2-hop path under the current constraints. See diagnostics below.\n")
    else:
        L.append("| Rank | ID | Motif | Score | Grounding | Measurability | Claim |")
        L.append("|---:|---|---|---:|---:|---:|---|")
        for i, c in enumerate(cands, 1):
            s = c.scores
            L.append(f"| {i} | {c.hypothesis_id} | {_short_motif(c.motif)} | "
                     f"{s.get('rank_score', 0):.3f} | {s.get('grounding_score', 0):.2f} | "
                     f"{s.get('measurability_score', 0):.2f} | {_truncate(c.claim, 80)} |")
        L.append("")

    if cands:
        L.append("## Ranked candidates\n")
        for i, c in enumerate(cands, 1):
            L.extend(_render_candidate(c, i, result))

    L.append("## Diagnostics\n")
    L.append("### Frames skipped (by reason)")
    fs = d.get("frames_skipped", {})
    if fs:
        for r, n in sorted(fs.items(), key=lambda x: -x[1]):
            L.append(f"- {r}: {n}")
    else:
        L.append("- (none)")
    L.append("\n### Candidates filtered (by reason)")
    cf = d.get("candidates_filtered", {})
    if cf:
        for r, n in sorted(cf.items(), key=lambda x: -x[1]):
            L.append(f"- {r}: {n}")
    else:
        L.append("- (none)")
    L.append("\n### Scope")
    for w in d.get("warnings", []):
        L.append(f"- {w}")
    L.append("")
    return "\n".join(L)


def _render_candidate(c: HypothesisCandidate, rank: int, result) -> list[str]:
    s = c.scores
    L = [f"### {c.hypothesis_id} — {_short_motif(c.motif)}\n"]
    L.append(f"**Claim.** {c.claim}\n")
    L.append(f"**Predicted direction:** `{c.predicted_direction}` · "
             f"**rank score:** {s.get('rank_score', 0):.3f} · "
             f"**support papers:** {', '.join(c.support_paper_ids)}\n")

    L.append("**Evidence path.**")
    edges_by_id = result.graph.graph.get("edges_by_id", {})
    for eid in c.support_edge_ids:
        e = edges_by_id.get(eid)
        if e:
            L.append(f"- `{e.source_entity.canonical_name}` --{e.relation} "
                     f"({e.evidence_mode})--> `{e.target_entity.canonical_name}` "
                     f"[{e.paper_id}] — \"{_truncate(e.source_quote, 100)}\"")
    L.append("")

    td = c.test_design
    L.append("**Proposed test.**")
    L.append(f"- Assay: {td.get('assay', '')}; sample: {td.get('sample', '')}; "
             f"cell population: {td.get('cell_population', '')}")
    L.append(f"- Conditions: {', '.join(td.get('conditions', []))}")
    ro = "; ".join(f"{r.get('entity')} ({r.get('modality')}"
                   + (f", {r.get('role')}" if r.get('role') else "") + ")"
                   for r in td.get("readouts", []))
    L.append(f"- Readouts: {ro}")
    L.append("")

    if c.warnings:
        L.append("**Warnings.**")
        for w in c.warnings:
            L.append(f"- {w}")
        L.append("")

    L.append("**Scores.** " + ", ".join(
        f"{k}={s.get(k, 0):.2f}" for k in
        ("grounding_score", "context_match_score", "measurability_score", "evidence_design_score",
         "local_nonredundancy_score", "specificity_score", "risk_penalty")) + "\n")

    L.append("**Evidence trace.**\n")
    L.append("```mermaid")
    L.append(visualize.candidate_trace_mermaid(c, result))
    L.append("```\n")
    return L


def _short_motif(motif: str) -> str:
    return {
        "perturbation_to_marker_state_completion": "perturbation→marker/state",
        "disease_signature_reversal": "disease-signature reversal",
        "signature_consolidation": "signature consolidation",
        "cite_seq_marker_bridge": "CITE-seq marker bridge",
    }.get(motif, motif)


def _truncate(text: str, n: int) -> str:
    text = (text or "").replace("\n", " ").replace("|", "/")
    return text if len(text) <= n else text[: n - 1] + "…"
