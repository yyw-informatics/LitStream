"""The orchestrator: grounded evidence records -> ranked hypothesis candidates + diagnostics.

``ContextBoundHypothesisGenerator.run`` is the in-memory pipeline; ``run_to_dir`` wraps it with I/O
and artifact writing and is what the CLI / app wiring call. Both succeed on an empty corpus (writing
an honest empty report), and neither mutates the input evidence files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import HypothesisConfig
from .filters import filter_candidates
from .frame_extractor import extract_frames
from .graph_builder import build_evidence_graph
from .grounding import verify_frames
from .normalize import Normalizer
from .ranker import rank_candidates
from .schema import HypothesisRunResult
from .templates import generate_candidates


def filter_records_by_relevance(records: list[dict], config: HypothesisConfig) -> list[dict]:
    return [r for r in records
            if isinstance(r, dict) and config.relevance_ok(r.get("relevance", "NOT_USEFUL"))]


class ContextBoundHypothesisGenerator:
    """ContextBoundHypothesisGenerator v0.1."""

    def __init__(self, config: HypothesisConfig | None = None, norm: Normalizer | None = None):
        self.config = config or HypothesisConfig()
        self.norm = norm or Normalizer()

    def run(self, evidence_records: list[dict], grounder: Any | None = None) -> HypothesisRunResult:
        cfg, norm = self.config, self.norm
        records = filter_records_by_relevance(evidence_records, cfg)

        frames, frame_diag = extract_frames(records, cfg, norm)
        verified_frames, grounding_diag = verify_frames(frames, cfg, grounder)

        graph = build_evidence_graph(verified_frames, cfg, norm, records=records)

        raw_candidates = generate_candidates(graph, verified_frames, cfg, norm)
        filtered, filter_diag = filter_candidates(raw_candidates, graph, verified_frames, cfg, norm)
        ranked = rank_candidates(filtered, graph, verified_frames, cfg, norm)

        for i, c in enumerate(ranked, start=1):
            c.hypothesis_id = f"HYP-{i:05d}"

        diagnostics = _assemble_diagnostics(
            len(evidence_records), len(records), frame_diag, grounding_diag,
            len(raw_candidates), len(ranked), filter_diag, graph, cfg,
        )
        return HypothesisRunResult(frames=verified_frames, graph=graph,
                                   candidates=ranked, diagnostics=diagnostics)


def _assemble_diagnostics(records_read, records_used, frame_diag, grounding_diag,
                          raw_count, retained_count, filter_diag, graph, cfg) -> dict[str, Any]:
    frames_skipped = dict(frame_diag.get("frames_skipped", {}))
    for fr in grounding_diag.get("skipped", []):
        frames_skipped[fr["reason"]] = frames_skipped.get(fr["reason"], 0) + 1
    skipped_findings = list(frame_diag.get("skipped", [])) + list(grounding_diag.get("skipped", []))
    adequacy = graph.graph.get("adequacy", {})
    return {
        "records_read": records_read,
        "records_used": records_used,
        "findings_seen": frame_diag.get("findings_seen", 0),
        "frames_extracted": frame_diag.get("frames_extracted", 0),
        "frames_grounded": grounding_diag.get("frames_grounded", 0),
        "frames_skipped": frames_skipped,
        "candidates_generated_raw": raw_count,
        "candidates_retained": retained_count,
        "candidates_filtered": filter_diag.get("by_reason", {}),
        "graph_adequacy": {k: v for k, v in adequacy.items() if k != "pivot_ids"},
        "novelty_scope": cfg.novelty_scope,
        "warnings": [
            "Novelty assessed only against this LitStream input corpus (local_corpus_only).",
            "No global PubMed / external-database novelty check was performed.",
            "Hypotheses are untested candidates, not validated discoveries.",
        ],
        # detail sections (not part of the headline summary)
        "frame_extraction": {k: v for k, v in frame_diag.items() if k != "skipped"},
        "grounding": {k: v for k, v in grounding_diag.items() if k != "skipped"},
        "filtering": {k: v for k, v in filter_diag.items() if k != "dropped"},
        "filtered_detail": filter_diag.get("dropped", []),
        "skipped_findings": skipped_findings,
        "config": cfg.to_dict(),
    }


def run_to_dir(
    evidence_dir: str | Path, out_dir: str | Path, config: HypothesisConfig | None = None,
    *, synthesis_path: str | Path | None = None, grounder: Any | None = None,
) -> dict[str, Any]:
    """Load evidence -> run -> write all artifacts. Returns a summary dict (like the grounding
    wrapper). Always writes a report + diagnostics, even with zero candidates."""
    from . import io, report, visualize
    cfg = config or HypothesisConfig()
    out = io.ensure_dir(out_dir)
    records = io.load_evidence_records(evidence_dir)
    synthesis = io.load_synthesis(synthesis_path)

    result = ContextBoundHypothesisGenerator(cfg).run(records, grounder=grounder)
    if synthesis is not None:
        result.diagnostics["synthesis_loaded"] = True

    paths = report.write_report(result, out, cfg, synthesis=synthesis)
    fig_paths = visualize.write_visuals(result, out, cfg)
    paths.update(fig_paths)

    return {
        "evidence_dir": str(evidence_dir),
        "out_dir": str(out),
        "records": len(records),
        "frames_grounded": result.diagnostics["frames_grounded"],
        "candidates": len(result.candidates),
        "report": paths.get("markdown", ""),
        "paths": paths,
    }
