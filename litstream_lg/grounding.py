"""Post-mine grounding for the live agent — opt-in per routine.

After the mine phase writes per-paper `*_evidence.md`, this structures each into
`*_evidence.json` and runs the find-then-verify cascade from `litstream_evidence`,
writing `*_evidence.regrounded.json` next to each plus a `grounding_report.md` that
flags facts the paper does not support (candidate fabrications). Non-destructive: the
markdown is untouched.

Enable per routine with `ground: true`; defaults run the full path (LLM extraction +
entailment-verified numeric facts) and can be tuned via a `grounding:` block:

    ground: true
    grounding:
      backend: llm
      entity_verifier: overlap # overlap (presence) | minicheck (entailment)
      value_verifier: minicheck
      embeddings: hf

Requires the `grounding` extra (sentence-transformers + minicheck). The cascade is presence
for entity mentions and MiniCheck entailment for numeric/claim items.
"""

from __future__ import annotations

from pathlib import Path


def ground_mine_output(project_dir, project: str, *, backend: str = "llm", embeddings: str = "hf",
                       entity_verifier: str = "overlap", value_verifier: str = "minicheck",
                       minicheck_model: str = "flan-t5-large") -> dict:
    """Structure + ground the mine phase output; return a summary dict. Lazy-imports the evidence
    core so the (heavy) grounding deps stay opt-in."""
    from litstream_evidence.structure_evidence import make_structurer, run as structure_run
    from litstream_evidence.ground_retrieval import make_embeddings, make_verifier, run as ground_run

    pdir = Path(project_dir)
    structure_run(project, pdir, make_structurer(backend))
    rows = ground_run(project, pdir, make_embeddings(embeddings),
                      make_verifier(entity_verifier, minicheck_model),
                      make_verifier(value_verifier, minicheck_model))
    report = _write_report(pdir, project, rows, backend, entity_verifier, value_verifier)
    return {"papers": len(rows), "grounded": sum(r["grounded"] for r in rows),
            "flagged": sum(r["flagged"] for r in rows), "report": str(report)}


def _write_report(pdir: Path, project: str, rows: list[dict], backend: str, ev: str, vv: str) -> Path:
    g, f = sum(r["grounded"] for r in rows), sum(r["flagged"] for r in rows)
    lines = [f"# Grounding report — {project}", "",
             f"backend `{backend}` · entity verifier `{ev}` · value verifier `{vv}`", "",
             f"**Total: {g} grounded / {f} flagged** across {len(rows)} paper(s). Flagged items have "
             "no supporting passage in the source — candidate fabrications; see each "
             "`*_evidence.regrounded.json`.", "",
             "| paper | grounded | flagged |", "|---|---|---|"]
    lines += [f"| {r['paper']} | {r['grounded']} | {r['flagged']} |" for r in rows]
    path = pdir / f"projects/{project}/grounding_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
