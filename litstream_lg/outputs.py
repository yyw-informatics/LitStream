"""Output verification for pipeline steps.

The mine phase must leave per-paper `*_evidence.md` files (tolerant of the model's
naming, then normalized); later phases must leave their specific output file.
"""

from __future__ import annotations

from pathlib import Path

PHASE_OUTPUT = {
    "mine": "projects/{project}/literature",
    "synthesize": "projects/{project}/literature/0_synthesis_literature.md",
    "evaluate": "projects/{project}/bioinformatics/fitness_summary.md",
    "design": "projects/{project}/analysis_plan.md",
}

_NON_EVIDENCE = {"literature_summary.md", "0_synthesis_literature.md", "fitness_summary.md"}


def _evidence_files(lit_dir: Path) -> list[Path]:
    """Per-paper evidence markdown in literature/, tolerant of the model's naming."""
    if not lit_dir.is_dir():
        return []
    return [p for p in lit_dir.glob("*.md")
            if not p.name.startswith(".") and p.name not in _NON_EVIDENCE]


def normalize_mine_output(project_dir: Path, project: str) -> int:
    """Ensure every per-paper evidence file ends in _evidence.md so the synthesize
    phase (which globs *_evidence.md) finds it. Returns how many were renamed."""
    lit = project_dir / f"projects/{project}/literature"
    renamed = 0
    for p in _evidence_files(lit):
        if not p.name.endswith("_evidence.md"):
            target = p.with_name(p.stem + "_evidence.md")
            if not target.exists():
                p.rename(target)
                renamed += 1
    return renamed


def output_exists(phase: str, project_dir: Path, project: str) -> bool:
    target = project_dir / PHASE_OUTPUT[phase].format(project=project)
    if target.suffix:
        return target.is_file() and target.stat().st_size > 200
    return len(_evidence_files(target)) > 0
