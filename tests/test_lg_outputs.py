"""Tests for output verification.

output_exists is False before a step writes and True after; for file-producing
steps it enforces a >200-byte threshold. normalize_mine_output renames descriptively
named evidence files to the *_evidence.md convention.
"""

from __future__ import annotations

from litstream_lg.outputs import normalize_mine_output, output_exists

PROJECT = "demo"


def _lit(tmp_path):
    return tmp_path / f"projects/{PROJECT}/literature"


def test_mine_false_before_any_evidence(tmp_path):
    assert output_exists("mine", tmp_path, PROJECT) is False


def test_mine_true_after_evidence_written(tmp_path):
    lit = _lit(tmp_path)
    lit.mkdir(parents=True)
    (lit / "paper1_evidence.md").write_text("short ok, mine has no size gate")
    assert output_exists("mine", tmp_path, PROJECT) is True


def test_mine_ignores_non_evidence_files(tmp_path):
    lit = _lit(tmp_path)
    lit.mkdir(parents=True)
    (lit / "0_synthesis_literature.md").write_text("synthesis, not per-paper")
    (lit / "literature_summary.md").write_text("summary, not per-paper")
    assert output_exists("mine", tmp_path, PROJECT) is False


def test_synthesize_false_before_file(tmp_path):
    assert output_exists("synthesize", tmp_path, PROJECT) is False


def test_synthesize_true_after_large_file(tmp_path):
    lit = _lit(tmp_path)
    lit.mkdir(parents=True)
    (lit / "0_synthesis_literature.md").write_text("x" * 201)
    assert output_exists("synthesize", tmp_path, PROJECT) is True


def test_synthesize_tiny_file_under_threshold_is_false(tmp_path):
    lit = _lit(tmp_path)
    lit.mkdir(parents=True)
    (lit / "0_synthesis_literature.md").write_text("x" * 200)   # exactly 200 -> not > 200
    assert output_exists("synthesize", tmp_path, PROJECT) is False


def test_design_and_evaluate_paths(tmp_path):
    assert output_exists("design", tmp_path, PROJECT) is False
    plan = tmp_path / f"projects/{PROJECT}/analysis_plan.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("y" * 300)
    assert output_exists("design", tmp_path, PROJECT) is True

    assert output_exists("evaluate", tmp_path, PROJECT) is False
    fit = tmp_path / f"projects/{PROJECT}/bioinformatics/fitness_summary.md"
    fit.parent.mkdir(parents=True)
    fit.write_text("z" * 300)
    assert output_exists("evaluate", tmp_path, PROJECT) is True


# ---------------------------------------------------------------------------
# normalize_mine_output — rename to the *_evidence.md convention
# ---------------------------------------------------------------------------

def test_normalize_renames_descriptive_md_to_evidence(tmp_path):
    lit = _lit(tmp_path)
    lit.mkdir(parents=True)
    (lit / "smith2024.md").write_text("evidence for smith 2024")
    renamed = normalize_mine_output(tmp_path, PROJECT)
    assert renamed == 1
    assert not (lit / "smith2024.md").exists()
    assert (lit / "smith2024_evidence.md").exists()


def test_normalize_leaves_already_normalized_files(tmp_path):
    lit = _lit(tmp_path)
    lit.mkdir(parents=True)
    (lit / "p_evidence.md").write_text("already named")
    renamed = normalize_mine_output(tmp_path, PROJECT)
    assert renamed == 0
    assert (lit / "p_evidence.md").exists()


def test_normalize_skips_known_non_evidence(tmp_path):
    lit = _lit(tmp_path)
    lit.mkdir(parents=True)
    (lit / "0_synthesis_literature.md").write_text("synthesis")
    (lit / "literature_summary.md").write_text("summary")
    renamed = normalize_mine_output(tmp_path, PROJECT)
    assert renamed == 0
    assert (lit / "0_synthesis_literature.md").exists()
    assert (lit / "literature_summary.md").exists()


def test_normalize_then_output_exists_true(tmp_path):
    lit = _lit(tmp_path)
    lit.mkdir(parents=True)
    (lit / "foo.md").write_text("verdict")
    normalize_mine_output(tmp_path, PROJECT)
    assert output_exists("mine", tmp_path, PROJECT) is True
