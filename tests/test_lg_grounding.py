"""Tests for the post-mine grounding wrapper (litstream_lg/grounding.py).
No torch, no model, no API: fake embedder and structurer on an empty project."""

from pathlib import Path

from litstream_lg.grounding import ground_mine_output


def test_ground_mine_output_empty_project(tmp_path):
    (tmp_path / "projects" / "p" / "literature").mkdir(parents=True)
    summary = ground_mine_output(tmp_path, "p", backend="stub", embeddings="fake")
    assert summary == {"papers": 0, "grounded": 0, "flagged": 0,
                       "report": str(tmp_path / "projects/p/grounding_report.md")}
    report = Path(summary["report"]).read_text()
    assert "# Grounding report — p" in report
    assert "0 grounded / 0 flagged" in report
