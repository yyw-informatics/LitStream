"""Run a routine through the LangGraph pipeline.

    mamba run -n litstream-lg python -m litstream_lg.run \
        --routine litstream/config/routines/weekly-citeseq.yaml \
        --project-dir /path/to/kb-skills-working-dir

Requires ANTHROPIC_API_KEY (+ provider keys for any non-Claude phase models).
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import yaml

from langgraph.checkpoint.sqlite import SqliteSaver

from litstream.config.env import load_env
from .graph import build_graph
from .state import PipelineState

ROOT = Path(__file__).resolve().parents[1]


def run_routine_cfg(cfg: dict, project_dir: str, *, db_path: str, library_dir: str,
                    skills_dir: str | None = None, cap_usd: float = 50.0,
                    pdf_limit: int | None = None, checkpoint_db: str | None = None,
                    thread_id: str | None = None) -> dict:
    """Drive the graph to completion for an already-parsed routine cfg. Returns the final
    state dict (status, acquired, phases_done, cost_cents, digest_path, ...). This is the
    entry the scheduler fires; `run_routine` is the same thing from a YAML path."""
    load_env()
    project = cfg["project"]
    init: PipelineState = {
        "routine": cfg, "project": project,
        "project_dir": str(Path(project_dir).resolve()),
        "skills_dir": str(skills_dir or ROOT / "kb-skills-bioinformatics" / "skills"),
        "context_rel": cfg.get("context", f"projects/{project}/context.md"),
        "db_path": str(db_path), "library_dir": str(library_dir),
        "cap_usd": cap_usd, "pdf_limit": pdf_limit,
        "pending_phases": list(cfg.get("phases", [])), "phases_done": [],
        "status": "running", "cost_cents": 0.0,
    }
    tid = thread_id or f"{cfg['name']}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    cfg_run = {"configurable": {"thread_id": tid}, "recursion_limit": 60}
    chk = checkpoint_db or str(Path(db_path).resolve().parent / "litstream_lg_checkpoints.sqlite")
    with SqliteSaver.from_conn_string(chk) as saver:
        graph = build_graph(checkpointer=saver)
        return graph.invoke(init, config=cfg_run)


def run_routine(routine_path: str, project_dir: str, **kw) -> dict:
    """Run a routine from a YAML path — read the file, then drive the graph."""
    cfg = yaml.safe_load(Path(routine_path).read_text())
    return run_routine_cfg(cfg, project_dir, **kw)


def main() -> None:
    ap = argparse.ArgumentParser(description="LitStream (LangGraph) pipeline driver")
    ap.add_argument("--routine", required=True, help="path to a routine YAML")
    ap.add_argument("--project-dir", required=True,
                    help="working dir containing projects/<name>/ (papers, context, KB)")
    ap.add_argument("--skills-dir", default=None,
                    help="kb-skills skills/ dir (default: ./kb-skills-bioinformatics/skills)")
    ap.add_argument("--db", default=str(ROOT / "litstream_lg.db"))
    ap.add_argument("--library", default=str(ROOT / "library"))
    ap.add_argument("--cap-usd", type=float, default=50.0)
    ap.add_argument("--pdf-limit", type=int, default=None)
    args = ap.parse_args()

    final = run_routine(args.routine, args.project_dir, db_path=args.db,
                        library_dir=args.library, skills_dir=args.skills_dir,
                        cap_usd=args.cap_usd, pdf_limit=args.pdf_limit)
    print(f"\n  {final['routine']['name']} → {final.get('status')}")
    for k in ("acquired", "pdfs", "phases_done", "cost_cents", "grounding_result", "digest_path", "note"):
        if final.get(k) not in (None, "", []):
            v = f"${final[k]/100:.4f}" if k == "cost_cents" else final[k]
            print(f"    {k}: {v}")
    print()


if __name__ == "__main__":
    main()
