"""CLI for the hypothesis-candidate generator.

Invoke as ``python -m litstream.hypotheses <run|eval> ...`` or via the console scripts
``litstream-hypothesize`` / ``litstream-hypothesize-eval``.

    python -m litstream.hypotheses run --evidence-dir DIR --out-dir DIR [--config cfg.yml] ...
    python -m litstream.hypotheses eval frames --gold gold.jsonl --evidence-dir DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _build_config(args):
    from .config import HypothesisConfig
    cfg = HypothesisConfig.from_yaml(args.config)
    overrides: dict = {}
    if args.max_candidates is not None:
        overrides["max_candidates"] = args.max_candidates
    if args.min_relevance is not None:
        overrides["min_paper_relevance"] = args.min_relevance
    if args.require_grounded_frames is not None:
        overrides["require_grounded_frames"] = _bool(args.require_grounded_frames)
    if args.write_graphml is not None:
        overrides["write_graphml"] = _bool(args.write_graphml)
    if args.write_figures is not None:
        overrides["write_figures"] = _bool(args.write_figures)
    if args.llm_verbalizer is not None:
        overrides["llm_verbalizer"] = _bool(args.llm_verbalizer)
    if args.grounder is not None:
        overrides["grounder"] = args.grounder
    return cfg.with_overrides(**overrides)


def _add_run_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--evidence-dir", required=True, help="directory of *_evidence.json records")
    ap.add_argument("--out-dir", required=True, help="output directory for the report artifacts")
    ap.add_argument("--config", default=None, help="hypothesis_config.yml (optional)")
    ap.add_argument("--synthesis", default=None, help="synthesis.json (optional, for the run summary)")
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--min-relevance", choices=["HIGH", "MODERATE", "LOW"], default=None)
    ap.add_argument("--require-grounded-frames", default=None)
    ap.add_argument("--grounder", choices=["overlap", "minicheck", "stub"], default=None)
    ap.add_argument("--write-graphml", default=None)
    ap.add_argument("--write-figures", default=None)
    ap.add_argument("--llm-verbalizer", default=None)


def run_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="litstream-hypothesize",
                                 description="Generate ranked hypothesis candidates from evidence.")
    _add_run_args(ap)
    args = ap.parse_args(argv)
    from .pipeline import run_to_dir
    cfg = _build_config(args)
    summary = run_to_dir(args.evidence_dir, args.out_dir, cfg, synthesis_path=args.synthesis)
    print(f"\n  Hypotheses: {summary['candidates']} candidate(s) from {summary['records']} record(s) "
          f"({summary['frames_grounded']} grounded frames).")
    print(f"  Report: {summary['report']}\n")
    return 0   # succeed even with zero candidates


def eval_main(argv=None) -> int:
    from . import evaluate
    ap = argparse.ArgumentParser(prog="litstream-hypothesize-eval",
                                 description="Offline evaluation of the hypothesis generator.")
    sub = ap.add_subparsers(dest="kind", required=True)

    f = sub.add_parser("frames", help="gold frame-extraction accuracy")
    f.add_argument("--gold", required=True)
    f.add_argument("--evidence-dir", required=True)

    h = sub.add_parser("hidden-edge", help="hidden-edge recovery (Recall@k, MRR)")
    h.add_argument("--evidence-dir", required=True)
    h.add_argument("--gold-hidden", default=None)
    h.add_argument("--k", type=int, default=10)

    n = sub.add_parser("null", help="null-model comparison")
    n.add_argument("--evidence-dir", required=True)
    n.add_argument("--seed", type=int, default=0)

    args = ap.parse_args(argv)
    if args.kind == "frames":
        res = evaluate.eval_frames(args.gold, args.evidence_dir)
    elif args.kind == "hidden-edge":
        res = evaluate.eval_hidden_edge(args.evidence_dir, args.gold_hidden, k=args.k)
    else:
        res = evaluate.eval_null_models(args.evidence_dir, seed=args.seed)
    import json
    print(json.dumps(res, indent=2))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("eval", "hypothesize-eval"):
        return eval_main(argv[1:])
    if argv and argv[0] in ("run", "generate", "hypothesize"):
        return run_main(argv[1:])
    return run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
