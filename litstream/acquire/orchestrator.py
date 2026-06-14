"""Acquire orchestrator — run the query across sources, dedup into the library,
link new papers to a project.

    mamba run -n litstream python -m litstream.acquire.orchestrator \
        --routine litstream/config/routines/weekly-citeseq.yaml

Reads sources.yaml for journal filters + the SS key from the environment. PDFs are
NOT fetched here (deferred to pipeline time for triage survivors).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from litstream.config.env import load_env
from litstream.ledger.cost import CostLedger
from litstream.library.store import PaperStore
from litstream.tasks.models import build_model
from .sources import ArxivSource, EuropePMCSource, SemanticScholarSource
from .source_policy import SourcePolicy
from .triage import triage_project

ROOT = Path(__file__).resolve().parents[2]


def _load_deepseek(resolve_key):
    """Build the DeepSeek triage model + its price from task_models.yaml."""
    cfg = yaml.safe_load((ROOT / "litstream" / "config" / "task_models.yaml").read_text())
    spec = next(s for s in cfg["task_models"] if s["name"] == "deepseek")
    p = spec["price_usd_per_mtok"]
    return build_model(spec, resolve_key), spec["model"], (p["input"], p["output"], p["cached_input"])


ALL_SOURCES = ["arxiv", "europepmc", "europepmc_ppr", "semantic_scholar"]


@dataclass
class AcquireSummary:
    project: str
    since: str
    per_source: dict = field(default_factory=dict)   # source -> raw hit count
    fetched: int = 0       # total records returned (pre-dedup)
    new_papers: int = 0    # total papers in library
    new_to_project: int = 0  # newly linked to this project
    new_paper_ids: list = field(default_factory=list)  # newly linked (for triage)
    errors: dict = field(default_factory=dict)


def acquire(*, queries: list[str], project: str, since: dt.date, store: PaperStore,
            sources: list[str] | None = None, journals: list[dict] | None = None,
            ss_key: str | None = None, limit_per_source: int = 20) -> AcquireSummary:
    """Run each query across the ENABLED sources, dedup into the library, link to
    project. Sources are instantiated once so their rate limiters persist across
    queries (important for Semantic Scholar's 1 req/s)."""
    sources = sources if sources is not None else ALL_SOURCES
    summ = AcquireSummary(project=project, since=since.isoformat())
    arx, epmc, ss = ArxivSource(), EuropePMCSource(), SemanticScholarSource(ss_key)

    def fetch(name: str, q: str):
        if name == "arxiv":
            return arx.search(q, since=since, limit=limit_per_source)
        if name == "europepmc":
            return epmc.search(q, since=since, limit=limit_per_source, preprint=False, journals=journals)
        if name == "europepmc_ppr":
            return epmc.search(q, since=since, limit=limit_per_source, preprint=True)
        if name == "semantic_scholar":
            return ss.search(q, since=since, limit=min(limit_per_source, 15))
        raise ValueError(f"unknown source {name!r}")

    for q in queries:
        for name in sources:
            try:
                records = fetch(name, q)
                summ.per_source[name] = summ.per_source.get(name, 0) + len(records)
                summ.fetched += len(records)
                for rec in records:
                    try:
                        pid = store.upsert(rec)
                    except ValueError:
                        continue  # no usable identifier — skip
                    if store.add_to_project(project, pid):
                        summ.new_to_project += 1
                        summ.new_paper_ids.append(pid)
            except Exception as exc:
                summ.errors[name] = f"{type(exc).__name__}: {exc}"

    summ.new_papers = store.stats()["papers"]
    return summ


def _load_journals(sources_yaml: Path) -> list[dict]:
    cfg = yaml.safe_load(sources_yaml.read_text())
    jf = cfg.get("journal_filters", {})
    return [*jf.get("core_methods", []), *jf.get("high_impact_biology", [])]


def main() -> None:
    ap = argparse.ArgumentParser(description="LitStream Acquire")
    ap.add_argument("--routine", required=True)
    ap.add_argument("--db", default=str(ROOT / "litstream.db"))
    ap.add_argument("--library", default=str(ROOT / "library"))
    ap.add_argument("--limit", type=int, default=20, help="max results per source")
    ap.add_argument("--cap-usd", type=float, default=50.0, help="monthly budget cap")
    args = ap.parse_args()

    load_env()
    cfg = yaml.safe_load(Path(args.routine).read_text())
    # since_date (fixed window, e.g. initial 2024→now build) overrides lookback_days.
    if cfg.get("since_date"):
        since = dt.date.fromisoformat(str(cfg["since_date"]))
    else:
        since = dt.date.today() - dt.timedelta(days=cfg.get("lookback_days", 8))
    journals = _load_journals(ROOT / "litstream" / "config" / "sources.yaml")
    queries = cfg.get("search_queries") or [cfg["query"]]   # API keywords ≠ NL query
    routine, project = cfg["name"], cfg["project"]

    store = PaperStore(args.db, library_dir=args.library)
    policy = SourcePolicy(store._conn)

    # 1. decide which sources to fetch (mute the consistently-noisy ones).
    candidates = cfg.get("sources", ALL_SOURCES)
    decisions = policy.decide(routine, candidates)
    effective = [s for s, d in decisions.items() if d in ("run", "reprobe")]
    muted = [s for s, d in decisions.items() if d == "muted_skip"]

    print(f"\n  Acquire · {routine} · {project} · since {since.isoformat()}")
    print("  " + "─" * 56)
    if muted:
        print(f"    muted (skipped): {', '.join(muted)}")
    reprobe = [s for s, d in decisions.items() if d == "reprobe"]
    if reprobe:
        print(f"    re-probing muted source(s): {', '.join(reprobe)}")

    # 2. fetch from enabled sources → dedup into library → link to project.
    summ = acquire(queries=queries, project=project, since=since, store=store,
                   sources=effective, journals=journals,
                   ss_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"), limit_per_source=args.limit)
    for src, n in summ.per_source.items():
        print(f"    {src:<18} {n:>3} hits")
    for src, err in summ.errors.items():
        print(f"    {src:<18} ERROR: {err}")
    print(f"  fetched {summ.fetched} → {summ.new_to_project} new to project, "
          f"{summ.new_papers} in library")

    # 3. triage the new papers (DeepSeek RANK) — feeds the source policy.
    ledger = CostLedger(args.db)
    ledger.set_policy(cap_usd=args.cap_usd)
    model, ds_model, price = _load_deepseek(os.environ.get)
    ledger.seed_pricing(ds_model, *price)
    run_id = ledger.start_run(project=project, routine=routine, invocation="manual")
    results = triage_project(store, project=project, focus=cfg["query"], model=model,
                             ledger=ledger, run_id=run_id)
    ledger.finish_run(run_id)

    kept = sum(1 for r in results if r["score"] >= 0.5)
    print(f"  triaged {len(results)} new papers → {kept} kept (score≥0.5), "
          f"cost ${ledger.run_cost_cents(run_id)/100:.4f}")

    # 4. update source-quality stats (attribute each triaged paper to the sources
    #    that returned it, this run) and apply the mute threshold.
    seen: dict[str, int] = {}
    keepc: dict[str, int] = {}
    for r in results:
        for s in r["sources"]:
            if s in effective:
                seen[s] = seen.get(s, 0) + 1
                if r["score"] >= 0.5:
                    keepc[s] = keepc.get(s, 0) + 1
    policy.update(routine, decisions, seen, keepc)

    print("\n  Source quality (cumulative):")
    for row in policy.status(routine):
        rate = (row["kept"] / row["seen"]) if row["seen"] else 0.0
        flag = " ← MUTED" if row["muted"] else ""
        print(f"    {row['source']:<18} kept {row['kept']:>3}/{row['seen']:<3} "
              f"({rate:>4.0%}) over {row['runs']} run(s){flag}")
    print()


if __name__ == "__main__":
    main()
