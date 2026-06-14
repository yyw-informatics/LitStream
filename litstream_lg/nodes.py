"""LangGraph orchestration for the LitStream pipeline.

Pipeline steps return partial state updates, while routing functions decide whether
the run continues, completes, or exits on budget.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import litstream
from langchain.agents import create_agent

from litstream.config.env import load_env
from litstream.ledger.cost import CostLedger
from litstream.library.store import PaperStore
from litstream.acquire.orchestrator import (
    acquire, ALL_SOURCES, _load_journals, _load_deepseek)
from litstream.acquire.triage import triage_project
from litstream.acquire.source_policy import SourcePolicy
from litstream.acquire.pdf import PdfFetcher
from litstream.skills import render_phase_parts
from litstream.deliver.digest import build_digest
from litstream.deliver.notify import deliver

from .models import (make_chat_model, LedgerCallbackHandler, BudgetExceeded,
                     caching_middleware)
from .routing import resolve, price_for
from .outputs import output_exists, normalize_mine_output
from .tools import make_file_tools
from .state import PipelineState

_CONFIG = Path(litstream.__file__).resolve().parent / "config"
_RECURSION_LIMIT = 50


def _ledger(state: PipelineState) -> CostLedger:
    led = CostLedger(state["db_path"])
    led.set_policy(cap_usd=state.get("cap_usd", 50.0))
    return led


def acquire_node(state: PipelineState) -> dict:
    load_env()
    cfg, project, routine = state["routine"], state["project"], state["routine"]["name"]
    led = _ledger(state)
    if not led.preflight().ok:
        return {"status": "aborted_budget", "note": "month cap reached; run not started",
                "acquired": 0, "pending_phases": [], "cost_cents": led.month_to_date_cents()}

    store = PaperStore(state["db_path"], library_dir=state["library_dir"])
    policy = SourcePolicy(store._conn)
    decisions = policy.decide(routine, cfg.get("sources", ALL_SOURCES))
    effective = [s for s, d in decisions.items() if d in ("run", "reprobe")]
    since = (dt.date.fromisoformat(str(cfg["since_date"])) if cfg.get("since_date")
             else dt.date.today() - dt.timedelta(days=cfg.get("lookback_days", 8)))
    journals = _load_journals(_CONFIG / "sources.yaml")

    summ = acquire(queries=cfg.get("search_queries") or [cfg["query"]], project=project,
                   since=since, store=store, sources=effective, journals=journals,
                   ss_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
                   limit_per_source=cfg.get("max_new_papers", 20))

    run_id = led.start_run(project=project, routine=routine,
                           invocation=cfg.get("invocation", "scheduled"))
    model, ds_model, price = _load_deepseek(os.environ.get)
    led.seed_pricing(ds_model, *price)
    results = triage_project(store, project=project, focus=cfg["query"], model=model,
                             ledger=led, run_id=run_id)
    seen: dict = {}; kept: dict = {}
    for r in results:
        for s in r["sources"]:
            if s in effective:
                seen[s] = seen.get(s, 0) + 1
                if r["score"] >= 0.5:
                    kept[s] = kept.get(s, 0) + 1
    policy.update(routine, decisions, seen, kept)

    return {"run_id": run_id, "acquired": summ.new_to_project, "status": "running",
            "cost_cents": led.run_cost_cents(run_id)}


def pdf_node(state: PipelineState) -> dict:
    project = state["project"]
    store = PaperStore(state["db_path"], library_dir=state["library_dir"])
    papers_dir = Path(state["project_dir"]) / f"projects/{project}/papers"
    pdf = PdfFetcher(store, library_dir=state["library_dir"]).fetch_for_project(
        project, papers_dir, limit=state.get("pdf_limit"))
    return {"pdfs": pdf.fetched}


def agentic_phase_node(state: PipelineState) -> dict:
    phase = state["pending_phases"][0]
    rest = state["pending_phases"][1:]
    cfg, project, run_id = state["routine"], state["project"], state["run_id"]
    led = _ledger(state)

    if not led.phase_gate().ok:
        return {"status": "aborted_budget", "pending_phases": [],
                "note": f"month cap hit before '{phase}'",
                "cost_cents": led.run_cost_cents(run_id)}
    cap = cfg.get("max_cost_per_run_usd")
    if cap is not None and led.run_cost_cents(run_id) >= cap * 100:
        return {"status": "aborted_budget", "pending_phases": [],
                "note": f"per-run cap ${cap:.2f} hit before '{phase}'",
                "cost_cents": led.run_cost_cents(run_id)}

    model_name = cfg.get("models", {}).get(phase, cfg.get("default_model"))
    resolved = resolve(model_name)
    if (price := price_for(model_name)):
        led.seed_pricing(resolved.model, *price)
    system, user = render_phase_parts(phase, skills_dir=Path(state["skills_dir"]),
                                       project=project, context_rel=state["context_rel"])
    cap_cents = cap * 100 if cap is not None else None
    cb = LedgerCallbackHandler(led, run_id, model=resolved.model, phase=phase,
                               cap_cents=cap_cents)
    model = make_chat_model(model_name, callbacks=[cb])
    agent = create_agent(model, make_file_tools(state["project_dir"]), system_prompt=system,
                         middleware=caching_middleware(model_name))

    try:
        agent.invoke({"messages": [{"role": "user", "content": user}]},
                     config={"recursion_limit": _RECURSION_LIMIT})
    except BudgetExceeded as exc:
        return {"status": "aborted_budget", "pending_phases": [],
                "note": f"phase '{phase}': {exc}", "cost_cents": led.run_cost_cents(run_id)}
    except Exception as exc:
        return {"status": "failed", "pending_phases": [],
                "note": f"phase '{phase}' errored: {type(exc).__name__}: {exc}",
                "cost_cents": led.run_cost_cents(run_id)}

    pdir = Path(state["project_dir"])
    grounding_result = None
    if phase == "mine":
        normalize_mine_output(pdir, project)
        if cfg.get("ground"):
            grounding_result = _ground_mine(cfg, pdir, project)
    if not output_exists(phase, pdir, project):
        return {"status": "failed", "pending_phases": [],
                "note": f"phase '{phase}' produced no output",
                "cost_cents": led.run_cost_cents(run_id)}

    audit_result = None
    if phase == "synthesize" and cfg.get("audit_synthesis"):
        audit_result = _audit_synthesis(cfg, pdir, project)

    out = {"phases_done": [phase], "pending_phases": rest, "status": "running",
           "cost_cents": led.run_cost_cents(run_id)}
    if grounding_result is not None:
        out["grounding_result"] = grounding_result
    if audit_result is not None:
        out["audit_result"] = audit_result
    return out


def _ground_mine(cfg: dict, pdir: Path, project: str) -> dict:
    """Structure + ground the mine output (only when the routine sets `ground: true`). Non-fatal:
    grounding is an add-on audit, so any failure — e.g. the `grounding` extra not installed — is
    logged and reported, never raised into the phase."""
    g = cfg.get("grounding", {}) or {}
    try:
        from .grounding import ground_mine_output
        summary = ground_mine_output(
            pdir, project, backend=g.get("backend", "llm"), embeddings=g.get("embeddings", "hf"),
            entity_verifier=g.get("entity_verifier", "overlap"),
            value_verifier=g.get("value_verifier", "minicheck"),
            minicheck_model=g.get("minicheck_model", "flan-t5-large"))
        print(f"[ground] {summary['grounded']} grounded / {summary['flagged']} flagged across "
              f"{summary['papers']} paper(s) → {summary['report']}", flush=True)
        return summary
    except Exception as exc:
        msg = f"grounding skipped: {type(exc).__name__}: {exc}"
        print(f"[ground] {msg}", flush=True)
        return {"error": msg}


def _audit_synthesis(cfg: dict, pdir: Path, project: str) -> dict:
    """Audit the synthesis citations (only when the routine sets `audit_synthesis: true`). Flag-only
    and non-fatal: it writes a `0_synthesis_audit.md` overlay and never edits the synthesis, and any
    failure — e.g. the `grounding` extra not installed — is logged and reported, never raised."""
    a = cfg.get("synthesis_audit", {}) or {}
    try:
        from .synthesis_audit import audit_synthesis_output
        summary = audit_synthesis_output(
            pdir, project, verifier=a.get("verifier", "overlap"),
            minicheck_model=a.get("minicheck_model", "flan-t5-large"))
        if summary.get("audited"):
            print(f"[audit] {summary['flagged']} claim(s) flagged of {summary['n_claims']} parsed "
                  f"→ {summary['report']}", flush=True)
        return summary
    except Exception as exc:
        msg = f"synthesis audit skipped: {type(exc).__name__}: {exc}"
        print(f"[audit] {msg}", flush=True)
        return {"error": msg}


def deliver_node(state: PipelineState) -> dict:
    cfg, routine, project = state["routine"], state["routine"]["name"], state["project"]
    run_id = state.get("run_id")
    status = state.get("status", "completed")
    if status == "running":
        status = "completed"
    led = CostLedger(state["db_path"])
    if run_id:
        led.finish_run(run_id, status)
    now = dt.datetime.now(dt.timezone.utc)
    digests_dir = Path(state["db_path"]).resolve().parent / "digests" / routine
    out = {"status": status,
           "cost_cents": led.run_cost_cents(run_id) if run_id else state.get("cost_cents", 0.0)}
    try:
        path, md = build_digest(db_path=state["db_path"], project=project, routine=routine,
                                project_dir=Path(state["project_dir"]),
                                digests_dir=digests_dir, now=now)
        deliver(cfg.get("deliver", ["digest_md"]),
                subject=f"LitStream·LG · {routine} · {now:%Y-%m-%d}", body=md,
                config={"email_to": cfg.get("email_to"),
                        "slack_webhook": os.environ.get("LITSTREAM_SLACK_WEBHOOK")})
        out["digest_path"] = str(path)
    except Exception as exc:
        out["note"] = (state.get("note", "") or "") + f" [digest error: {exc}]"
    return out


def route_after_acquire(state: PipelineState) -> str:
    return "deliver" if state.get("status") == "aborted_budget" else "pdf"


def route_after_pdf(state: PipelineState) -> str:
    return "agentic" if state.get("pending_phases") else "deliver"


def route_after_phase(state: PipelineState) -> str:
    if state.get("status") in ("failed", "aborted_budget"):
        return "deliver"
    return "agentic" if state.get("pending_phases") else "deliver"
