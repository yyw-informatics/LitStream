"""Shared state for one LitStream pipeline run.

A single typed channel dict. Lists that accumulate across nodes use reducers so a node
returns only its delta (e.g. `{"phases_done": ["mine"]}`) and LangGraph appends.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class PipelineState(TypedDict, total=False):
    routine: dict           # the parsed routine YAML
    project: str
    project_dir: str        # working dir containing projects/<name>/
    skills_dir: str
    context_rel: str        # projects/<name>/context.md, relative to project_dir
    db_path: str
    library_dir: str
    cap_usd: float
    pdf_limit: int | None

    acquired: int           # new papers linked to the project
    pdfs: int               # PDFs fetched for triage survivors

    run_id: str
    pending_phases: list[str]
    phases_done: Annotated[list[str], operator.add]

    cost_cents: float       # authoritative running total (read from the ledger)
    status: str             # running | completed | failed | aborted_budget
    note: str
    digest_path: str
    grounding_result: dict  # post-mine grounding summary (only if routine sets `ground: true`)
