"""Triage pre-filter: score each newly-acquired abstract against a project's focus.

Uses DeepSeek (selected on a CSMeD/cost benchmark) to rank rather than hard-exclude:
every paper gets a label plus a 0-1 relevance score in project_papers, so the
pipeline can later mine the top-N within budget. Because recall is imperfect, a low
score is a deprioritization rather than a deletion; the paper stays in the library.

The per-paper labels also feed the source-quality policy (source_policy.py): a
source whose papers consistently score low gets auto-muted.
"""

from __future__ import annotations

import re

from litstream.eval.triage_eval import parse_label   # shared label parser

SYSTEM = ("You are a precise literature-triage filter for a bioinformatics project. "
          "Judge methods/relevance strictly. Output exactly one line.")

_SCORE_RE = re.compile(r"(?<!\d)(0?\.\d+|1\.0+|0|1)(?!\d)")
_DEFAULT_SCORE = {"RELEVANT": 0.8, "BORDERLINE": 0.5, "NOT_RELEVANT": 0.1, "UNKNOWN": 0.3}


def build_prompt(focus: str, title: str, abstract: str) -> str:
    return (
        f"PROJECT FOCUS:\n{focus}\n\n"
        "Decide whether this paper is worth deep-reading for the focus above. "
        "Respond with EXACTLY one line: a label (RELEVANT, BORDERLINE, or "
        "NOT_RELEVANT) then a relevance score 0.0-1.0. Example: RELEVANT 0.85\n\n"
        f"TITLE: {title}\nABSTRACT: {abstract}"
    )


def parse_score(text: str, label: str) -> float:
    m = _SCORE_RE.search(text)
    if m:
        try:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            pass
    return _DEFAULT_SCORE.get(label, 0.3)


def triage_project(store, *, project: str, focus: str, model, ledger, run_id: str,
                   keep_threshold: float = 0.5) -> list[dict]:
    """Score all un-triaged papers for a project. Returns per-paper results
    (paper_id, label, score, sources) for the source policy to aggregate."""
    rows = store._conn.execute(
        """SELECT pp.paper_id, p.title, p.abstract, p.sources
           FROM project_papers pp JOIN papers p ON p.paper_id = pp.paper_id
           WHERE pp.project = ? AND pp.triage_label IS NULL""", (project,)).fetchall()

    import json
    results: list[dict] = []
    for r in rows:
        if not (r["abstract"] or "").strip():
            # No abstract to judge: park as low-priority borderline without spending a call.
            store.set_triage(project, r["paper_id"], "BORDERLINE", score=0.3,
                             model="(no-abstract)", status="dropped")
            results.append({"paper_id": r["paper_id"], "label": "BORDERLINE",
                            "score": 0.3, "sources": json.loads(r["sources"])})
            continue
        res = model.complete(build_prompt(focus, r["title"], r["abstract"]),
                             system=SYSTEM, max_tokens=24)
        ledger.record(run_id, res.model, phase="triage", role="acquire",
                      input_tokens=res.input_tokens, output_tokens=res.output_tokens,
                      cached_input_tokens=res.cached_input_tokens)
        label = parse_label(res.text)
        score = parse_score(res.text, label)
        status = "kept" if score >= keep_threshold else "dropped"
        store.set_triage(project, r["paper_id"], label, score=score,
                         model=res.model, status=status)
        results.append({"paper_id": r["paper_id"], "label": label, "score": score,
                        "sources": json.loads(r["sources"])})
    return results
