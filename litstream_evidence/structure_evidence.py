"""Turn an evidence note into a structured `<stem>_evidence.json` and check that each item is backed
by a quote in the source paper.

The text -> structured-record step is pluggable:
  - `llm` (default): JSON-mode extraction via a LangChain chat model with
    `.with_structured_output(EvidenceRecord)` — see `LLMStructurer`.
  - `stub`: deterministic, model-free parsing of the evidence frontmatter with light pattern matching.

Every extracted item carries a `source_quote`; `ground_record` flags items whose quote is not
present in the paper. The markdown evidence note is left untouched — the JSON is a sidecar.

    python -m litstream_evidence.structure_evidence --project lg_smoke \
        --project-dir /tmp/lg_smoke_proj --backend llm
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Protocol

import yaml

from litstream_evidence.pdf_text import extract_text
from litstream_evidence.evidence_schema import LIST_FIELDS, empty_record, validate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def _parse_frontmatter(md: str) -> tuple[dict, str]:
    m = _FM_RE.match(md or "")
    if not m:
        return {}, md or ""
    try:
        fm = yaml.safe_load(m.group(1)) or {}
        if not isinstance(fm, dict):
            fm = {}
    except Exception:
        fm = {}
    return fm, m.group(2)


def _map_relevance(s: str) -> str:
    """Map a model's free-text relevance ('HIGH RELEVANCE', 'Not useful') to the enum."""
    u = (s or "").upper()
    for key, val in (("NOT", "NOT_USEFUL"), ("LOW", "LOW"),
                     ("MOD", "MODERATE"), ("HIGH", "HIGH")):
        if key in u:
            return val
    return "NOT_USEFUL"


def _aslist(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    s = str(v)
    return [p.strip() for p in s.split(",") if p.strip()] if "," in s else [s]


def _window(source: str, term: str, pad: int = 40) -> str:
    """A short snippet of `source` around the first case-insensitive hit of `term`, else ''."""
    if not term:
        return ""
    i = source.lower().find(term.lower())
    if i < 0:
        return ""
    a, b = max(0, i - pad), min(len(source), i + len(term) + pad)
    return re.sub(r"\s+", " ", source[a:b]).strip()


class Structurer(Protocol):
    def structure(self, evidence_md: str, source_text: str) -> dict: ...


class StubStructurer:
    """Deterministic, no-LLM converter for offline tests. Pulls the structured fields from the
    frontmatter and does light pattern matching for markers/genes. Source quotes are windows taken
    from the SOURCE text, so present terms ground and absent ones don't."""

    _GENES = ["FOXP3", "CD4", "CD8A", "CD8", "GZMB", "PRF1", "NKG7", "MS4A1", "CD19",
              "CD14", "IL2RA", "CTLA4", "IL7R", "CCR7", "FCGR3A"]

    def structure(self, evidence_md: str, source_text: str) -> dict:
        fm, body = _parse_frontmatter(evidence_md)
        rec = empty_record()
        rec["relevance"] = _map_relevance(str(fm.get("relevance", "")))
        rec["species"] = _aslist(fm.get("species"))
        rec["tissue"] = _aslist(fm.get("tissue"))

        for name in _aslist(fm.get("cell_types")):
            lead = name.split("(")[0].strip()
            rec["cell_types"].append({"name": name, "source_quote": _window(source_text, lead)})

        seen: set = set()
        for m in re.finditer(r"\b(CD\d+[a-z]?|HLA-DR|PD-1|PD-L1)\b", body):
            mk = m.group(0)
            if mk.lower() in seen:
                continue
            seen.add(mk.lower())
            rec["surface_markers"].append(
                {"marker": mk, "maps_to_gene": "", "source_quote": _window(source_text, mk)})

        sp = rec["species"][0] if rec["species"] else ""
        for g in self._GENES:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(g)}(?![A-Za-z0-9])", body, re.IGNORECASE):
                rec["genes"].append({"symbol": g, "species": sp,
                                     "source_quote": _window(source_text, g)})
        return rec


_LLM_PROMPT = (
    "You extract structured facts from a single-cell / CITE-seq methods paper.\n"
    "Below is an analyst's note about the paper, then the paper's text.\n\n"
    "Rules:\n"
    "- Fill ONLY facts actually stated in the paper. Leave a list empty if it has nothing.\n"
    "- For every item, `source_quote` MUST be a short phrase copied VERBATIM from the PAPER "
    "TEXT (not from the note). Do not invent entries.\n\n"
    "=== ANALYST NOTE ===\n{note}\n\n=== PAPER TEXT ===\n{paper}\n"
)


class LLMStructurer:
    """LLM converter: a LangChain chat model with `.with_structured_output(EvidenceRecord)` fills
    the field list directly (the provider's tool-use / JSON mode under the hood, with LangChain
    validating the result). Accepts any LangChain chat model; the prompt constrains every
    `source_quote` to the paper text."""

    def __init__(self, model, max_chars: int = 45_000, note_chars: int = 8_000):
        from litstream_evidence.evidence_models import EvidenceRecord
        self.structured = model.with_structured_output(EvidenceRecord)
        self.max_chars = max_chars
        self.note_chars = note_chars

    def structure(self, evidence_md: str, source_text: str) -> dict:
        prompt = _LLM_PROMPT.format(note=(evidence_md or "")[:self.note_chars],
                                    paper=(source_text or "")[:self.max_chars])
        result = self.structured.invoke(prompt)
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)


def _default_chat_model():
    """Build the default LangChain chat model for the LLM converter (lazy import of the chat stack)."""
    try:
        from litstream_lg.models import make_chat_model
    except Exception as exc:
        raise RuntimeError("the 'llm' converter needs the LangChain stack (litstream_lg + "
                           "langchain-anthropic). Install it, or pass your own model.") from exc
    return make_chat_model("claude-haiku-4-5-20251001")


def make_structurer(name: str, model=None) -> Structurer:
    if name == "stub":
        return StubStructurer()
    if name == "llm":
        return LLMStructurer(model if model is not None else _default_chat_model())
    raise ValueError(f"unknown backend {name!r} (use 'stub' or 'llm')")


def ground_record(rec: dict, source_text: str) -> dict:
    """For each item, is its `source_quote` actually present in the source? Returns a report
    with per-field grounded/ungrounded counts and the ungrounded items (candidate fabrications)."""
    src = _norm(source_text)
    report: dict = {"grounded": 0, "ungrounded": 0, "by_field": {}, "ungrounded_items": []}
    for field in LIST_FIELDS:
        ok = bad = 0
        for it in rec.get(field, []):
            q = _norm(it.get("source_quote", ""))
            if q and q in src:
                ok += 1
            else:
                bad += 1
                report["ungrounded_items"].append({"field": field, "item": it})
        report["by_field"][field] = {"grounded": ok, "ungrounded": bad}
        report["grounded"] += ok
        report["ungrounded"] += bad
    return report


def structure_evidence(stem: str, evidence_md: str, source_text: str,
                       structurer: Structurer) -> tuple[dict, dict]:
    rec = structurer.structure(evidence_md, source_text)
    rec.setdefault("paper_id", stem)
    if not rec.get("paper_id"):
        rec["paper_id"] = stem
    report = ground_record(rec, source_text)
    report["schema_errors"] = validate(rec)
    return rec, report


def run(project: str, project_dir: Path, structurer: Structurer) -> list[dict]:
    lit = project_dir / f"projects/{project}/literature"
    papers = project_dir / f"projects/{project}/papers"
    ev_files = sorted(lit.glob("*_evidence.md")) if lit.is_dir() else []
    rows: list[dict] = []
    for ev in ev_files:
        stem = ev.name[: -len("_evidence.md")]
        pdf = papers / f"{stem}.pdf"
        if not pdf.exists():
            log(f"{stem}: no source PDF ({pdf.name}) — skipping"); continue
        try:
            source_text = extract_text(pdf)
        except Exception as exc:
            log(f"{stem}: PDF extract failed: {exc} — skipping"); continue
        rec, report = structure_evidence(stem, ev.read_text(), source_text, structurer)
        (lit / f"{stem}_evidence.json").write_text(json.dumps(rec, indent=2))
        n_items = sum(len(rec.get(f, [])) for f in LIST_FIELDS)
        rows.append({"paper": stem, "items": n_items, "grounded": report["grounded"],
                     "ungrounded": report["ungrounded"],
                     "schema_errors": len(report["schema_errors"])})
        log(f"{stem}: {n_items} items, {report['grounded']} grounded / {report['ungrounded']} "
            f"ungrounded, {len(report['schema_errors'])} schema errors → {stem}_evidence.json")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Structure MINE evidence notes into evidence.json + ground-check them")
    ap.add_argument("--project", required=True)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--backend", default="stub", choices=["stub", "llm"])
    args = ap.parse_args()

    from litstream.config.env import load_env
    load_env()
    rows = run(args.project, Path(args.project_dir).resolve(), make_structurer(args.backend))
    if not rows:
        log("no (evidence, pdf) pairs found — nothing structured"); return
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "structure_evidence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["paper", "items", "grounded", "ungrounded", "schema_errors"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tot_items = sum(r["items"] for r in rows)
    tot_ground = sum(r["grounded"] for r in rows)
    tot_err = sum(r["schema_errors"] for r in rows)
    print(f"\n  structured {len(rows)} paper(s): {tot_items} items, "
          f"{tot_ground} grounded, {tot_err} schema errors. "
          f"JSON sidecars written next to each *_evidence.md; summary → results/structure_evidence.csv\n")


if __name__ == "__main__":
    main()
