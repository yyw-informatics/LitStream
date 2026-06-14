"""MINE entity-recall — measure the MINE stage's targeted-extraction quality against a
specialist-NER silver standard, with no ground-truth labels required.

For each mined paper we run a biomedical NER over (a) the SOURCE paper text and (b) the
MINE `*_evidence.md`, then report per entity type:
  - recall     = of the entities the NER found in the SOURCE, the fraction MINE also
                 surfaced  (did MINE miss what the paper contains?)
  - extra_rate = of the entities MINE surfaced, the fraction the NER did NOT find AND that
                 don't appear literally in the source  (candidate hallucination)

This measures AGREEMENT with a NER model, not ground truth — the NER has its own false
negatives and blind spots. Calibrate the absolute numbers against a small human-gold set
(see the generated SUMMARY) before trusting them.

The NER backend is pluggable. The default `stub` backend is a built-in CITE-seq lexicon
(dictionary matcher), so this runs with no model or network; `gliner-biomed` lazy-loads
GLiNER once you `pip install gliner`.

    python -m litstream.eval.mine_entity_recall --project lg_smoke \
        --project-dir /tmp/lg_smoke_proj --backend stub
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from litstream_evidence.pdf_text import extract_text

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

DEFAULT_LABELS = ["gene", "surface_marker", "cell_type", "species"]


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    front = ["paper", "label", "n_silver", "n_mine", "recall", "extra_rate", "match"]
    keys = front + [k for k in {k for r in rows for k in r} if k not in front]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    log(f"wrote {path.name} ({len(rows)} rows)")


# ---- entity model + normalization ----------------------------------------------

@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    span: tuple[int, int] | None = None
    norm: str = ""


def normalize(text: str) -> str:
    """Surface-form normalizer: collapse whitespace, casefold, strip edge punctuation.

    Comparison is surface-form only. Ontology linkers (HGNC for genes, Cell Ontology for
    cell types, NCBI Taxonomy for species, an ADT antibody-panel dictionary for surface
    markers) plug in here when entity normalization is added."""
    t = re.sub(r"\s+", " ", text or "").strip().casefold()
    return t.strip(" .,:;()[]{}")


# ---- NER backends --------------------------------------------------------------

class SilverNER(Protocol):
    def extract(self, text: str, labels: list[str]) -> list[Entity]: ...


# Compact CITE-seq lexicon: keeps the dictionary backend deterministic and model-free.
_STUB_LEXICON: dict[str, list[str]] = {
    "gene": ["CD4", "CD8A", "CD8", "FOXP3", "GZMB", "PRF1", "NKG7", "MS4A1", "CD19",
             "CD14", "FCGR3A", "ITGAX", "CLEC9A", "IL7R", "CCR7", "SELL", "KLRB1"],
    "surface_marker": ["CD3", "CD4", "CD8", "CD11c", "CD14", "CD16", "CD19", "CD25",
                       "CD45RA", "CD45RO", "CD56", "CD127", "CD279", "PD-1", "HLA-DR"],
    "cell_type": ["T cell", "T cells", "CD4 T cell", "CD8 T cell", "Treg",
                  "regulatory T cell", "B cell", "B cells", "NK cell", "natural killer",
                  "monocyte", "monocytes", "dendritic cell", "dendritic cells",
                  "macrophage", "neutrophil", "hematopoietic stem cell", "HSC",
                  "progenitor", "MAIT"],
    "species": ["human", "Homo sapiens", "mouse", "Mus musculus", "murine", "rat"],
}


class StubNER:
    """Deterministic word-boundary dictionary matcher over a compact CITE-seq lexicon.

    Runs with no model or network, which keeps the harness and tests fast and offline.
    Use the `gliner-biomed` backend for model-based silver-standard numbers."""

    def __init__(self, lexicon: dict[str, list[str]] | None = None):
        self.lexicon = lexicon if lexicon is not None else _STUB_LEXICON

    def extract(self, text: str, labels: list[str]) -> list[Entity]:
        out: list[Entity] = []
        for label in labels:
            for term in self.lexicon.get(label, []):
                pat = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
                for m in re.finditer(pat, text or "", re.IGNORECASE):
                    out.append(Entity(text=m.group(0), label=label,
                                      span=(m.start(), m.end()), norm=normalize(term)))
        return out


class GlinerBiomedNER:
    """GLiNER-biomed backend — lazy-imports `gliner` so the module loads without it.

    The model id, threshold, and the schema-key -> natural-language label map are
    configurable here."""

    def __init__(self, model_id: str = "Ihor/gliner-biomed-base-v1.0", threshold: float = 0.5):
        from gliner import GLiNER  # lazy: only when this backend is selected
        self.model = GLiNER.from_pretrained(model_id)
        self.threshold = threshold
        self._label_text = {"gene": "gene", "surface_marker": "surface protein marker",
                            "cell_type": "cell type", "species": "species"}

    def extract(self, text: str, labels: list[str]) -> list[Entity]:
        types = [self._label_text.get(l, l) for l in labels]
        rev = {self._label_text.get(l, l): l for l in labels}
        out: list[Entity] = []
        for s in self.model.predict_entities(text, types, threshold=self.threshold):
            out.append(Entity(text=s["text"], label=rev.get(s["label"], s["label"]),
                              span=(s.get("start"), s.get("end")), norm=normalize(s["text"])))
        return out


def make_backend(name: str) -> SilverNER:
    if name == "stub":
        return StubNER()
    if name == "gliner-biomed":
        return GlinerBiomedNER()
    raise ValueError(f"unknown backend {name!r} (use 'stub' or 'gliner-biomed')")


# ---- evidence parsing ----------------------------------------------------------

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def parse_evidence(md_text: str) -> tuple[dict, str]:
    """Split model-generated evidence markdown into (frontmatter_dict, body). Tolerant:
    absent/invalid YAML -> ({}, whole_text)."""
    m = _FM_RE.match(md_text or "")
    if not m:
        return {}, md_text or ""
    try:
        fm = yaml.safe_load(m.group(1)) or {}
        if not isinstance(fm, dict):
            fm = {}
    except Exception:
        fm = {}
    return fm, m.group(2)


def _frontmatter_text(fm: dict) -> str:
    """Flatten frontmatter values (species, cell_types, tissue, ...) into searchable text
    so the NER picks up structured entities the prose body might not repeat."""
    parts: list[str] = []
    for v in fm.values():
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v is not None:
            parts.append(str(v))
    return " . ".join(parts)


# ---- scoring -------------------------------------------------------------------

def score_paper(stem: str, source_text: str, evidence_md: str, backend: SilverNER,
                labels: list[str], relaxed: bool = False) -> list[dict]:
    """One row per label: recall (vs silver-in-source) + extra_rate (vs silver + source)."""
    fm, body = parse_evidence(evidence_md)
    mine_text = _frontmatter_text(fm) + "\n" + body
    silver = backend.extract(source_text, labels)
    mine = backend.extract(mine_text, labels)
    src_norm = normalize(source_text)

    def norms(ents: list[Entity], label: str) -> set[str]:
        return {e.norm for e in ents if e.label == label and e.norm}

    def matches(a: str, b: str) -> bool:
        return (a in b or b in a) if relaxed else (a == b)

    rows: list[dict] = []
    for label in labels:
        s = norms(silver, label)          # silver entities found in the SOURCE
        mm = norms(mine, label)           # entities MINE surfaced
        covered = {y for y in s if any(matches(x, y) for x in mm)}
        matched_mine = {x for x in mm if any(matches(x, y) for y in s)}
        extra = {x for x in (mm - matched_mine) if x not in src_norm}
        recall = (len(covered) / len(s)) if s else None
        extra_rate = (len(extra) / len(mm)) if mm else None
        rows.append({
            "paper": stem, "label": label, "n_silver": len(s), "n_mine": len(mm),
            "recall": None if recall is None else round(recall, 3),
            "extra_rate": None if extra_rate is None else round(extra_rate, 3),
            "match": "relaxed" if relaxed else "strict",
            "missed": "; ".join(sorted(s - covered))[:300],
            "extra": "; ".join(sorted(extra))[:300],
        })
    return rows


def run(project: str, project_dir: Path, backend: SilverNER, labels: list[str],
        relaxed: bool = False) -> list[dict]:
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
        rows.extend(score_paper(stem, source_text, ev.read_text(), backend, labels, relaxed))
        log(f"{stem}: scored {len(labels)} labels")
    return rows


def summarize(rows: list[dict], backend_name: str, match: str) -> str:
    agg: dict = defaultdict(lambda: {"recall": [], "extra": [], "n_silver": 0, "n_mine": 0})
    for r in rows:
        a = agg[r["label"]]
        if r["recall"] not in (None, ""):
            a["recall"].append(float(r["recall"]))
        if r["extra_rate"] not in (None, ""):
            a["extra"].append(float(r["extra_rate"]))
        a["n_silver"] += int(r["n_silver"]); a["n_mine"] += int(r["n_mine"])
    lines = [f"# MINE entity-recall ({match} match · backend={backend_name})", "",
             f"papers scored: **{len({r['paper'] for r in rows})}**", "",
             "| label | mean recall | mean extra-rate | Σ silver | Σ mine |",
             "|---|---|---|---|---|"]
    for label, a in sorted(agg.items()):
        mr = round(statistics.mean(a["recall"]), 3) if a["recall"] else "—"
        me = round(statistics.mean(a["extra"]), 3) if a["extra"] else "—"
        lines.append(f"| {label} | {mr} | {me} | {a['n_silver']} | {a['n_mine']} |")
    lines += [
        "",
        "> **recall** = of the entities the silver NER found in the SOURCE paper, the "
        "fraction MINE also surfaced (did MINE miss what the paper contains?).",
        "> **extra-rate** = of the entities MINE surfaced, the fraction the silver NER did "
        "NOT find AND that don't appear literally in the source (candidate hallucination).",
        "",
        "> ⚠️ **This measures AGREEMENT with a NER model, not ground truth.** The silver NER "
        "has its own false negatives and blind spots, so these numbers are only meaningful "
        "*relative to that model*. Calibrate against a small **human-gold** set (20–50 "
        "hand-labeled evidence files) before trusting them — that gold set is the real truth "
        "anchor.",
        "",
        f"*backend: {backend_name} · match: {match}*",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MINE entity-recall vs a specialist-NER silver standard")
    ap.add_argument("--project", required=True)
    ap.add_argument("--project-dir", required=True,
                    help="dir containing projects/<project>/{literature,papers}/")
    ap.add_argument("--backend", default="stub", choices=["stub", "gliner-biomed"])
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    ap.add_argument("--match", default="strict", choices=["strict", "relaxed"])
    args = ap.parse_args()

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    backend = make_backend(args.backend)
    rows = run(args.project, Path(args.project_dir).resolve(), backend, labels,
               relaxed=(args.match == "relaxed"))
    if not rows:
        log("no (evidence, pdf) pairs found — nothing scored"); return
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "mine_entity_recall.csv", rows)
    summary = summarize(rows, args.backend, args.match)
    (RESULTS / "mine_entity_recall_SUMMARY.md").write_text(summary)
    log("wrote mine_entity_recall_SUMMARY.md")
    print("\n" + summary + "\n")


if __name__ == "__main__":
    main()
