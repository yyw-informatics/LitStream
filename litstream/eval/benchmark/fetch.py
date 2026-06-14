"""Download + cache the benchmark data (stdlib only — no pip installs).

Each ensure_* downloads once into litstream/eval/benchmark/data/<name>/ and is a no-op after.
Run `python3 -m litstream.eval.benchmark.fetch` to pre-download everything.
"""

from __future__ import annotations

import io
import json
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

# benchmark data lives inside the package so it ports as one self-contained unit
DATA = Path(__file__).resolve().parent / "data"

_BIORED_ZIP = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip"
_JNLPBA_TEST = ("https://raw.githubusercontent.com/cambridgeltl/"
                "MTL-Bioinformatics-2016/master/data/JNLPBA/test.tsv")
_MEASEVAL_ZIP = "https://codeload.github.com/harperco/MeasEval/zip/refs/heads/main"
_MSLR_TAR = "https://ai2-s2-mslr.s3.us-west-2.amazonaws.com/mslr_data.tar.gz"


def _get(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (trusted hosts)
        return r.read()


def ensure_biored() -> Path:
    out = DATA / "biored" / "Test.PubTator"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(_get(_BIORED_ZIP))) as z:
        name = next(n for n in z.namelist() if n.endswith("Test.PubTator"))
        out.write_bytes(z.read(name))
    return out


def ensure_jnlpba() -> Path:
    out = DATA / "jnlpba" / "test.tsv"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_get(_JNLPBA_TEST))
    return out


def ensure_measeval() -> Path:
    """Returns the MeasEval data root (the dir containing trial/ train/ eval/)."""
    root = DATA / "measeval"
    if (root / "eval").exists() or (root / "train").exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(_get(_MEASEVAL_ZIP))) as z:
        for n in z.namelist():
            # keep only data/<split>/{text,tsv}/* ; flatten the MeasEval-main/data/ prefix
            if "/data/" in n and not n.endswith("/"):
                rel = n.split("/data/", 1)[1]
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(n))
    return root


def ensure_mslr(subset: str = "cochrane") -> Path:
    """Download the MSLR-2022 multi-document medical-study summarization data and
    extract one subset. The SYNTHESIZE analog: each review groups many input studies
    (Title + Abstract) under one ReviewID into a single Target review summary.

    subset: "cochrane" (~4.7k reviews, smaller, cleaner references) or
            "ms2" (~17.9k reviews, larger, with per-review Background).

    Files per subset: {train,dev,test}-inputs.csv (ReviewID, PMID, Title, Abstract)
    and {train,dev}-targets.csv (ReviewID, Target). The test split is the shared-task
    holdout (inputs only, no targets), so score on dev. The ~253 MB tarball is streamed
    member-by-member to avoid holding it in memory; cached per subset to disk.
    """
    if subset not in ("cochrane", "ms2"):
        raise ValueError(f"subset must be 'cochrane' or 'ms2', got {subset!r}")
    out = DATA / "mslr" / subset
    if (out / "dev-inputs.csv").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_MSLR_TAR, timeout=600) as resp:  # noqa: S310 (trusted host)
        # r|gz = streaming gzip: members arrive in order; extract only this subset's CSVs
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for m in tar:
                if not m.isfile() or not m.name.endswith(".csv"):
                    continue
                if subset not in m.name.split("/"):
                    continue
                src = tar.extractfile(m)
                if src is None:
                    continue
                with (out / Path(m.name).name).open("wb") as fh:
                    shutil.copyfileobj(src, fh)  # chunked copy — ms2 inputs are large
    return out


_CXG_API = "https://api.cellxgene.cziscience.com/curation/v1"


def ensure_cellxgene(limit: int | None = None) -> Path:
    """Crawl the CZ CELLxGENE Discover Curation REST API (no auth) into a cached
    collection-level file: one record per collection = {text, cl_labels, cl_ids}.
    text = collection name + description (+ dataset titles); gold = union of the CL
    cell types across the collection's datasets. Cached to disk."""
    out = DATA / "cellxgene" / "collections.json"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = json.loads(_get(f"{_CXG_API}/collections", timeout=120))
    records = []
    for c in (cols[:limit] if limit else cols):
        try:
            d = json.loads(_get(f"{_CXG_API}/collections/{c['collection_id']}", timeout=120))
        except Exception:  # noqa: BLE001 — skip a flaky collection, keep crawling
            continue
        labels: dict[str, str] = {}                    # CL id -> label, deduped across datasets
        titles = []
        for ds in d.get("datasets", []) or []:
            for ct in ds.get("cell_type") or []:
                cid = ct.get("ontology_term_id", "")
                if cid.startswith("CL:"):
                    labels[cid] = ct.get("label", "")
            if ds.get("title"):
                titles.append(ds["title"])
        if not labels:
            continue
        text = "\n".join(filter(None, [d.get("name"), d.get("description"),
                                       " ".join(dict.fromkeys(titles))]))
        records.append({"collection_id": c["collection_id"], "text": text,
                        "doi": d.get("doi"), "cl_ids": sorted(labels),
                        "cl_labels": [labels[i] for i in sorted(labels)]})
    out.write_text(json.dumps(records, indent=0))
    return out


_EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _abstract_for_doi(doi: str) -> str:
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    q = urllib.parse.quote(f'DOI:"{doi}"')
    try:
        data = json.loads(_get(f"{_EPMC}?query={q}&resultType=core&format=json", timeout=60))
    except Exception:  # noqa: BLE001
        return ""
    results = (data.get("resultList") or {}).get("result") or []
    return (results[0].get("abstractText") or "").strip() if results else ""


def ensure_cellxgene_abstracts() -> Path:
    """Enrich the cellxgene crawl with the studies' paper abstracts (via Europe PMC,
    by DOI). text = abstract (paper prose, like MINE's input), falling back to the study
    description when no abstract resolves. Gold (CL cell types) is unchanged."""
    out = DATA / "cellxgene" / "collections_abstracts.json"
    if out.exists():
        return out
    base = json.loads(ensure_cellxgene().read_text())
    enriched = []
    for r in base:
        ab = _abstract_for_doi(r["doi"]) if r.get("doi") else ""
        enriched.append({**r, "text": ab or r["text"], "has_abstract": bool(ab)})
    out.write_text(json.dumps(enriched, indent=0))
    return out


if __name__ == "__main__":
    for name, fn in (("biored", ensure_biored), ("jnlpba", ensure_jnlpba),
                     ("measeval", ensure_measeval), ("cellxgene", ensure_cellxgene),
                     ("mslr-cochrane", ensure_mslr)):
        try:
            p = fn()
            print(f"{name}: ready at {p}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: FAILED — {exc}")
