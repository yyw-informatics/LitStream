"""Reading evidence records + small write helpers. Never mutates input files."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def load_evidence_records(evidence_dir: str | Path) -> list[dict]:
    """Load per-paper evidence records from a directory of ``*.json`` files.

    Each file may hold a single record (dict) or a list of records, so smoke fixtures and the live
    ``projects/<p>/literature/*_evidence.json`` layout both load. When both
    ``<stem>_evidence.json`` and ``<stem>_evidence.regrounded.json`` exist, the regrounded sidecar
    wins (its source_quotes are the verified passages). ``paper_id`` falls back to the file stem."""
    d = Path(evidence_dir)
    if d.is_file():
        return _records_from_file(d)
    if not d.is_dir():
        return []

    files = sorted(d.glob("*.json"))
    regrounded_stems = {
        f.name[: -len("_evidence.regrounded.json")]
        for f in files if f.name.endswith("_evidence.regrounded.json")
    }
    out: list[dict] = []
    seen_ids: set[str] = set()
    for f in files:
        if f.name.endswith("_evidence.json"):
            stem = f.name[: -len("_evidence.json")]
            if stem in regrounded_stems:
                continue
        for rec in _records_from_file(f):
            pid = rec.get("paper_id") or _stem_id(f.name)
            rec["paper_id"] = pid
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            out.append(rec)
    return out


def _records_from_file(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    if isinstance(data, list):
        recs = [r for r in data if isinstance(r, dict)]
    elif isinstance(data, dict):
        recs = [data]
    else:
        return []
    for i, rec in enumerate(recs):
        rec.setdefault("paper_id", _stem_id(path.name) if len(recs) == 1 else f"{_stem_id(path.name)}#{i}")
    return recs


def _stem_id(filename: str) -> str:
    for suffix in ("_evidence.regrounded.json", "_evidence.json", ".json"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def load_synthesis(path: str | Path | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses (Entity/BioContext/...) to plain JSON-able structures."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    return obj


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.write_text(json.dumps(to_jsonable(obj), indent=2))
    return p


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    p = Path(path)
    with p.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(to_jsonable(row)) + "\n")
    return p


def write_csv(path: str | Path, rows: list[dict], columns: list[str]) -> Path:
    p = Path(path)
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: _csv_cell(row.get(c, "")) for c in columns})
    return p


def _csv_cell(v: Any) -> str:
    if isinstance(v, (list, tuple, set)):
        return "; ".join(str(x) for x in v)
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)
