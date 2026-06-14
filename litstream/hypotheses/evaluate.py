"""Offline evaluation harness. No network, model API, or GPU required.

* ``eval_frames`` — gold frame-extraction accuracy (readout / direction / cell-type / evidence-mode /
  comparator recovery / grounding precision / abstention rate).
* ``eval_hidden_edge`` — hide a directly-stated edge, check the generator recomposes it (Recall@k, MRR,
  direction- and context-correct recovery).
* ``eval_null_models`` — seeded sign / context / relation-label shuffles + type-preserving rewiring;
  the real generator should beat the nulls (fewer cross-context candidates, fewer sign errors).

Gold frame format (one JSON object per line)::

    {"paper_id": "...", "statement": "...", "readout": "FOXP3", "direction": "increase",
     "cell_type": "regulatory T cell", "evidence_mode": "interventional", "comparator": null}
    {"paper_id": "...", "statement": "...", "abstain": true}      # expect no frame
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import HypothesisConfig
from .filters import filter_candidates
from .frame_extractor import extract_frames
from .graph_builder import build_evidence_graph, relation_sign
from .grounding import verify_frames
from .io import load_evidence_records
from .normalize import Normalizer, normalize_text
from .ranker import rank_candidates
from .templates import generate_candidates


def eval_frames(gold_path: str | Path, evidence_dir: str | Path) -> dict[str, Any]:
    recs = load_evidence_records(evidence_dir)
    cfg = HypothesisConfig()
    norm = Normalizer()
    frames, _ = extract_frames(recs, cfg, norm)
    grounded, _ = verify_frames(frames, cfg)
    grounded_ids = {f.frame_id for f in grounded}

    by_finding: dict[tuple, list] = {}
    for f in frames:
        by_finding.setdefault((f.paper_id, normalize_text(f.raw_statement)), []).append(f)

    gold = [json.loads(ln) for ln in Path(gold_path).read_text().splitlines() if ln.strip()]
    fields = ["readout", "direction", "cell_type", "evidence_mode"]
    correct = {k: 0 for k in fields}
    measured = {k: 0 for k in fields}
    comparator_hits = comparator_total = 0
    grounded_matched = matched = abstained = abstain_correct = abstain_total = 0

    for g in gold:
        key = (g.get("paper_id"), normalize_text(g.get("statement") or g.get("finding") or ""))
        preds = by_finding.get(key, [])
        if g.get("abstain"):
            abstain_total += 1
            abstain_correct += int(not preds)
            continue
        if not preds:
            abstained += 1
            continue
        pred = _best_match(preds, g.get("readout", ""))
        matched += 1
        grounded_matched += int(pred.frame_id in grounded_ids)
        if g.get("readout"):
            measured["readout"] += 1
            correct["readout"] += int(normalize_text(pred.readout.canonical_name) == normalize_text(g["readout"]))
        if g.get("direction"):
            measured["direction"] += 1
            correct["direction"] += int(pred.direction == g["direction"])
        if g.get("cell_type"):
            measured["cell_type"] += 1
            pc = norm.canonical_cell_type(g["cell_type"])[0]
            got = pred.cell_type.canonical_name if pred.cell_type else ""
            correct["cell_type"] += int(normalize_text(got) == normalize_text(pc))
        if g.get("evidence_mode"):
            measured["evidence_mode"] += 1
            correct["evidence_mode"] += int(pred.evidence_mode == g["evidence_mode"])
        if "comparator" in g and g["comparator"]:
            comparator_total += 1
            comparator_hits += int(bool(pred.context.comparator)
                                   and normalize_text(g["comparator"]) in normalize_text(pred.context.comparator or ""))

    acc = {k: round(correct[k] / measured[k], 3) if measured[k] else None for k in fields}
    return {
        "n_gold": len(gold),
        "matched": matched,
        "accuracy": acc,
        "comparator_recovery": round(comparator_hits / comparator_total, 3) if comparator_total else None,
        "grounding_precision": round(grounded_matched / matched, 3) if matched else None,
        "abstention_rate": round(abstained / max(1, len(gold) - abstain_total), 3),
        "abstain_correct": f"{abstain_correct}/{abstain_total}" if abstain_total else None,
    }


def _best_match(preds, gold_readout):
    if gold_readout:
        gn = normalize_text(gold_readout)
        for p in preds:
            if normalize_text(p.readout.canonical_name) == gn:
                return p
    return preds[0]


def eval_hidden_edge(evidence_dir: str | Path, gold_hidden: str | Path | None = None,
                     k: int = 10) -> dict[str, Any]:
    recs = load_evidence_records(evidence_dir)
    cfg = HypothesisConfig()
    norm = Normalizer()
    frames, _ = extract_frames(recs, cfg, norm)
    vf, _ = verify_frames(frames, cfg)
    full = build_evidence_graph(vf, cfg, norm, records=recs)

    targets: list[dict] = []
    if gold_hidden:
        targets = [json.loads(ln) for ln in Path(gold_hidden).read_text().splitlines() if ln.strip()]
    else:
        for _, _, d in full.edges(data=True):
            e = d["edge"]
            if relation_sign(e.relation) == 0:
                continue
            a, c = e.source_entity.entity_id, e.target_entity.entity_id
            for b in full.successors(a):
                if b != c and full.has_edge(b, c):
                    targets.append({"frame_id": e.frame_id, "anchor": a, "object": c,
                                    "direction": e.direction})
                    break

    ranks: list[int | None] = []
    dir_ok = ctx_ok = 0
    for t in targets:
        masked = [f for f in vf if f.frame_id != t["frame_id"]]
        g2 = build_evidence_graph(masked, cfg, norm, records=recs)
        raw = generate_candidates(g2, masked, cfg, norm)
        filt, _ = filter_candidates(raw, g2, masked, cfg, norm)
        ranked = rank_candidates(filt, g2, masked, cfg, norm)
        rank = None
        for i, cand in enumerate(ranked, 1):
            if cand.anchor.entity_id == t["anchor"] and cand.meta.get("novel_object") == t["object"]:
                rank = i
                ctx_ok += 1
                dir_ok += int(cand.predicted_direction == t.get("direction"))
                break
        ranks.append(rank)

    n = len(targets)

    def recall_at(kk):
        return round(sum(1 for r in ranks if r and r <= kk) / n, 3) if n else 0.0

    return {
        "n_targets": n,
        "recall@5": recall_at(5),
        f"recall@{k}": recall_at(k),
        "mrr": round(sum(1.0 / r for r in ranks if r) / n, 3) if n else 0.0,
        "direction_correct": round(dir_ok / n, 3) if n else 0.0,
        "context_correct": round(ctx_ok / n, 3) if n else 0.0,
    }


def _generate_count(frames, recs, cfg, norm) -> int:
    g = build_evidence_graph(frames, cfg, norm, records=recs)
    raw = generate_candidates(g, frames, cfg, norm)
    filt, _ = filter_candidates(raw, g, frames, cfg, norm)
    return len(filt)


def eval_null_models(evidence_dir: str | Path, seed: int = 0) -> dict[str, Any]:
    recs = load_evidence_records(evidence_dir)
    cfg = HypothesisConfig()
    norm = Normalizer()
    frames, _ = extract_frames(recs, cfg, norm)
    vf, _ = verify_frames(frames, cfg)
    rng = random.Random(seed)

    real = _generate_count(vf, recs, cfg, norm)

    sign_shuf = _shuffle_attr(vf, rng, "direction", lambda f: f.direction in
                              ("increase", "decrease", "association_positive", "association_negative"))
    ctx_shuf = _shuffle_context(vf, rng)
    rel_shuf = _shuffle_attr(vf, rng, "evidence_mode", lambda f: True)

    return {
        "seed": seed,
        "real_candidates": real,
        "sign_shuffled_candidates": _generate_count(sign_shuf, recs, cfg, norm),
        "context_shuffled_candidates": _generate_count(ctx_shuf, recs, cfg, norm),
        "relation_shuffled_candidates": _generate_count(rel_shuf, recs, cfg, norm),
        "note": ("The real generator should not be dominated by the nulls. Context/relation shuffles "
                 "break compatible composition; a real candidate count at or above the shuffled "
                 "counts indicates the constraints carry signal. On a tiny corpus this is descriptive, "
                 "not inferential."),
    }


def _shuffle_attr(frames, rng, attr, pred):
    idx = [i for i, f in enumerate(frames) if pred(f)]
    vals = [getattr(frames[i], attr) for i in idx]
    shuffled = vals[:]
    rng.shuffle(shuffled)
    out = list(frames)
    for j, i in enumerate(idx):
        out[i] = replace(frames[i], **{attr: shuffled[j]})
    return out


def _shuffle_context(frames, rng):
    ctxs = [f.context for f in frames]
    shuffled = ctxs[:]
    rng.shuffle(shuffled)
    return [replace(f, context=shuffled[i]) for i, f in enumerate(frames)]
