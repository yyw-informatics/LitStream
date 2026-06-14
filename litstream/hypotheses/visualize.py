"""Interpretable visuals. No UMAP or embedding dependencies in core.

* Per-hypothesis evidence trace — Mermaid ``.mmd`` text (solid = verified finding, dashed = generated
  candidate / proposed readout). Pure text, no dependency.
* Method pipeline — a static Mermaid diagram of the stages.
* Portfolio plot — matplotlib scatter (grounding × measurability) **only if matplotlib is importable**;
  otherwise a Markdown-table fallback is written and noted (matplotlib is not a core dependency).
* GraphML — a sanitized copy of the evidence graph (primitive attrs only) via networkx.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import io
from .config import HypothesisConfig
from .schema import Entity, HypothesisCandidate, HypothesisRunResult

_NODE_CLASS = {"PERTURBATION": "perturbation", "DISEASE": "disease",
               "CELL_TYPE": "cell", "CELL_STATE": "cell"}
_CLASSDEFS = [
    "    classDef context fill:#eeeeee,stroke:#666;",
    "    classDef perturbation fill:#dceeff,stroke:#333;",
    "    classDef disease fill:#ffe0e0,stroke:#333;",
    "    classDef readout fill:#e7f7e7,stroke:#333;",
    "    classDef cell fill:#fff3d6,stroke:#333;",
    "    classDef hypothesis fill:#f5e6ff,stroke:#333,stroke-width:2px;",
]


def _safe(text: str) -> str:
    return (text or "").replace('"', "'").replace("|", "/").replace("[", "(").replace("]", ")")


def _wrap(text: str, every: int = 6) -> str:
    words = _safe(text).split()
    out = []
    for i in range(0, len(words), every):
        out.append(" ".join(words[i:i + every]))
    return "<br/>".join(out)


def _ent_kind(ent: Entity) -> str:
    return {"PERTURBATION": "Perturbation", "DISEASE": "Disease", "GENE_RNA": "Readout (RNA)",
            "SURFACE_PROTEIN": "Readout (ADT)", "SIGNATURE": "Signature", "CELL_TYPE": "Cell type",
            "CELL_STATE": "Cell state", "CELL_FREQUENCY": "Cell frequency"}.get(ent.type, "Entity")


def _node_class(ent: Entity) -> str:
    return _NODE_CLASS.get(ent.type, "readout")


def candidate_trace_mermaid(c: HypothesisCandidate, result: HypothesisRunResult) -> str:
    edges_by_id = result.graph.graph.get("edges_by_id", {})
    lines = ["flowchart LR"]
    ctx = c.context
    ctx_label = "<br/>".join(filter(None, [
        " ".join(ctx.species), " ".join(ctx.tissue), " ".join(ctx.cell_type),
        " ".join(ctx.disease), " ".join(ctx.perturbation)])) or "context"
    lines.append(f'    Ctx["Context<br/>{_safe(ctx_label)}"]:::context')

    nodes: dict[str, str] = {}

    def node(ent: Entity) -> str:
        if ent.entity_id in nodes:
            return nodes[ent.entity_id]
        nid = f"N{len(nodes)}"
        nodes[ent.entity_id] = nid
        lines.append(f'    {nid}["{_ent_kind(ent)}<br/>{_safe(ent.canonical_name)}"]:::{_node_class(ent)}')
        return nid

    for eid in c.support_edge_ids:
        e = edges_by_id.get(eid)
        if not e:
            continue
        sn, tn = node(e.source_entity), node(e.target_entity)
        dirw = e.direction if e.direction != "unknown" else e.relation.lower()
        lines.append(f'    {sn} -->|"verified finding<br/>{_safe(e.paper_id)}<br/>'
                     f'{e.evidence_mode}<br/>{dirw}"| {tn}')

    lines.append(f'    H["Hypothesis<br/>{_wrap(c.claim)}"]:::hypothesis')
    an = nodes.get(c.anchor.entity_id)
    if an:
        lines.append(f'    {an} -. "generated candidate" .-> H')
    for r in c.readouts:
        rn = nodes.get(r.entity_id)
        if rn:
            lines.append(f'    {rn} -. "test readout" .-> H')
    lines.append("    Ctx --- H")
    lines += _CLASSDEFS
    return "\n".join(lines)


def method_pipeline_mermaid() -> str:
    return "\n".join([
        "flowchart TD",
        '    A["Evidence records<br/>(grounded, per paper)"] --> B["Frame extraction<br/>(deterministic rules)"]',
        '    B --> C["Grounding<br/>(verify vs source quote)"]',
        '    C --> D["Evidence graph<br/>(typed, signed edges)"]',
        '    D --> E["Candidate generation<br/>(4 motif templates)"]',
        '    E --> F["Hard filters<br/>(context, novelty, testability)"]',
        '    F --> G["Multi-axis ranking"]',
        '    G --> H["Report + traces"]',
    ])


def write_visuals(result: HypothesisRunResult, out_dir: str | Path,
                  config: HypothesisConfig) -> dict[str, str]:
    out = io.ensure_dir(out_dir)
    paths: dict[str, str] = {}

    if config.write_graphml:
        gp = _write_graphml(result, out / "hypothesis_graph.graphml")
        if gp:
            paths["graphml"] = str(gp)

    if not config.write_figures or config.visualization_backend == "none":
        return paths

    figs = io.ensure_dir(out / "figures")
    (figs / "method_pipeline.mmd").write_text(method_pipeline_mermaid())
    paths["method_pipeline"] = str(figs / "method_pipeline.mmd")

    for c in result.candidates:
        (figs / f"{c.hypothesis_id}_trace.mmd").write_text(candidate_trace_mermaid(c, result))

    portfolio = _write_portfolio(result, figs, config)
    if portfolio:
        paths["portfolio"] = str(portfolio)
    return paths


def _write_graphml(result: HypothesisRunResult, path: Path) -> Path | None:
    """Sanitized copy — GraphML only accepts primitive attribute types."""
    try:
        import networkx as nx
    except Exception:
        return None
    g = result.graph
    clean = nx.MultiDiGraph()
    for n, data in g.nodes(data=True):
        ent = data.get("entity")
        clean.add_node(n, label=getattr(ent, "canonical_name", str(n)),
                       type=getattr(ent, "type", ""), species=getattr(ent, "species", "") or "")
    for u, v, data in g.edges(data=True):
        e = data.get("edge")
        clean.add_edge(u, v,
                       relation=getattr(e, "relation", ""), direction=getattr(e, "direction", ""),
                       paper_id=getattr(e, "paper_id", ""), evidence_mode=getattr(e, "evidence_mode", ""),
                       grounding_score=float(getattr(e, "grounding_score", 0.0)))
    try:
        nx.write_graphml(clean, path)
        return path
    except Exception:
        return None


def _write_portfolio(result: HypothesisRunResult, figs: Path, config: HypothesisConfig) -> Path | None:
    cands = result.candidates
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        # matplotlib not installed (it is not a core dependency) — write an interpretable table instead
        md = ["# Hypothesis portfolio (matplotlib unavailable — table fallback)\n",
              "x≈grounding, y≈measurability, score=rank_score.\n",
              "| ID | motif | grounding | measurability | rank_score |",
              "|---|---|---:|---:|---:|"]
        for c in cands:
            s = c.scores
            md.append(f"| {c.hypothesis_id} | {c.motif} | {s.get('grounding_score', 0):.2f} | "
                      f"{s.get('measurability_score', 0):.2f} | {s.get('rank_score', 0):.3f} |")
        p = figs / "hypothesis_portfolio.md"
        p.write_text("\n".join(md))
        return p

    if not cands:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No candidates", ha="center", va="center")
        ax.set_axis_off()
        p = figs / "hypothesis_portfolio.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return p

    motifs = sorted({c.motif for c in cands})
    cmap = {m: plt.cm.tab10(i % 10) for i, m in enumerate(motifs)}
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in cands:
        s = c.scores
        ax.scatter(s.get("grounding_score", 0), s.get("measurability_score", 0),
                   s=40 + 260 * s.get("rank_score", 0), color=cmap[c.motif],
                   alpha=0.7, edgecolors="black", linewidths=0.5)
    for c in cands[:10]:
        s = c.scores
        ax.annotate(c.hypothesis_id, (s.get("grounding_score", 0), s.get("measurability_score", 0)),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("grounding score")
    ax.set_ylabel("measurability score")
    ax.set_title("Hypothesis portfolio (size = rank score)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[m],
                          markersize=8, label=m) for m in motifs]
    ax.legend(handles=handles, fontsize=7, loc="lower left")
    p = figs / "hypothesis_portfolio.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p
