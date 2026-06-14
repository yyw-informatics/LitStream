"""Post-synthesis citation audit for the live agent.

After the synthesize phase writes `0_synthesis_literature.md`, this runs the A2 citation
checker (`litstream.eval.citation_check`) over it: each cited `[Author Year]` claim is
resolved via Appendix A to its `*_evidence.md` and an entailment verifier asks *does the
source support it?*. The result is a `0_synthesis_audit.md` overlay listing only the claims
that fail — unsupported / partial / contradicted — plus the headline precision/recall/coverage.

FLAG-ONLY and non-destructive: the synthesis prose is never edited (auto-drop/repair is
deliberately out of scope). Enable per routine with:

    audit_synthesis: true
    synthesis_audit:
      verifier: overlap
      minicheck_model: flan-t5-large

The `minicheck` verifier needs the `grounding` extra; `overlap` is dependency-free. The eval
module is lazy-imported so those deps stay opt-in.
"""

from __future__ import annotations

from pathlib import Path

_FAILING = ("Partial", "Contradictory")


def audit_synthesis_output(project_dir, project: str, *, verifier: str = "overlap",
                           minicheck_model: str = "flan-t5-large", _verifier=None) -> dict:
    """Run the A2 citation check over the synthesis and write `0_synthesis_audit.md`. Returns a
    summary dict. Lazy-imports the eval module so the grounding deps stay opt-in. The synthesis
    markdown is read, never written."""
    from litstream.eval.citation_check import run, make_verifier

    pdir = Path(project_dir)
    v = _verifier if _verifier is not None else make_verifier(verifier, minicheck_model)
    pairings, m = run(project, pdir, v)
    if not m:
        return {"audited": False, "note": "no synthesis found"}

    flagged = [p for p in pairings
               if p.label in _FAILING or (p.label == "Irrelevant" and p.evidence_file)]
    report = _write_report(pdir, project, m, flagged, verifier)
    return {"audited": True, "n_claims": m["n_claims"], "flagged": len(flagged),
            "citation_precision": m["citation_precision"], "report": str(report)}


def _write_report(pdir: Path, project: str, m: dict, flagged: list, verifier: str) -> Path:
    pct = lambda v: "—" if v is None else f"{v:.1%}"
    lines = [
        f"# Synthesis citation audit — {project}", "",
        f"verifier `{verifier}` · claims parsed **{m['n_claims']}** · scored claim↔source pairs "
        f"**{m['n_scored_pairs']}** · flagged **{len(flagged)}**", "",
        "Flag-only overlay: the synthesis is NOT edited. Each row is a cited claim whose source does "
        "not fully support it — review against the evidence file before trusting the claim.", "",
        f"citation precision **{pct(m['citation_precision'])}** · claim recall "
        f"**{pct(m['claim_recall'])}** · coverage **{pct(m['coverage'])}** "
        f"({m['evidence_used']}/{m['evidence_total']})", "",
    ]
    if flagged:
        lines += ["| Claim | Citation | Evidence file | Verdict |", "|---|---|---|---|"]
        lines += [f"| {p.claim} | {p.citation} | `{p.evidence_file}` | {p.label} |"
                  for p in flagged]
    else:
        lines.append("No unsupported, partial, or contradicted claims found.")
    lines += [
        "",
        "> **Scope.** This measures *the cited source supports the claim*, not world-truth, and the "
        "checker itself is unvalidated until a human attribution key measures its own precision/"
        "recall (see `litstream/eval/attribution_key.example.jsonl`). The entailment verifier is "
        "weak on numeric/direction/species claims — treat flags as candidates for human review.", "",
    ]
    path = pdir / f"projects/{project}/literature/0_synthesis_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
