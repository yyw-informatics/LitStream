"""Load SKILL.md files as prompt templates.

Each skill's SKILL.md body is read directly from disk and injected as explicit
instructions for each pipeline phase.
"""

from __future__ import annotations

from pathlib import Path

PHASE_SKILL = {
    "mine": "mine-paper",
    "synthesize": "synthesize-literature",
    "evaluate": "evaluate-fit",
    "design": "design-analysis",
}

PHASE_TASK = {
    "mine": ("Run in batch mode over EVERY PDF in projects/{project}/papers/ , screened "
             "against {context}. For each paper write projects/{project}/literature/"
             "<stem>_evidence.md where <stem> is the PDF filename without extension. The "
             "'_evidence.md' suffix is MANDATORY and exact — the next phase finds files by "
             "globbing *_evidence.md, so any other name is invisible to it. Example: "
             "papers/foo.pdf -> projects/{project}/literature/foo_evidence.md. A NOT-USEFUL "
             "paper still gets a foo_evidence.md with a one-paragraph verdict."),
    "synthesize": ("Read {context} and ALL projects/{project}/literature/*_evidence.md, then "
                   "write the cross-paper synthesis to projects/{project}/literature/"
                   "0_synthesis_literature.md."),
    "evaluate": ("Assess EVERY method under knowledge_base/ against {context}. Write each "
                 "method's projects/{project}/bioinformatics/<method>_fitness_assessment.md and "
                 "the aggregate projects/{project}/bioinformatics/fitness_summary.md."),
    "design": ("Read projects/{project}/literature/0_synthesis_literature.md, projects/{project}/"
               "bioinformatics/fitness_summary.md, {context}, and the knowledge_base/, then write "
               "the ordered plan to projects/{project}/analysis_plan.md."),
}


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].lstrip("\n")
    return text


def load_skill_body(skill_name: str, skills_dir: Path) -> str:
    path = Path(skills_dir) / skill_name / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(f"skill not found: {path}")
    return strip_frontmatter(path.read_text())


def render_phase_parts(phase: str, *, skills_dir: Path, project: str,
                       context_rel: str) -> tuple[str, str]:
    """Return (system, user): system carries the stable skill methodology
    (cacheable); user carries the per-run task plus execution rules."""
    skill = PHASE_SKILL[phase]
    body = load_skill_body(skill, skills_dir)
    task = PHASE_TASK[phase].format(project=project, context=context_rel)
    system = (
        f"You are executing the '{skill}' procedure for project '{project}'. The "
        f"<procedure> block is your authoritative method — follow it exactly.\n\n"
        f"<procedure name=\"{skill}\">\n{body}\n</procedure>"
    )
    user = (
        f"CONCRETE TASK: {task}\n\n"
        "EXECUTION RULES (unattended run):\n"
        "- Work fully autonomously. Do NOT ask interactive questions; pick sensible defaults.\n"
        "- Process all items sequentially in THIS session. Do NOT spawn sub-agents.\n"
        "- Use your file tools (read_file/Read, glob/Glob, grep/Grep, write_file/Write, "
        "edit_file/Edit) for all input/output.\n"
        "- Create output directories as needed. Stop once the output file(s) above exist."
    )
    return system, user


def render_phase_prompt(phase: str, *, skills_dir: Path, project: str, context_rel: str) -> str:
    system, user = render_phase_parts(phase, skills_dir=skills_dir, project=project,
                                      context_rel=context_rel)
    return system + "\n\n" + user
