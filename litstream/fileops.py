"""Provider-agnostic file tools for the agentic loop.

A read/write/edit/glob/grep tool set implemented in Python and exposed in both
Anthropic and OpenAI tool-schema formats, so any provider's function-calling can
drive it. read_file auto-extracts PDF text. All paths are confined to the run's cwd.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from litstream_evidence.pdf_text import extract_text


def _safe(cwd: str | Path, path: str) -> Path:
    base = Path(cwd).resolve()
    # Confine logically (normpath, not resolve) so a workspace symlink pointing into
    # the global paper library is allowed, while '../' escapes are still blocked.
    p = Path(os.path.normpath(base / path))
    if base != p and base not in p.parents:
        raise ValueError(f"path escapes working dir: {path}")
    return p


def _read_file(cwd, path, max_chars: int = 60_000):
    p = _safe(cwd, path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    if p.suffix.lower() == ".pdf":
        return extract_text(p, max_chars=max_chars)
    return p.read_text(errors="replace")[:max_chars]


def _write_file(cwd, path, content):
    p = _safe(cwd, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {path}"


def _edit_file(cwd, path, old_string, new_string):
    p = _safe(cwd, path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    s = p.read_text()
    if old_string not in s:
        return "ERROR: old_string not found in file"
    p.write_text(s.replace(old_string, new_string, 1))
    return "edited"


def _glob(cwd, pattern):
    base = Path(cwd).resolve()
    out = [str(p.relative_to(base)) for p in base.rglob("*")
           if fnmatch.fnmatch(str(p.relative_to(base)), pattern)]
    return "\n".join(out[:300]) or "(no matches)"


def _grep(cwd, pattern, path="."):
    base = _safe(cwd, path)
    rx = re.compile(pattern)
    hits: list[str] = []
    files = [base] if base.is_file() else base.rglob("*")
    for f in files:
        if not f.is_file() or f.suffix.lower() == ".pdf":
            continue
        try:
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f.relative_to(Path(cwd).resolve())}:{i}: {line.strip()[:160]}")
                    if len(hits) >= 200:
                        return "\n".join(hits)
        except Exception:
            continue
    return "\n".join(hits) or "(no matches)"


# name -> (fn, description, JSON-schema properties, required tool args)
TOOLS: dict = {
    "read_file": (_read_file, "Read a file's contents. PDFs are auto-extracted to text.",
                  {"path": {"type": "string", "description": "path relative to working dir"}}, ["path"]),
    "write_file": (_write_file, "Write content to a file, creating parent dirs.",
                   {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    "edit_file": (_edit_file, "Replace the first occurrence of old_string with new_string in a file.",
                  {"path": {"type": "string"}, "old_string": {"type": "string"},
                   "new_string": {"type": "string"}}, ["path", "old_string", "new_string"]),
    "glob": (_glob, "List files matching a glob pattern (recursive).",
             {"pattern": {"type": "string", "description": "e.g. projects/x/literature/*_evidence.md"}}, ["pattern"]),
    "grep": (_grep, "Search file contents for a regex; returns file:line matches.",
             {"pattern": {"type": "string"}, "path": {"type": "string", "description": "dir or file (default .)"}}, ["pattern"]),
}


def anthropic_schema() -> list[dict]:
    return [{"name": n, "description": d,
             "input_schema": {"type": "object", "properties": props, "required": req}}
            for n, (_, d, props, req) in TOOLS.items()]


def openai_schema() -> list[dict]:
    return [{"type": "function", "function": {
                "name": n, "description": d,
                "parameters": {"type": "object", "properties": props, "required": req}}}
            for n, (_, d, props, req) in TOOLS.items()]


def execute(name: str, cwd, args: dict) -> str:
    if name not in TOOLS:
        return f"ERROR: unknown tool {name}"
    fn = TOOLS[name][0]
    try:
        return str(fn(cwd, **args))
    except TypeError as e:
        return f"ERROR: bad arguments for {name}: {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
