"""make_file_tools — five LangChain StructuredTools confined to a cwd.

Verifies the wiring: five tools, the right names, real read/write/edit/glob/grep
against a tmp cwd, and that logical-normpath confinement blocks '../' escapes.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from litstream_lg.tools import make_file_tools


def _by_name(cwd):
    return {t.name: t for t in make_file_tools(str(cwd))}


def test_returns_five_structured_tools(tmp_path):
    tools = make_file_tools(str(tmp_path))
    assert len(tools) == 5
    assert all(isinstance(t, StructuredTool) for t in tools)
    assert {t.name for t in tools} == {"read_file", "write_file", "edit_file", "glob", "grep"}


def test_write_then_read_roundtrip(tmp_path):
    tools = _by_name(tmp_path)
    msg = tools["write_file"].invoke({"path": "sub/note.md", "content": "hello world"})
    assert "wrote" in msg
    assert (tmp_path / "sub/note.md").read_text() == "hello world"
    # read it back through the tool
    assert tools["read_file"].invoke({"path": "sub/note.md"}) == "hello world"


def test_edit_replaces_first_occurrence(tmp_path):
    tools = _by_name(tmp_path)
    tools["write_file"].invoke({"path": "a.txt", "content": "foo bar foo"})
    out = tools["edit_file"].invoke({"path": "a.txt", "old_string": "foo", "new_string": "baz"})
    assert "edited" in out
    assert (tmp_path / "a.txt").read_text() == "baz bar foo"   # only the first


def test_glob_finds_evidence_files(tmp_path):
    tools = _by_name(tmp_path)
    tools["write_file"].invoke({"path": "projects/x/literature/p1_evidence.md", "content": "e"})
    tools["write_file"].invoke({"path": "projects/x/literature/notes.txt", "content": "n"})
    out = tools["glob"].invoke({"pattern": "projects/x/literature/*_evidence.md"})
    assert "p1_evidence.md" in out
    assert "notes.txt" not in out


def test_grep_returns_file_line_matches(tmp_path):
    tools = _by_name(tmp_path)
    tools["write_file"].invoke({"path": "doc.md", "content": "alpha\nbeta needle\ngamma"})
    out = tools["grep"].invoke({"pattern": "needle"})
    assert "doc.md:2" in out
    assert "needle" in out


def test_read_missing_file_reports_error(tmp_path):
    tools = _by_name(tmp_path)
    out = tools["read_file"].invoke({"path": "nope.md"})
    assert out.startswith("ERROR")


def test_confinement_blocks_parent_escape_on_write(tmp_path):
    """A '../escape' write must be blocked by logical-normpath confinement.

    The StructuredTool surfaces the underlying ValueError; either way the escape file
    must not be created outside cwd.
    """
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "secret.txt"
    tools = _by_name(work)
    try:
        tools["write_file"].invoke({"path": "../secret.txt", "content": "pwned"})
    except ValueError as exc:
        assert "escapes working dir" in str(exc)
    assert not outside.exists()       # the key invariant: nothing written outside cwd


def test_confinement_blocks_parent_escape_on_read(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "secret.txt").write_text("classified")
    tools = _by_name(work)
    try:
        out = tools["read_file"].invoke({"path": "../secret.txt"})
    except ValueError as exc:
        assert "escapes working dir" in str(exc)
    else:
        assert "classified" not in out
