"""File tools for the LangGraph agent.

Wraps the shared confined file ops as `StructuredTool`s; LangChain normalizes the tool
schemas across providers. The path-confinement (logical-normpath) and PDF-text
extraction live in the underlying `litstream.fileops` functions.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from litstream.fileops import _read_file, _write_file, _edit_file, _glob, _grep


def make_file_tools(cwd: str) -> list[StructuredTool]:
    """Return the five file tools bound to `cwd`.

    A fresh list per run binds the working directory as a closure variable, keeping it
    out of the tool args so the agent cannot reach outside it.
    """

    def read_file(path: str) -> str:
        """Read a file's contents; PDFs are auto-extracted to text.

        Args:
            path: Path to the file to read, relative to the working directory.
        """
        return _read_file(cwd, path)

    def write_file(path: str, content: str) -> str:
        """Write content to a file, creating parent directories as needed.

        Args:
            path: Destination path, relative to the working directory.
            content: Full text to write to the file.
        """
        return _write_file(cwd, path, content)

    def edit_file(path: str, old_string: str, new_string: str) -> str:
        """Replace the first occurrence of old_string with new_string in a file.

        Args:
            path: Path to the file to edit, relative to the working directory.
            old_string: Exact text to find and replace.
            new_string: Replacement text.
        """
        return _edit_file(cwd, path, old_string, new_string)

    def glob(pattern: str) -> str:
        """List files matching a glob pattern, recursively.

        Args:
            pattern: Glob pattern, e.g. 'projects/x/literature/*_evidence.md'.
        """
        return _glob(cwd, pattern)

    def grep(pattern: str, path: str = ".") -> str:
        """Search file contents for a regex; returns file:line matches.

        Args:
            pattern: Regular expression to search for.
            path: Directory or file to search within (default '.').
        """
        return _grep(cwd, pattern, path)

    return [StructuredTool.from_function(fn, parse_docstring=True) for fn in
            (read_file, write_file, edit_file, glob, grep)]
