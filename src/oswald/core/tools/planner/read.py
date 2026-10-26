"""Read-only inspection tools exclusively granted to the Planner agent."""

from __future__ import annotations

from pathlib import Path

from oswald.core.codeindex.indexer import CodeIndexer
from oswald.core.llm.types import ToolDefinition
from oswald.core.tools.coder.read import (
    list_files as _list_files,
)
from oswald.core.tools.coder.read import (
    read_file_range as _read_file_range,
)
from oswald.core.tools.coder.read import (
    read_symbol as _read_symbol,
)
from oswald.core.tools.coder.read import (
    search_code as _search_code,
)

PLANNER_TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="read_file_range",
        description="Read lines from a workspace file (1-indexed). Useful for inspecting code slices.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file in workspace",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed, inclusive)",
                    "default": 1,
                },
                "end_line": {
                    "type": "integer",
                    "description": "Ending line number (inclusive). Omit to read up to budget ceiling.",
                },
            },
            "required": ["file_path"],
        },
    ),
    ToolDefinition(
        name="search_code",
        description="Search for a literal string or regex pattern across workspace files.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "String or regex pattern to search for",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g. '*.py')",
                },
                "is_regex": {
                    "type": "boolean",
                    "description": "Whether query is a regular expression",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="list_files",
        description="List files inside a workspace directory.",
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Relative directory path (defaults to root '.')",
                    "default": ".",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively into subdirectories",
                    "default": True,
                },
            },
        },
    ),
    ToolDefinition(
        name="read_symbol",
        description="Look up function, class, or method signature and definition across the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "symbol_name": {
                    "type": "string",
                    "description": "Name of the symbol (function, method, class) to look up",
                },
            },
            "required": ["symbol_name"],
        },
    ),
]


def read_file_range(
    workspace_root: Path,
    file_path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """Read line slice from workspace file (strictly read-only)."""
    return _read_file_range(
        workspace_root=workspace_root,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
    )


def search_code(
    workspace_root: Path,
    query: str,
    file_pattern: str | None = None,
    is_regex: bool = False,
) -> str:
    """Search code across workspace (strictly read-only)."""
    return _search_code(
        workspace_root=workspace_root,
        query=query,
        file_pattern=file_pattern,
        is_regex=is_regex,
    )


def list_files(
    workspace_root: Path,
    directory: str = ".",
    recursive: bool = True,
) -> str:
    """List directory contents in workspace (strictly read-only)."""
    return _list_files(
        workspace_root=workspace_root,
        directory=directory,
        recursive=recursive,
    )


def read_symbol(
    workspace_root: Path,
    symbol_name: str,
    indexer: CodeIndexer | None = None,
) -> str:
    """Read symbol definition from workspace or code indexer (strictly read-only)."""
    return _read_symbol(
        workspace_root=workspace_root,
        symbol_name=symbol_name,
        indexer=indexer,
    )
