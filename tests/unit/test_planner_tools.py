"""Unit tests for Planner Agent tools (read-only isolation and correctness)."""

from pathlib import Path

import pytest

from oswald.core.tools.planner import (
    PLANNER_TOOL_DEFINITIONS,
    list_files,
    read_file_range,
    search_code,
)


@pytest.fixture
def sample_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "planner_ws"
    ws.mkdir()
    (ws / "main.py").write_text("def hello() -> str:\n    return 'world'\n", encoding="utf-8")
    (ws / "utils.py").write_text("MAGIC_NUMBER = 42\n", encoding="utf-8")
    sub = ws / "subpkg"
    sub.mkdir()
    (sub / "mod.py").write_text("class Sample:\n    pass\n", encoding="utf-8")
    return ws


def test_planner_tool_definitions_has_only_read_tools() -> None:
    tool_names = {t.name for t in PLANNER_TOOL_DEFINITIONS}
    assert tool_names == {"read_file_range", "search_code", "list_files", "read_symbol"}
    assert "edit_file" not in tool_names
    assert "write_test_file" not in tool_names
    assert "run_test_command" not in tool_names


def test_planner_read_file_range(sample_workspace: Path) -> None:
    content = read_file_range(sample_workspace, "main.py", start_line=1, end_line=1)
    assert "def hello()" in content


def test_planner_list_files(sample_workspace: Path) -> None:
    output = list_files(sample_workspace, ".", recursive=True)
    assert "main.py" in output
    assert "utils.py" in output
    assert "mod.py" in output


def test_planner_search_code(sample_workspace: Path) -> None:
    output = search_code(sample_workspace, "MAGIC_NUMBER")
    assert "MAGIC_NUMBER = 42" in output
