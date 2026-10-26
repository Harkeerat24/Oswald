"""Unit tests for TesterAgent and its dedicated test tools."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from oswald.core.agents.coder import InMemoryEventSink
from oswald.core.agents.tester import TesterAgent, TesterInput
from oswald.core.artifacts.models import CodeChanges, PlanSpec
from oswald.core.llm.adapters.fake import FakeLLMClient
from oswald.core.llm.exceptions import LLMError
from oswald.core.llm.types import LLMResponse, StopReason, ToolCall
from oswald.core.sandbox.exceptions import SandboxError, SandboxTimeoutError
from oswald.core.sandbox.fake import FakeSandboxRuntime
from oswald.core.sandbox.models import ExecResult, SandboxConfig
from oswald.core.tools.tester import (
    WorkspaceSnapshot,
    execute_test_command,
    is_test_related_path,
    write_test_file,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "app.py").write_text("def greet() -> str:\n    return 'hello'\n", encoding="utf-8")
    return root


def _input(
    task_id: str = "tester-1",
    target_files: list[str] | None = None,
    modified_files: list[str] | None = None,
) -> TesterInput:
    targets = target_files if target_files is not None else ["app.py"]
    mods = modified_files if modified_files is not None else ["app.py"]
    return TesterInput(
        task_id=task_id,
        spec=PlanSpec(
            task_id=task_id,
            summary="Test greeting",
            target_files=targets,
            content="# Spec",
        ),
        changes=CodeChanges(
            task_id=task_id,
            summary="Greeting added",
            modified_files=mods,
            diff="--- a/app.py\n+++ b/app.py\n+def greet(): ...",
            content="# Changes",
            test_attention=["Verify greeting"],
        ),
    )


# --- write_test_file and isolation tests ---


def test_write_test_file_allows_test_conventions_and_blocks_source(workspace: Path) -> None:
    created = write_test_file(workspace, "qa/GreetingTest.java", "", "class GreetingTest {}\n")
    assert created.success is True
    assert (workspace / "qa" / "GreetingTest.java").exists()

    blocked = write_test_file(workspace, "app.py", "", "bad")
    assert blocked.success is False
    assert "PermissionDenied" in blocked.message


def test_write_test_file_allows_declared_custom_test_root(workspace: Path) -> None:
    result = write_test_file(
        workspace,
        "qa/greeting.java",
        "",
        "class Greeting {}\n",
        test_root="qa",
    )
    assert result.success is True
    assert (workspace / "qa" / "greeting.java").exists()


def test_write_test_file_blocks_source_roots_and_forbidden_paths(workspace: Path) -> None:
    # Cannot declare 'src' or '.' as test root
    bad_root = write_test_file(
        workspace,
        "src/test_foo.py",
        "",
        "print(1)",
        test_root="src",
    )
    assert bad_root.success is False
    assert "application source root" in bad_root.message

    # Cannot touch files targeted by spec / changes
    blocked_source = write_test_file(
        workspace,
        "app.py",
        "return 'hello'",
        "return 'bye'",
        forbidden_paths={"app.py"},
    )
    assert blocked_source.success is False
    assert "application source file" in blocked_source.message


def test_write_test_file_blocks_protected_files(workspace: Path) -> None:
    (workspace / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    blocked = write_test_file(workspace, "pyproject.toml", "name='test'", "name='evil'")
    assert blocked.success is False
    assert "protected project file" in blocked.message


def test_write_test_file_surgical_edits_and_line_endings(workspace: Path) -> None:
    test_file = workspace / "tests" / "test_app.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_one():\r\n    assert 1 == 1\r\n", encoding="utf-8")

    # Surgical replacement with LF replacing CRLF content
    res = write_test_file(
        workspace,
        "tests/test_app.py",
        "assert 1 == 1",
        "assert 1 == 1 and True",
    )
    assert res.success is True
    raw_content = test_file.read_bytes().decode("utf-8")
    assert "assert 1 == 1 and True" in raw_content
    assert "\r\n" in raw_content  # Preserved original CRLF line endings

    # Ambiguous match check
    test_file.write_text("dup\ndup\n", encoding="utf-8")
    ambiguous = write_test_file(workspace, "tests/test_app.py", "dup", "single")
    assert ambiguous.success is False
    assert "matches 2 occurrences" in ambiguous.message

    # Already exists on creation without overwrite
    exists_err = write_test_file(workspace, "tests/test_app.py", "", "new content")
    assert exists_err.success is False
    assert "already exists" in exists_err.message

    # Overwrite allowed when overwrite=True
    overwrite_res = write_test_file(
        workspace, "tests/test_app.py", "", "overwritten content", overwrite=True
    )
    assert overwrite_res.success is True
    assert (workspace / "tests" / "test_app.py").read_text(
        encoding="utf-8"
    ) == "overwritten content"


def test_write_test_file_blocks_path_traversal(workspace: Path) -> None:
    traversal = write_test_file(workspace, "../outside_test.py", "", "bad")
    assert traversal.success is False
    assert "SecurityError" in traversal.message


# --- execute_test_command and Snapshot tests ---


@pytest.mark.asyncio
async def test_execute_test_command_variations(workspace: Path) -> None:
    sandbox = FakeSandboxRuntime(SandboxConfig(task_id="t1", workspace_path=workspace))
    await sandbox.start()

    # Empty command
    empty = await execute_test_command(sandbox, [])
    assert empty.exit_code == 1
    assert "must contain at least one argument" in empty.output

    # Normal command
    sandbox.queue_response(stdout="test output", exit_code=0)
    normal = await execute_test_command(sandbox, ["pytest", "tests"])
    assert normal.exit_code == 0
    assert "test output" in normal.output

    # Mock timeout
    sandbox_mock = MagicMock()
    sandbox_mock.exec_command = MagicMock(side_effect=SandboxTimeoutError("Timed out"))
    timeout = await execute_test_command(sandbox_mock, ["pytest"], timeout_seconds=10.0)
    assert timeout.exit_code == 124
    assert "Command timed out" in timeout.output

    # Mock execution error
    sandbox_mock.exec_command = MagicMock(side_effect=SandboxError("Docker failed"))
    err = await execute_test_command(sandbox_mock, ["pytest"])
    assert err.exit_code == 1
    assert "Sandbox execution error" in err.output


def test_snapshot_restores_non_test_mutations(workspace: Path) -> None:
    snapshot = WorkspaceSnapshot.capture(workspace)
    (workspace / "app.py").write_text("changed", encoding="utf-8")
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_app.py").write_text("def test_ok(): pass\n", encoding="utf-8")

    affected = snapshot.restore_non_test_changes(workspace)

    assert affected == ["app.py"]
    assert "return 'hello'" in (workspace / "app.py").read_text(encoding="utf-8")
    assert (workspace / "tests" / "test_app.py").exists()


def test_snapshot_allows_coverage_and_test_artifacts(workspace: Path) -> None:
    snapshot = WorkspaceSnapshot.capture(workspace)
    # Generate coverage and junit files
    (workspace / ".coverage").write_text("!coverage data", encoding="utf-8")
    (workspace / "junit.xml").write_text("<xml/>", encoding="utf-8")

    affected = snapshot.restore_non_test_changes(workspace)
    assert affected == []
    assert (workspace / ".coverage").exists()
    assert (workspace / "junit.xml").exists()


def test_is_test_related_path_multi_platform_and_languages() -> None:
    # Python
    assert is_test_related_path("tests/unit/test_app.py") is True
    assert is_test_related_path("test/app_test.py") is True
    assert is_test_related_path("conftest.py") is True
    assert is_test_related_path("src/app.py") is False

    # TypeScript / JavaScript
    assert is_test_related_path("src/components/Button.test.tsx") is True
    assert is_test_related_path("src/__tests__/app.spec.js") is True
    assert is_test_related_path("src/components/Button.tsx") is False

    # Go
    assert is_test_related_path("pkg/server_test.go") is True
    assert is_test_related_path("pkg/server.go") is False

    # Java / Kotlin / Scala / C#
    assert is_test_related_path("qa/GreetingTest.java") is True
    assert is_test_related_path("src/main/Greeting.java") is False
    assert is_test_related_path("src/OrderTests.cs") is True
    assert is_test_related_path("src/OrderService.cs") is False
    assert is_test_related_path("src/ParserSpec.scala") is True

    # Rust
    assert is_test_related_path("tests/integration_test.rs") is True
    assert is_test_related_path("src/lib_test.rs") is True
    assert is_test_related_path("src/main.rs") is False

    # C / C++
    assert is_test_related_path("test/parser_test.cpp") is True
    assert is_test_related_path("src/test_crypto.c") is True
    assert is_test_related_path("src/parser.cpp") is False
    assert is_test_related_path("src/crypto.c") is False

    # Ruby / PHP
    assert is_test_related_path("spec/user_spec.rb") is True
    assert is_test_related_path("tests/UserTest.php") is True
    assert is_test_related_path("app/models/user.rb") is False

    # Swift / Dart / Zig / Elixir / Julia
    assert is_test_related_path("src/AppTests.swift") is True
    assert is_test_related_path("src/client_test.dart") is True
    assert is_test_related_path("src/alloc_test.zig") is True
    assert is_test_related_path("test/algo_test.exs") is True
    assert is_test_related_path("runtests.jl") is True
    assert is_test_related_path("src/alloc.zig") is False
    assert is_test_related_path("lib/client.dart") is False

    # Internal codebase check: tester.py is source code, not a test!
    assert is_test_related_path("src/oswald/core/agents/tester.py") is False
    assert is_test_related_path("src/oswald/config.py") is False

    # Non-test files inside test directories (fixtures, mocks, helpers, seeds)
    assert is_test_related_path("tests/fixtures/data.json") is False
    assert is_test_related_path("tests/helpers/mock_server.py") is False
    assert is_test_related_path("tests/utils/database_seed.ts") is False
    assert is_test_related_path("tests/seed.sql") is False
    assert is_test_related_path("test/fixtures/expected.txt") is False
    assert is_test_related_path("e2e/support/auth_service.go") is False
    assert is_test_related_path("tests/common/utils.py") is False


def test_write_test_file_blocks_non_test_files_inside_test_dir(workspace: Path) -> None:
    (workspace / "tests" / "helpers").mkdir(parents=True)
    (workspace / "tests" / "helpers" / "mock_server.py").write_text(
        'print("server")', encoding="utf-8"
    )

    # Editing existing non-test helper inside tests/ is rejected
    res = write_test_file(
        workspace,
        "tests/helpers/mock_server.py",
        'print("server")',
        'print("new server")',
    )
    assert res.success is False
    assert "not recognized as a test file" in res.message


def test_snapshot_restores_non_test_files_inside_test_dir(workspace: Path) -> None:
    (workspace / "tests" / "fixtures").mkdir(parents=True)
    (workspace / "tests" / "fixtures" / "data.json").write_text('{"initial": 1}', encoding="utf-8")
    (workspace / "tests" / "helpers").mkdir(parents=True)
    (workspace / "tests" / "helpers" / "mock_server.py").write_text(
        'print("server")', encoding="utf-8"
    )

    snapshot = WorkspaceSnapshot.capture(workspace)

    # Mutate data.json and mock_server.py
    (workspace / "tests" / "fixtures" / "data.json").write_text('{"mutated": 2}', encoding="utf-8")
    (workspace / "tests" / "helpers" / "mock_server.py").write_text(
        'print("corrupted")', encoding="utf-8"
    )

    # Author a valid test file
    (workspace / "tests" / "test_feature.py").write_text("def test_ok(): pass\n", encoding="utf-8")

    affected = snapshot.restore_non_test_changes(workspace)

    assert sorted(affected) == ["tests/fixtures/data.json", "tests/helpers/mock_server.py"]
    assert (workspace / "tests" / "fixtures" / "data.json").read_text(
        encoding="utf-8"
    ) == '{"initial": 1}'
    assert (workspace / "tests" / "helpers" / "mock_server.py").read_text(
        encoding="utf-8"
    ) == 'print("server")'
    assert (workspace / "tests" / "test_feature.py").exists()


# --- TesterAgent turn loop and lifecycle tests ---


@pytest.mark.asyncio
async def test_tester_authors_runs_and_reports_tests_happy_path(workspace: Path) -> None:
    llm = FakeLLMClient()
    sandbox = FakeSandboxRuntime(SandboxConfig(task_id="tester-1", workspace_path=workspace))
    await sandbox.start()
    events = InMemoryEventSink()
    agent = TesterAgent(llm=llm, sandbox=sandbox, workspace_root=workspace, event_sink=events)

    llm.enqueue(
        LLMResponse(
            content="Write a test",
            tool_calls=[
                ToolCall(
                    id="write-1",
                    name="write_test_file",
                    arguments={
                        "file_path": "tests/test_app.py",
                        "old_str": "",
                        "new_str": "def test_greet():\n    assert True\n",
                    },
                )
            ],
            stop_reason=StopReason.TOOL_USE,
        )
    )
    sandbox.queue_response(stdout="= 1 passed in 0.01s =")
    llm.enqueue(
        LLMResponse(
            content="Run tests",
            tool_calls=[
                ToolCall(
                    id="run-1",
                    name="run_test_command",
                    arguments={"command": ["pytest", "tests/test_app.py"]},
                )
            ],
            stop_reason=StopReason.TOOL_USE,
        )
    )
    llm.enqueue(
        LLMResponse(
            content='{"status":"COMPLETED","content":"# Test Report\\n\\nGreeting verified."}',
            stop_reason=StopReason.END_TURN,
        )
    )

    result = await agent.run(_input())

    assert result.passed is True
    assert result.exit_code == 0
    assert result.tests_run == 1
    assert result.tests_failed == 0
    assert result.test_files == ["tests/test_app.py"]
    assert sandbox.executed_commands == [["pytest", "tests/test_app.py"]]
    assert "Greeting verified" in result.content

    # Verify event emission
    event_types = [e.event_type for e in events.events]
    assert "tester.started" in event_types
    assert "tester.tool_executed" in event_types
    assert "tester.completed" in event_types


@pytest.mark.asyncio
async def test_tester_recovers_from_intermediate_failing_command(workspace: Path) -> None:
    llm = FakeLLMClient()
    sandbox = FakeSandboxRuntime(SandboxConfig(task_id="tester-1", workspace_path=workspace))
    await sandbox.start()
    agent = TesterAgent(llm=llm, sandbox=sandbox, workspace_root=workspace)

    # Turn 1: run test that fails
    sandbox.queue_response(stdout="= 1 failed in 0.02s =", exit_code=1)
    llm.enqueue(
        LLMResponse(
            content="Run initial test",
            tool_calls=[
                ToolCall(
                    id="run-1",
                    name="run_test_command",
                    arguments={"command": ["pytest", "tests/test_app.py"]},
                )
            ],
            stop_reason=StopReason.TOOL_USE,
        )
    )

    # Turn 2: fix test file
    llm.enqueue(
        LLMResponse(
            content="Fix test",
            tool_calls=[
                ToolCall(
                    id="write-1",
                    name="write_test_file",
                    arguments={
                        "file_path": "tests/test_app.py",
                        "old_str": "",
                        "new_str": "def test_ok(): assert True\n",
                    },
                )
            ],
            stop_reason=StopReason.TOOL_USE,
        )
    )

    # Turn 3: re-run test cleanly
    sandbox.queue_response(stdout="= 1 passed in 0.01s =", exit_code=0)
    llm.enqueue(
        LLMResponse(
            content="Re-run tests",
            tool_calls=[
                ToolCall(
                    id="run-2",
                    name="run_test_command",
                    arguments={"command": ["pytest", "tests/test_app.py"]},
                )
            ],
            stop_reason=StopReason.TOOL_USE,
        )
    )

    # Turn 4: complete
    llm.enqueue(
        LLMResponse(
            content='{"status":"COMPLETED","content":"# Test Report\\n\\nFixed and passed."}',
            stop_reason=StopReason.END_TURN,
        )
    )

    result = await agent.run(_input())

    # Crucial check: latest execution passed, so overall result is passed!
    assert result.passed is True
    assert result.exit_code == 0
    assert result.tests_run == 1
    assert result.tests_failed == 0


@pytest.mark.asyncio
async def test_tester_reports_failing_tests_accurately(workspace: Path) -> None:
    llm = FakeLLMClient()
    sandbox = FakeSandboxRuntime(SandboxConfig(task_id="tester-1", workspace_path=workspace))
    await sandbox.start()
    agent = TesterAgent(llm=llm, sandbox=sandbox, workspace_root=workspace)

    sandbox.queue_response(stdout="= 2 failed, 1 passed in 0.05s =", exit_code=1)
    llm.enqueue(
        LLMResponse(
            content="Run tests",
            tool_calls=[
                ToolCall(
                    id="run-1",
                    name="run_test_command",
                    arguments={"command": ["pytest"]},
                )
            ],
            stop_reason=StopReason.TOOL_USE,
        )
    )
    llm.enqueue(
        LLMResponse(
            content='{"status":"COMPLETED","content":"# Test Report\\n\\nBugs detected in greeting."}',
            stop_reason=StopReason.END_TURN,
        )
    )

    result = await agent.run(_input())

    assert result.passed is False
    assert result.exit_code == 1
    assert result.tests_run == 3
    assert result.tests_failed == 2
    assert "Bugs detected in greeting" in result.content


@pytest.mark.asyncio
async def test_circuit_breaker_not_tripped_by_normal_test_failures(workspace: Path) -> None:
    llm = FakeLLMClient()
    sandbox = FakeSandboxRuntime(SandboxConfig(task_id="tester-1", workspace_path=workspace))
    await sandbox.start()
    agent = TesterAgent(llm=llm, sandbox=sandbox, workspace_root=workspace)

    # Run pytest 3 times with test failures
    for i in range(3):
        sandbox.queue_response(stdout="= 1 failed in 0.01s =", exit_code=1)
        llm.enqueue(
            LLMResponse(
                content=f"Run {i}",
                tool_calls=[
                    ToolCall(
                        id=f"run-{i}",
                        name="run_test_command",
                        arguments={"command": ["pytest"]},
                    )
                ],
                stop_reason=StopReason.TOOL_USE,
            )
        )

    # Complete after 3 failed test runs
    llm.enqueue(
        LLMResponse(
            content='{"status":"COMPLETED","content":"# Test Report\\n\\nPersistent failures reported."}',
            stop_reason=StopReason.END_TURN,
        )
    )

    result = await agent.run(_input())
    # Did NOT escalate via circuit breaker! It completed normally reporting test failures.
    assert "Circuit breaker tripped" not in result.content
    assert result.passed is False
    assert result.exit_code == 1


@pytest.mark.asyncio
async def test_circuit_breaker_tripped_by_repeated_tool_errors(workspace: Path) -> None:
    llm = FakeLLMClient()
    sandbox = FakeSandboxRuntime(SandboxConfig(task_id="tester-1", workspace_path=workspace))
    await sandbox.start()
    agent = TesterAgent(llm=llm, sandbox=sandbox, workspace_root=workspace)

    # 3 identical invalid edits that fail
    for i in range(3):
        llm.enqueue(
            LLMResponse(
                content=f"Bad write {i}",
                tool_calls=[
                    ToolCall(
                        id=f"write-{i}",
                        name="write_test_file",
                        arguments={
                            "file_path": "tests/missing.py",
                            "old_str": "nonexistent",
                            "new_str": "something",
                        },
                    )
                ],
                stop_reason=StopReason.TOOL_USE,
            )
        )

    result = await agent.run(_input())
    assert result.passed is False
    assert "Circuit breaker tripped" in result.content


@pytest.mark.asyncio
async def test_tester_handles_unauthorized_non_test_mutations(workspace: Path) -> None:
    class MutatingSandbox(FakeSandboxRuntime):
        async def exec_command(
            self,
            cmd: list[str],
            timeout_seconds: float = 120.0,
            env: dict[str, str] | None = None,
        ) -> ExecResult:
            (workspace / "app.py").write_text("mutated by command", encoding="utf-8")
            return await super().exec_command(cmd, timeout_seconds=timeout_seconds, env=env)

    llm = FakeLLMClient()
    sandbox = MutatingSandbox(SandboxConfig(task_id="tester-1", workspace_path=workspace))
    await sandbox.start()
    agent = TesterAgent(llm=llm, sandbox=sandbox, workspace_root=workspace)

    sandbox.queue_response(stdout="= 1 passed =", exit_code=0)
    llm.enqueue(
        LLMResponse(
            content="Run test",
            tool_calls=[
                ToolCall(
                    id="run-1",
                    name="run_test_command",
                    arguments={"command": ["pytest", "tests/test_app.py"]},
                )
            ],
            stop_reason=StopReason.TOOL_USE,
        )
    )
    llm.enqueue(
        LLMResponse(
            content='{"status":"COMPLETED","content":"# Test Report\\n\\nTests passed."}',
            stop_reason=StopReason.END_TURN,
        )
    )

    result = await agent.run(_input())
    # Mutation was reverted
    assert (workspace / "app.py").read_text(
        encoding="utf-8"
    ) == "def greet() -> str:\n    return 'hello'\n"
    # Result marked passed=False due to unauthorized mutation
    assert result.passed is False
    assert "Unauthorized non-test mutations were restored" in result.output


@pytest.mark.asyncio
async def test_tester_handles_llm_error_gracefully(workspace: Path) -> None:
    llm = FakeLLMClient()
    sandbox = FakeSandboxRuntime(SandboxConfig(task_id="tester-1", workspace_path=workspace))
    await sandbox.start()
    agent = TesterAgent(llm=llm, sandbox=sandbox, workspace_root=workspace)

    llm.enqueue(LLMError("API rate limit exceeded"))

    result = await agent.run(_input())
    assert result.passed is False
    assert result.exit_code == 1
    assert "LLM failure" in result.content


@pytest.mark.asyncio
async def test_tester_no_commands_executed(workspace: Path) -> None:
    llm = FakeLLMClient()
    sandbox = FakeSandboxRuntime(SandboxConfig(task_id="tester-1", workspace_path=workspace))
    await sandbox.start()
    agent = TesterAgent(llm=llm, sandbox=sandbox, workspace_root=workspace)

    llm.enqueue(
        LLMResponse(
            content='{"status":"COMPLETED","content":"# Test Report\\n\\nNo tests run."}',
            stop_reason=StopReason.END_TURN,
        )
    )

    result = await agent.run(_input())
    assert result.passed is False
    assert result.exit_code == 1
    assert "No test command was executed" in result.output


@pytest.mark.asyncio
async def test_tester_allows_editing_test_files_present_in_spec_and_changes(
    workspace: Path,
) -> None:
    # If a test file was targeted by PlanSpec or present in CodeChanges (e.g. attempt 2),
    # TesterAgent must not forbid editing it!
    test_file = workspace / "tests" / "test_existing.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_old(): pass\n", encoding="utf-8")

    inp = _input(
        target_files=["app.py", "tests/test_existing.py"],
        modified_files=["app.py", "tests/test_existing.py"],
    )

    llm = FakeLLMClient()
    sandbox = FakeSandboxRuntime(SandboxConfig(task_id="tester-1", workspace_path=workspace))
    await sandbox.start()
    agent = TesterAgent(llm=llm, sandbox=sandbox, workspace_root=workspace)

    # Turn 1: overwrite/edit existing test
    llm.enqueue(
        LLMResponse(
            content="Edit test",
            tool_calls=[
                ToolCall(
                    id="write-1",
                    name="write_test_file",
                    arguments={
                        "file_path": "tests/test_existing.py",
                        "old_str": "def test_old(): pass",
                        "new_str": "def test_updated(): pass",
                    },
                )
            ],
            stop_reason=StopReason.TOOL_USE,
        )
    )
    sandbox.queue_response(stdout="= 1 passed in 0.01s =", exit_code=0)
    llm.enqueue(
        LLMResponse(
            content="Run test",
            tool_calls=[
                ToolCall(
                    id="run-1",
                    name="run_test_command",
                    arguments={"command": ["pytest", "tests/test_existing.py"]},
                )
            ],
            stop_reason=StopReason.TOOL_USE,
        )
    )
    llm.enqueue(
        LLMResponse(
            content='{"status":"COMPLETED","content":"# Report\\n\\nExisting test updated."}',
            stop_reason=StopReason.END_TURN,
        )
    )

    result = await agent.run(inp)
    assert result.passed is True
    assert "def test_updated(): pass" in test_file.read_text(encoding="utf-8")


async def test_diagnostics_and_missing_runner_do_not_poison_unittest(workspace: Path) -> None:
    import json

    llm = FakeLLMClient()
    sandbox = FakeSandboxRuntime(SandboxConfig(task_id="tester-1", workspace_path=workspace))
    sandbox.queue_response(stdout="Python 3.12")
    sandbox.queue_response(exit_code=127, stdout="exec: pytest: executable file not found in $PATH")
    sandbox.queue_response(stdout="Ran 1 test in 0.001s\n\nOK")
    await sandbox.start()
    calls: list[tuple[str, dict[str, Any]]] = [
        ("run_test_command", {"command": ["git", "status", "--porcelain"]}),
        ("run_diagnostic_command", {"command": ["python", "--version"]}),
        ("run_test_command", {"command": ["pytest", "test_app.py"]}),
        (
            "write_test_file",
            {"file_path": "test_app.py", "old_str": "", "new_str": "import unittest\n"},
        ),
        ("run_test_command", {"command": ["python", "-m", "unittest", "discover", "-v"]}),
    ]
    for i, (name, arguments) in enumerate(calls):
        llm.enqueue(
            LLMResponse(
                stop_reason=StopReason.TOOL_USE,
                tool_calls=[ToolCall(id=str(i), name=name, arguments=arguments)],
            )
        )
    llm.enqueue_text(json.dumps({"status": "COMPLETED", "content": "One behavior verified"}))
    result = await TesterAgent(llm, sandbox, workspace).run(_input())
    assert result.passed and result.tests_run == 1 and result.tests_failed == 0
    assert not result.execution_error
    assert "Git inspection belongs" in result.output and "unavailable" in result.output
    assert not any(command[0] == "git" for command in sandbox.executed_commands)


@pytest.mark.parametrize("diagnostic", [False, True])
async def test_zero_tests_and_diagnostics_cannot_pass(workspace: Path, diagnostic: bool) -> None:
    llm = FakeLLMClient()
    sandbox = FakeSandboxRuntime(SandboxConfig(task_id="tester-1", workspace_path=workspace))
    sandbox.queue_response(stdout="Python 3.12" if diagnostic else "Ran 0 tests in 0.001s\n\nOK")
    await sandbox.start()
    llm.enqueue(
        LLMResponse(
            stop_reason=StopReason.TOOL_USE,
            tool_calls=[
                ToolCall(
                    id="run",
                    name="run_diagnostic_command" if diagnostic else "run_test_command",
                    arguments={
                        "command": ["python", "--version"]
                        if diagnostic
                        else ["python", "-m", "unittest"]
                    },
                )
            ],
        )
    )
    llm.enqueue_text('{"status": "COMPLETED", "content": "Claimed success"}')
    result = await TesterAgent(llm, sandbox, workspace).run(_input())
    assert not result.passed and result.execution_error
    assert result.tests_run == 0


@pytest.mark.parametrize(
    "output,expected",
    [
        ("Ran 3 tests in 0.01s\nOK (skipped=1)", (3, 0)),
        ("Ran 3 tests in 0.01s\nFAILED (failures=1, errors=1)", (3, 2)),
    ],
)
def test_unittest_metrics(output: str, expected: tuple[int, int]) -> None:
    from oswald.core.tools.tester.runtime import CommandResult

    assert (
        TesterAgent._extract_metrics([CommandResult(["python", "-m", "unittest"], 0, output, 0.01)])
        == expected
    )


async def test_unittest_discovery_recovery_and_inline_diagnostics(workspace: Path) -> None:
    llm = FakeLLMClient()
    sandbox = FakeSandboxRuntime(SandboxConfig(task_id="tester-1", workspace_path=workspace))
    await sandbox.start()
    for code, text in (
        (5, "Ran 0 tests in 0.001s\nNO TESTS RAN"),
        (1, "ImportError: Start directory is not importable: '/workspace/tests'"),
        (
            0,
            "Ran 1 test in 0.001s\nFAILED (errors=1)",
        ),  # A synthetic inline diagnostic, not the suite.
        (0, "Ran 1 test in 0.001s\nOK"),
    ):
        sandbox.queue_response(exit_code=code, stdout=text)
    commands = [
        ["python", "-m", "unittest", "discover"],
        ["python", "-m", "unittest", "discover", "-s", "tests", "-t", "."],
        ["python", "-c", "print('synthetic test output')"],
        ["python", "-m", "unittest", "discover", "-s", "tests"],
    ]
    for i, command in enumerate(commands):
        llm.enqueue(
            LLMResponse(
                stop_reason=StopReason.TOOL_USE,
                tool_calls=[
                    ToolCall(id=str(i), name="run_test_command", arguments={"command": command}),
                ],
            )
        )
    llm.enqueue_text('{"status": "COMPLETED", "content": "Discovery corrected, one test passed"}')
    result = await TesterAgent(llm, sandbox, workspace).run(_input())
    assert result.passed and result.tests_run == 1 and result.tests_failed == 0
    assert not result.execution_error
    assert "No tests ran" in result.output
