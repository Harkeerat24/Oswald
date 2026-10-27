"""Actual CLI, four real agents, Git, and Docker with scripted model replies.

Run pytest -m integration tests/integration/test_orchestrator_flow.py.
No API key, paid model request, or pytest image dependency is needed.
"""

# Fixture files and Git setup are tiny and local.
# ruff: noqa: ASYNC240, ASYNC221

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from oswald.cli.__main__ import run
from oswald.core.artifacts.models import PlanSpec, TestResults
from oswald.core.artifacts.storage import FileSystemArtifactStore
from oswald.core.llm.adapters.fake import FakeLLMClient
from oswald.core.llm.types import LLMResponse, Role, StopReason, TokenUsage, ToolCall
from oswald.core.orchestrator import RunResult
from oswald.core.sandbox import DockerSandboxRuntime
from oswald.core.sandbox.fake import FakeSandboxRuntime
from oswald.core.sandbox.models import SandboxConfig
from oswald.core.sandbox.protocol import SandboxRuntime
from oswald.observability.logging import configure_logging


def docker_ready() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    try:
        return (
            subprocess.run(  # noqa: S603
                [docker, "info"],
                capture_output=True,
                timeout=5,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def enqueue_tools(client: FakeLLMClient, *calls: ToolCall) -> None:
    client.enqueue(
        LLMResponse(
            tool_calls=list(calls),
            stop_reason=StopReason.TOOL_USE,
            usage=TokenUsage(prompt_tokens=20, completion_tokens=5),
        )
    )


@pytest.mark.parametrize("docker", [False, pytest.param(True, marks=pytest.mark.integration)])
@pytest.mark.parametrize("correction", [False, True], ids=["first-attempt", "correction"])
async def test_real_agents_complete_and_correct(
    tmp_path: Path,
    docker: bool,
    correction: bool,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if docker and not docker_ready():
        pytest.skip("Docker daemon is unavailable")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def value():\n    return 1\n")
    (repo / ".gitignore").write_text("__pycache__/\n.oswald/\n")
    git = shutil.which("git") or "git"
    for git_args in (
        ["init", "-b", "main"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "fixture"],
    ):
        subprocess.run([git, *git_args], cwd=repo, capture_output=True, check=True)  # noqa: S603
    # Planner must read committed source in its task worktree, not this dirty checkout.
    (repo / "app.py").write_text("def value():\n    return 999\n")
    output = tmp_path / "reports"
    store = FileSystemArtifactStore(output)
    runtimes: list[SandboxRuntime] = []

    def sandbox(config: SandboxConfig) -> SandboxRuntime:
        runtime = (
            DockerSandboxRuntime(config)
            if docker
            else FakeSandboxRuntime(config, execute_locally=True)
        )
        runtimes.append(runtime)
        return runtime

    llm = FakeLLMClient()
    for stage in ("planner", "coder", "tester", "reviewer"):
        monkeypatch.setenv(f"{stage.upper()}_MODEL", f"fake/{stage}")
    enqueue_tools(
        llm,
        ToolCall(id="list", name="list_files", arguments={"directory": "."}),
        ToolCall(id="inspect", name="read_file_range", arguments={"file_path": "app.py"}),
    )
    llm.enqueue_text(
        json.dumps(
            {
                "summary": "Return 2",
                "target_files": ["app.py", "test_app.py"],
                "tasks": [
                    {
                        "step": 1,
                        "description": "Return 2",
                        "target_file": "app.py",
                        "action": "modify",
                    }
                ],
                "acceptance_criteria": ["value() returns 2"],
                "has_open_questions": False,
                "open_questions": [],
                "content": "Make value() return 2 and verify with unittest.",
            }
        )
    )
    edits = (
        (("return 1", "return 3"), ("return 3", "return 2"))
        if correction
        else (("return 1", "return 2"),)
    )
    for attempt, (old, new) in enumerate(edits, start=1):
        enqueue_tools(
            llm,
            ToolCall(
                id="edit",
                name="edit_file",
                arguments={
                    "file_path": "app.py",
                    "old_str": old,
                    "new_str": new,
                },
            ),
        )
        llm.enqueue_text(
            json.dumps(
                {
                    "status": "COMPLETED",
                    "summary": "Implemented",
                    "content": "Implementation report",
                }
            )
        )
        if attempt == 1:
            enqueue_tools(
                llm,
                ToolCall(
                    id="git-diagnostic",
                    name="run_test_command",
                    arguments={"command": ["git", "status"]},
                ),
            )
            enqueue_tools(
                llm,
                ToolCall(
                    id="runtime",
                    name="run_diagnostic_command",
                    arguments={"command": ["python", "--version"]},
                ),
            )
            if docker:
                enqueue_tools(
                    llm,
                    ToolCall(
                        id="missing-runner",
                        name="run_test_command",
                        arguments={"command": ["pytest", "test_app.py"]},
                    ),
                )
            enqueue_tools(
                llm,
                ToolCall(
                    id="write-test",
                    name="write_test_file",
                    arguments={
                        "file_path": "test_app.py",
                        "old_str": "",
                        "new_str": (
                            "import unittest\nfrom app import value\n\nclass ValueTest(unittest.TestCase):\n"
                            "    def test_value(self):\n        self.assertEqual(value(), 2)\n"
                        ),
                    },
                ),
            )
        enqueue_tools(
            llm,
            ToolCall(
                id="test",
                name="run_test_command",
                arguments={
                    "command": ["python", "-B", "-m", "unittest", "discover", "-v"],
                },
            ),
        )
        llm.enqueue_text(json.dumps({"status": "COMPLETED", "content": "Test report"}))
        failed = correction and attempt == 1
        llm.enqueue_text(
            json.dumps(
                {
                    "verdict": "CHANGES_REQUESTED" if failed else "APPROVED",
                    "summary": "Review report",
                    "content": "Check value()",
                    "blocking_issues": ["value() must return 2"] if failed else [],
                }
            )
        )
    # Exercise the production CLI composition and Orchestrator; only model IO is scripted.
    args = argparse.Namespace(
        repo=str(repo),
        output=str(output),
        plan=None,
        planner=None,
        image="python:3.12-slim",
        stage_timeout=60,
        request="Return 2",
        base_ref="HEAD",
    )
    with (
        patch("oswald.core.llm.adapters.litellm_adapter.LiteLLMAdapter", return_value=llm),
        patch(
            "oswald.cli.__main__.DockerSandboxRuntime",
            side_effect=sandbox,
        ),
        configure_logging(output),
    ):
        code = await run(args)
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed["status"] == "SUCCEEDED"
    assert "PLANNER" in captured.err and "REVIEWER" in captured.err
    assert "tool_calls=" in captured.err and "tokens=" in captured.err
    entries = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    assert entries[-1]["event"] == "run.completed"
    assert entries[-1]["tests_run"] == 1 and entries[-1]["tests_failed"] == 0
    assert entries[-1]["verdict"] == "APPROVED"
    manifest = next(output.glob("*/run.json"))
    result = RunResult.model_validate_json(manifest.read_text())
    assert code == 0 and result.status == "SUCCEEDED", result.reason
    assert result.attempts == len(edits)
    assert result.metrics.llm_calls == llm.call_count
    assert result.metrics.tool_calls == 7 + int(docker) + (2 if correction else 0)
    assert result.metrics.tokens_in is not None and result.metrics.tokens_in > 0
    assert result.metrics.tokens_out is not None and result.metrics.tokens_out > 0
    assert len(result.metrics.agents) == (7 if correction else 4)
    assert printed["metrics"] == result.metrics.model_dump()
    for key in ("tokens_in", "tokens_out", "tool_calls", "llm_calls"):
        assert entries[-1][key] == getattr(result.metrics, key)
    completed = [
        entry
        for entry in entries
        if entry["event"]
        in {
            "agent.completed",
            "agent.failed",
            "agent.stopped",
        }
    ]
    assert len(completed) == len(result.metrics.agents)
    assert [r.model for r in llm.requests[:2]] == ["fake/planner", "fake/planner"]
    inspected = next(m.content for m in llm.requests[1].messages if m.tool_call_id == "inspect")
    assert "return 1" in inspected and "999" not in inspected
    assert result.base_commit in llm.requests[0].messages[1].content
    assert "Return 2" in llm.requests[0].messages[1].content
    assert {r.model for r in llm.requests} == {
        "fake/planner",
        "fake/coder",
        "fake/tester",
        "fake/reviewer",
    }
    assert {t.name for t in llm.requests[0].tools} == {
        "read_file_range",
        "search_code",
        "list_files",
        "read_symbol",
    }
    spec = await store.load_latest(result.task_id, "spec", PlanSpec)
    assert spec.task_id == result.task_id and spec.tasks[0].target_file == "app.py"
    coder_requests = [r for r in llm.requests if r.model == "fake/coder"]
    assert "Tester authors and runs those" in coder_requests[0].messages[1].content
    authorized = (
        coder_requests[0]
        .messages[1]
        .content.split("## Authorized Target Files")[1]
        .split("## Planned Tasks")[0]
    )
    assert "test_app.py" not in authorized
    if correction:
        assert "Fix Loop Findings" in coder_requests[2].messages[1].content
    review_request = next(r for r in reversed(llm.requests) if r.model == "fake/reviewer")
    assert "test_app.py" in json.loads(review_request.messages[1].content)["changes"]["diff"]
    assert any(m.role == Role.TOOL for m in llm.requests[1].messages)
    tests = await store.load_latest(result.task_id, "test_results", TestResults)
    assert (
        tests.passed
        and tests.tests_run == 1
        and "test_value" in tests.output
        and "OK" in tests.output
    )
    final_patch = Path(result.patch_path).read_text()
    assert "+    return 2" in final_patch and "test_app.py" in final_patch
    assert not Path(result.workspace_root).exists()
    assert runtimes[0].config.workspace_path == Path(result.workspace_root)
    assert not await runtimes[0].is_running()
    assert (repo / "app.py").read_text() == "def value():\n    return 999\n"
    # Restore only our fixture's intentional dirty change to check applicability to base.
    (repo / "app.py").write_text("def value():\n    return 1\n")
    subprocess.run(  # noqa: S603
        [git, "apply", "--check", result.patch_path],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    branches = subprocess.run(  # noqa: S603
        [git, "branch", "--list", "oswald/task-*"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    assert not branches.stdout.strip()
    if docker:
        assert list((output / result.task_id / "logs").glob("*.log"))
