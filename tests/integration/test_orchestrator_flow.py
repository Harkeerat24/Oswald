"""Actual agent classes + Git + Docker, with scripted model replies and a pending Planner stub.

Run with pytest -m integration tests/integration/test_orchestrator.py.
No API key, model request, pytest image dependency, or network access is needed.
"""

# Fixture files are tiny and local.
# ruff: noqa: ASYNC240, ASYNC221

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from oswald.core.agents.coder import CoderAgent
from oswald.core.agents.reviewer import ReviewerAgent
from oswald.core.agents.tester import TesterAgent
from oswald.core.artifacts.models import PlanSpec, TestResults
from oswald.core.artifacts.storage import FileSystemArtifactStore
from oswald.core.llm.adapters.fake import FakeLLMClient
from oswald.core.llm.types import LLMResponse, StopReason, ToolCall
from oswald.core.orchestrator import Agents, Orchestrator, PlannerInput
from oswald.core.sandbox import DockerSandboxRuntime
from oswald.core.sandbox.fake import FakeSandboxRuntime
from oswald.core.sandbox.models import SandboxConfig
from oswald.core.sandbox.protocol import SandboxRuntime
from oswald.core.vcs.workspace import WorkspaceDiff


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


class PendingPlanner:
    async def run(self, data: PlannerInput) -> PlanSpec:
        # Inspect exactly the path which Coder edits and Docker bind-mounts.
        assert (data.workspace_root / "app.py").read_text() == "def value():\n    return 1\n"
        return PlanSpec(
            task_id=data.task_id,
            summary="Return 2",
            target_files=["app.py"],
            acceptance_criteria=["value() returns 2"],
            content="Make value() return 2.",
        )


def enqueue_tools(client: FakeLLMClient, *calls: ToolCall) -> None:
    client.enqueue(LLMResponse(tool_calls=list(calls), stop_reason=StopReason.TOOL_USE))


@pytest.mark.parametrize("docker", [False, pytest.param(True, marks=pytest.mark.integration)])
async def test_real_agents_complete_and_correct(tmp_path: Path, docker: bool) -> None:
    if docker and not docker_ready():
        pytest.skip("Docker daemon is unavailable")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def value():\n    return 1\n")
    (repo / ".gitignore").write_text("__pycache__/\n.oswald/\n")
    git = shutil.which("git") or "git"
    for args in (
        ["init", "-b", "main"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "fixture"],
    ):
        subprocess.run([git, *args], cwd=repo, capture_output=True, check=True)  # noqa: S603
    output = tmp_path / "reports"
    store = FileSystemArtifactStore(output)
    runtimes: list[SandboxRuntime] = []
    clients: list[FakeLLMClient] = []

    def sandbox(config: SandboxConfig) -> SandboxRuntime:
        runtime = (
            DockerSandboxRuntime(config)
            if docker
            else FakeSandboxRuntime(config, execute_locally=True)
        )
        runtimes.append(runtime)
        return runtime

    def agents(root: Path, runtime: SandboxRuntime, diff: WorkspaceDiff) -> Agents:
        coder, tester, reviewer = FakeLLMClient(), FakeLLMClient(), FakeLLMClient()
        clients.extend((coder, tester, reviewer))
        # First implementation is wrong; correction must consume Tester + Reviewer feedback.
        for old, new in (("return 1", "return 3"), ("return 3", "return 2")):
            enqueue_tools(
                coder,
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
            coder.enqueue_text(
                json.dumps(
                    {
                        "status": "COMPLETED",
                        "summary": "Implemented",
                        "content": "Implementation report",
                    }
                )
            )
        enqueue_tools(
            tester,
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
        for _ in range(2):
            enqueue_tools(
                tester,
                ToolCall(
                    id="test",
                    name="run_test_command",
                    arguments={
                        "command": ["python", "-B", "-m", "unittest", "discover", "-v"],
                    },
                ),
            )
            tester.enqueue_text(json.dumps({"status": "COMPLETED", "content": "Test report"}))
        for verdict in ("CHANGES_REQUESTED", "APPROVED"):
            reviewer.enqueue_text(
                json.dumps(
                    {
                        "verdict": verdict,
                        "summary": "Review report",
                        "content": "Check value()",
                        "blocking_issues": ["value() must return 2"]
                        if verdict == "CHANGES_REQUESTED"
                        else [],
                    }
                )
            )
        return Agents(
            PendingPlanner(),
            CoderAgent(coder, runtime, diff, root),
            TesterAgent(tester, runtime, root),
            ReviewerAgent(reviewer, model="fake/test"),
        )

    result = await Orchestrator(repo, output, store, agents, sandbox).run("Return 2")
    assert result.status == "SUCCEEDED", result.reason
    assert result.attempts == 2
    assert "Fix Loop Findings" in clients[0].requests[2].messages[1].content
    assert (
        "test_app.py" in json.loads(clients[2].requests[-1].messages[1].content)["changes"]["diff"]
    )
    tests = await store.load_latest(result.task_id, "test_results", TestResults)
    assert tests.passed and "test_value" in tests.output and "OK" in tests.output
    final_patch = Path(result.patch_path).read_text()
    assert "+    return 2" in final_patch and "test_app.py" in final_patch
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
    assert not Path(result.workspace_root).exists()
    assert not await runtimes[0].is_running()
    assert (repo / "app.py").read_text() == "def value():\n    return 1\n"
    if docker:
        assert list((output / result.task_id / "logs").glob("*.log"))
