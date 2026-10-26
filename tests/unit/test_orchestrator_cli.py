"""CLI argument handling, production composition, and machine-readable outcome."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from oswald.cli.__main__ import SavedPlanPlanner, main, run
from oswald.core.agents.coder import CoderAgent
from oswald.core.agents.reviewer import ReviewerAgent
from oswald.core.agents.tester import TesterAgent
from oswald.core.orchestrator import RunResult
from oswald.core.sandbox.fake import FakeSandboxRuntime
from oswald.core.sandbox.models import SandboxConfig
from oswald.core.vcs.workspace import FakeWorkspaceDiff


def test_cli_requires_explicit_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["oswald", "Do something"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize("success", [True, False])
async def test_cli_composes_real_agents_and_returns_exit_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    success: bool,
) -> None:
    args = argparse.Namespace(
        repo=str(tmp_path),
        output=str(tmp_path / "reports"),
        plan=str(tmp_path / "plan.json"),
        planner=None,
        image="python:3.12-slim",
        stage_timeout=60,
        request="Fix value",
        base_ref="HEAD",
    )
    result = RunResult(
        task_id="test",
        status="SUCCEEDED" if success else "STOPPED",
        stage="reviewer",
        reason="result",
        attempts=1,
    )
    with patch("oswald.cli.__main__.Orchestrator") as pipeline:
        pipeline.return_value.run = AsyncMock(return_value=result)
        code = await run(args)
    assert code == (0 if success else 1)
    assert json.loads(capsys.readouterr().out)["status"] == result.status
    factory = pipeline.call_args.args[3]
    runtime = FakeSandboxRuntime(SandboxConfig(task_id="test", workspace_path=tmp_path))
    agents = factory(tmp_path, runtime, FakeWorkspaceDiff())
    assert isinstance(agents.planner, SavedPlanPlanner)
    assert isinstance(agents.coder, CoderAgent)
    assert isinstance(agents.tester, TesterAgent)
    assert isinstance(agents.reviewer, ReviewerAgent)
    assert agents.coder.workspace_root == agents.tester.workspace_root == tmp_path
    assert agents.coder.sandbox is agents.tester.sandbox is runtime
