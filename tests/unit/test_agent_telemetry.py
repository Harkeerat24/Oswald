"""Offline accounting, concurrency, and timer-lifecycle regressions."""

import asyncio
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from oswald.core.agents.coder import CoderAgent, InMemoryEventSink, NoOpEventSink, RunEvent
from oswald.core.agents.planner import PlannerAgent, PlannerInput
from oswald.core.agents.telemetry import RunProgress, agent_scope, complete, tool_call
from oswald.core.artifacts.models import TestResults
from oswald.core.llm.adapters.fake import FakeLLMClient
from oswald.core.llm.exceptions import LLMError
from oswald.core.llm.types import LLMRequest, LLMResponse, Message, Role, TokenUsage, ToolCall
from oswald.core.sandbox.fake import FakeSandboxRuntime
from oswald.core.sandbox.models import SandboxConfig
from oswald.core.vcs.workspace import FakeWorkspaceDiff


def response(incoming: int = 100, outgoing: int = 20) -> LLMResponse:
    return LLMResponse(usage=TokenUsage(prompt_tokens=incoming, completion_tokens=outgoing))


async def test_counts_responses_and_dispatched_tools_without_double_counting() -> None:
    llm = FakeLLMClient()
    llm.enqueue(response().model_copy(update={"tool_calls": [ToolCall(id="1", name="unused")]}))
    llm.enqueue(response(200, 30))
    run = RunProgress("task")
    with capture_logs() as logs:
        async with agent_scope("coder", "task", run=run):
            await complete(
                llm,
                LLMRequest(model="fake", messages=[Message(role=Role.USER, content="offline")]),
                turn=1,
            )
            with tool_call("read_file_range"):
                pass
            with tool_call("edit_file") as tool:
                tool.success = False
            await complete(
                llm,
                LLMRequest(model="fake", messages=[Message(role=Role.USER, content="offline")]),
                turn=2,
            )
    totals = run.snapshot()
    assert (totals.tokens_in, totals.tokens_out) == (300, 50)
    assert (totals.llm_calls, totals.tool_calls, totals.tool_errors) == (2, 2, 1)
    finished = logs[-1]
    assert finished["event"] == "agent.completed"
    assert finished["elapsed_seconds"] >= 0
    assert all(line["agent"] == "coder" and line["attempt"] == 1 for line in logs)
    assert [line["turn"] for line in logs if line["event"] == "agent.llm_completed"] == [1, 2]


@pytest.mark.parametrize("available", [False, True])
async def test_missing_usage_is_unknown_but_reported_zero_is_zero(available: bool) -> None:
    llm = FakeLLMClient()
    llm.enqueue(response(0, 0) if available else LLMResponse())
    run = RunProgress("task")
    with capture_logs():
        async with agent_scope("planner", "task", run=run):
            await complete(
                llm, LLMRequest(model="fake", messages=[Message(role=Role.USER, content="offline")])
            )
    assert run.snapshot().tokens_in == (0 if available else None)
    assert run.snapshot().tokens_out == (0 if available else None)


async def test_provider_explicitly_marks_missing_usage() -> None:
    llm = FakeLLMClient()
    llm.enqueue(response().model_copy(update={"provider_metadata": {"usage_available": False}}))
    run = RunProgress("task")
    with capture_logs():
        async with agent_scope("planner", "task", run=run):
            await complete(
                llm, LLMRequest(model="fake", messages=[Message(role=Role.USER, content="offline")])
            )
    assert run.snapshot().tokens_in is None


async def test_failed_request_does_not_claim_complete_token_accounting() -> None:
    llm = FakeLLMClient()
    llm.enqueue(LLMError("offline failure"))
    run = RunProgress("task")
    with capture_logs() as logs, pytest.raises(LLMError):
        async with agent_scope("planner", "task", run=run):
            await complete(
                llm, LLMRequest(model="fake", messages=[Message(role=Role.USER, content="offline")])
            )
    assert logs[-1]["event"] == "agent.failed"
    assert logs[-1]["error_type"] == "LLMError"
    assert run.snapshot().tokens_in is None and run.snapshot().llm_calls == 1


async def test_nested_builtin_scope_reuses_orchestrator_counters() -> None:
    run = RunProgress("task")
    with capture_logs() as logs:
        async with agent_scope("coder", "task", run=run) as outer:  # noqa: SIM117
            async with agent_scope("coder", "task") as inner:
                assert inner is outer
                with tool_call("edit_file"):
                    pass
    assert len(run.agents) == 1
    assert sum(line["event"] == "agent.started" for line in logs) == 1
    assert sum(line["event"] == "agent.completed" for line in logs) == 1
    assert run.snapshot().tool_calls == 1


async def test_correction_attempts_reset_counters_but_run_totals_accumulate() -> None:
    run = RunProgress("task")
    with capture_logs():
        for attempt in (1, 2):
            async with agent_scope("coder", "task", attempt, run=run) as progress:
                assert progress.tool_calls == 0
                with tool_call("edit_file"):
                    pass
    assert [entry.attempt for entry in run.agents] == [1, 2]
    assert [entry.tool_calls for entry in run.agents] == [1, 1]
    assert run.snapshot().tool_calls == 2


async def test_concurrent_runs_keep_independent_identity_and_counters() -> None:
    async def execute(task: str, count: int) -> RunProgress:
        run = RunProgress(task)
        async with agent_scope("coder", task, run=run):
            for _ in range(count):
                with tool_call("read_file_range"):
                    await asyncio.sleep(0)
        return run

    with capture_logs() as logs:
        first, second = await asyncio.gather(execute("first", 2), execute("second", 3))
    assert first.snapshot().tool_calls == 2
    assert second.snapshot().tool_calls == 3
    completed = {line["task_id"]: line for line in logs if line["event"] == "agent.completed"}
    assert completed["first"]["tool_calls"] == 2
    assert completed["second"]["tool_calls"] == 3


@pytest.mark.parametrize("outcome", ["completed", "timeout", "cancelled"])
async def test_heartbeat_reports_current_action_and_stops_after_exit(outcome: str) -> None:
    run = RunProgress("task", interval=0.005)
    before = asyncio.all_tasks()
    with capture_logs() as logs:
        try:
            async with agent_scope("tester", "task", run=run):
                with tool_call("run_test_command"):
                    await asyncio.sleep(0.025)
                    if outcome == "timeout":
                        async with asyncio.timeout(0.005):
                            await asyncio.Event().wait()
                    if outcome == "cancelled":
                        raise asyncio.CancelledError
        except (TimeoutError, asyncio.CancelledError):
            pass
        running = [line for line in logs if line["event"] == "agent.running"]
        assert running and all(line["action"] == "tool:run_test_command" for line in running)
        assert running[-1]["elapsed_seconds"] > 0
        count = len(logs)
        await asyncio.sleep(0.015)
        assert len(logs) == count
    assert asyncio.all_tasks() == before
    assert (
        run.agents[0].outcome
        == {"completed": "completed", "timeout": "failed", "cancelled": "cancelled"}[outcome]
    )
    assert run.snapshot().tool_errors == (0 if outcome == "completed" else 1)
    # An unrelated log outside the scope must not inherit the finished agent.
    from oswald.core.agents.telemetry import add_progress_context

    assert add_progress_context(None, "info", {"event": "outside"}) == {"event": "outside"}


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_progress_intervals_rejected(interval: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        RunProgress("task", interval=interval)


async def test_default_sink_does_not_retain_events_and_explicit_sink_still_does(
    tmp_path: Path,
) -> None:
    llm = FakeLLMClient()
    runtime = FakeSandboxRuntime(SandboxConfig(task_id="task", workspace_path=tmp_path))
    coder = CoderAgent(llm, runtime, FakeWorkspaceDiff(), tmp_path)
    planner = PlannerAgent(llm, tmp_path)
    assert isinstance(coder.event_sink, NoOpEventSink)
    assert isinstance(planner.event_sink, NoOpEventSink)
    retained = InMemoryEventSink()
    for _ in range(1000):
        event = RunEvent(task_id="task", event_type="coder.tool_executed")
        await coder.event_sink.emit(event)
        await retained.emit(event)
    assert not hasattr(coder.event_sink, "events")
    assert len(retained.events) == 1000


async def test_real_planner_logs_usage_and_tool_failures(tmp_path: Path) -> None:
    llm = FakeLLMClient()
    llm.enqueue(
        response().model_copy(
            update={
                "tool_calls": [ToolCall(id="1", name="unknown_tool")],
            }
        )
    )
    llm.enqueue_text('{"summary":"Done","content":"Plan","target_files":[]}')
    agent = PlannerAgent(llm, tmp_path)
    with capture_logs() as logs:
        await agent.run(PlannerInput(task_id="task", prompt="Inspect"))
    assert logs[-1]["tool_calls"] == 1
    assert logs[-1]["tool_errors"] == 1
    assert logs[-1]["llm_calls"] == 2
    assert logs[-1]["tokens_in"] == 110


async def test_zero_test_count_is_visible_without_inventing_collection_evidence() -> None:
    result = TestResults(task_id="task", passed=False, tests_run=0, content="No results")
    with capture_logs() as logs:
        async with agent_scope("tester", "task") as progress:
            progress.result(result)
    assert logs[-1]["event"] == "agent.failed"
    assert logs[-1]["tests_run"] == 0
    assert logs[-1]["test_collection"] == "No tests reported"


async def test_pending_model_request_reports_last_confirmed_usage_then_cancels() -> None:
    entered = asyncio.Event()

    class WaitingClient:
        async def complete(self, request: LLMRequest) -> LLMResponse:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    run = RunProgress("task", interval=0.005)

    async def execute() -> None:
        async with agent_scope("coder", "task", run=run) as progress:
            progress.tokens_in, progress.tokens_out = 100, 20
            await complete(
                WaitingClient(),
                LLMRequest(
                    model="fake",
                    messages=[Message(role=Role.USER, content="offline")],
                ),
            )

    with capture_logs() as logs:
        task = asyncio.create_task(execute())
        await entered.wait()
        await asyncio.sleep(0.025)
        running = [line for line in logs if line["event"] == "agent.running"]
        assert running and running[-1]["action"] == "waiting_for_model"
        assert (running[-1]["tokens_in"], running[-1]["tokens_out"]) == (100, 20)
        assert running[-1]["llm_calls"] == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert run.agents[0].outcome == "cancelled"
    assert run.snapshot().tokens_in is None
