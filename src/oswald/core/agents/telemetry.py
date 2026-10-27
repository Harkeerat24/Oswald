"""Small, context-local counters for readable agent progress; never stores event history."""

import asyncio
import math
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import structlog
from pydantic import BaseModel, Field
from structlog.typing import EventDict

from oswald.core.artifacts.models import CodeChanges, PlanSpec, ReviewVerdict, TestResults
from oswald.core.llm.protocol import LLMClient
from oswald.core.llm.types import LLMRequest, LLMResponse

logger = structlog.get_logger()


class UsageStats(BaseModel):
    elapsed_seconds: float = 0
    tokens_in: int | None = 0
    tokens_out: int | None = 0
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0


class AgentStats(UsageStats):
    agent: str
    attempt: int
    outcome: str


class RunStats(UsageStats):
    agents: list[AgentStats] = Field(default_factory=list)


@dataclass
class RunProgress:
    task_id: str
    interval: float = 10
    started: float = field(default_factory=time.monotonic)
    agents: list[AgentStats] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("Progress interval must be finite and positive")

    def snapshot(self) -> RunStats:
        return RunStats(
            elapsed_seconds=round(time.monotonic() - self.started, 3),
            tokens_in=(
                sum(a.tokens_in for a in self.agents if a.tokens_in is not None)
                if all(a.tokens_in is not None for a in self.agents)
                else None
            ),
            tokens_out=(
                sum(a.tokens_out for a in self.agents if a.tokens_out is not None)
                if all(a.tokens_out is not None for a in self.agents)
                else None
            ),
            llm_calls=sum(a.llm_calls for a in self.agents),
            tool_calls=sum(a.tool_calls for a in self.agents),
            tool_errors=sum(a.tool_errors for a in self.agents),
            agents=list(self.agents),
        )


@dataclass
class AgentProgress:
    task_id: str
    agent: str
    attempt: int
    started: float = field(default_factory=time.monotonic)
    last_event: float = field(default_factory=time.monotonic)
    tokens_in: int = 0
    tokens_out: int = 0
    usage_known: bool = True
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    action: str = "processing"
    details: dict[str, Any] = field(default_factory=dict)
    outcome: str = "completed"

    def fields(self) -> dict[str, Any]:
        return dict(
            task_id=self.task_id,
            agent=self.agent,
            attempt=self.attempt,
            elapsed_seconds=round(time.monotonic() - self.started, 3),
            tokens_in=self.tokens_in if self.usage_known else None,
            tokens_out=self.tokens_out if self.usage_known else None,
            llm_calls=self.llm_calls,
            tool_calls=self.tool_calls,
            tool_errors=self.tool_errors,
        )

    def log(self, event: str, *, level: str = "info", **extra: Any) -> None:
        self.last_event = time.monotonic()
        getattr(logger, level)(event, **self.fields(), **extra)

    def result(self, artifact: object) -> None:
        # Only small control fields enter logs; never copy reports, diffs, or tool arguments.
        if isinstance(artifact, CodeChanges):
            self.details["files_changed"] = len(artifact.modified_files)
            if artifact.status == "ESCALATED":
                self.outcome = "stopped"
                self.details["reason"] = "Agent escalated; see saved artifact"
        elif isinstance(artifact, PlanSpec):
            self.details["target_files"] = len(artifact.target_files)
            if artifact.has_open_questions or artifact.open_questions:
                self.outcome = "stopped"
                self.details["reason"] = "Unresolved planning questions"
        elif isinstance(artifact, TestResults):
            self.details.update(
                tests_run=artifact.tests_run,
                tests_failed=artifact.tests_failed,
                exit_code=artifact.exit_code,
                passed=artifact.passed,
            )
            if artifact.tests_run == 0:
                self.details["test_collection"] = "No tests reported"
            if not artifact.passed or artifact.execution_error:
                self.outcome = "failed"
        elif isinstance(artifact, ReviewVerdict):
            self.details.update(
                verdict=artifact.verdict, blocking_issues=len(artifact.blocking_issues)
            )
            if artifact.verdict != "APPROVED":
                self.outcome = "stopped"


_active: ContextVar[AgentProgress | None] = ContextVar("agent_progress", default=None)


def add_progress_context(_logger: Any, _method: str, event: EventDict) -> EventDict:
    """Enrich existing warnings/errors and provider retries with the active agent identity."""
    progress = _active.get()
    if progress is not None:
        event = {**progress.fields(), **event}
    return event


async def _heartbeat(progress: AgentProgress, interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        if time.monotonic() - progress.last_event >= interval:
            progress.log("agent.running", action=progress.action)


@asynccontextmanager
async def agent_scope(
    agent: str,
    task_id: str,
    attempt: int = 1,
    *,
    run: RunProgress | None = None,
) -> AsyncIterator[AgentProgress]:
    """Own one progress timer, including exception/cancellation cleanup.

    The orchestrator's outer scope also covers saved/custom planners and stage deadlines.
    Built-in agents reuse that scope and work independently when called directly.
    """
    active = _active.get()
    if active is not None and (active.agent, active.task_id, active.attempt) == (
        agent,
        task_id,
        attempt,
    ):
        yield active
        return
    progress = AgentProgress(task_id, agent, attempt)
    token = _active.set(progress)
    progress.log("agent.started")
    heartbeat = asyncio.create_task(_heartbeat(progress, run.interval if run else 10))
    try:
        yield progress
    except asyncio.CancelledError:
        progress.outcome = "cancelled"
        progress.details = {"reason": "Cancelled during agent execution"}
        raise
    except Exception as exc:
        progress.outcome = "failed"
        progress.details = {"error_type": type(exc).__name__}
        raise
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
        progress.log(
            "agent." + progress.outcome,
            level="error"
            if progress.outcome == "failed"
            else "warning"
            if progress.outcome in {"cancelled", "stopped"}
            else "info",
            **progress.details,
        )
        if run is not None:
            fields = progress.fields()
            fields.pop("task_id")
            run.agents.append(AgentStats(**fields, outcome=progress.outcome))
        _active.reset(token)


async def complete(llm: LLMClient, request: LLMRequest, *, turn: int = 1) -> LLMResponse:
    """Account for completed responses once, regardless of injected LLM implementation."""
    progress = _active.get()
    if progress is not None:
        progress.llm_calls += 1
        progress.action = "waiting_for_model"
    try:
        response = await llm.complete(request)
    except BaseException:
        if progress is not None:
            # A failed/cancelled request may have consumed tokens without returning usage.
            progress.usage_known = False
        raise
    finally:
        if progress is not None:
            progress.action = "processing"
    if progress is not None:
        known = response.provider_metadata.get(
            "usage_available",
            {"prompt_tokens", "completion_tokens"} <= response.usage.model_fields_set,
        )
        progress.usage_known = progress.usage_known and bool(known)
        progress.tokens_in += response.usage.prompt_tokens
        progress.tokens_out += response.usage.completion_tokens
        progress.log("agent.llm_completed", turn=turn, model=request.model)
    return response


@dataclass
class ToolProgress:
    success: bool = True


@contextmanager
def tool_call(name: str) -> Iterator[ToolProgress]:
    """Count dispatches, rather than model proposals; log before potentially slow tools."""
    progress = _active.get()
    tool = ToolProgress()
    if progress is not None:
        progress.tool_calls += 1
        progress.action = "tool:" + name[:80]
        progress.log("agent.tool_started", tool=name[:80])
    try:
        yield tool
    except BaseException:
        tool.success = False
        raise
    finally:
        if progress is not None:
            progress.tool_errors += int(not tool.success)
            progress.action = "processing"
            progress.log(
                "agent.tool_completed" if tool.success else "agent.tool_failed",
                level="debug" if tool.success else "warning",
                tool=name[:80],
            )
