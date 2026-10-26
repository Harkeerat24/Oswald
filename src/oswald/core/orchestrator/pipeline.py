"""Bounded async pipeline; agents decide content, this module owns lifecycle and routing."""

import asyncio
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from oswald.core.agents.coder import CoderInput
from oswald.core.agents.reviewer import ReviewerInput
from oswald.core.agents.tester import TesterInput
from oswald.core.artifacts.models import (
    ArtifactRef,
    BaseArtifact,
    CodeChanges,
    Finding,
    PlanSpec,
    Provenance,
    ReviewVerdict,
    TestResults,
)
from oswald.core.artifacts.storage import ArtifactStore
from oswald.core.sandbox.models import SandboxConfig
from oswald.core.sandbox.protocol import SandboxRuntime
from oswald.core.vcs.workspace import WorkspaceDiff, WorktreeWorkspaceDiff
from oswald.core.vcs.worktree import WorktreeManager


class PlannerInput(BaseModel):
    """Future Planner adapter must inspect workspace_root, never the source checkout."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    request: str = Field(min_length=1)
    workspace_root: Path
    base_commit: str


class Agent[Input, Output](Protocol):
    async def run(self, data: Input) -> Output: ...


@dataclass(frozen=True)
class Agents:
    planner: Agent[PlannerInput, PlanSpec]
    coder: Agent[CoderInput, CodeChanges]
    tester: Agent[TesterInput, TestResults]
    reviewer: Agent[ReviewerInput, ReviewVerdict]


class RunResult(BaseArtifact):
    status: Literal["SUCCEEDED", "STOPPED", "FAILED", "CANCELLED"]
    stage: str
    reason: str
    attempts: int
    base_commit: str = ""
    workspace_root: str = ""
    patch_path: str = ""
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    cleanup_errors: list[str] = Field(default_factory=list)


AgentFactory = Callable[[Path, SandboxRuntime, WorkspaceDiff], Agents]


class Orchestrator:
    """Run one task with at most one correction, persisting before deleting resources.

    A private worktree parent avoids collisions with other WorktreeManager users.
    No commits, pushes, or merges are performed. Reports live outside the worktree.
    """

    def __init__(
        self,
        repo_path: Path,
        output_dir: Path,
        store: ArtifactStore,
        agents: AgentFactory,
        sandbox: Callable[[SandboxConfig], SandboxRuntime],
        *,
        image: str = "python:3.12-slim",
        stage_timeout: float = 900,
    ) -> None:
        if stage_timeout <= 0:
            raise ValueError("stage_timeout must be positive")
        self.repo_path = repo_path.resolve()
        self.output_dir = output_dir.resolve()
        self.store = store
        self.agents = agents
        self.sandbox = sandbox
        self.image = image
        self.stage_timeout = stage_timeout

    async def run(self, request: str, *, base_ref: str = "HEAD") -> RunResult:
        if not request.strip():
            raise ValueError("Task request must not be blank")
        task_id = uuid4().hex
        report_dir = self.output_dir / task_id
        await asyncio.to_thread(report_dir.mkdir, parents=True, exist_ok=False)
        manager: WorktreeManager | None = None
        runtime: SandboxRuntime | None = None
        root: Path | None = None
        base_commit = ""
        stage = "workspace"
        attempt = 0
        refs: list[ArtifactRef] = []
        status: Literal["SUCCEEDED", "STOPPED", "FAILED", "CANCELLED"] = "STOPPED"
        reason = ""
        patch_path = ""
        cleanup_errors: list[str] = []

        async def save[T: BaseArtifact](name: str, artifact: T, model: type[T]) -> T:
            # Revalidate even model_construct()/model_copy() outputs at every handoff.
            validated = model.model_validate_json(artifact.model_dump_json())
            if validated.task_id != task_id:
                raise ValueError(f"{name} belongs to another task")
            ref = await self.store.save(
                task_id,
                name,
                validated,
                produced_by=Provenance(agent=stage, attempt=max(attempt, 1)),
                inputs=tuple(refs[-1:]),
            )
            refs.append(ref)
            attempt_dir = report_dir / f"attempt-{attempt}"
            await asyncio.to_thread(attempt_dir.mkdir, exist_ok=True)
            await asyncio.to_thread(
                (attempt_dir / f"{name}.json").write_text,
                validated.model_dump_json(indent=2),
                encoding="utf-8",
            )
            return validated

        async def call[T](operation: Awaitable[T]) -> T:
            async with asyncio.timeout(self.stage_timeout):
                return await operation

        try:
            manager = WorktreeManager(self.repo_path, report_dir / "workspace")
            context = await call(manager.create_worktree(task_id, base_ref=base_ref))
            root = context.worktree_path
            base_commit = context.base_ref
            await asyncio.to_thread(
                (report_dir / "request.json").write_text,
                PlannerInput(
                    task_id=task_id, request=request, workspace_root=root, base_commit=base_commit
                ).model_dump_json(indent=2),
                encoding="utf-8",
            )
            stage = "sandbox"
            runtime = self.sandbox(
                SandboxConfig(task_id=task_id, workspace_path=root, image=self.image)
            )
            await call(runtime.start())
            diff = WorktreeWorkspaceDiff(manager)
            stage = "planner"
            agents = self.agents(root, runtime, diff)
            spec = await save(
                "spec",
                await call(
                    agents.planner.run(
                        PlannerInput(
                            task_id=task_id,
                            request=request,
                            workspace_root=root,
                            base_commit=base_commit,
                        )
                    )
                ),
                PlanSpec,
            )
            if spec.has_open_questions or spec.open_questions:
                reason = "Unresolved planning questions: " + "; ".join(spec.open_questions)
            else:
                findings: list[Finding] = []
                for attempt in (1, 2):
                    stage = "coder"
                    changes = await save(
                        "code_changes",
                        await call(
                            agents.coder.run(
                                CoderInput(
                                    task_id=task_id,
                                    attempt=attempt,
                                    spec=spec,
                                    findings=findings,
                                )
                            )
                        ),
                        CodeChanges,
                    )
                    if changes.status == "ESCALATED":
                        reason = changes.escalation or changes.summary
                        break
                    stage = "tester"
                    tests = await save(
                        "test_results",
                        await call(
                            agents.tester.run(
                                TesterInput(
                                    task_id=task_id,
                                    attempt=attempt,
                                    spec=spec,
                                    changes=changes,
                                )
                            )
                        ),
                        TestResults,
                    )
                    if tests.execution_error:
                        status, reason = "FAILED", tests.execution_error
                        break
                    # Tester can author tests: Reviewer must see the current cumulative patch.
                    current_diff = await call(manager.get_diff(task_id))
                    reviewed_changes = changes.model_copy(update={"diff": current_diff.diff})
                    reviewed_changes = await save("review_changes", reviewed_changes, CodeChanges)
                    stage = "reviewer"
                    review = await save(
                        "review",
                        await call(
                            agents.reviewer.run(
                                ReviewerInput(
                                    task_id=task_id,
                                    attempt=attempt,
                                    spec=spec,
                                    changes=reviewed_changes,
                                    test_results=tests,
                                )
                            )
                        ),
                        ReviewVerdict,
                    )
                    clean_tests = tests.passed and tests.exit_code == 0 and tests.tests_failed == 0
                    if review.verdict == "REJECTED":
                        reason = review.summary + ": " + "; ".join(review.blocking_issues)
                        break
                    if clean_tests and review.verdict == "APPROVED" and not review.blocking_issues:
                        status, reason = "SUCCEEDED", "Tests passed and Reviewer approved"
                        break
                    if not clean_tests:
                        stage = "tester"
                        reason = "Tests failed: " + tests.content
                    else:
                        reason = review.summary + ": " + "; ".join(review.blocking_issues)
                    if attempt == 2:
                        reason = "Correction limit reached (2 total attempts). " + reason
                        break
                    findings = []
                    if not clean_tests:
                        findings.append(
                            Finding(
                                source="tester",
                                severity="major",
                                problem=tests.content,
                                evidence=tests.output[-500:],
                                expected_correction="Correct the implementation so the tests pass.",
                            )
                        )
                    if review.verdict != "APPROVED" or review.blocking_issues:
                        findings.extend(
                            Finding(
                                source="reviewer",
                                severity="major",
                                problem=issue,
                                expected_correction=review.content,
                            )
                            for issue in (review.blocking_issues or [review.summary])
                        )
        except asyncio.CancelledError:
            status, reason = "CANCELLED", f"Cancelled during {stage}"
            raise
        except Exception as exc:
            status, reason = "FAILED", f"{type(exc).__name__}: {exc}"
        finally:
            # Stop even after partial startup; preserve changes if snapshot or teardown fails.
            if runtime is not None:
                try:
                    await call(runtime.stop())
                except Exception as exc:
                    cleanup_errors.append(f"sandbox: {exc}")
            if root is None and manager is not None:
                candidate = manager.get_worktree_path(task_id)
                if await asyncio.to_thread(candidate.is_dir):
                    root = candidate
            if root is not None and manager is not None:
                try:
                    logs = root / ".oswald" / "logs"
                    if await asyncio.to_thread(logs.is_dir):
                        await asyncio.to_thread(
                            shutil.copytree, logs, report_dir / "logs", symlinks=True
                        )
                    final_diff = await call(manager.get_diff(task_id))
                    target = report_dir / "final.patch"
                    await asyncio.to_thread(target.write_text, final_diff.diff, encoding="utf-8")
                    patch_path = str(target)
                except Exception as exc:
                    cleanup_errors.append(f"patch: {exc}; worktree retained at {root}")
                if patch_path and not cleanup_errors:
                    try:
                        await call(manager.remove_worktree(task_id))
                        if await asyncio.to_thread(root.exists):
                            raise RuntimeError(f"Worktree still exists at {root}")
                    except Exception as exc:
                        cleanup_errors.append(f"worktree: {exc}")
            if cleanup_errors and status != "CANCELLED":
                status = "FAILED"
                reason += "; Cleanup failed: " + "; ".join(cleanup_errors)
            result = RunResult(
                task_id=task_id,
                status=status,
                stage=stage,
                reason=reason,
                attempts=attempt,
                base_commit=base_commit,
                workspace_root=str(root or ""),
                patch_path=patch_path,
                artifacts=refs,
                cleanup_errors=cleanup_errors,
            )
            await asyncio.to_thread(
                (report_dir / "run.json").write_text,
                result.model_dump_json(indent=2),
                encoding="utf-8",
            )
        return result
