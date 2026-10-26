"""Translate a pinned orchestration request into the Planner agent's input contract."""

from oswald.core.agents.planner import PlannerAgent
from oswald.core.agents.planner import PlannerInput as AgentPlannerInput
from oswald.core.artifacts.models import PlanSpec
from oswald.core.orchestrator.pipeline import PlannerInput


class PlannerAdapter:
    """Bind the real Planner to the task worktree supplied by the orchestrator."""

    def __init__(self, agent: PlannerAgent) -> None:
        self.agent = agent

    async def run(self, data: PlannerInput) -> PlanSpec:
        if self.agent.workspace_root != data.workspace_root:
            raise ValueError("Planner workspace must match the isolated task worktree")
        return await self.agent.run(
            AgentPlannerInput(
                task_id=data.task_id,
                prompt=data.request,
                context=f"Repository base commit: {data.base_commit}. Inspect the isolated workspace read-only.",
            )
        )
