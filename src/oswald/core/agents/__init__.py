"""Core agents package.

Responsibility:
    One module per role (Planner, Coder, Tester, Reviewer); prompts loaded from markdown files.

Forbidden imports:
    Must not import from `oswald.api`, `oswald.workers`, or `oswald.db`.
"""

from oswald.core.agents.coder import (
    BudgetExceededError,
    BudgetGuard,
    CoderAgent,
    CoderInput,
    EventSink,
    InMemoryEventSink,
    NoOpBudgetGuard,
    RunEvent,
)
from oswald.core.agents.planner import PlannerAgent, PlannerInput
from oswald.core.agents.reviewer import ReviewerAgent, ReviewerInput
from oswald.core.agents.tester import TesterAgent, TesterInput

__all__ = [
    "BudgetExceededError",
    "BudgetGuard",
    "CoderAgent",
    "CoderInput",
    "EventSink",
    "InMemoryEventSink",
    "NoOpBudgetGuard",
    "PlannerAgent",
    "PlannerInput",
    "ReviewerAgent",
    "ReviewerInput",
    "RunEvent",
    "TesterAgent",
    "TesterInput",
]
