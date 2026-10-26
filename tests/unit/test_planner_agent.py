"""Unit tests for PlannerAgent execution loop, PlanSpec generation, and edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oswald.core.agents.coder import InMemoryEventSink
from oswald.core.agents.planner import PlannerAgent, PlannerInput
from oswald.core.artifacts.models import PlanSpec
from oswald.core.llm.adapters.fake import FakeLLMClient
from oswald.core.llm.types import (
    LLMResponse,
    StopReason,
    TokenUsage,
    ToolCall,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "main.py").write_text("def run():\n    pass\n", encoding="utf-8")
    (ws / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    return ws


@pytest.mark.asyncio
async def test_planner_agent_happy_path(workspace: Path) -> None:
    """Planner receives task, executes read-only tool, and returns valid PlanSpec."""
    fake_plan = {
        "summary": "Add greet function to main.py",
        "target_files": ["main.py"],
        "tasks": [
            {
                "step": 1,
                "description": "Implement greet function",
                "target_file": "main.py",
                "action": "modify",
                "details": "Add greet(name: str) -> str",
            }
        ],
        "acceptance_criteria": ["greet('world') returns 'Hello world'"],
        "has_open_questions": False,
        "open_questions": [],
        "content": "# Specification for Greet Function\n\nImplement greet function in main.py.",
    }

    llm = FakeLLMClient()
    # Turn 1: Model asks to list files in workspace
    llm.enqueue(
        LLMResponse(
            content="Let me check files first.",
            tool_calls=[ToolCall(id="call_1", name="list_files", arguments={"directory": "."})],
            stop_reason=StopReason.TOOL_USE,
            usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        )
    )
    # Turn 2: Model returns valid PlanSpec JSON block
    llm.enqueue(
        LLMResponse(
            content=f"```json\n{json.dumps(fake_plan, indent=2)}\n```",
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(prompt_tokens=80, completion_tokens=80, total_tokens=160),
        )
    )

    sink = InMemoryEventSink()
    agent = PlannerAgent(llm=llm, workspace_root=workspace, event_sink=sink)

    inp = PlannerInput(task_id="task-001", prompt="Add a greeting function")
    spec = await agent.run(inp)

    assert isinstance(spec, PlanSpec)
    assert spec.task_id == "task-001"
    assert spec.summary == "Add greet function to main.py"
    assert spec.target_files == ["main.py"]
    assert len(spec.tasks) == 1
    assert spec.tasks[0].target_file == "main.py"
    assert spec.tasks[0].action == "modify"
    assert spec.has_open_questions is False
    assert spec.open_questions == []
    assert "Specification for Greet Function" in spec.content

    # Verify event telemetry
    event_types = [e.event_type for e in sink.events]
    assert "planner.started" in event_types
    assert "planner.tool_call" in event_types
    assert "planner.completed" in event_types


@pytest.mark.asyncio
async def test_planner_agent_requests_clarification(workspace: Path) -> None:
    """Planner detects ambiguous request and produces PlanSpec with has_open_questions=True."""
    clarification_plan = {
        "summary": "Implement payment gateway adapter (pending provider choice)",
        "target_files": ["src/payments.py"],
        "tasks": [],
        "acceptance_criteria": [],
        "has_open_questions": True,
        "open_questions": [
            "Which payment provider should be supported: Stripe, PayPal, or Square?",
            "What webhook secret management strategy should be used?",
        ],
        "content": "# Specification (Draft)\nRequires choice of payment provider.",
    }

    llm = FakeLLMClient()
    llm.enqueue(
        LLMResponse(
            content=f"```json\n{json.dumps(clarification_plan)}\n```",
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(prompt_tokens=50, completion_tokens=40, total_tokens=90),
        )
    )

    sink = InMemoryEventSink()
    agent = PlannerAgent(llm=llm, workspace_root=workspace, event_sink=sink)

    inp = PlannerInput(task_id="task-002", prompt="Add payment gateway")
    spec = await agent.run(inp)

    assert spec.has_open_questions is True
    assert len(spec.open_questions) == 2
    assert "Stripe, PayPal, or Square" in spec.open_questions[0]


@pytest.mark.asyncio
async def test_planner_agent_recovers_from_invalid_json(workspace: Path) -> None:
    """Planner retries when receiving non-JSON output and recovers on second turn."""
    valid_plan = {
        "summary": "Fix bug in calculation",
        "target_files": ["calc.py"],
        "tasks": [
            {
                "step": 1,
                "description": "Fix zero division",
                "target_file": "calc.py",
                "action": "modify",
            }
        ],
        "acceptance_criteria": ["Division by zero raises ZeroDivisionError"],
        "has_open_questions": False,
        "open_questions": [],
        "content": "# Fix spec\nHandle division by zero cleanly.",
    }

    llm = FakeLLMClient()
    # Turn 1: Invalid text without JSON
    llm.enqueue(
        LLMResponse(
            content="I think we should fix calc.py by modifying lines 10-15.",
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
        )
    )
    # Turn 2: Recovered with valid JSON after prompt retry
    llm.enqueue(
        LLMResponse(
            content=f"```json\n{json.dumps(valid_plan)}\n```",
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(prompt_tokens=80, completion_tokens=60, total_tokens=140),
        )
    )

    agent = PlannerAgent(llm=llm, workspace_root=workspace, max_schema_retries=2)

    spec = await agent.run(PlannerInput(task_id="task-003", prompt="Fix division bug"))
    assert spec.summary == "Fix bug in calculation"
    assert spec.target_files == ["calc.py"]
    assert spec.tasks[0].step == 1


@pytest.mark.asyncio
async def test_planner_agent_fallback_on_unrecoverable_response(workspace: Path) -> None:
    """Planner emits fallback PlanSpec with has_open_questions=True if model repeatedly fails JSON."""
    llm = FakeLLMClient()
    llm.enqueue(
        LLMResponse(
            content="I cannot provide JSON.",
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(prompt_tokens=30, completion_tokens=10, total_tokens=40),
        )
    )
    llm.enqueue(
        LLMResponse(
            content="Still cannot provide JSON.",
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        )
    )

    agent = PlannerAgent(llm=llm, workspace_root=workspace, max_schema_retries=1)

    spec = await agent.run(PlannerInput(task_id="task-004", prompt="Unclear task"))
    assert spec.has_open_questions is True
    assert len(spec.open_questions) > 0
    assert "task-004" in spec.content
