"""Unit tests for provider-neutral LLM types."""

import pytest
from pydantic import ValidationError

from oswald.core.llm.types import (
    LLMRequest,
    LLMResponse,
    Message,
    Role,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


def test_role_enum_values() -> None:
    """Verify all defined Role enum values and string representations."""
    assert Role.SYSTEM.value == "system"
    assert Role.USER.value == "user"
    assert Role.ASSISTANT.value == "assistant"
    assert Role.TOOL.value == "tool"


def test_stop_reason_enum_values() -> None:
    """Verify all StopReason enum variants."""
    assert StopReason.END_TURN.value == "end_turn"
    assert StopReason.TOOL_USE.value == "tool_use"
    assert StopReason.MAX_TOKENS.value == "max_tokens"
    assert StopReason.CONTENT_FILTER.value == "content_filter"
    assert StopReason.UNKNOWN.value == "unknown"


def test_tool_definition_creation_and_immutability() -> None:
    """Verify ToolDefinition fields, immutability, and rejection of extra attributes."""
    tool = ToolDefinition(
        name="read_file",
        description="Reads file contents",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    assert tool.name == "read_file"
    assert tool.description == "Reads file contents"
    assert tool.parameters["type"] == "object"

    with pytest.raises(ValidationError):
        tool.name = "write_file"

    with pytest.raises(ValidationError):
        ToolDefinition.model_validate(
            {"name": "read_file", "description": "desc", "unexpected_field": "invalid"}
        )


def test_tool_call_creation_and_immutability() -> None:
    """Verify ToolCall fields, immutability, and rejection of extra attributes."""
    tc = ToolCall(
        id="call_123",
        name="grep",
        arguments={"pattern": "TODO"},
    )
    assert tc.id == "call_123"
    assert tc.name == "grep"
    assert tc.arguments == {"pattern": "TODO"}

    with pytest.raises(ValidationError):
        tc.name = "other"

    with pytest.raises(ValidationError):
        ToolCall.model_validate({"id": "call_123", "name": "grep", "extra_arg": "forbidden"})


def test_message_creation_immutability_and_extra_forbid() -> None:
    """Verify Message creation with defaults, tool response fields, and validation."""
    msg = Message(role=Role.USER, content="Hello world")
    assert msg.role == Role.USER
    assert msg.content == "Hello world"
    assert msg.tool_calls == []
    assert msg.tool_call_id is None

    # Assistant message with tool call
    assistant_msg = Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[ToolCall(id="call_1", name="search", arguments={"q": "test"})],
    )
    assert len(assistant_msg.tool_calls) == 1
    assert assistant_msg.tool_calls[0].name == "search"

    # Tool result message
    tool_msg = Message(
        role=Role.TOOL,
        content="Search result body",
        tool_call_id="call_1",
    )
    assert tool_msg.tool_call_id == "call_1"

    # Frozen check
    with pytest.raises(ValidationError):
        msg.content = "New text"

    # Extra forbid check
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "user", "content": "Hi", "invalid_meta": 123})


def test_token_usage_validation_and_immutability() -> None:
    """Verify TokenUsage non-negative constraints and immutability."""
    usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150

    # Negative tokens rejected
    with pytest.raises(ValidationError):
        TokenUsage(prompt_tokens=-1)

    with pytest.raises(ValidationError):
        TokenUsage(completion_tokens=-5)

    with pytest.raises(ValidationError):
        TokenUsage(total_tokens=-10)

    # Immutability
    with pytest.raises(ValidationError):
        usage.total_tokens = 200

    # Extra fields
    with pytest.raises(ValidationError):
        TokenUsage.model_validate({"prompt_tokens": 10, "cost": 0.01})


def test_llm_response_creation_and_immutability() -> None:
    """Verify LLMResponse structure, defaults, and immutability."""
    resp = LLMResponse(
        content="Generated text",
        stop_reason=StopReason.END_TURN,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        model="gemini/gemini-2.0-flash",
        provider_metadata={"latency": 0.35},
    )
    assert resp.content == "Generated text"
    assert resp.stop_reason == StopReason.END_TURN
    assert resp.usage.total_tokens == 30
    assert resp.model == "gemini/gemini-2.0-flash"
    assert resp.provider_metadata["latency"] == 0.35

    with pytest.raises(ValidationError):
        resp.content = "Mutated"

    with pytest.raises(ValidationError):
        LLMResponse.model_validate({"content": "Hi", "unexpected": "field"})


def test_llm_request_validation_rules() -> None:
    """Verify LLMRequest validation boundaries for messages, max_tokens, and temperature."""
    # Valid minimal request
    valid_msg = Message(role=Role.USER, content="Hello")
    req = LLMRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[valid_msg],
    )
    assert req.model == "claude-3-5-sonnet-20241022"
    assert req.max_tokens == 16384
    assert req.temperature == 0.0
    assert req.tools == []
    assert req.stop_sequences == []

    # Messages must not be empty
    with pytest.raises(ValidationError):
        LLMRequest(model="test-model", messages=[])

    # max_tokens must be gt 0
    with pytest.raises(ValidationError):
        LLMRequest(model="test-model", messages=[valid_msg], max_tokens=0)

    with pytest.raises(ValidationError):
        LLMRequest(model="test-model", messages=[valid_msg], max_tokens=-10)

    # temperature boundaries [0.0, 2.0]
    with pytest.raises(ValidationError):
        LLMRequest(model="test-model", messages=[valid_msg], temperature=-0.1)

    with pytest.raises(ValidationError):
        LLMRequest(model="test-model", messages=[valid_msg], temperature=2.1)

    # Extra forbid
    with pytest.raises(ValidationError):
        LLMRequest.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "presence_penalty": 0.5,
            }
        )


def test_serialization_and_deserialization_roundtrip() -> None:
    """Verify serialization to JSON and deserialization back into typed models."""
    tool = ToolDefinition(name="calc", description="calculator", parameters={"type": "object"})
    req = LLMRequest(
        model="gemini/gemini-2.0-flash",
        messages=[
            Message(role=Role.SYSTEM, content="You are a helper."),
            Message(role=Role.USER, content="Calculate 2+2"),
        ],
        tools=[tool],
        max_tokens=1000,
        temperature=0.7,
        stop_sequences=["STOP"],
    )

    json_str = req.model_dump_json()
    reconstructed = LLMRequest.model_validate_json(json_str)

    assert reconstructed == req
    assert reconstructed.messages[0].role == Role.SYSTEM
    assert reconstructed.tools[0].name == "calc"
