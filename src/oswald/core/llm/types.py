"""Provider-neutral LLM types for OSWALD.

All agent modules depend exclusively on these data structures.
Vendor SDK types (e.g., litellm, anthropic, openai) must not cross
into agent implementations or orchestration state machines.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _ImmutableBase(BaseModel):
    """Base model with immutability and strict schema validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Role(StrEnum):
    """Message author role within a conversation turn."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolDefinition(_ImmutableBase):
    """Schema definition for an external tool exposed to the model."""

    name: str = Field(description="Unique tool function name")
    description: str = Field(description="Description of what this tool performs")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema defining the tool parameter structure",
    )


class ToolCall(_ImmutableBase):
    """A single tool invocation requested by the model."""

    id: str = Field(description="Unique identifier for this tool call")
    name: str = Field(description="Name of the tool to invoke")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Decoded argument dictionary passed by the model",
    )


class Message(_ImmutableBase):
    """A single message exchanged within a conversation."""

    role: Role = Field(description="Author role of the message")
    content: str = Field(default="", description="Textual body of the message")
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="List of tool calls initiated by this message (assistant role)",
    )
    tool_call_id: str | None = Field(
        default=None,
        description="Identifier of the tool call this message responds to (tool role only)",
    )


class TokenUsage(_ImmutableBase):
    """Token consumption metrics for a completion call."""

    prompt_tokens: int = Field(ge=0, default=0, description="Input tokens processed")
    completion_tokens: int = Field(ge=0, default=0, description="Output tokens generated")
    total_tokens: int = Field(ge=0, default=0, description="Total tokens consumed")


class StopReason(StrEnum):
    """Normalized reason indicating why generation terminated."""

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


class LLMResponse(_ImmutableBase):
    """Normalized response payload produced by an LLM invocation."""

    content: str = Field(default="", description="Generated text content")
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool invocations requested by the model",
    )
    stop_reason: StopReason = Field(
        default=StopReason.END_TURN,
        description="Normalized completion termination reason",
    )
    usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Token accounting for the request",
    )
    model: str = Field(default="", description="Model identifier that served the completion")
    provider_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-standard provider diagnostic metadata",
    )


class LLMRequest(_ImmutableBase):
    """Complete specification for a model completion request."""

    model: str = Field(description="Model identifier (e.g. gemini/gemini-2.0-flash)")
    messages: list[Message] = Field(
        min_length=1,
        description="Chronological conversation message sequence",
    )
    tools: list[ToolDefinition] = Field(
        default_factory=list,
        description="List of tools available for model invocation",
    )
    max_tokens: int = Field(
        default=16384,
        gt=0,
        description="Upper bound on output tokens to generate",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature controlling output randomness",
    )
    stop_sequences: list[str] = Field(
        default_factory=list,
        description="Optional literal sequences that immediately halt output",
    )
