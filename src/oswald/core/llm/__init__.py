"""Core LLM package.

Responsibility:
    LLMClient Protocol, provider-neutral types, and LiteLLM adapter.

Forbidden imports:
    Must not import from `oswald.api`, `oswald.workers`, or `oswald.db`.
"""

from oswald.config import Settings
from oswald.core.llm.adapters.litellm_adapter import LiteLLMAdapter, RetryConfig
from oswald.core.llm.exceptions import (
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
)
from oswald.core.llm.protocol import LLMClient
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


def create_llm_client(
    settings: Settings,
    *,
    run_id: str = "",
) -> LLMClient:
    """Create a production LLMClient instance wired with application settings."""
    return LiteLLMAdapter(
        api_key=settings.llm_provider_key,
        api_base=settings.llm_base_url,
        run_id=run_id,
        timeout_seconds=settings.llm_timeout_seconds,
        retry_config=RetryConfig(max_retries=settings.llm_max_retries),
    )


__all__ = [
    "LLMClient",
    "LLMError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "Role",
    "StopReason",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "create_llm_client",
]
