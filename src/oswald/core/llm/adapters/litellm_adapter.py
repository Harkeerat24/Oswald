"""LiteLLM adapter — production LLMClient implementation.

Implements LLMClient Protocol using LiteLLM raw completion

Vendor coupling boundary: this is the ONLY file that imports litellm.
"""

import asyncio
import json
import math
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import litellm
import structlog
from litellm.exceptions import APIError, RateLimitError
from litellm.exceptions import Timeout as LiteLLMTimeout

from oswald.core.llm.exceptions import (
    LLMProviderError,
    LLMRateLimitError,
)
from oswald.core.llm.types import (
    LLMRequest,
    LLMResponse,
    Message,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

logger = structlog.get_logger()

_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "end_turn": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.CONTENT_FILTER,
}


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy configuration for transient provider failures."""

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    jitter_factor: float = 0.5


def _compute_delay(attempt: int, config: RetryConfig) -> float:
    capped = min(config.base_delay_seconds * (2**attempt), config.max_delay_seconds)
    jitter = random.uniform(0.0, config.jitter_factor * capped)  # noqa: S311
    return float(capped + jitter)


async def retry_with_backoff[T](
    fn: Callable[[], Awaitable[T]],
    config: RetryConfig,
    *,
    run_id: str = "",
) -> T:
    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return await fn()
        except LLMRateLimitError as exc:
            last_exception = exc
            delay = (
                exc.retry_after_seconds
                if exc.retry_after_seconds is not None
                else _compute_delay(attempt, config)
            )
            logger.warning(
                "llm.rate_limited",
                run_id=run_id,
                attempt=attempt + 1,
                retry_after=delay,
            )
        except LLMProviderError as exc:
            last_exception = exc
            if not exc.is_retryable:
                raise
            delay = _compute_delay(attempt, config)
            logger.warning(
                "llm.provider_error_retryable",
                run_id=run_id,
                attempt=attempt + 1,
                status_code=exc.status_code,
                delay=delay,
            )

        if attempt < config.max_retries:
            await asyncio.sleep(delay)

    raise (
        last_exception
        if last_exception is not None
        else RuntimeError("Retry loop exhausted without result")
    )


class LiteLLMAdapter:
    """Production LLMClient backed by LiteLLM."""

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str | None = None,
        retry_config: RetryConfig | None = None,
        run_id: str = "",
        timeout_seconds: float = 300,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self._timeout_seconds = timeout_seconds
        self._api_key = api_key
        self._api_base = api_base
        self._retry_config = retry_config or RetryConfig()
        self._run_id = run_id

        # Suppress LiteLLM's verbose stdout logging in favor of structlog
        litellm.suppress_debug_info = True

    async def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs = self._build_litellm_kwargs(request)

        async def _call() -> LLMResponse:
            start = time.monotonic()
            try:
                logger.debug(
                    "llm.completion.started",
                    run_id=self._run_id,
                    model=request.model,
                    timeout_seconds=self._timeout_seconds,
                )
                async with asyncio.timeout(self._timeout_seconds):
                    raw = await litellm.acompletion(**kwargs)
                elapsed = time.monotonic() - start
                response = self._translate_response(raw, request.model, elapsed)

                logger.debug(
                    "llm.completion.success",
                    run_id=self._run_id,
                    model=request.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    elapsed_seconds=round(elapsed, 3),
                    stop_reason=response.stop_reason,
                )
                return response
            except (TimeoutError, LiteLLMTimeout) as exc:
                raise LLMProviderError(
                    f"Model request timed out after {self._timeout_seconds:g} seconds",
                    model=request.model,
                    is_retryable=True,
                ) from exc
            except RateLimitError as exc:
                retry_after: float | None = None
                headers = getattr(exc, "headers", None)
                if isinstance(headers, dict) and "retry-after" in headers:
                    try:
                        retry_after = float(headers["retry-after"])
                    except (ValueError, TypeError):
                        retry_after = None

                raise LLMRateLimitError(
                    str(exc),
                    model=request.model,
                    retry_after_seconds=retry_after,
                ) from exc
            except APIError as exc:
                raw_status = getattr(exc, "status_code", None)
                status = raw_status if isinstance(raw_status, int) else None
                is_retryable = status in (500, 502, 503, 529) if status is not None else False
                raise LLMProviderError(
                    str(exc),
                    model=request.model,
                    status_code=status,
                    is_retryable=is_retryable,
                ) from exc

        return await retry_with_backoff(
            _call,
            config=self._retry_config,
            run_id=self._run_id,
        )

    def _build_litellm_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        messages = [self._translate_message(m) for m in request.messages]

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "api_key": self._api_key,
            "timeout": self._timeout_seconds,
            "num_retries": 0,
        }

        # V4 defaults to thinking, which requires replaying reasoning_content across tools.
        # This bounded pipeline uses ordinary content/tool messages, so request non-thinking mode.
        if request.model.startswith(("deepseek/deepseek-v4-", "deepseek/deepseek-flash")):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        if self._api_base:
            kwargs["api_base"] = self._api_base

        if request.tools:
            kwargs["tools"] = [self._translate_tool_def(t) for t in request.tools]

        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        return kwargs

    @staticmethod
    def _translate_message(msg: Message) -> dict[str, Any]:
        result: dict[str, Any] = {"role": msg.role.value, "content": msg.content}

        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]

        if msg.tool_call_id is not None:
            result["tool_call_id"] = msg.tool_call_id

        return result

    @staticmethod
    def _translate_tool_def(tool: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    @staticmethod
    def _translate_response(raw: Any, model: str, elapsed: float) -> LLMResponse:
        choice = raw.choices[0]
        message = choice.message

        content = message.content or ""

        tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(message, "tool_calls", None)
        if raw_tool_calls:
            for tc in raw_tool_calls:
                args = tc.function.arguments
                parsed_args = json.loads(args) if isinstance(args, str) else args
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=parsed_args if isinstance(parsed_args, dict) else {},
                    )
                )

        finish = choice.finish_reason or ""
        stop_reason = _STOP_REASON_MAP.get(finish, StopReason.UNKNOWN)

        usage_data = getattr(raw, "usage", None)
        usage = TokenUsage(
            prompt_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage_data, "total_tokens", 0) or 0,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=model,
            provider_metadata={
                "elapsed_seconds": round(elapsed, 3),
                "raw_finish_reason": finish,
                "usage_available": (
                    getattr(usage_data, "prompt_tokens", None) is not None
                    and getattr(usage_data, "completion_tokens", None) is not None
                ),
            },
        )
