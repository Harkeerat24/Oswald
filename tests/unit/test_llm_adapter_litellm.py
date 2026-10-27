"""Unit tests for LiteLLMAdapter translation, execution, and error handling."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import litellm
import pytest
from litellm.exceptions import APIError, RateLimitError

from oswald.core.llm.adapters.litellm_adapter import LiteLLMAdapter, RetryConfig
from oswald.core.llm.exceptions import LLMProviderError, LLMRateLimitError
from oswald.core.llm.types import (
    LLMRequest,
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolDefinition,
)


def _make_dummy_response(
    content: str = "Test response",
    finish_reason: str = "stop",
    tool_calls: list[Any] | None = None,
    prompt_tokens: int = 15,
    completion_tokens: int = 25,
) -> SimpleNamespace:
    """Helper creating a LiteLLM-like response namespace."""
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(
        choices=[choice],
        usage=usage,
    )


async def test_litellm_adapter_builds_complete_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify LiteLLMAdapter correctly builds kwargs for litellm.acompletion."""
    mock_acompletion = AsyncMock(return_value=_make_dummy_response())
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    adapter = LiteLLMAdapter(
        api_key="sk-test-key",
        api_base="https://custom.api.base/v1",
        run_id="run-kwarg-test",
    )

    tool = ToolDefinition(
        name="web_search",
        description="Search web",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    req = LLMRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[
            Message(role=Role.SYSTEM, content="System prompt"),
            Message(role=Role.USER, content="User question"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(id="call_99", name="web_search", arguments={"query": "AI"})],
            ),
            Message(role=Role.TOOL, content="Results here", tool_call_id="call_99"),
        ],
        tools=[tool],
        max_tokens=2048,
        temperature=0.5,
        stop_sequences=["STOP_NOW"],
    )

    resp = await adapter.complete(req)
    assert resp.content == "Test response"
    assert mock_acompletion.await_count == 1

    called_kwargs = mock_acompletion.call_args.kwargs
    assert called_kwargs["model"] == "claude-3-5-sonnet-20241022"
    assert called_kwargs["api_key"] == "sk-test-key"
    assert called_kwargs["api_base"] == "https://custom.api.base/v1"
    assert called_kwargs["max_tokens"] == 2048
    assert called_kwargs["temperature"] == 0.5
    assert called_kwargs["stop"] == ["STOP_NOW"]

    # Verify message serialization
    messages = called_kwargs["messages"]
    assert len(messages) == 4
    assert messages[0] == {"role": "system", "content": "System prompt"}
    assert messages[1] == {"role": "user", "content": "User question"}
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"] == [
        {
            "id": "call_99",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "AI"}',
            },
        }
    ]
    assert messages[3] == {
        "role": "tool",
        "content": "Results here",
        "tool_call_id": "call_99",
    }

    # Verify tool serialization
    tools = called_kwargs["tools"]
    assert len(tools) == 1
    assert tools[0] == {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }


async def test_litellm_adapter_omits_optional_kwargs_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify optional fields like api_base, tools, and stop are omitted when empty."""
    mock_acompletion = AsyncMock(return_value=_make_dummy_response())
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    adapter = LiteLLMAdapter(api_key="sk-test-key", api_base=None)
    req = LLMRequest(
        model="gpt-4o",
        messages=[Message(role=Role.USER, content="Hello")],
    )

    await adapter.complete(req)
    called_kwargs = mock_acompletion.call_args.kwargs

    assert "api_base" not in called_kwargs
    assert "tools" not in called_kwargs
    assert "stop" not in called_kwargs


async def test_litellm_adapter_parses_response_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify tool call parsing with both stringified JSON and dict arguments."""
    tool_call_mock1 = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="read_file", arguments='{"path": "foo.py"}'),
    )
    tool_call_mock2 = SimpleNamespace(
        id="call_2",
        function=SimpleNamespace(name="list_dir", arguments={"path": "."}),
    )

    raw_response = _make_dummy_response(
        content=None,  # type: ignore[arg-type]
        finish_reason="tool_calls",
        tool_calls=[tool_call_mock1, tool_call_mock2],
    )
    mock_acompletion = AsyncMock(return_value=raw_response)
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    adapter = LiteLLMAdapter(api_key="sk-key")
    req = LLMRequest(model="gpt-4o", messages=[Message(role=Role.USER, content="Run tools")])

    res = await adapter.complete(req)

    assert res.content == ""
    assert res.stop_reason == StopReason.TOOL_USE
    assert len(res.tool_calls) == 2
    assert res.tool_calls[0].id == "call_1"
    assert res.tool_calls[0].name == "read_file"
    assert res.tool_calls[0].arguments == {"path": "foo.py"}
    assert res.tool_calls[1].id == "call_2"
    assert res.tool_calls[1].arguments == {"path": "."}


@pytest.mark.parametrize(
    ("finish_reason", "expected_stop_reason"),
    [
        ("stop", StopReason.END_TURN),
        ("end_turn", StopReason.END_TURN),
        ("tool_calls", StopReason.TOOL_USE),
        ("function_call", StopReason.TOOL_USE),
        ("length", StopReason.MAX_TOKENS),
        ("content_filter", StopReason.CONTENT_FILTER),
        ("unrecognized_reason", StopReason.UNKNOWN),
        (None, StopReason.UNKNOWN),
    ],
)
async def test_litellm_adapter_finish_reason_mapping(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str | None,
    expected_stop_reason: StopReason,
) -> None:
    """Verify mapping from LiteLLM finish_reason string to StopReason enum."""
    raw = _make_dummy_response(finish_reason=finish_reason or "")
    mock_acompletion = AsyncMock(return_value=raw)
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    adapter = LiteLLMAdapter(api_key="sk-key")
    req = LLMRequest(model="gpt-4o", messages=[Message(role=Role.USER, content="Test")])

    res = await adapter.complete(req)
    assert res.stop_reason == expected_stop_reason


async def test_litellm_adapter_maps_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify RateLimitError is mapped to LLMRateLimitError with retry_after."""
    exc = RateLimitError(
        message="Rate limit reached",
        llm_provider="openai",
        model="gpt-4o",
    )
    exc.headers = {"retry-after": "12.5"}

    mock_acompletion = AsyncMock(side_effect=exc)
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    adapter = LiteLLMAdapter(api_key="sk-key", retry_config=RetryConfig(max_retries=0))
    req = LLMRequest(model="gpt-4o", messages=[Message(role=Role.USER, content="Hello")])

    with pytest.raises(LLMRateLimitError) as exc_info:
        await adapter.complete(req)

    assert exc_info.value.model == "gpt-4o"
    assert exc_info.value.retry_after_seconds == 12.5
    assert "Rate limit reached" in str(exc_info.value)


async def test_litellm_adapter_maps_rate_limit_error_invalid_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify non-numeric retry-after headers safely default to None."""
    exc = RateLimitError(
        message="Rate limit reached",
        llm_provider="openai",
        model="gpt-4o",
    )
    exc.headers = {"retry-after": "not-a-number"}

    mock_acompletion = AsyncMock(side_effect=exc)
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    adapter = LiteLLMAdapter(api_key="sk-key", retry_config=RetryConfig(max_retries=0))
    req = LLMRequest(model="gpt-4o", messages=[Message(role=Role.USER, content="Hello")])

    with pytest.raises(LLMRateLimitError) as exc_info:
        await adapter.complete(req)

    assert exc_info.value.retry_after_seconds is None


async def test_litellm_adapter_maps_rate_limit_error_missing_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify headers without retry-after key safely default to None."""
    exc = RateLimitError(
        message="Rate limit reached",
        llm_provider="openai",
        model="gpt-4o",
    )
    exc.headers = {"content-type": "application/json"}

    mock_acompletion = AsyncMock(side_effect=exc)
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    adapter = LiteLLMAdapter(api_key="sk-key", retry_config=RetryConfig(max_retries=0))
    req = LLMRequest(model="gpt-4o", messages=[Message(role=Role.USER, content="Hello")])

    with pytest.raises(LLMRateLimitError) as exc_info:
        await adapter.complete(req)

    assert exc_info.value.retry_after_seconds is None


async def test_litellm_adapter_maps_rate_limit_error_no_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify missing headers attribute safely defaults to None."""
    exc = RateLimitError(
        message="Rate limit reached",
        llm_provider="openai",
        model="gpt-4o",
    )
    exc.headers = None

    mock_acompletion = AsyncMock(side_effect=exc)
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    adapter = LiteLLMAdapter(api_key="sk-key", retry_config=RetryConfig(max_retries=0))
    req = LLMRequest(model="gpt-4o", messages=[Message(role=Role.USER, content="Hello")])

    with pytest.raises(LLMRateLimitError) as exc_info:
        await adapter.complete(req)

    assert exc_info.value.retry_after_seconds is None


@pytest.mark.parametrize(
    ("status_code", "expected_retryable"),
    [
        (500, True),
        (502, True),
        (503, True),
        (529, True),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (None, False),
        ("invalid_status", False),
    ],
)
async def test_litellm_adapter_maps_api_error(
    monkeypatch: pytest.MonkeyPatch,
    status_code: Any,
    expected_retryable: bool,
) -> None:
    """Verify APIError is mapped to LLMProviderError with status and retryability."""
    exc = APIError(
        status_code=status_code,
        message="API error occurred",
        llm_provider="openai",
        model="gpt-4o",
    )

    mock_acompletion = AsyncMock(side_effect=exc)
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    adapter = LiteLLMAdapter(api_key="sk-key", retry_config=RetryConfig(max_retries=0))
    req = LLMRequest(model="gpt-4o", messages=[Message(role=Role.USER, content="Hello")])

    with pytest.raises(LLMProviderError) as exc_info:
        await adapter.complete(req)

    assert exc_info.value.model == "gpt-4o"
    assert exc_info.value.is_retryable == expected_retryable
    if isinstance(status_code, int):
        assert exc_info.value.status_code == status_code
    else:
        assert exc_info.value.status_code is None


async def test_request_timeout_is_bounded_and_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    calls = 0

    async def complete(**kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        assert kwargs["timeout"] == 0.01 and kwargs["num_retries"] == 0
        if calls == 1:
            await asyncio.Event().wait()
        return _make_dummy_response()

    monkeypatch.setattr(litellm, "acompletion", complete)
    adapter = LiteLLMAdapter(
        api_key="test",
        timeout_seconds=0.01,
        retry_config=RetryConfig(max_retries=1, base_delay_seconds=0, jitter_factor=0),
    )
    result = await adapter.complete(
        LLMRequest(model="fake/test", messages=[Message(role=Role.USER, content="Hi")])
    )
    assert result.content == "Test response" and calls == 2


async def test_request_timeout_exhausts_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    async def hang(**kwargs: Any) -> None:
        await asyncio.Event().wait()

    complete = AsyncMock(side_effect=hang)
    monkeypatch.setattr(litellm, "acompletion", complete)
    adapter = LiteLLMAdapter(
        api_key="test",
        timeout_seconds=0.01,
        retry_config=RetryConfig(max_retries=1, base_delay_seconds=0, jitter_factor=0),
    )
    with pytest.raises(LLMProviderError, match=r"timed out after 0\.01 seconds"):
        await adapter.complete(
            LLMRequest(model="fake/test", messages=[Message(role=Role.USER, content="Hi")])
        )
    assert complete.await_count == 2


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_request_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        LiteLLMAdapter(api_key="test", timeout_seconds=timeout)


@pytest.mark.parametrize("model", ["deepseek/deepseek-v4-flash", "deepseek/deepseek-flash"])
async def test_deepseek_flash_uses_content_mode(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    complete = AsyncMock(return_value=_make_dummy_response())
    monkeypatch.setattr(litellm, "acompletion", complete)
    await LiteLLMAdapter(api_key="test").complete(
        LLMRequest(model=model, messages=[Message(role=Role.USER, content="Hi")]),
    )
    assert complete.call_args.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("missing", ["usage", "prompt_tokens", "completion_tokens", "none"])
def test_usage_availability_distinguishes_missing_fields_from_reported_zero(missing: str) -> None:
    raw = _make_dummy_response(prompt_tokens=0, completion_tokens=0)
    if missing == "usage":
        del raw.usage
    elif missing != "none":
        delattr(raw.usage, missing)
    translated = LiteLLMAdapter._translate_response(raw, "fake", 0.1)
    assert translated.provider_metadata["usage_available"] is (missing == "none")
    assert translated.usage.prompt_tokens == 0 and translated.usage.completion_tokens == 0


async def test_request_pacing_includes_retries_and_concurrent_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from oswald.core.llm.adapters import litellm_adapter as adapter_module

    now = [0.0]
    starts: list[float] = []

    async def sleep(delay: float) -> None:
        now[0] += delay

    async def completion(**kwargs: Any) -> SimpleNamespace:
        starts.append(now[0])
        if len(starts) == 1:
            raise RateLimitError(message="Slow down", llm_provider="gemini", model="test")
        return _make_dummy_response()

    monkeypatch.setattr(adapter_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(
        adapter_module,
        "asyncio",
        SimpleNamespace(
            Lock=asyncio.Lock,
            timeout=asyncio.timeout,
            sleep=sleep,
        ),
    )
    monkeypatch.setattr(litellm, "acompletion", completion)
    adapter = LiteLLMAdapter(
        api_key="test",
        min_request_interval_seconds=13,
        retry_config=RetryConfig(max_retries=1, base_delay_seconds=0, jitter_factor=0),
    )
    request = LLMRequest(model="fake/test", messages=[Message(role=Role.USER, content="Hi")])
    results = await asyncio.gather(adapter.complete(request), adapter.complete(request))
    assert all(result.content == "Test response" for result in results)
    assert starts == [0, 13, 26]


async def test_service_unavailable_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.exceptions import ServiceUnavailableError

    completion = AsyncMock(
        side_effect=[
            ServiceUnavailableError(message="High demand", llm_provider="gemini", model="test"),
            _make_dummy_response(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", completion)
    adapter = LiteLLMAdapter(
        api_key="test",
        retry_config=RetryConfig(max_retries=1, base_delay_seconds=0, jitter_factor=0),
    )
    result = await adapter.complete(
        LLMRequest(model="fake/test", messages=[Message(role=Role.USER, content="Hi")])
    )
    assert result.content == "Test response"
    assert completion.await_count == 2


@pytest.mark.parametrize("interval", [-1, float("inf"), float("nan")])
def test_invalid_request_interval_is_rejected(interval: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        LiteLLMAdapter(api_key="test", min_request_interval_seconds=interval)
