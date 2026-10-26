"""Unit tests for create_llm_client factory function."""

from oswald.config import Settings
from oswald.core.llm import create_llm_client
from oswald.core.llm.adapters.litellm_adapter import LiteLLMAdapter
from oswald.core.llm.protocol import LLMClient


def test_create_llm_client_with_custom_settings() -> None:
    """Verify create_llm_client wires API key, base URL, and run_id into LiteLLMAdapter."""
    settings = Settings(
        llm_provider_key="custom-secret-key",
        llm_base_url="http://localhost:11434/v1",
    )

    client = create_llm_client(settings, run_id="run-factory-42")

    assert isinstance(client, LLMClient)
    assert isinstance(client, LiteLLMAdapter)
    assert client._api_key == "custom-secret-key"
    assert client._api_base == "http://localhost:11434/v1"
    assert client._run_id == "run-factory-42"
    assert client._timeout_seconds == settings.llm_timeout_seconds
    assert client._retry_config.max_retries == settings.llm_max_retries


def test_create_llm_client_with_default_settings() -> None:
    """Verify create_llm_client works cleanly with default Settings values."""
    settings = Settings()

    client = create_llm_client(settings)

    assert isinstance(client, LLMClient)
    assert isinstance(client, LiteLLMAdapter)
    assert client._api_key == settings.llm_provider_key
    assert client._api_base is None
    assert client._run_id == ""
