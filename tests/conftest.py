"""Root pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_fixture() -> str:
    """Sample fixture available across test suites."""
    return "oswald"


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep repository checks independent of a developer's live API configuration."""
    from oswald.config import Settings

    monkeypatch.setattr(Settings, "model_config", {**Settings.model_config, "env_file": None})
    for field in Settings.model_fields.values():
        if isinstance(field.validation_alias, str):
            monkeypatch.delenv(field.validation_alias, raising=False)
