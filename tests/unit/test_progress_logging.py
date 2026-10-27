"""Readable stderr, parseable stdout, scoped configuration, and bounded disk logs."""

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import structlog

from oswald.cli.__main__ import main
from oswald.core.agents.telemetry import RunProgress, agent_scope, tool_call
from oswald.observability.logging import CompactRenderer, configure_logging


async def test_compact_stderr_and_structured_disk_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    previous_config = structlog.get_config()
    previous_handlers = list(logging.getLogger("oswald").handlers)
    with configure_logging(tmp_path):
        async with agent_scope("coder", "task-123456", 2, run=RunProgress("task-123456")):
            with tool_call("read_file_range"):
                pass
        structlog.get_logger("oswald.test").debug("hidden.debug")
        print('{"status":"SUCCEEDED"}')
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "SUCCEEDED"
    assert "CODER" in captured.err
    assert "elapsed=" in captured.err and "tokens=0/0" in captured.err
    assert "tool_calls=1" in captured.err and "attempt=2" in captured.err
    assert "hidden.debug" not in captured.err
    assert "tool_completed" not in captured.err  # Successful completion details are verbose-only.
    entries = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert entries[-1]["event"] == "agent.completed"
    assert entries[-1]["task_id"] == "task-123456"
    assert entries[-1]["elapsed_seconds"] >= 0
    assert "T" in entries[-1]["timestamp"]  # Disk retains full UTC timestamp.
    assert logging.getLogger("oswald").handlers == previous_handlers
    assert structlog.get_config() == previous_config


async def test_existing_error_logs_are_enriched_and_single_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with configure_logging(tmp_path):
        async with agent_scope("tester", "task", 2):
            structlog.get_logger("oswald.test").warning("existing.error", error="first\nsecond")
    captured = capsys.readouterr()
    entries = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    warning = next(entry for entry in entries if entry["event"] == "existing.error")
    assert warning["agent"] == "tester" and warning["attempt"] == 2
    assert warning["error"] == "first\nsecond"
    assert len(captured.err.splitlines()) == 3


def test_verbose_provider_logs_and_disk_rotation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with configure_logging(tmp_path, verbose=True):
        target = logging.getLogger("oswald")
        disk = next(
            handler for handler in target.handlers if isinstance(handler, RotatingFileHandler)
        )
        assert disk.maxBytes == 1024 * 1024 and disk.backupCount == 2
        disk.maxBytes = 512
        for index in range(20):
            structlog.get_logger("oswald.test").debug(
                "llm.completion.started", index=index, detail="x" * 100
            )
    assert "llm.completion.started" in capsys.readouterr().err
    files = list(tmp_path.glob("progress.jsonl*"))
    assert len(files) == 3
    for path in files:
        assert path.stat().st_size <= 512
        for line in path.read_text().splitlines():
            assert json.loads(line)["event"] == "llm.completion.started"


def test_renderer_handles_unknown_usage_and_bounds_error_text() -> None:
    renderer = CompactRenderer()
    line = renderer(
        None,
        "info",
        {
            "event": "agent.failed",
            "agent": "coder",
            "elapsed_seconds": 12.4,
            "tokens_in": None,
            "tokens_out": None,
            "tool_calls": 2,
            "error": "multiline\n" + "x" * 5000,
        },
    )
    assert "tokens=unknown/unknown" in line and "tool_calls=2" in line
    assert "\n" not in line and len(line) < 400


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_progress_interval(
    monkeypatch: pytest.MonkeyPatch, interval: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["oswald", "Inspect", f"--progress-interval={interval}"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_cli_progress_and_verbose_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "oswald",
            "Inspect",
            "--output",
            str(tmp_path),
            "--progress-interval",
            "2",
            "--verbose",
        ],
    )
    with patch("oswald.cli.__main__.run", new_callable=AsyncMock, return_value=0) as execute:
        assert main() == 0
    args = execute.call_args.args[0]
    assert args.progress_interval == 2 and args.verbose is True
