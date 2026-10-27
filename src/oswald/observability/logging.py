"""Compact stderr logs plus a bounded structured log, using the existing structlog pipeline."""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog
from structlog.typing import EventDict, Processor

from oswald.core.agents.telemetry import add_progress_context


class CompactRenderer:
    """Render single-line terminal logs while JSON retains full structured fields."""

    def __call__(self, _logger: Any, _name: str, event: EventDict) -> str:
        data = dict(event)
        timestamp = data.pop("timestamp", "")
        level = str(data.pop("level", "info")).upper()
        name = str(data.pop("event", ""))
        agent = str(data.pop("agent", "run" if name.startswith("run.") else "system")).upper()
        action = name.removeprefix("agent.").removeprefix("run.")
        parts = [f"{timestamp} {level:<7} {agent:<8} {action:<14}"]
        if "elapsed_seconds" in data:
            parts.append(f"elapsed={data.pop('elapsed_seconds'):.1f}s")
        if "tokens_in" in data:
            incoming = data.pop("tokens_in")
            outgoing = data.pop("tokens_out", None)
            parts.append(
                f"tokens={incoming if incoming is not None else 'unknown'}/{outgoing if outgoing is not None else 'unknown'}"
            )
        for key in ("tool_calls", "tool_errors", "llm_calls", "attempt", "task_id"):
            if key in data:
                value = data.pop(key)
                if key == "tool_errors" and value == 0:
                    continue
                if key == "task_id":
                    value = str(value)[:8]
                parts.append(f"{key}={value}")
        for key, value in data.items():
            # Keep arbitrary exception messages on one bounded terminal line.
            text = json.dumps(value, ensure_ascii=False, default=str)
            if len(text) > 200:
                text = text[:197] + "..."
            parts.append(f"{key}={text}")
        return " ".join(parts)


@contextmanager
def configure_logging(output_dir: Path, *, verbose: bool = False) -> Iterator[None]:
    """Write logs to stderr and at most three 1 MiB JSONL files; leave stdout for results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    previous = structlog.get_config()
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        add_progress_context,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    console = logging.StreamHandler()
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                _terminal_time,
                CompactRenderer(),
            ]
        )
    )
    disk = RotatingFileHandler(
        output_dir / "progress.jsonl",
        maxBytes=1024 * 1024,
        backupCount=2,
        encoding="utf-8",
        delay=True,
    )
    disk.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ]
        )
    )
    target = logging.getLogger("oswald")
    previous_level, previous_propagate = target.level, target.propagate
    for handler in (console, disk):
        target.addHandler(handler)
    target.setLevel(level)
    target.propagate = False
    try:
        yield
    finally:
        for handler in (console, disk):
            target.removeHandler(handler)
            handler.close()
        target.setLevel(previous_level)
        target.propagate = previous_propagate
        structlog.configure(**previous)


def _terminal_time(_logger: Any, _name: str, event: EventDict) -> EventDict:
    timestamp = event.get("timestamp", "")
    if isinstance(timestamp, str) and "T" in timestamp:
        event["timestamp"] = timestamp.split("T", 1)[1][:8] + "Z"
    return event
