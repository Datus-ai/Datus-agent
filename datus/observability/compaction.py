# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Compaction evidence using counts already measured by the compactor."""

import time
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any
from uuid import uuid4

from datus.observability.manager import get_observability_manager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class _Compaction:
    metrics: dict[str, Any]
    stack: ExitStack
    started: bool = False
    span: Any = None


_COMPACT: ContextVar[_Compaction | None] = ContextVar("datus_compact", default=None)


def current_compact_id() -> str | None:
    state = _COMPACT.get()
    return state.metrics.get("compact_id") if state else None


def compact_started(**metrics):
    state = _COMPACT.get()
    if state is None:
        return
    state.metrics.update(metrics)
    if not state.started:
        state.started = True
        attrs = {f"datus.compact.{key}": value for key, value in state.metrics.items() if value is not None}
        state.span = state.stack.enter_context(get_observability_manager().span("compact", attrs))
        logger.info("compact.started", **state.metrics)


def compact_metrics(**metrics):
    if (state := _COMPACT.get()) is not None:
        state.metrics.update(metrics)


def observe_compaction(func):
    @wraps(func)
    async def wrapped(self, *args, **kwargs):
        from datus.observability.model_call import remember_compact

        with ExitStack() as stack:
            state = _Compaction(
                metrics={
                    "compact_id": uuid4().hex,
                    "session_id": getattr(self, "session_id", None),
                    "trigger": kwargs.get("reason", "manual"),
                },
                stack=stack,
            )
            token = _COMPACT.set(state)
            started = time.monotonic()
            result = None
            try:
                result = await func(self, *args, **kwargs)
                return result
            finally:
                _COMPACT.reset(token)
                if state.started:
                    result = result if isinstance(result, dict) else {}
                    metrics = state.metrics
                    metrics.update(
                        mode=result.get("mode", metrics.get("mode", "unknown")),
                        status="success" if result.get("success") else "error",
                        duration_ms=round((time.monotonic() - started) * 1000, 2),
                    )
                    for key in ("archived_count", "summary_token", "history_jsonl"):
                        if key in result:
                            metrics[key] = result[key]
                    if isinstance(result.get("items"), list):
                        metrics["items_after"] = len(result["items"])
                    names = {
                        tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
                        for tool in getattr(self, "tools", []) or []
                    }
                    metrics["recovery_tool_available"] = "read_file" in names
                    if (
                        metrics["mode"] == "major"
                        and result.get("history_jsonl")
                        and not metrics["recovery_tool_available"]
                    ):
                        logger.warning(
                            "compact.recovery_tool_unavailable",
                            compact_id=metrics["compact_id"],
                            session_id=metrics["session_id"],
                            required_tool="read_file",
                        )
                    remember_compact(metrics["compact_id"])
                    if state.span is not None:
                        for key, value in metrics.items():
                            if value is not None:
                                state.span.set_attribute(f"datus.compact.{key}", value)
                    logger.info("compact.finished", **metrics)

    return wrapped
