# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Literal

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

HAS_LANGSMITH = False
try:
    from langsmith.client import RUN_TYPE_T

    HAS_LANGSMITH = True
except ImportError:
    RUN_TYPE_T = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


def optional_traceable(name: str = "", run_type: RUN_TYPE_T = "chain"):
    """
    Optional traceable decorator that wraps functions with LangSmith tracing.

    Args:
        name: The name of the trace. Defaults to the function name.
        run_type: The type of run (e.g., "chain", "llm", "tool").
    """

    def decorator(func):
        if not HAS_LANGSMITH:
            return func
        try:
            from langsmith import traceable

            trace_name = name or getattr(func, "__name__", "agent_operation")
            return traceable(name=trace_name, run_type=run_type)(func)
        except ImportError:
            return func

    return decorator


_tracing_initialized = False
_tracing_processor = None


def setup_tracing():
    """Set up LangSmith tracing with DatusTracingProcessor.

    Creates a DatusTracingProcessor (subclass of OpenAIAgentsTracingProcessor)
    that captures trace URLs on trace end, and registers it via set_trace_processors.

    Safe to call multiple times; initialization only happens once.
    """
    global _tracing_initialized, _tracing_processor
    if _tracing_initialized:
        return
    _tracing_initialized = True

    if not HAS_LANGSMITH:
        return

    try:
        from agents import set_trace_processors
        from langsmith.wrappers import OpenAIAgentsTracingProcessor

        class DatusTracingProcessor(OpenAIAgentsTracingProcessor):
            """Extended tracing processor that captures trace URLs."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._last_trace_url: str | None = None

            def on_trace_end(self, trace) -> None:
                # Capture trace URL from RunTree before super() pops it
                run = self._runs.get(trace.trace_id)
                if run:
                    try:
                        self._last_trace_url = run.get_url()
                        logger.info(f"LangSmith Trace: {self._last_trace_url}")
                    except Exception as e:
                        logger.debug(f"Failed to get trace URL: {e}")
                super().on_trace_end(trace)

        _tracing_processor = DatusTracingProcessor()
        set_trace_processors([_tracing_processor])
        logger.info("LangSmith DatusTracingProcessor enabled for SDK tracing")
    except ImportError:
        logger.warning("OpenAIAgentsTracingProcessor not available")


def get_trace_url() -> str | None:
    """Return the last captured LangSmith trace URL, or None."""
    if _tracing_processor is not None:
        return getattr(_tracing_processor, "_last_trace_url", None)
    return None
