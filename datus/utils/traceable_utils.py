# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Literal, Optional, Type, Union

from openai import AsyncOpenAI, OpenAI

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

HAS_LANGSMITH = False
try:
    from langsmith.client import RUN_TYPE_T

    HAS_LANGSMITH = True
except ImportError:
    RUN_TYPE_T = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


def create_openai_client(
    cls: Type[Union[OpenAI, AsyncOpenAI]],
    api_key: str,
    base_url: Optional[str],
    default_headers: Union[dict[str, str], None] = None,
) -> Union[OpenAI, AsyncOpenAI]:
    # OpenAI client accepts None for base_url (uses default URL)
    client = cls(api_key=api_key, base_url=base_url, default_headers=default_headers)
    if not HAS_LANGSMITH:
        return client
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except ImportError:
        logger.warning("langsmith wrapper not available")
        return client


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


# Global tracing processor instance
_tracing_processor = None


def create_tracing_processor(**kwargs):
    """Create a DatusTracingProcessor that captures trace URLs on trace end.

    This processor subclasses OpenAIAgentsTracingProcessor and intercepts
    on_trace_end() to capture the trace URL directly from the RunTree object,
    bypassing all contextvars propagation issues (asyncio.run boundaries,
    async generator yield boundaries, etc.).

    Returns:
        A DatusTracingProcessor instance, or None if dependencies are unavailable.
    """
    global _tracing_processor
    if _tracing_processor is not None:
        return _tracing_processor

    if not HAS_LANGSMITH:
        return None

    try:
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

        _tracing_processor = DatusTracingProcessor(**kwargs)
        return _tracing_processor
    except ImportError:
        logger.warning("OpenAIAgentsTracingProcessor not available")
        return None


def get_last_trace_url() -> str | None:
    """Get the last trace URL captured by the tracing processor.

    This retrieves the trace URL from the most recently completed trace,
    captured in on_trace_end(). Works reliably across all execution
    boundaries (asyncio.run, async generators, etc.).

    Returns:
        The trace URL string, or None if unavailable.
    """
    if _tracing_processor is not None:
        return _tracing_processor._last_trace_url
    return None
