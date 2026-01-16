# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Literal, Optional, Type, Union
import asyncio
import inspect
from functools import wraps

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


def optional_traceable(name: str = "", run_type: RUN_TYPE_T = "chain", log_trace_url: bool = False):
    """
    Optional traceable decorator that wraps functions with LangSmith tracing.

    This decorator always uses @traceable to preserve proper trace nesting.
    When log_trace_url=True, it additionally captures and logs the trace URL
    using get_current_run_tree() after function execution.

    Args:
        name: The name of the trace. Defaults to the function name.
        run_type: The type of run (e.g., "chain", "llm", "tool").
        log_trace_url: If True, log the trace URL after function execution.
    """

    def decorator(func):
        if not HAS_LANGSMITH:
            return func
        try:
            from langsmith import get_current_run_tree, traceable

            # Use provided run_name or fallback to function name
            trace_name = name or getattr(func, "__name__", "agent_operation")

            def _log_trace_url():
                """Log trace URL from current run tree."""
                try:
                    rt = get_current_run_tree()
                    if rt:
                        url = rt.get_url()
                        if url:
                            logger.info(f"LangSmith Trace: {url}")
                            return url
                except Exception as e:
                    logger.debug(f"Failed to get trace URL: {e}")
                return None

            if not log_trace_url:
                # Simple case: just apply traceable decorator
                return traceable(name=trace_name, run_type=run_type)(func)

            # With log_trace_url: wrap function to capture URL after execution
            # Still use @traceable to preserve nesting hierarchy

            if inspect.isasyncgenfunction(func):
                # Handle async generator functions
                @traceable(name=trace_name, run_type=run_type)
                @wraps(func)
                async def async_gen_wrapper(*args, **kwargs):
                    _log_trace_url()
                    async for item in func(*args, **kwargs):
                        yield item

                return async_gen_wrapper
            elif asyncio.iscoroutinefunction(func):
                # Handle async functions
                @traceable(name=trace_name, run_type=run_type)
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    _log_trace_url()
                    return await func(*args, **kwargs)

                return async_wrapper
            else:
                # Handle sync functions
                @traceable(name=trace_name, run_type=run_type)
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    _log_trace_url()
                    return func(*args, **kwargs)

                return sync_wrapper

        except ImportError:
            # If langsmith is not available, just return the original function
            return func

    return decorator


def get_current_trace_url() -> str | None:
    """
    Get the URL of the current LangSmith trace from the run tree.

    Returns:
        The trace URL string, or None if unavailable.

    Usage:
        url = get_current_trace_url()
        if url:
            print(f"View trace: {url}")
    """
    if not HAS_LANGSMITH:
        return None

    try:
        from langsmith import get_current_run_tree

        rt = get_current_run_tree()
        if rt:
            return rt.get_url()
    except Exception:
        pass

    return None
