# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import inspect
import logging
from typing import Literal

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

RUN_TYPE_T = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


def optional_traceable(name: str = "", run_type: RUN_TYPE_T = "chain", context_builder=None):
    """
    Optional traceable decorator that wraps functions with Datus observability.

    The Datus observability path creates an OpenTelemetry span when configured.

    Args:
        name: The name of the trace. Defaults to the function name.
        run_type: The type of run (e.g., "chain", "llm", "tool").
    """
    import functools

    def decorator(func):
        trace_name = name or getattr(func, "__name__", "agent_operation")

        def _prepare_trace_context(args, kwargs):
            token = None
            trace_ctx = None
            try:
                from datus.utils.trace_context import get_trace_context, set_trace_context

                trace_ctx = get_trace_context()
                if trace_ctx is None and context_builder is not None:
                    trace_ctx = context_builder(*args, **kwargs)
                    if trace_ctx is not None:
                        token = set_trace_context(trace_ctx)
            except Exception as e:
                logger.debug(f"Failed to prepare trace context: {e}")
                trace_ctx = None
            return trace_ctx, token

        def _reset_trace_context(token) -> None:
            if token is None:
                return
            try:
                from datus.utils.trace_context import reset_trace_context

                reset_trace_context(token)
            except Exception as e:
                logger.debug(f"Failed to reset trace context: {e}")

        def _span_attributes(trace_ctx) -> dict:
            from datus.utils.trace_context import build_trace_span_attributes

            return build_trace_span_attributes(operation=trace_name, run_type=str(run_type), ctx=trace_ctx)

        def _run_id(trace_ctx) -> str | None:
            if trace_ctx is None:
                return None
            metadata = trace_ctx.metadata or {}
            run_id = metadata.get("run_id") or metadata.get("benchmark_run_id")
            if run_id:
                return str(run_id)
            return str(trace_ctx.session_id) if trace_ctx.session_id else None

        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def _asyncgen_observability_wrapper(*args, **kwargs):
                trace_ctx, token = _prepare_trace_context(args, kwargs)
                try:
                    from datus.observability.manager import get_observability_manager

                    observability = get_observability_manager()
                    with observability.span(trace_name, _span_attributes(trace_ctx), run_id=_run_id(trace_ctx)):
                        async for item in func(*args, **kwargs):
                            yield item
                finally:
                    _reset_trace_context(token)

            return _asyncgen_observability_wrapper

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def _async_observability_wrapper(*args, **kwargs):
                trace_ctx, token = _prepare_trace_context(args, kwargs)
                try:
                    from datus.observability.manager import get_observability_manager

                    observability = get_observability_manager()
                    with observability.span(trace_name, _span_attributes(trace_ctx), run_id=_run_id(trace_ctx)):
                        result = func(*args, **kwargs)
                        if inspect.isawaitable(result):
                            return await result
                        return result
                finally:
                    _reset_trace_context(token)

            return _async_observability_wrapper

        @functools.wraps(func)
        def _observability_wrapper(*args, **kwargs):
            trace_ctx, token = _prepare_trace_context(args, kwargs)
            try:
                from datus.observability.manager import get_observability_manager

                observability = get_observability_manager()
                with observability.span(trace_name, _span_attributes(trace_ctx), run_id=_run_id(trace_ctx)):
                    return func(*args, **kwargs)
            finally:
                _reset_trace_context(token)

        return _observability_wrapper

    return decorator


_tracing_initialized = False


def _suppress_noisy_otel_warnings() -> None:
    """Suppress noisy OpenTelemetry span lifecycle warnings.

    OpenTelemetry logs "Setting attribute on ended span." at WARNING level when
    instrumentation tries to attach attributes after a span has closed. It is
    useful while debugging instrumentation, but too noisy for normal Datus runs.
    Keep ERROR-level OpenTelemetry diagnostics visible.
    """
    logging.getLogger("opentelemetry.sdk.trace").setLevel(logging.ERROR)


def _disable_sdk_tracing(reason: str) -> None:
    """Disable the OpenAI Agents SDK's default tracing to avoid atexit deadlock."""
    try:
        from agents import set_tracing_disabled

        set_tracing_disabled(True)
        logger.debug(f"OpenAI Agents SDK tracing disabled: {reason}")
    except ImportError:
        pass


def setup_tracing(observability_config=None):
    """Set up tracing from the configured Datus observability adapters.

    LangSmith/Langfuse credentials may still come from environment variables,
    but only adapters listed under ``agent.observability.tracing.adapters``
    initialize external trace export.

    Safe to call multiple times; initialization only happens once.
    """
    global _tracing_initialized
    _suppress_noisy_otel_warnings()
    if _tracing_initialized:
        return

    if _setup_configured_observability(observability_config):
        _tracing_initialized = True
        return
    _disable_sdk_tracing("observability tracing not configured")
    tracing = getattr(observability_config, "tracing", None)
    if tracing is not None and getattr(tracing, "explicit", False):
        _tracing_initialized = True


def _setup_configured_observability(observability_config) -> bool | None:
    """Initialize the new observability manager when config explicitly asks for it.

    Returns:
        True: config path handled tracing and legacy setup should not run.
        True: tracing was configured and enabled.
        False: tracing should remain disabled.
    """
    if observability_config is None or not getattr(observability_config, "explicit", False):
        return False

    tracing = getattr(observability_config, "tracing", None)
    if tracing is None or not getattr(tracing, "explicit", False):
        return False

    if not getattr(tracing, "enabled", False):
        return False

    try:
        from datus.observability.manager import configure_observability

        if configure_observability(observability_config):
            return True
    except Exception as e:
        logger.warning(f"Configured observability initialization failed: {e}")

    return False


def get_trace_reference():
    """Return the current stable trace reference, if tracing is active."""
    try:
        from datus.observability.manager import get_trace_reference as get_observability_trace_reference

        ref = get_observability_trace_reference()
        if ref:
            return ref
    except Exception as e:
        logger.debug(f"Failed to get configured observability trace reference: {e}")
    return None
