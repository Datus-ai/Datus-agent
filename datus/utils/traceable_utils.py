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

HAS_LANGFUSE = False
try:
    import langfuse as _langfuse_mod  # noqa: F401

    HAS_LANGFUSE = True
except ImportError:
    pass

_LANGFUSE_TYPE_MAP = {
    "chain": "span",
    "tool": "tool",
    "llm": "generation",
    "retriever": "span",
    "embedding": "span",
    "agent": "span",
    "prompt": "span",
    "parser": "span",
}

_langfuse_enabled = False


def optional_traceable(name: str = "", run_type: RUN_TYPE_T = "chain"):
    """
    Optional traceable decorator that wraps functions with LangSmith and/or Langfuse tracing.

    LangSmith wrapping is applied eagerly at decoration time (import-time).
    Langfuse wrapping is deferred to the first call so that credentials loaded
    via load_dotenv() after module import are honoured.

    Args:
        name: The name of the trace. Defaults to the function name.
        run_type: The type of run (e.g., "chain", "llm", "tool").
    """
    import functools

    def decorator(func):
        wrapped = func

        if HAS_LANGSMITH:
            try:
                from langsmith import traceable

                trace_name = name or getattr(func, "__name__", "agent_operation")
                wrapped = traceable(name=trace_name, run_type=run_type)(wrapped)
            except ImportError:
                pass

        if HAS_LANGFUSE:
            _inner = wrapped
            _observed = None

            @functools.wraps(_inner)
            def _langfuse_lazy(*args, **kwargs):
                nonlocal _observed
                if _langfuse_enabled:
                    if _observed is None:
                        from langfuse import observe

                        t = name or getattr(func, "__name__", "agent_operation")
                        _observed = observe(name=t, as_type=_LANGFUSE_TYPE_MAP.get(run_type, "span"))(_inner)
                    return _observed(*args, **kwargs)
                return _inner(*args, **kwargs)

            wrapped = _langfuse_lazy

        return wrapped

    return decorator


_tracing_initialized = False
_tracing_processor = None


def _is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is explicitly enabled via environment variables."""
    import os

    tracing_enabled = (
        os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
        or os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    )
    has_api_key = bool(os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY"))
    return tracing_enabled and has_api_key


def _is_langfuse_enabled() -> bool:
    """Check if Langfuse tracing is enabled via environment variables."""
    import os

    if not HAS_LANGFUSE:
        return False
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(os.environ.get("LANGFUSE_SECRET_KEY"))


def _disable_sdk_tracing(reason: str) -> None:
    """Disable the OpenAI Agents SDK's default tracing to avoid atexit deadlock."""
    try:
        from agents import set_tracing_disabled

        set_tracing_disabled(True)
        logger.debug(f"OpenAI Agents SDK tracing disabled: {reason}")
    except ImportError:
        pass


def _setup_langfuse_tracing(*, langsmith_active: bool = False) -> None:
    """Configure Langfuse tracing via LiteLLM callback and optionally OpenAI Agents SDK instrumentor.

    LiteLLM callback is always registered to capture direct litellm.completion()
    calls (used by non-agentic nodes).  When OpenInference is available, the Agents
    SDK instrumentor is added for full agent/tool/handoff tracing.

    Args:
        langsmith_active: True when DatusTracingProcessor was already installed
            via set_trace_processors (which removes the SDK default exporter).
    """
    global _langfuse_enabled
    import base64
    import os

    import litellm

    if not os.environ.get("LANGFUSE_OTEL_HOST"):
        base_url = os.environ.get("LANGFUSE_HOST", os.environ.get("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com"))
        os.environ["LANGFUSE_OTEL_HOST"] = base_url

    callback_name = "langfuse_otel"
    if callback_name not in (litellm.success_callback or []):
        litellm.success_callback = litellm.success_callback or []
        litellm.success_callback.append(callback_name)
    if callback_name not in (litellm.failure_callback or []):
        litellm.failure_callback = litellm.failure_callback or []
        litellm.failure_callback.append(callback_name)

    try:
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        otel_host = os.environ["LANGFUSE_OTEL_HOST"]
        auth = base64.b64encode(
            f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
        ).decode()
        exporter = OTLPSpanExporter(
            endpoint=f"{otel_host}/api/public/otel/v1/traces",
            headers={
                "Authorization": f"Basic {auth}",
                "x-langfuse-ingestion-version": "4",
            },
        )
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

        # When LangSmith is active, set_trace_processors already replaced the
        # SDK default exporter → use exclusive_processor=False to add alongside.
        # Otherwise use exclusive_processor=True to replace the default exporter
        # and prevent traces from also going to OpenAI's dashboard.
        exclusive = not langsmith_active
        OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider, exclusive_processor=exclusive)
        logger.info("Langfuse tracing enabled (LiteLLM callback + OpenAI Agents SDK instrumentor)")
    except ImportError:
        if not (HAS_LANGSMITH and _is_tracing_enabled()):
            _disable_sdk_tracing("openinference not installed and no LangSmith, only LiteLLM callbacks active")
        logger.info("Langfuse tracing enabled (LiteLLM callback only, openinference not installed)")

    _langfuse_enabled = True


def setup_tracing():
    """Set up tracing with LangSmith and/or Langfuse.

    LangSmith: Creates a DatusTracingProcessor (subclass of OpenAIAgentsTracingProcessor)
    that captures trace URLs on trace end, and registers it via set_trace_processors.

    Langfuse: Registers LiteLLM callbacks and OpenAI Agents SDK instrumentor (additive,
    coexists with LangSmith).

    Safe to call multiple times; initialization only happens once.
    """
    global _tracing_initialized, _tracing_processor
    if _tracing_initialized:
        return
    _tracing_initialized = True

    langsmith_enabled = HAS_LANGSMITH and _is_tracing_enabled()
    langfuse_enabled = _is_langfuse_enabled()

    if not langsmith_enabled and not langfuse_enabled:
        if not HAS_LANGSMITH:
            _disable_sdk_tracing("langsmith not installed")
        else:
            logger.debug("LangSmith tracing not enabled (set LANGSMITH_TRACING=true and LANGCHAIN_API_KEY to enable)")
            _disable_sdk_tracing("LANGSMITH_TRACING/api key not set")
        return

    if langsmith_enabled:
        try:
            from agents import set_trace_processors
            from langsmith.wrappers import OpenAIAgentsTracingProcessor

            class DatusTracingProcessor(OpenAIAgentsTracingProcessor):
                """Extended tracing processor that captures trace URLs."""

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._last_trace_url: str | None = None

                def on_trace_end(self, trace) -> None:
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
    if langfuse_enabled:
        _setup_langfuse_tracing(langsmith_active=langsmith_enabled)


def _get_langfuse_trace_url() -> str | None:
    """Get trace URL from Langfuse using the SDK helper (includes project segment)."""
    try:
        from langfuse import get_client

        lf = get_client()
        trace_id = lf.get_current_trace_id()
        if trace_id:
            return lf.get_trace_url(trace_id=trace_id)
    except Exception as e:
        logger.debug(f"Failed to get Langfuse trace URL: {e}")
    return None


def get_trace_url() -> str | None:
    """Return the last captured trace URL (LangSmith or Langfuse), or None."""
    if _tracing_processor is not None:
        url = getattr(_tracing_processor, "_last_trace_url", None)
        if url:
            return url

    if _langfuse_enabled:
        return _get_langfuse_trace_url()

    return None
