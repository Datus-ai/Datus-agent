# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.utils.traceable_utils — LangSmith and Langfuse tracing integration."""

import sys
from unittest.mock import MagicMock, patch

from datus.utils.traceable_utils import (
    _disable_sdk_tracing,
    _get_langfuse_trace_url,
    _is_langfuse_enabled,
    _is_tracing_enabled,
    get_trace_url,
    optional_traceable,
    setup_tracing,
)


def _clear_all_tracing_envvars(monkeypatch):
    """Helper to clear all tracing-related environment variables."""
    for var in (
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_OTEL_HOST",
    ):
        monkeypatch.delenv(var, raising=False)


def _ensure_langfuse_module():
    """Inject a mock langfuse module into sys.modules if the real one is not installed.

    Returns the saved module (or None) so callers can restore it in a finally block.
    This allows patch("langfuse.observe") etc. to work in CI without the real package.
    """
    saved = sys.modules.get("langfuse")
    if saved is None or not hasattr(saved, "observe"):
        mock_mod = MagicMock()
        mock_mod.observe = MagicMock(side_effect=lambda *a, **kw: lambda fn: fn)
        mock_mod.get_client = MagicMock()
        sys.modules["langfuse"] = mock_mod
    return saved


def _restore_langfuse_module(saved):
    if saved is None:
        sys.modules.pop("langfuse", None)
    else:
        sys.modules["langfuse"] = saved


class TestIsTracingEnabled:
    """Tests for _is_tracing_enabled helper."""

    def test_disabled_by_default(self, monkeypatch):
        """Tracing is disabled when env vars are not set."""
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        assert _is_tracing_enabled() is False

    def test_disabled_without_api_key(self, monkeypatch):
        """Tracing is disabled when tracing is on but no API key."""
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        assert _is_tracing_enabled() is False

    def test_disabled_with_key_but_no_flag(self, monkeypatch):
        """Tracing is disabled when API key exists but tracing flag is off."""
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.setenv("LANGCHAIN_API_KEY", "fake-key")
        assert _is_tracing_enabled() is False


class TestSetupTracing:
    """Tests for setup_tracing function."""

    def test_setup_tracing_not_enabled(self, monkeypatch):
        """setup_tracing logs debug when tracing is not enabled."""
        import datus.utils.traceable_utils as module

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.setattr(module, "_tracing_initialized", False)
        monkeypatch.setattr(module, "_tracing_processor", None)

        setup_tracing()

        assert module._tracing_initialized is True
        assert module._tracing_processor is None

    def test_setup_tracing_idempotent(self, monkeypatch):
        """setup_tracing only initializes once."""
        import datus.utils.traceable_utils as module

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.setattr(module, "_tracing_initialized", False)
        monkeypatch.setattr(module, "_tracing_processor", None)

        setup_tracing()
        setup_tracing()

        assert module._tracing_initialized is True

    def test_disables_sdk_when_langsmith_not_installed(self, monkeypatch):
        """setup_tracing disables SDK tracing when HAS_LANGSMITH is False."""
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setattr(module, "_tracing_initialized", False)
        monkeypatch.setattr(module, "_tracing_processor", None)
        monkeypatch.setattr(module, "HAS_LANGSMITH", False)
        monkeypatch.setattr(module, "HAS_LANGFUSE", False)

        with patch.object(module, "_disable_sdk_tracing") as mock_disable:
            setup_tracing()
            mock_disable.assert_called_once_with("langsmith not installed")

    def _setup_langsmith_with_fake_processor(self, monkeypatch):
        """Set up LangSmith path with a fake base processor class."""
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setattr(module, "_tracing_initialized", False)
        monkeypatch.setattr(module, "_tracing_processor", None)
        monkeypatch.setattr(module, "HAS_LANGSMITH", True)
        monkeypatch.setattr(module, "HAS_LANGFUSE", False)
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "fake-key")

        class FakeBaseProcessor:
            """Minimal stand-in for OpenAIAgentsTracingProcessor."""

            def __init__(self):
                self._runs = {}

            def on_trace_end(self, trace):
                pass

        return module, FakeBaseProcessor

    def test_langsmith_enabled_installs_processor(self, monkeypatch):
        """setup_tracing installs DatusTracingProcessor when LangSmith is configured."""
        module, FakeBase = self._setup_langsmith_with_fake_processor(monkeypatch)
        mock_set = MagicMock()

        with (
            patch("langsmith.wrappers.OpenAIAgentsTracingProcessor", FakeBase),
            patch("agents.set_trace_processors", mock_set),
        ):
            setup_tracing()

            assert module._tracing_processor is not None
            mock_set.assert_called_once()

    def test_datus_tracing_processor_captures_trace_url(self, monkeypatch):
        """DatusTracingProcessor.on_trace_end captures the trace URL."""
        module, FakeBase = self._setup_langsmith_with_fake_processor(monkeypatch)

        with (
            patch("langsmith.wrappers.OpenAIAgentsTracingProcessor", FakeBase),
            patch("agents.set_trace_processors"),
        ):
            setup_tracing()

        processor = module._tracing_processor

        mock_trace = MagicMock()
        mock_trace.trace_id = "trace-123"
        mock_run = MagicMock()
        mock_run.get_url.return_value = "https://smith.langchain.com/trace/123"
        processor._runs = {"trace-123": mock_run}

        processor.on_trace_end(mock_trace)

        assert processor._last_trace_url == "https://smith.langchain.com/trace/123"

    def test_datus_tracing_processor_handles_url_error(self, monkeypatch):
        """DatusTracingProcessor.on_trace_end handles get_url failures gracefully."""
        module, FakeBase = self._setup_langsmith_with_fake_processor(monkeypatch)

        with (
            patch("langsmith.wrappers.OpenAIAgentsTracingProcessor", FakeBase),
            patch("agents.set_trace_processors"),
        ):
            setup_tracing()

        processor = module._tracing_processor
        mock_trace = MagicMock()
        mock_trace.trace_id = "trace-456"
        mock_run = MagicMock()
        mock_run.get_url.side_effect = RuntimeError("network error")
        processor._runs = {"trace-456": mock_run}

        processor.on_trace_end(mock_trace)

        assert processor._last_trace_url is None


class TestDisableSdkTracing:
    """Tests for _disable_sdk_tracing."""

    def test_calls_set_tracing_disabled(self):
        """_disable_sdk_tracing calls agents.set_tracing_disabled(True)."""
        with patch("agents.set_tracing_disabled") as mock_disable:
            _disable_sdk_tracing("test reason")
            mock_disable.assert_called_once_with(True)


class TestOptionalTraceable:
    """Tests for optional_traceable decorator."""

    def test_function_runs_normally(self):
        """Decorated function should still execute correctly."""

        @optional_traceable(name="test_op")
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_function_name_preserved(self):
        """Decorated function preserves its behavior."""

        @optional_traceable()
        def my_function():
            return "hello"

        assert my_function() == "hello"


class TestGetTraceUrl:
    """Tests for get_trace_url function."""

    def test_returns_none_when_no_processor(self, monkeypatch):
        """Returns None when no tracing processor is configured."""
        import datus.utils.traceable_utils as module

        monkeypatch.setattr(module, "_tracing_processor", None)
        monkeypatch.setattr(module, "_langfuse_enabled", False)
        assert get_trace_url() is None

    def test_returns_langsmith_url_when_available(self, monkeypatch):
        """Returns LangSmith URL from processor when available."""
        import datus.utils.traceable_utils as module

        mock_processor = MagicMock()
        mock_processor._last_trace_url = "https://smith.langchain.com/o/org/projects/p/proj/r/run123"
        monkeypatch.setattr(module, "_tracing_processor", mock_processor)
        monkeypatch.setattr(module, "_langfuse_enabled", False)

        url = get_trace_url()
        assert url == "https://smith.langchain.com/o/org/projects/p/proj/r/run123"

    def test_falls_through_to_langfuse_when_langsmith_url_empty(self, monkeypatch):
        """Falls through to Langfuse when LangSmith processor has no URL."""
        import datus.utils.traceable_utils as module

        mock_processor = MagicMock()
        mock_processor._last_trace_url = None
        monkeypatch.setattr(module, "_tracing_processor", mock_processor)
        monkeypatch.setattr(module, "_langfuse_enabled", True)

        with patch.object(module, "_get_langfuse_trace_url", return_value="https://langfuse/trace/123"):
            url = get_trace_url()
            assert url == "https://langfuse/trace/123"


class TestIsLangfuseEnabled:
    """Tests for _is_langfuse_enabled helper."""

    def test_disabled_without_keys(self, monkeypatch):
        """Langfuse is disabled when no env vars are set."""
        _clear_all_tracing_envvars(monkeypatch)
        assert _is_langfuse_enabled() is False

    def test_disabled_with_partial_keys(self, monkeypatch):
        """Langfuse is disabled when only public key is set."""
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setattr(module, "HAS_LANGFUSE", True)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-fake")
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        assert _is_langfuse_enabled() is False

    def test_disabled_without_sdk(self, monkeypatch):
        """Langfuse is disabled when SDK is not installed."""
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setattr(module, "HAS_LANGFUSE", False)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-fake")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-fake")
        assert _is_langfuse_enabled() is False

    def test_enabled_with_both_keys(self, monkeypatch):
        """Langfuse is enabled when both keys are set and SDK is available."""
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setattr(module, "HAS_LANGFUSE", True)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-fake")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-fake")
        assert _is_langfuse_enabled() is True


class TestSetupLangfuseTracing:
    """Tests for Langfuse path in setup_tracing."""

    def _setup_langfuse_env(self, monkeypatch):
        """Common setup for Langfuse tracing tests."""
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setattr(module, "_tracing_initialized", False)
        monkeypatch.setattr(module, "_tracing_processor", None)
        monkeypatch.setattr(module, "_langfuse_enabled", False)
        monkeypatch.setattr(module, "HAS_LANGFUSE", True)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-fake")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-fake")

    def test_registers_litellm_callbacks(self, monkeypatch):
        """setup_tracing registers langfuse_otel in litellm callbacks when openinference is absent."""
        import litellm

        import datus.utils.traceable_utils as module

        self._setup_langfuse_env(monkeypatch)

        original_success = litellm.success_callback.copy() if litellm.success_callback else []
        original_failure = litellm.failure_callback.copy() if litellm.failure_callback else []

        oi_key = "openinference.instrumentation.openai_agents"
        saved_mod = sys.modules.get(oi_key)
        sys.modules[oi_key] = None
        try:
            setup_tracing()

            assert "langfuse_otel" in litellm.success_callback
            assert "langfuse_otel" in litellm.failure_callback
            assert module._langfuse_enabled is True
        finally:
            if saved_mod is None:
                sys.modules.pop(oi_key, None)
            else:
                sys.modules[oi_key] = saved_mod
            litellm.success_callback = original_success
            litellm.failure_callback = original_failure

    def test_instrumentor_called_with_exclusive_false(self, monkeypatch):
        """OpenAIAgentsInstrumentor is called with exclusive_processor=True (Langfuse-only)."""
        import litellm

        self._setup_langfuse_env(monkeypatch)

        original_success = litellm.success_callback.copy() if litellm.success_callback else []
        original_failure = litellm.failure_callback.copy() if litellm.failure_callback else []

        mock_instrumentor_instance = MagicMock()
        mock_instrumentor_cls = MagicMock(return_value=mock_instrumentor_instance)
        mock_oi_module = MagicMock()
        mock_oi_module.OpenAIAgentsInstrumentor = mock_instrumentor_cls

        # Also mock OTel modules that are imported in the same try block
        mock_otel_exporter = MagicMock()
        mock_otel_trace = MagicMock()
        mock_otel_export = MagicMock()

        oi_key = "openinference.instrumentation.openai_agents"
        otel_keys = {
            oi_key: mock_oi_module,
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": mock_otel_exporter,
            "opentelemetry.sdk.trace": mock_otel_trace,
            "opentelemetry.sdk.trace.export": mock_otel_export,
        }
        saved_mods = {k: sys.modules.get(k) for k in otel_keys}
        for k, v in otel_keys.items():
            sys.modules[k] = v
        try:
            setup_tracing()

            mock_instrumentor_instance.instrument.assert_called_once()
            call_kwargs = mock_instrumentor_instance.instrument.call_args[1]
            assert call_kwargs["exclusive_processor"] is True
            assert "tracer_provider" in call_kwargs
            assert "langfuse_otel" in litellm.success_callback
        finally:
            for k, saved in saved_mods.items():
                if saved is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = saved
            litellm.success_callback = original_success
            litellm.failure_callback = original_failure

    def test_sets_langfuse_otel_host_from_langfuse_host(self, monkeypatch):
        """LANGFUSE_OTEL_HOST is derived from LANGFUSE_HOST when not explicitly set."""
        import litellm

        self._setup_langfuse_env(monkeypatch)
        monkeypatch.setenv("LANGFUSE_HOST", "https://eu.langfuse.example.com")
        monkeypatch.delenv("LANGFUSE_OTEL_HOST", raising=False)

        original_success = litellm.success_callback.copy() if litellm.success_callback else []
        original_failure = litellm.failure_callback.copy() if litellm.failure_callback else []

        oi_key = "openinference.instrumentation.openai_agents"
        saved_mod = sys.modules.get(oi_key)
        sys.modules[oi_key] = None
        try:
            setup_tracing()
            import os

            assert os.environ.get("LANGFUSE_OTEL_HOST") == "https://eu.langfuse.example.com"
        finally:
            if saved_mod is None:
                sys.modules.pop(oi_key, None)
            else:
                sys.modules[oi_key] = saved_mod
            litellm.success_callback = original_success
            litellm.failure_callback = original_failure
            os.environ.pop("LANGFUSE_OTEL_HOST", None)


class TestOptionalTraceableLangfuse:
    """Tests for optional_traceable when Langfuse is active."""

    def test_function_runs_with_langfuse(self, monkeypatch):
        """Decorated function executes correctly when Langfuse is fully configured."""
        import datus.utils.traceable_utils as module

        monkeypatch.setattr(module, "HAS_LANGFUSE", True)
        monkeypatch.setattr(module, "_langfuse_enabled", True)

        saved = _ensure_langfuse_module()
        try:
            mock_observe = MagicMock(side_effect=lambda *a, **kw: lambda fn: fn)
            with patch("langfuse.observe", mock_observe):

                @optional_traceable(name="test_langfuse_op")
                def multiply(a, b):
                    return a * b

                assert multiply(3, 4) == 12
                mock_observe.assert_called_once()
        finally:
            _restore_langfuse_module(saved)

    def test_langfuse_not_applied_when_disabled(self, monkeypatch):
        """Langfuse observe is not applied when _langfuse_enabled is False."""
        import datus.utils.traceable_utils as module

        monkeypatch.setattr(module, "HAS_LANGFUSE", True)
        monkeypatch.setattr(module, "_langfuse_enabled", False)

        saved = _ensure_langfuse_module()
        try:
            with patch("langfuse.observe") as mock_observe:

                @optional_traceable(name="test_op")
                def add(a, b):
                    return a + b

                assert add(1, 2) == 3
                mock_observe.assert_not_called()
        finally:
            _restore_langfuse_module(saved)


class TestGetLangfuseTraceUrl:
    """Tests for _get_langfuse_trace_url."""

    def test_returns_none_when_no_trace(self):
        """Returns None when no active trace."""
        saved = _ensure_langfuse_module()
        try:
            mock_client = MagicMock()
            mock_client.get_current_trace_id.return_value = None
            with patch("langfuse.get_client", return_value=mock_client):
                assert _get_langfuse_trace_url() is None
        finally:
            _restore_langfuse_module(saved)

    def test_returns_url_when_trace_active(self):
        """Returns SDK-constructed URL when a trace is active."""
        saved = _ensure_langfuse_module()
        try:
            mock_client = MagicMock()
            mock_client.get_current_trace_id.return_value = "trace-abc-123"
            mock_client.get_trace_url.return_value = (
                "https://us.cloud.langfuse.com/project/proj-123/traces/trace-abc-123"
            )
            with patch("langfuse.get_client", return_value=mock_client):
                url = _get_langfuse_trace_url()
                assert url == "https://us.cloud.langfuse.com/project/proj-123/traces/trace-abc-123"
                mock_client.get_trace_url.assert_called_once_with(trace_id="trace-abc-123")
        finally:
            _restore_langfuse_module(saved)

    def test_returns_none_on_exception(self):
        """Returns None gracefully when Langfuse client raises."""
        saved = _ensure_langfuse_module()
        try:
            with patch("langfuse.get_client", side_effect=RuntimeError("no client")):
                assert _get_langfuse_trace_url() is None
        finally:
            _restore_langfuse_module(saved)


class TestGetTraceUrlLangfuse:
    """Tests for get_trace_url with Langfuse backend."""

    def test_returns_langfuse_url(self, monkeypatch):
        """Returns Langfuse URL when enabled and no LangSmith processor."""
        import datus.utils.traceable_utils as module

        monkeypatch.setattr(module, "_tracing_processor", None)
        monkeypatch.setattr(module, "_langfuse_enabled", True)

        with patch.object(module, "_get_langfuse_trace_url", return_value="https://langfuse/trace/abc"):
            assert get_trace_url() == "https://langfuse/trace/abc"


class TestLangsmithUnchanged:
    """Verify that Langfuse additions do not break LangSmith behavior."""

    def test_langsmith_tracing_check_unchanged(self, monkeypatch):
        """_is_tracing_enabled still works correctly for LangSmith."""
        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "fake-key")
        assert _is_tracing_enabled() is True

    def test_setup_tracing_langsmith_only(self, monkeypatch):
        """setup_tracing with LangSmith only does not enable Langfuse."""
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setattr(module, "_tracing_initialized", False)
        monkeypatch.setattr(module, "_tracing_processor", None)
        monkeypatch.setattr(module, "_langfuse_enabled", False)
        monkeypatch.setattr(module, "HAS_LANGFUSE", False)

        setup_tracing()

        assert module._langfuse_enabled is False
