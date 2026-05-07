# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.utils.traceable_utils — LangSmith and Langfuse tracing integration."""

from unittest.mock import MagicMock, patch

from datus.utils.traceable_utils import (
    _is_langfuse_enabled,
    _is_tracing_enabled,
    get_trace_url,
    optional_traceable,
    setup_tracing,
)


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
        # Reset the initialization flag to allow re-entry
        monkeypatch.setattr(module, "_tracing_initialized", False)
        monkeypatch.setattr(module, "_tracing_processor", None)

        setup_tracing()

        # After calling, it should be initialized
        assert module._tracing_initialized is True
        # But no processor since tracing is not enabled
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
        setup_tracing()  # second call should be no-op

        assert module._tracing_initialized is True


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
    ):
        monkeypatch.delenv(var, raising=False)


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
        """setup_tracing registers langfuse_otel in litellm callbacks when Langfuse is configured."""
        import sys

        import litellm

        import datus.utils.traceable_utils as module

        self._setup_langfuse_env(monkeypatch)

        original_success = litellm.success_callback.copy() if litellm.success_callback else []
        original_failure = litellm.failure_callback.copy() if litellm.failure_callback else []

        # Hide openinference so _setup_langfuse_tracing hits the ImportError branch.
        # This works even in CI where the package is not installed.
        oi_key = "openinference.instrumentation.openai_agents"
        saved_mod = sys.modules.get(oi_key)
        sys.modules[oi_key] = None  # force ImportError on import
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
        """OpenAIAgentsInstrumentor is called with exclusive_processor=False."""
        import sys

        import litellm

        self._setup_langfuse_env(monkeypatch)

        original_success = litellm.success_callback.copy() if litellm.success_callback else []
        original_failure = litellm.failure_callback.copy() if litellm.failure_callback else []

        # Inject a mock module so the import inside _setup_langfuse_tracing succeeds
        # even in CI where openinference is not installed.
        mock_instrumentor_instance = MagicMock()
        mock_instrumentor_cls = MagicMock(return_value=mock_instrumentor_instance)
        mock_oi_module = MagicMock()
        mock_oi_module.OpenAIAgentsInstrumentor = mock_instrumentor_cls

        oi_key = "openinference.instrumentation.openai_agents"
        saved_mod = sys.modules.get(oi_key)
        sys.modules[oi_key] = mock_oi_module
        try:
            setup_tracing()

            mock_instrumentor_instance.instrument.assert_called_once_with(exclusive_processor=False)
        finally:
            if saved_mod is None:
                sys.modules.pop(oi_key, None)
            else:
                sys.modules[oi_key] = saved_mod
            litellm.success_callback = original_success
            litellm.failure_callback = original_failure


class TestOptionalTraceableLangfuse:
    """Tests for optional_traceable when Langfuse is active."""

    def test_function_runs_with_langfuse(self, monkeypatch):
        """Decorated function executes correctly when Langfuse is fully configured."""
        import datus.utils.traceable_utils as module

        monkeypatch.setattr(module, "HAS_LANGFUSE", True)
        monkeypatch.setattr(module, "_langfuse_enabled", True)

        mock_observe = MagicMock(side_effect=lambda *a, **kw: lambda fn: fn)
        with patch("langfuse.observe", mock_observe):

            @optional_traceable(name="test_langfuse_op")
            def multiply(a, b):
                return a * b

            assert multiply(3, 4) == 12
            mock_observe.assert_called_once()

    def test_langfuse_not_applied_when_disabled(self, monkeypatch):
        """Langfuse observe is not applied when _langfuse_enabled is False."""
        import datus.utils.traceable_utils as module

        monkeypatch.setattr(module, "HAS_LANGFUSE", True)
        monkeypatch.setattr(module, "_langfuse_enabled", False)

        with patch("langfuse.observe") as mock_observe:

            @optional_traceable(name="test_op")
            def add(a, b):
                return a + b

            assert add(1, 2) == 3
            mock_observe.assert_not_called()


class TestGetTraceUrlLangfuse:
    """Tests for get_trace_url with Langfuse backend."""

    def test_returns_none_when_no_trace(self, monkeypatch):
        """Returns None when Langfuse is enabled but no active trace."""
        import datus.utils.traceable_utils as module

        monkeypatch.setattr(module, "_tracing_processor", None)
        monkeypatch.setattr(module, "_langfuse_enabled", True)

        mock_client = MagicMock()
        mock_client.get_current_trace_id.return_value = None
        with patch("langfuse.get_client", return_value=mock_client):
            assert get_trace_url() is None

    def test_returns_url_when_trace_active(self, monkeypatch):
        """Returns SDK-constructed URL when Langfuse has an active trace."""
        import datus.utils.traceable_utils as module

        monkeypatch.setattr(module, "_tracing_processor", None)
        monkeypatch.setattr(module, "_langfuse_enabled", True)

        mock_client = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace-abc-123"
        mock_client.get_trace_url.return_value = "https://us.cloud.langfuse.com/project/proj-123/traces/trace-abc-123"
        with patch("langfuse.get_client", return_value=mock_client):
            url = get_trace_url()
            assert url == "https://us.cloud.langfuse.com/project/proj-123/traces/trace-abc-123"
            mock_client.get_trace_url.assert_called_once_with(trace_id="trace-abc-123")


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
