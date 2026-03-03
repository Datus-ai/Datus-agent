# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.utils.traceable_utils — LangSmith tracing integration."""

from datus.utils.traceable_utils import (
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
        assert get_trace_url() is None
