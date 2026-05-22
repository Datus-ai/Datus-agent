# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.utils.traceable_utils."""

import logging
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from datus.observability.config import ObservabilityConfig
from datus.utils.trace_context import TraceContext, get_trace_context
from datus.utils.traceable_utils import _disable_sdk_tracing, optional_traceable, setup_tracing


def _clear_all_tracing_envvars(monkeypatch):
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


class TestSetupTracing:
    def test_setup_tracing_without_observability_disables_sdk_tracing(self, monkeypatch):
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setattr(module, "_tracing_initialized", False)

        with patch.object(module, "_disable_sdk_tracing") as mock_disable:
            setup_tracing()

        mock_disable.assert_called_once_with("observability tracing not configured")
        assert module._tracing_initialized is False

    def test_setup_tracing_ignores_legacy_env_auto_detection(self, monkeypatch):
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-fake")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-fake")
        monkeypatch.setattr(module, "_tracing_initialized", False)

        with (
            patch.object(module, "_disable_sdk_tracing") as mock_disable,
            patch("datus.observability.manager.configure_observability") as mock_configure,
            patch("agents.set_trace_processors") as mock_set_processors,
        ):
            setup_tracing()

        mock_configure.assert_not_called()
        mock_set_processors.assert_not_called()
        mock_disable.assert_called_once_with("observability tracing not configured")

    def test_configured_observability_uses_manager(self, monkeypatch):
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setattr(module, "_tracing_initialized", False)
        cfg = ObservabilityConfig.from_dict(
            {
                "tracing": {
                    "enabled": True,
                    "adapters": [{"type": "otlp", "endpoint": "http://collector/v1/traces"}],
                }
            }
        )

        with (
            patch("datus.observability.manager.configure_observability", return_value=True) as mock_configure,
            patch.object(module, "_disable_sdk_tracing") as mock_disable,
        ):
            setup_tracing(cfg)

        mock_configure.assert_called_once_with(cfg)
        mock_disable.assert_not_called()
        assert module._tracing_initialized is True

    def test_disabled_observability_disables_sdk_tracing(self, monkeypatch):
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setattr(module, "_tracing_initialized", False)
        cfg = ObservabilityConfig.from_dict({"tracing": {"enabled": False}})

        with (
            patch.object(module, "_disable_sdk_tracing") as mock_disable,
            patch("datus.observability.manager.configure_observability") as mock_configure,
        ):
            setup_tracing(cfg)

        mock_configure.assert_not_called()
        mock_disable.assert_called_once_with("observability tracing not configured")

    def test_empty_observability_does_not_enable_legacy_env_tracing(self, monkeypatch):
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "fake-key")
        monkeypatch.setattr(module, "_tracing_initialized", False)
        cfg = ObservabilityConfig.from_dict({})

        with (
            patch.object(module, "_disable_sdk_tracing") as mock_disable,
            patch("agents.set_trace_processors") as mock_set_processors,
        ):
            setup_tracing(cfg)

        mock_set_processors.assert_not_called()
        mock_disable.assert_called_once_with("observability tracing not configured")
        assert module._tracing_initialized is False

    def test_setup_tracing_without_config_does_not_block_later_config(self, monkeypatch):
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        monkeypatch.setattr(module, "_tracing_initialized", False)
        cfg = ObservabilityConfig.from_dict(
            {
                "tracing": {
                    "enabled": True,
                    "adapters": [{"type": "otlp", "endpoint": "http://collector/v1/traces"}],
                }
            }
        )

        with (
            patch.object(module, "_disable_sdk_tracing") as mock_disable,
            patch("datus.observability.manager.configure_observability", return_value=True) as mock_configure,
        ):
            setup_tracing()
            setup_tracing(cfg)

        mock_disable.assert_called_once()
        mock_configure.assert_called_once_with(cfg)
        assert module._tracing_initialized is True

    def test_setup_tracing_suppresses_noisy_otel_span_warnings(self, monkeypatch):
        import datus.utils.traceable_utils as module

        _clear_all_tracing_envvars(monkeypatch)
        otel_logger = logging.getLogger("opentelemetry.sdk.trace")
        original_level = otel_logger.level
        otel_logger.setLevel(logging.NOTSET)
        monkeypatch.setattr(module, "_tracing_initialized", False)

        try:
            setup_tracing()
            assert otel_logger.level == logging.ERROR
        finally:
            otel_logger.setLevel(original_level)


class TestDisableSdkTracing:
    def test_calls_set_tracing_disabled(self):
        with patch("agents.set_tracing_disabled") as mock_disable:
            _disable_sdk_tracing("test reason")
            mock_disable.assert_called_once_with(True)


class TestOptionalTraceable:
    def test_function_runs_normally(self):
        @optional_traceable(name="test_op")
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_function_name_preserved(self):
        @optional_traceable()
        def my_function():
            return "hello"

        assert my_function() == "hello"

    def test_context_builder_scopes_context(self, monkeypatch):
        import datus.utils.traceable_utils as module

        monkeypatch.setattr(module, "_tracing_initialized", False)
        seen = {}

        @optional_traceable(
            name="test_op",
            context_builder=lambda *_args, **_kwargs: TraceContext(
                name="workflow/test",
                session_id="workflow:run:test",
                tags=("workflow",),
                metadata={"workflow": "test"},
            ),
        )
        def op():
            seen["ctx"] = get_trace_context()
            return "ok"

        assert op() == "ok"
        assert seen["ctx"].name == "workflow/test"
        assert get_trace_context() is None

    @pytest.mark.asyncio
    async def test_async_generator_keeps_span_and_context_during_iteration(self):
        class FakeObservability:
            def __init__(self):
                self.active = False
                self.span_calls = []

            @contextmanager
            def span(self, name, attributes=None, *, run_id=None):
                self.span_calls.append((name, attributes, run_id))
                self.active = True
                try:
                    yield object()
                finally:
                    self.active = False

        observability = FakeObservability()

        @optional_traceable(
            name="stream_op",
            context_builder=lambda *_args, **_kwargs: TraceContext(
                name="workflow/stream",
                session_id="workflow:run:stream",
                tags=("stream",),
                metadata={"run_id": "run-123"},
            ),
        )
        async def stream():
            yield {
                "active": observability.active,
                "ctx": get_trace_context().session_id,
            }
            yield {
                "active": observability.active,
                "ctx": get_trace_context().session_id,
            }

        with patch("datus.observability.manager.get_observability_manager", return_value=observability):
            generator = stream()
            assert observability.active is False
            items = [item async for item in generator]

        assert items == [
            {"active": True, "ctx": "workflow:run:stream"},
            {"active": True, "ctx": "workflow:run:stream"},
        ]
        assert observability.active is False
        assert observability.span_calls == [
            (
                "stream_op",
                {
                    "datus.operation": "stream_op",
                    "datus.run_type": "chain",
                    "datus.trace.name": "workflow/stream",
                    "datus.session_id": "workflow:run:stream",
                    "datus.tags": "stream",
                    "datus.metadata.run_id": "run-123",
                    "datus.run_id": "run-123",
                },
                "run-123",
            )
        ]
        assert get_trace_context() is None
