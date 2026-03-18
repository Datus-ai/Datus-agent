# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.metric.metric_init."""

from enum import Enum
from unittest.mock import MagicMock

import pytest

from datus.storage.metric.metric_init import BIZ_NAME, _action_status_value, init_semantic_yaml_metrics

# ---------------------------------------------------------------------------
# _action_status_value
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestActionStatusValue:
    """Tests for the _action_status_value helper."""

    def test_none_status(self):
        """Returns None when action.status is None."""
        action = MagicMock(status=None)
        assert _action_status_value(action) is None

    def test_no_status_attr(self):
        """Returns None when action has no status attribute."""
        action = object()
        assert _action_status_value(action) is None

    def test_enum_status(self):
        """Returns enum .value when status is an Enum."""

        class St(Enum):
            DONE = "done"

        action = MagicMock()
        action.status = St.DONE
        assert _action_status_value(action) == "done"

    def test_string_status(self):
        """Returns str(status) for plain string status."""
        action = MagicMock()
        action.status = "processing"
        assert _action_status_value(action) == "processing"

    def test_object_with_value_attr(self):
        """Returns status.value for objects with value attribute."""

        class CustomStatus:
            value = "custom"

        action = MagicMock()
        action.status = CustomStatus()
        assert _action_status_value(action) == "custom"


# ---------------------------------------------------------------------------
# BIZ_NAME constant
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestBizNameConstant:
    """Tests for module-level constant."""

    def test_biz_name(self):
        """BIZ_NAME is metric_init."""
        assert BIZ_NAME == "metric_init"


# ---------------------------------------------------------------------------
# init_semantic_yaml_metrics - file not found
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSemanticYamlMetrics:
    """Tests for init_semantic_yaml_metrics function."""

    def test_file_not_found(self, tmp_path):
        """Returns (False, error) when YAML file does not exist."""
        nonexistent = str(tmp_path / "nonexistent.yaml")
        mock_config = MagicMock()

        success, error = init_semantic_yaml_metrics(nonexistent, mock_config)

        assert success is False
        assert "not found" in error

    def test_existing_file_calls_process(self, tmp_path):
        """When file exists, delegates to process_semantic_yaml_file."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("tables:\n  - name: test\n")
        mock_config = MagicMock()

        # The import happens inside the function body from the semantic_model package
        from unittest.mock import patch

        with patch(
            "datus.storage.semantic_model.semantic_model_init.process_semantic_yaml_file",
            return_value=(True, ""),
        ) as mock_process:
            success, error = init_semantic_yaml_metrics(str(yaml_file), mock_config)

        assert success is True
        assert error == ""
        mock_process.assert_called_once_with(str(yaml_file), mock_config, include_semantic_objects=False)


# ---------------------------------------------------------------------------
# init_success_story_metrics_async - importability and coroutine check
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSuccessStoryMetricsAsync:
    """Tests for init_success_story_metrics_async importability and interface."""

    def test_async_function_is_importable(self):
        """init_success_story_metrics_async can be imported from the module."""
        from datus.storage.metric.metric_init import init_success_story_metrics_async

        assert init_success_story_metrics_async is not None

    def test_async_function_is_coroutine(self):
        """init_success_story_metrics_async is a coroutine function (async def)."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics_async

        assert inspect.iscoroutinefunction(init_success_story_metrics_async)

    def test_async_function_signature_has_no_args_param(self):
        """init_success_story_metrics_async signature does not include argparse.Namespace args."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics_async

        sig = inspect.signature(init_success_story_metrics_async)
        param_names = list(sig.parameters.keys())
        assert "args" not in param_names
        assert "agent_config" in param_names
        assert "success_story" in param_names

    def test_async_optional_params_present(self):
        """init_success_story_metrics_async exposes subject_tree, emit, extra_instructions."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics_async

        sig = inspect.signature(init_success_story_metrics_async)
        param_names = list(sig.parameters.keys())
        assert "subject_tree" in param_names
        assert "emit" in param_names
        assert "extra_instructions" in param_names


# ---------------------------------------------------------------------------
# init_success_story_metrics sync wrapper - new signature
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSuccessStoryMetricsSync:
    """Tests for init_success_story_metrics sync wrapper with decoupled signature."""

    def test_sync_function_is_importable(self):
        """init_success_story_metrics can be imported."""
        from datus.storage.metric.metric_init import init_success_story_metrics

        assert init_success_story_metrics is not None

    def test_sync_function_is_not_coroutine(self):
        """init_success_story_metrics is a plain sync function."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics

        assert not inspect.iscoroutinefunction(init_success_story_metrics)

    def test_sync_function_signature_has_no_args_param(self):
        """init_success_story_metrics signature does not include argparse.Namespace args."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics

        sig = inspect.signature(init_success_story_metrics)
        param_names = list(sig.parameters.keys())
        assert "args" not in param_names
        assert "agent_config" in param_names
        assert "success_story" in param_names

    def test_sync_returns_three_tuple(self, tmp_path):
        """Sync wrapper returns a 3-tuple (bool, str, Optional[dict]) for a missing CSV."""
        from unittest.mock import patch

        from datus.storage.metric.metric_init import init_success_story_metrics

        missing = str(tmp_path / "no_file.csv")
        mock_config = MagicMock()

        # Patch the async function to avoid creating an unawaited coroutine
        with patch(
            "datus.storage.metric.metric_init.init_success_story_metrics_async",
            return_value=(False, "file error", None),
        ):
            result = init_success_story_metrics(mock_config, missing)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_sync_accepts_all_kwargs(self, tmp_path):
        """Sync wrapper accepts subject_tree, emit, extra_instructions kwargs."""
        from unittest.mock import patch

        from datus.storage.metric.metric_init import init_success_story_metrics

        mock_config = MagicMock()
        emit_events = []

        # Patch the async function to avoid creating an unawaited coroutine
        with patch(
            "datus.storage.metric.metric_init.init_success_story_metrics_async",
            return_value=(True, "", {"metrics": []}),
        ):
            result = init_success_story_metrics(
                mock_config,
                "dummy.csv",
                subject_tree=["Finance"],
                emit=emit_events.append,
                extra_instructions="Focus on revenue metrics.",
            )

        assert isinstance(result, tuple)
        assert len(result) == 3
