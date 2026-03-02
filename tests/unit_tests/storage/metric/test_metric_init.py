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
