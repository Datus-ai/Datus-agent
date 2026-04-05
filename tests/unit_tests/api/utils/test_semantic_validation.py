"""Tests for datus.api.utils.semantic_validation."""

from unittest.mock import patch

import pytest

from datus.api.utils import semantic_validation
from datus.api.utils.semantic_validation import validate_semantic_yaml


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset the module-level metricflow availability cache between tests."""
    semantic_validation._METRICFLOW_AVAILABLE = None
    yield
    semantic_validation._METRICFLOW_AVAILABLE = None


# ---------------------------------------------------------------------------
# Fallback path (no metricflow)
# ---------------------------------------------------------------------------


class TestFallbackValidation:
    """Tests when metricflow is NOT available."""

    def test_valid_yaml_passes(self):
        semantic_validation._METRICFLOW_AVAILABLE = False
        is_valid, errors = validate_semantic_yaml(
            yaml_content="metric:\n  name: revenue\n  type: simple\n",
            file_path="/tmp/revenue.yml",
            datus_home="/tmp/datus",
            namespace="default",
        )
        assert is_valid is True
        assert errors == []

    def test_invalid_yaml_syntax_fails(self):
        semantic_validation._METRICFLOW_AVAILABLE = False
        is_valid, errors = validate_semantic_yaml(
            yaml_content=":\n  - ][",
            file_path="/tmp/bad.yml",
            datus_home="/tmp/datus",
            namespace="default",
        )
        assert is_valid is False
        assert len(errors) > 0

    def test_empty_yaml_passes(self):
        semantic_validation._METRICFLOW_AVAILABLE = False
        is_valid, errors = validate_semantic_yaml(
            yaml_content="",
            file_path="/tmp/empty.yml",
            datus_home="/tmp/datus",
            namespace="default",
        )
        assert is_valid is True
        assert errors == []


# ---------------------------------------------------------------------------
# Deep validation path (metricflow available, mocked)
# ---------------------------------------------------------------------------


class TestDeepValidation:
    """Tests when metricflow IS available (mocked via _validate_deep)."""

    @patch("datus.api.utils.semantic_validation._check_metricflow", return_value=True)
    @patch.object(semantic_validation, "_validate_deep", return_value=(True, []))
    def test_deep_validation_passes(self, _mock_deep, _mock_check):
        is_valid, errors = validate_semantic_yaml(
            yaml_content="metric:\n  name: test\n",
            file_path="/tmp/test.yml",
            datus_home="/tmp/datus",
            namespace="default",
        )
        assert is_valid is True
        assert errors == []

    @patch("datus.api.utils.semantic_validation._check_metricflow", return_value=True)
    @patch.object(
        semantic_validation,
        "_validate_deep",
        return_value=(False, ["Missing required field 'type' in metric definition"]),
    )
    def test_deep_validation_lint_failure(self, _mock_deep, _mock_check):
        is_valid, errors = validate_semantic_yaml(
            yaml_content="metric:\n  name: test\n",
            file_path="/tmp/test.yml",
            datus_home="/tmp/datus",
            namespace="default",
        )
        assert is_valid is False
        assert "Missing required field" in errors[0]

    @patch("datus.api.utils.semantic_validation._check_metricflow", return_value=True)
    @patch.object(
        semantic_validation,
        "_validate_deep",
        return_value=(False, ["Unknown measure 'nonexistent_measure' referenced in metric"]),
    )
    def test_deep_validation_cross_ref_failure(self, _mock_deep, _mock_check):
        is_valid, errors = validate_semantic_yaml(
            yaml_content="metric:\n  name: test\n  type: simple\n  type_params:\n    measure: nonexistent_measure\n",
            file_path="/tmp/test.yml",
            datus_home="/tmp/datus",
            namespace="default",
        )
        assert is_valid is False
        assert "nonexistent_measure" in errors[0]

    @patch("datus.api.utils.semantic_validation._check_metricflow", return_value=True)
    @patch.object(
        semantic_validation,
        "_validate_deep",
        return_value=(False, ["Semantic validation failed: ratio numerator type mismatch"]),
    )
    def test_deep_validation_semantic_failure(self, _mock_deep, _mock_check):
        is_valid, errors = validate_semantic_yaml(
            yaml_content="metric:\n  name: test\n",
            file_path="/tmp/test.yml",
            datus_home="/tmp/datus",
            namespace="default",
        )
        assert is_valid is False
        assert "Semantic validation failed" in errors[0]


# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------


class TestMetricflowDetection:
    """Tests for _check_metricflow availability detection."""

    def test_detection_caches_result(self):
        semantic_validation._METRICFLOW_AVAILABLE = True
        assert semantic_validation._check_metricflow() is True
        semantic_validation._METRICFLOW_AVAILABLE = False
        assert semantic_validation._check_metricflow() is False

    def test_detection_returns_false_when_import_fails(self):
        """Verify _check_metricflow returns False when metricflow cannot be imported."""
        blocked = {
            "metricflow": None,
            "metricflow.model": None,
            "metricflow.model.model_validator": None,
            "metricflow.model.parsing": None,
            "metricflow.model.parsing.config_linter": None,
            "metricflow.model.parsing.dir_to_model": None,
        }
        with patch.dict("sys.modules", blocked):
            semantic_validation._METRICFLOW_AVAILABLE = None
            result = semantic_validation._check_metricflow()
            assert result is False
            assert semantic_validation._METRICFLOW_AVAILABLE is False
