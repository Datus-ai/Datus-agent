"""Tests for datus.api.utils.semantic_validation."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
            datasource="default",
        )
        assert is_valid is True
        assert errors == []

    def test_invalid_yaml_syntax_fails(self):
        semantic_validation._METRICFLOW_AVAILABLE = False
        is_valid, errors = validate_semantic_yaml(
            yaml_content=":\n  - ][",
            file_path="/tmp/bad.yml",
            datus_home="/tmp/datus",
            datasource="default",
        )
        assert is_valid is False
        assert len(errors) == 1

    def test_empty_yaml_passes(self):
        semantic_validation._METRICFLOW_AVAILABLE = False
        is_valid, errors = validate_semantic_yaml(
            yaml_content="",
            file_path="/tmp/empty.yml",
            datus_home="/tmp/datus",
            datasource="default",
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
            datasource="default",
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
            datasource="default",
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
            datasource="default",
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
            datasource="default",
        )
        assert is_valid is False
        assert "Semantic validation failed" in errors[0]


class TestDeepValidationPathIsolation:
    """Tests for metricflow-backed file collection boundaries."""

    def test_non_semantic_models_path_does_not_scan_parent_directory(self, monkeypatch, tmp_path):
        captured = _install_fake_metricflow(monkeypatch)
        semantic_validation._METRICFLOW_AVAILABLE = True

        is_valid, errors = validate_semantic_yaml(
            yaml_content="metric:\n  name: test\n  type: simple\n",
            file_path=str(tmp_path / "test.yml"),
            datus_home=str(tmp_path / "home"),
            datasource="default",
        )

        assert is_valid is True
        assert errors == []
        assert captured["collected_dirs"] == []
        assert len(captured["parsed_file_paths"]) == 1
        assert captured["parsed_file_paths"][0].endswith("test.yml")

    def test_semantic_models_path_collects_project_semantic_root(self, monkeypatch, tmp_path):
        captured = _install_fake_metricflow(monkeypatch)
        semantic_validation._METRICFLOW_AVAILABLE = True
        semantic_root = tmp_path / "subject" / "semantic_models"
        metric_path = semantic_root / "analytics" / "metrics" / "test.yml"

        is_valid, errors = validate_semantic_yaml(
            yaml_content="metric:\n  name: test\n  type: simple\n",
            file_path=str(metric_path),
            datus_home=str(tmp_path / "home"),
            datasource="analytics",
        )

        assert is_valid is True
        assert errors == []
        assert captured["collected_dirs"] == [str(semantic_root)]
        assert captured["parsed_file_paths"][0] == str(semantic_root / "existing.yml")
        assert captured["parsed_file_paths"][-1].endswith("test.yml")


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


def _install_fake_metricflow(monkeypatch):
    captured = {"collected_dirs": [], "parsed_file_paths": []}

    class FakeConfigLinter:
        def lint_file(self, _file_path):
            return []

    class FakeModelValidator:
        def validate_model(self, _model):
            return SimpleNamespace(issues=None)

    def fake_collect_yaml_config_file_paths(directory):
        captured["collected_dirs"].append(directory)
        return [str(Path(directory) / "existing.yml")]

    def fake_parse_yaml_file_paths_to_model(file_paths, raise_issues_as_exceptions=False):
        captured["parsed_file_paths"] = list(file_paths)
        assert raise_issues_as_exceptions is False
        return SimpleNamespace(issues=None, model=object())

    model_validator = ModuleType("metricflow.model.model_validator")
    model_validator.ModelValidator = FakeModelValidator

    config_linter = ModuleType("metricflow.model.parsing.config_linter")
    config_linter.ConfigLinter = FakeConfigLinter

    dir_to_model = ModuleType("metricflow.model.parsing.dir_to_model")
    dir_to_model.collect_yaml_config_file_paths = fake_collect_yaml_config_file_paths
    dir_to_model.parse_yaml_file_paths_to_model = fake_parse_yaml_file_paths_to_model

    modules = {
        "metricflow": ModuleType("metricflow"),
        "metricflow.model": ModuleType("metricflow.model"),
        "metricflow.model.model_validator": model_validator,
        "metricflow.model.parsing": ModuleType("metricflow.model.parsing"),
        "metricflow.model.parsing.config_linter": config_linter,
        "metricflow.model.parsing.dir_to_model": dir_to_model,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return captured
