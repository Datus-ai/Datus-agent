# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.semantic_model.semantic_model_init."""

from unittest.mock import MagicMock, patch

import pytest

from datus.storage.semantic_model.semantic_model_init import (
    init_semantic_yaml_semantic_model,
    process_semantic_yaml_file,
)

# ---------------------------------------------------------------------------
# init_semantic_yaml_semantic_model
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSemanticYamlSemanticModel:
    """Tests for init_semantic_yaml_semantic_model function."""

    def test_file_not_found(self, tmp_path):
        """Returns (False, error) when YAML file does not exist."""
        nonexistent = str(tmp_path / "missing.yaml")
        mock_config = MagicMock()

        success, error = init_semantic_yaml_semantic_model(nonexistent, mock_config)

        assert success is False
        assert "not found" in error

    def test_existing_file_delegates_to_process(self, tmp_path):
        """When file exists, calls process_semantic_yaml_file with include_metrics=False."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: test\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.process_semantic_yaml_file",
            return_value=(True, ""),
        ) as mock_process:
            success, error = init_semantic_yaml_semantic_model(str(yaml_file), mock_config)

        assert success is True
        assert error == ""
        mock_process.assert_called_once_with(str(yaml_file), mock_config, include_metrics=False)


# ---------------------------------------------------------------------------
# process_semantic_yaml_file
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestProcessSemanticYamlFile:
    """Tests for process_semantic_yaml_file function."""

    def test_file_not_found(self, tmp_path):
        """Returns (False, error) when file does not exist."""
        nonexistent = str(tmp_path / "missing.yaml")
        mock_config = MagicMock()

        success, error = process_semantic_yaml_file(nonexistent, mock_config)

        assert success is False
        assert "not found" in error

    def test_sync_success(self, tmp_path):
        """Returns (True, '') when sync succeeds."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True, "message": "synced"},
        ):
            success, error = process_semantic_yaml_file(str(yaml_file), mock_config)

        assert success is True
        assert error == ""

    def test_sync_failure(self, tmp_path):
        """Returns (False, error) when sync reports failure."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": False, "error": "validation failed"},
        ):
            success, error = process_semantic_yaml_file(str(yaml_file), mock_config)

        assert success is False
        assert "validation failed" in error

    def test_sync_exception(self, tmp_path):
        """Returns (False, error) when sync raises an exception."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            side_effect=RuntimeError("connection error"),
        ):
            success, error = process_semantic_yaml_file(str(yaml_file), mock_config)

        assert success is False
        assert "connection error" in error

    def test_default_includes_both(self, tmp_path):
        """By default, include_semantic_objects and include_metrics are both True."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True, "message": "ok"},
        ) as mock_sync:
            process_semantic_yaml_file(str(yaml_file), mock_config)

        mock_sync.assert_called_once_with(
            str(yaml_file),
            mock_config,
            include_semantic_objects=True,
            include_metrics=True,
        )

    def test_exclude_metrics(self, tmp_path):
        """include_metrics=False is forwarded to _sync_semantic_to_db."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True, "message": "ok"},
        ) as mock_sync:
            process_semantic_yaml_file(str(yaml_file), mock_config, include_metrics=False)

        mock_sync.assert_called_once_with(
            str(yaml_file),
            mock_config,
            include_semantic_objects=True,
            include_metrics=False,
        )

    def test_exclude_semantic_objects(self, tmp_path):
        """include_semantic_objects=False is forwarded to _sync_semantic_to_db."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True, "message": "ok"},
        ) as mock_sync:
            process_semantic_yaml_file(str(yaml_file), mock_config, include_semantic_objects=False)

        mock_sync.assert_called_once_with(
            str(yaml_file),
            mock_config,
            include_semantic_objects=False,
            include_metrics=True,
        )

    def test_sync_unknown_error(self, tmp_path):
        """When sync returns failure with no error key, uses 'Unknown error'."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": False},
        ):
            success, error = process_semantic_yaml_file(str(yaml_file), mock_config)

        assert success is False
        assert "Unknown error" in error
