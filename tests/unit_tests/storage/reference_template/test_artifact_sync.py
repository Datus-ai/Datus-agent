# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/storage/reference_template/artifact_sync.py."""

import json
from unittest.mock import MagicMock, patch

import yaml

from datus.storage.reference_template.artifact_sync import sync_reference_template_artifact_to_kb


class TestSyncReferenceTemplateArtifactToKb:
    def test_valid_yaml_synced(self, tmp_path):
        yaml_file = tmp_path / "tpl.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "sql": "SELECT * FROM t WHERE x = '{{val}}'",
                    "name": "test_reference_template",
                    "summary": "Test reference template",
                    "search_text": "test reference template val",
                    "subject_tree": "Sales/Revenue",
                    "comment": "Helpful template",
                    "tags": "test",
                }
            )
        )

        with (
            patch("datus.storage.reference_template.artifact_sync.ReferenceTemplateRAG") as mock_rag_cls,
            patch(
                "datus.storage.reference_template.artifact_sync.gen_reference_template_id",
                return_value="new_tpl_id",
            ),
            patch(
                "datus.storage.reference_template.artifact_sync.extract_template_parameters",
                return_value=[{"name": "val", "type": "string"}],
            ),
        ):
            mock_rag = mock_rag_cls.return_value
            result = sync_reference_template_artifact_to_kb(str(yaml_file), MagicMock())

        assert result["success"] is True
        assert "Synced reference template" in result["message"]
        mock_rag.upsert_batch.assert_called_once()
        stored = mock_rag.upsert_batch.call_args.args[0][0]
        assert stored == {
            "id": "new_tpl_id",
            "name": "test_reference_template",
            "template": "SELECT * FROM t WHERE x = '{{val}}'",
            "parameters": json.dumps([{"name": "val", "type": "string"}]),
            "comment": "Helpful template",
            "summary": "Test reference template",
            "search_text": "test reference template val",
            "filepath": str(yaml_file),
            "subject_path": ["Sales", "Revenue"],
            "tags": "test",
        }

    def test_missing_sql_field(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(yaml.dump({"name": "no_sql"}))

        result = sync_reference_template_artifact_to_kb(str(yaml_file), MagicMock())

        assert result["success"] is False
        assert "No reference_template data" in result["error"]

    def test_blank_sql_returns_error(self, tmp_path):
        yaml_file = tmp_path / "blank.yaml"
        yaml_file.write_text(yaml.dump({"sql": "   ", "name": "blank"}))

        result = sync_reference_template_artifact_to_kb(str(yaml_file), MagicMock())

        assert result["success"] is False
        assert "non-empty string" in result["error"]

    def test_existing_explicit_id_is_kept_and_upserted(self, tmp_path):
        """The YAML file is the source of truth: re-syncing an existing ID
        updates the KB row via upsert instead of skipping it."""
        yaml_file = tmp_path / "dup.yaml"
        yaml_file.write_text(yaml.dump({"id": "existing_tpl_id", "sql": "SELECT 1", "name": "dup_tpl"}))

        with (
            patch("datus.storage.reference_template.artifact_sync.ReferenceTemplateRAG") as mock_rag_cls,
            patch(
                "datus.storage.reference_template.artifact_sync.extract_template_parameters",
                return_value=[],
            ),
        ):
            mock_rag = mock_rag_cls.return_value
            result = sync_reference_template_artifact_to_kb(str(yaml_file), MagicMock())

        assert result["success"] is True
        mock_rag.upsert_batch.assert_called_once()
        stored = mock_rag.upsert_batch.call_args.args[0][0]
        assert stored["id"] == "existing_tpl_id"

    def test_storage_error_returns_failure(self, tmp_path):
        yaml_file = tmp_path / "boom.yaml"
        yaml_file.write_text(yaml.dump({"sql": "SELECT 1", "name": "boom_tpl"}))

        with (
            patch("datus.storage.reference_template.artifact_sync.ReferenceTemplateRAG") as mock_rag_cls,
            patch(
                "datus.storage.reference_template.artifact_sync.gen_reference_template_id",
                return_value="boom_id",
            ),
            patch(
                "datus.storage.reference_template.artifact_sync.extract_template_parameters",
                return_value=[],
            ),
        ):
            mock_rag_cls.return_value.upsert_batch.side_effect = RuntimeError("boom")
            result = sync_reference_template_artifact_to_kb(str(yaml_file), MagicMock())

        assert result["success"] is False
        assert result["error"] == "boom"

    def test_missing_file_returns_failure(self, tmp_path):
        result = sync_reference_template_artifact_to_kb(str(tmp_path / "nope.yaml"), MagicMock())
        assert result["success"] is False
