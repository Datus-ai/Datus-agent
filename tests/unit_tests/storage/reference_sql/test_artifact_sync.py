# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/storage/reference_sql/artifact_sync.py."""

from unittest.mock import MagicMock, patch

import yaml

from datus.storage.reference_sql.artifact_sync import sync_reference_sql_artifact_to_kb


class TestSyncReferenceSqlArtifactToKb:
    def test_valid_yaml_synced(self, tmp_path):
        yaml_file = tmp_path / "ref.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "sql": "SELECT * FROM t WHERE x = '{{val}}'",
                    "name": "test_reference_sql",
                    "summary": "Test reference sql",
                    "search_text": "test reference sql val",
                    "subject_tree": "Sales/Revenue",
                    "comment": "Helpful sql",
                    "tags": "test",
                }
            )
        )

        with (
            patch("datus.storage.reference_sql.artifact_sync.ReferenceSqlRAG") as mock_rag_cls,
            patch(
                "datus.storage.reference_sql.artifact_sync.gen_reference_sql_id",
                return_value="new_id",
            ),
        ):
            mock_rag = mock_rag_cls.return_value
            result = sync_reference_sql_artifact_to_kb(str(yaml_file), MagicMock())

        assert result["success"] is True
        assert "Synced reference SQL" in result["message"]
        mock_rag.upsert_batch.assert_called_once()
        stored = mock_rag.upsert_batch.call_args.args[0][0]
        assert stored == {
            "id": "new_id",
            "name": "test_reference_sql",
            "sql": "SELECT * FROM t WHERE x = '{{val}}'",
            "comment": "Helpful sql",
            "summary": "Test reference sql",
            "search_text": "test reference sql val",
            "filepath": str(yaml_file),
            "subject_path": ["Sales", "Revenue"],
            "tags": "test",
        }

    def test_missing_sql_field(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(yaml.dump({"name": "no_sql"}))

        result = sync_reference_sql_artifact_to_kb(str(yaml_file), MagicMock())

        assert result["success"] is False
        assert "No reference_sql data" in result["error"]

    def test_existing_explicit_id_is_kept_and_upserted(self, tmp_path):
        """The YAML file is the source of truth: re-syncing an existing ID
        updates the KB row via upsert instead of skipping it."""
        yaml_file = tmp_path / "dup.yaml"
        yaml_file.write_text(
            yaml.dump({"id": "existing_id", "sql": "SELECT 1", "name": "dup", "summary": "x", "search_text": "x"})
        )

        with patch("datus.storage.reference_sql.artifact_sync.ReferenceSqlRAG") as mock_rag_cls:
            mock_rag = mock_rag_cls.return_value
            result = sync_reference_sql_artifact_to_kb(str(yaml_file), MagicMock())

        assert result["success"] is True
        mock_rag.upsert_batch.assert_called_once()
        stored = mock_rag.upsert_batch.call_args.args[0][0]
        assert stored["id"] == "existing_id"

    def test_placeholder_id_regenerated_from_sql(self, tmp_path):
        yaml_file = tmp_path / "auto.yaml"
        yaml_file.write_text(yaml.dump({"id": "auto_generated", "sql": "SELECT 2", "name": "auto"}))

        with (
            patch("datus.storage.reference_sql.artifact_sync.ReferenceSqlRAG") as mock_rag_cls,
            patch(
                "datus.storage.reference_sql.artifact_sync.gen_reference_sql_id",
                return_value="generated_id",
            ) as mock_gen,
        ):
            mock_rag = mock_rag_cls.return_value
            result = sync_reference_sql_artifact_to_kb(str(yaml_file), MagicMock())

        assert result["success"] is True
        mock_gen.assert_called_once_with("SELECT 2")
        stored = mock_rag.upsert_batch.call_args.args[0][0]
        assert stored["id"] == "generated_id"

    def test_storage_error_returns_failure(self, tmp_path):
        yaml_file = tmp_path / "boom.yaml"
        yaml_file.write_text(yaml.dump({"sql": "SELECT 1", "name": "boom"}))

        with patch("datus.storage.reference_sql.artifact_sync.ReferenceSqlRAG") as mock_rag_cls:
            mock_rag_cls.return_value.upsert_batch.side_effect = RuntimeError("boom")
            result = sync_reference_sql_artifact_to_kb(str(yaml_file), MagicMock())

        assert result["success"] is False
        assert result["error"] == "boom"

    def test_missing_file_returns_failure(self, tmp_path):
        result = sync_reference_sql_artifact_to_kb(str(tmp_path / "nope.yaml"), MagicMock())
        assert result["success"] is False
