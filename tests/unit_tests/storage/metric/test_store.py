# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/storage/metric/store.py — MetricStorage."""

import os
import tempfile

import pytest
import yaml

from datus.storage.embedding_models import get_metric_embedding_model
from datus.storage.metric.store import MetricStorage


@pytest.fixture
def metric_storage(tmp_path) -> MetricStorage:
    """Create a MetricStorage instance backed by a tmp_path vector store."""
    return MetricStorage(embedding_model=get_metric_embedding_model())


def _make_metric(idx: int, subject_path: list[str] | None = None, yaml_path: str = "") -> dict:
    """Build a single metric dict with required fields."""
    return {
        "subject_path": subject_path or ["Finance", "Revenue"],
        "id": f"metric:test_{idx}",
        "name": f"test_metric_{idx}",
        "semantic_model_name": "orders_model",
        "description": f"Test metric number {idx} for measuring things",
        "metric_type": "simple",
        "measure_expr": f"COUNT(DISTINCT col_{idx})",
        "base_measures": [f"measure_{idx}"],
        "dimensions": ["dim_a", "dim_b"],
        "entities": ["entity_x"],
        "catalog_name": "default",
        "database_name": "analytics",
        "schema_name": "public",
        "sql": f"SELECT COUNT(DISTINCT col_{idx}) FROM orders",
        "yaml_path": yaml_path,
    }


# ---------------------------------------------------------------------------
# MetricStorage schema construction
# ---------------------------------------------------------------------------


class TestMetricStorageSchema:
    """Tests for MetricStorage schema and initialization."""

    def test_table_name_is_metrics(self, metric_storage: MetricStorage):
        """The table should be named 'metrics'."""
        assert metric_storage.table_name == "metrics"

    def test_vector_source_name_is_description(self, metric_storage: MetricStorage):
        """Vector source should be 'description' field."""
        assert metric_storage.vector_source_name == "description"

    def test_vector_column_name_is_vector(self, metric_storage: MetricStorage):
        """Vector column should be named 'vector'."""
        assert metric_storage.vector_column_name == "vector"

    def test_schema_has_expected_fields(self, metric_storage: MetricStorage):
        """Schema should contain all expected metric fields."""
        expected_fields = {
            "id",
            "name",
            "semantic_model_name",
            "description",
            "vector",
            "metric_type",
            "measure_expr",
            "base_measures",
            "dimensions",
            "entities",
            "catalog_name",
            "database_name",
            "schema_name",
            "sql",
            "yaml_path",
            "updated_at",
            "subject_node_id",
            "created_at",
        }
        schema_names = set(metric_storage._schema.names)
        for field in expected_fields:
            assert field in schema_names, f"Field '{field}' missing from schema"


# ---------------------------------------------------------------------------
# batch_store_metrics validation
# ---------------------------------------------------------------------------


class TestBatchStoreMetricsValidation:
    """Tests for batch_store_metrics input validation."""

    def test_batch_store_metrics_empty_list_noop(self, metric_storage: MetricStorage):
        """Storing an empty list should be a no-op."""
        metric_storage.batch_store_metrics([])
        # No exception should be raised

    def test_batch_store_metrics_missing_subject_path_raises(self, metric_storage: MetricStorage):
        """Missing subject_path should raise ValueError."""
        bad_metric = {
            "id": "metric:bad",
            "name": "bad_metric",
            "description": "no subject path",
            "semantic_model_name": "model",
        }
        with pytest.raises(ValueError, match="subject_path is required"):
            metric_storage.batch_store_metrics([bad_metric])

    def test_batch_store_metrics_empty_subject_path_raises(self, metric_storage: MetricStorage):
        """Empty subject_path list should raise ValueError."""
        bad_metric = {
            "subject_path": [],
            "id": "metric:bad",
            "name": "bad_metric",
            "description": "empty subject path",
            "semantic_model_name": "model",
        }
        with pytest.raises(ValueError, match="subject_path is required"):
            metric_storage.batch_store_metrics([bad_metric])

    def test_batch_store_metrics_none_subject_path_raises(self, metric_storage: MetricStorage):
        """None subject_path should raise ValueError."""
        bad_metric = {
            "subject_path": None,
            "id": "metric:bad",
            "name": "bad_metric",
            "description": "none subject path",
            "semantic_model_name": "model",
        }
        with pytest.raises(ValueError, match="subject_path is required"):
            metric_storage.batch_store_metrics([bad_metric])


# ---------------------------------------------------------------------------
# batch_upsert_metrics validation
# ---------------------------------------------------------------------------


class TestBatchUpsertMetricsValidation:
    """Tests for batch_upsert_metrics input validation."""

    def test_batch_upsert_metrics_empty_list_noop(self, metric_storage: MetricStorage):
        """Upserting an empty list should be a no-op."""
        metric_storage.batch_upsert_metrics([])

    def test_batch_upsert_metrics_missing_subject_path_raises(self, metric_storage: MetricStorage):
        """Missing subject_path should raise ValueError."""
        bad_metric = {
            "id": "metric:bad",
            "name": "bad_metric",
            "description": "no subject path",
            "semantic_model_name": "model",
        }
        with pytest.raises(ValueError, match="subject_path is required"):
            metric_storage.batch_upsert_metrics([bad_metric])


# ---------------------------------------------------------------------------
# YAML deletion logic in delete_metric
# ---------------------------------------------------------------------------


class TestDeleteMetricYaml:
    """Tests for YAML file handling in delete_metric."""

    def test_delete_metric_not_found_returns_failure(self, metric_storage: MetricStorage):
        """Deleting a non-existent metric should return success=False."""
        result = metric_storage.delete_metric(["Nonexistent"], "no_such_metric")
        assert result["success"] is False
        assert "not found" in result["message"]

    def test_delete_metric_removes_from_yaml_file(self, metric_storage: MetricStorage):
        """delete_metric should remove the metric entry from the yaml file."""
        # Create a temporary yaml file with two metrics
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8") as f:
            yaml_path = f.name
            docs = [
                {"metric": {"name": "to_delete", "description": "will be removed"}},
                {"metric": {"name": "to_keep", "description": "should stay"}},
            ]
            yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)

        try:
            # Store the metric with yaml_path
            metric = _make_metric(1, subject_path=["Finance", "Revenue"], yaml_path=yaml_path)
            metric["name"] = "to_delete"
            metric["id"] = "metric:to_delete"
            metric_storage.batch_store_metrics([metric])

            # Delete the metric
            result = metric_storage.delete_metric(["Finance", "Revenue"], "to_delete")
            assert result["success"] is True
            assert result.get("yaml_updated") is True

            # Verify yaml file still exists with remaining doc
            with open(yaml_path, "r", encoding="utf-8") as f:
                remaining = list(yaml.safe_load_all(f))
            remaining = [d for d in remaining if d is not None]
            assert len(remaining) == 1
            assert remaining[0]["metric"]["name"] == "to_keep"
        finally:
            if os.path.exists(yaml_path):
                os.remove(yaml_path)

    def test_delete_metric_removes_empty_yaml_file(self, metric_storage: MetricStorage):
        """If yaml file becomes empty after deletion, the file should be removed."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8") as f:
            yaml_path = f.name
            docs = [{"metric": {"name": "only_metric", "description": "sole entry"}}]
            yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)

        try:
            metric = _make_metric(1, subject_path=["Finance"], yaml_path=yaml_path)
            metric["name"] = "only_metric"
            metric["id"] = "metric:only_metric"
            metric_storage.batch_store_metrics([metric])

            result = metric_storage.delete_metric(["Finance"], "only_metric")
            assert result["success"] is True
            assert result.get("yaml_updated") is True
            assert result.get("yaml_deleted") is True
            assert not os.path.exists(yaml_path)
        finally:
            if os.path.exists(yaml_path):
                os.remove(yaml_path)

    def test_delete_metric_no_yaml_path(self, metric_storage: MetricStorage):
        """Deleting a metric without yaml_path should still succeed from vector store."""
        metric = _make_metric(1, subject_path=["Finance", "Revenue"])
        metric["name"] = "no_yaml_metric"
        metric["id"] = "metric:no_yaml"
        metric_storage.batch_store_metrics([metric])

        result = metric_storage.delete_metric(["Finance", "Revenue"], "no_yaml_metric")
        assert result["success"] is True
        assert result.get("yaml_updated", False) is False


# ---------------------------------------------------------------------------
# YAML update logic in update_entry
# ---------------------------------------------------------------------------


class TestUpdateMetricYaml:
    """Tests for YAML file sync in update_entry."""

    def test_update_metric_syncs_to_yaml_file(self, metric_storage: MetricStorage):
        """update_entry should write the updated description back to the yaml file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8") as f:
            yaml_path = f.name
            docs = [
                {"metric": {"name": "to_update", "description": "original description"}},
                {"metric": {"name": "other_metric", "description": "should stay unchanged"}},
            ]
            yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)

        try:
            metric = _make_metric(1, subject_path=["Finance", "Revenue"], yaml_path=yaml_path)
            metric["name"] = "to_update"
            metric["id"] = "metric:to_update"
            metric_storage.batch_store_metrics([metric])

            result = metric_storage.update_entry(
                ["Finance", "Revenue"], "to_update", {"description": "Updated description"}
            )
            assert result is True

            with open(yaml_path, "r", encoding="utf-8") as f:
                remaining = [d for d in yaml.safe_load_all(f) if d is not None]

            updated_doc = next(d for d in remaining if d["metric"]["name"] == "to_update")
            other_doc = next(d for d in remaining if d["metric"]["name"] == "other_metric")

            assert updated_doc["metric"]["description"] == "Updated description"
            assert other_doc["metric"]["description"] == "should stay unchanged"
        finally:
            if os.path.exists(yaml_path):
                os.remove(yaml_path)

    def test_update_metric_no_yaml_path_still_succeeds(self, metric_storage: MetricStorage):
        """update_entry without yaml_path should update vector DB and return True."""
        metric = _make_metric(3, subject_path=["Finance", "Revenue"])
        metric["name"] = "no_yaml_update"
        metric["id"] = "metric:no_yaml_update"
        metric["yaml_path"] = ""
        metric_storage.batch_store_metrics([metric])

        result = metric_storage.update_entry(["Finance", "Revenue"], "no_yaml_update", {"description": "new desc"})
        assert result is True

    def test_update_metric_nonexistent_yaml_file_still_succeeds(self, metric_storage: MetricStorage):
        """update_entry with a yaml_path pointing to a missing file should still return True."""
        metric = _make_metric(4, subject_path=["Finance"], yaml_path="/nonexistent/path/metrics.yml")
        metric["name"] = "ghost_metric"
        metric["id"] = "metric:ghost_metric"
        metric_storage.batch_store_metrics([metric])

        result = metric_storage.update_entry(["Finance"], "ghost_metric", {"description": "updated"})
        assert result is True

    def test_update_metric_skips_unmapped_fields_in_yaml(self, metric_storage: MetricStorage):
        """update_entry with fields not in the YAML mapping should leave the file unchanged."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8") as f:
            yaml_path = f.name
            original_docs = [{"metric": {"name": "stable_metric", "description": "stays the same"}}]
            yaml.safe_dump_all(original_docs, f, allow_unicode=True, sort_keys=False)

        try:
            metric = _make_metric(5, subject_path=["Finance"], yaml_path=yaml_path)
            metric["name"] = "stable_metric"
            metric["id"] = "metric:stable_metric"
            metric_storage.batch_store_metrics([metric])

            result = metric_storage.update_entry(["Finance"], "stable_metric", {"sql": "SELECT 1"})
            assert result is True

            with open(yaml_path, "r", encoding="utf-8") as f:
                docs_out = [d for d in yaml.safe_load_all(f) if d is not None]

            assert docs_out[0]["metric"]["description"] == "stays the same"
            assert "sql" not in docs_out[0]["metric"]
        finally:
            if os.path.exists(yaml_path):
                os.remove(yaml_path)

    def test_update_metric_corrupt_yaml_does_not_raise(self, metric_storage: MetricStorage):
        """_sync_metric_update_to_yaml catches exceptions on corrupt YAML files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8") as f:
            yaml_path = f.name
            f.write("{{invalid yaml")

        try:
            metric = _make_metric(10, subject_path=["Finance"], yaml_path=yaml_path)
            metric["name"] = "corrupt_yaml_metric"
            metric["id"] = "metric:corrupt"
            metric_storage.batch_store_metrics([metric])

            result = metric_storage.update_entry(["Finance"], "corrupt_yaml_metric", {"description": "new"})
            assert result is True
        finally:
            if os.path.exists(yaml_path):
                os.remove(yaml_path)

    # ---------------------------------------------------------------------------
    def test_update_metric_entry_not_found_does_not_touch_yaml(self, metric_storage: MetricStorage):
        """update_entry on a non-existent metric should raise and leave the YAML file untouched."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8") as f:
            yaml_path = f.name
            docs = [{"metric": {"name": "existing", "description": "original"}}]
            yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)

        try:
            with pytest.raises(ValueError):
                metric_storage.update_entry(["Finance", "Revenue"], "nonexistent_metric", {"description": "new"})

            with open(yaml_path, "r", encoding="utf-8") as f:
                remaining = [d for d in yaml.safe_load_all(f) if d is not None]
            assert remaining[0]["metric"]["description"] == "original"
        finally:
            if os.path.exists(yaml_path):
                os.remove(yaml_path)


# ---------------------------------------------------------------------------


class TestRenameMetricSubjectTreeYaml:
    """Tests for YAML subject_tree sync in rename."""

    def test_rename_move_replaces_subject_tree_tag(self, metric_storage: MetricStorage):
        """Moving a metric to a new parent path should replace subject_tree in locked_metadata.tags."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8") as f:
            yaml_path = f.name
            docs = [
                {
                    "metric": {
                        "name": "move_me",
                        "description": "to be moved",
                        "locked_metadata": {"tags": ["subject_tree: Finance/Revenue", "other_tag"]},
                    }
                }
            ]
            yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)

        try:
            # Pre-create the destination subject path so rename() can resolve it
            metric_storage.subject_tree.find_or_create_path(["Finance", "Costs"])

            metric = _make_metric(1, subject_path=["Finance", "Revenue"], yaml_path=yaml_path)
            metric["name"] = "move_me"
            metric["id"] = "metric:move_me"
            metric_storage.batch_store_metrics([metric])

            result = metric_storage.rename(
                ["Finance", "Revenue", "move_me"],
                ["Finance", "Costs", "move_me"],
            )
            assert result is True

            with open(yaml_path, "r", encoding="utf-8") as f:
                docs_out = [d for d in yaml.safe_load_all(f) if d is not None]
            tags = docs_out[0]["metric"]["locked_metadata"]["tags"]
            assert "subject_tree: Finance/Costs" in tags
            assert "subject_tree: Finance/Revenue" not in tags
            assert "other_tag" in tags  # unrelated tag preserved
        finally:
            if os.path.exists(yaml_path):
                os.remove(yaml_path)

    def test_rename_move_appends_subject_tree_tag_when_missing(self, metric_storage: MetricStorage):
        """Moving a metric with no existing subject_tree tag should append a new one."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8") as f:
            yaml_path = f.name
            docs = [{"metric": {"name": "untagged", "description": "no subject tree tag"}}]
            yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)

        try:
            metric_storage.subject_tree.find_or_create_path(["Finance", "Costs"])

            metric = _make_metric(2, subject_path=["Finance", "Revenue"], yaml_path=yaml_path)
            metric["name"] = "untagged"
            metric["id"] = "metric:untagged"
            metric_storage.batch_store_metrics([metric])

            result = metric_storage.rename(
                ["Finance", "Revenue", "untagged"],
                ["Finance", "Costs", "untagged"],
            )
            assert result is True

            with open(yaml_path, "r", encoding="utf-8") as f:
                docs_out = [d for d in yaml.safe_load_all(f) if d is not None]
            tags = docs_out[0]["metric"]["locked_metadata"]["tags"]
            assert tags == ["subject_tree: Finance/Costs"]
        finally:
            if os.path.exists(yaml_path):
                os.remove(yaml_path)

    def test_rename_only_does_not_touch_subject_tree_tag(self, metric_storage: MetricStorage):
        """Renaming without moving should leave YAML subject_tree untouched."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8") as f:
            yaml_path = f.name
            docs = [
                {
                    "metric": {
                        "name": "old_name",
                        "description": "same parent",
                        "locked_metadata": {"tags": ["subject_tree: Finance/Revenue"]},
                    }
                }
            ]
            yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)

        try:
            metric = _make_metric(3, subject_path=["Finance", "Revenue"], yaml_path=yaml_path)
            metric["name"] = "old_name"
            metric["id"] = "metric:old_name"
            metric_storage.batch_store_metrics([metric])

            result = metric_storage.rename(
                ["Finance", "Revenue", "old_name"],
                ["Finance", "Revenue", "new_name"],
            )
            assert result is True

            with open(yaml_path, "r", encoding="utf-8") as f:
                docs_out = [d for d in yaml.safe_load_all(f) if d is not None]
            tags = docs_out[0]["metric"]["locked_metadata"]["tags"]
            # YAML subject_tree tag unchanged — rename() does not touch it when parent is unchanged
            assert tags == ["subject_tree: Finance/Revenue"]
        finally:
            if os.path.exists(yaml_path):
                os.remove(yaml_path)

    def test_rename_move_without_yaml_path_still_succeeds(self, metric_storage: MetricStorage):
        """Moving a metric without a yaml_path should still return True."""
        metric_storage.subject_tree.find_or_create_path(["Finance", "Costs"])

        metric = _make_metric(4, subject_path=["Finance", "Revenue"])
        metric["name"] = "no_yaml_move"
        metric["id"] = "metric:no_yaml_move"
        metric["yaml_path"] = ""
        metric_storage.batch_store_metrics([metric])

        result = metric_storage.rename(
            ["Finance", "Revenue", "no_yaml_move"],
            ["Finance", "Costs", "no_yaml_move"],
        )
        assert result is True


# ---------------------------------------------------------------------------
# YAML subject_tree sync when a subject_node is renamed/moved
# ---------------------------------------------------------------------------


class TestSyncYamlSubjectTreeForSubtree:
    """Tests for sync_yaml_subject_tree_for_subtree — triggered after a
    subject_tree node is renamed/moved, to sync YAML files of all descendants.
    """

    def _write_metric_yaml(self, tmp_path, metric_name: str, subject_tree_tag: str) -> str:
        """Write a simple metric YAML with a subject_tree tag and return its path."""
        yaml_path = os.path.join(tmp_path, f"{metric_name}.yml")
        docs = [
            {
                "metric": {
                    "name": metric_name,
                    "description": f"desc for {metric_name}",
                    "locked_metadata": {"tags": [f"subject_tree: {subject_tree_tag}", "keep_me"]},
                }
            }
        ]
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)
        return yaml_path

    def _read_subject_tree_tags(self, yaml_path: str, metric_name: str) -> list:
        with open(yaml_path, "r", encoding="utf-8") as f:
            docs = [d for d in yaml.safe_load_all(f) if d is not None]
        for doc in docs:
            if doc.get("metric", {}).get("name") == metric_name:
                return doc["metric"]["locked_metadata"]["tags"]
        return []

    def test_rename_node_syncs_yaml_for_direct_and_descendant_metrics(self, metric_storage: MetricStorage, tmp_path):
        """Renaming an intermediate subject_node should update YAML subject_tree for
        metrics under that node AND metrics under descendant nodes."""
        tmp_path_str = str(tmp_path)

        # Create subject tree:  Finance / Revenue / Q1
        #                                         \ Detail (sub-folder of Q1)
        metric_storage.subject_tree.find_or_create_path(["Finance", "Revenue", "Q1"])
        metric_storage.subject_tree.find_or_create_path(["Finance", "Revenue", "Q1", "Detail"])

        # Write YAML files referencing the OLD subject_tree paths
        top_yaml = self._write_metric_yaml(tmp_path_str, "top_metric", "Finance/Revenue/Q1")
        deep_yaml = self._write_metric_yaml(tmp_path_str, "deep_metric", "Finance/Revenue/Q1/Detail")

        # Insert metrics pointing at each path
        top_metric = _make_metric(10, subject_path=["Finance", "Revenue", "Q1"], yaml_path=top_yaml)
        top_metric["name"] = "top_metric"
        top_metric["id"] = "metric:top_metric"

        deep_metric = _make_metric(11, subject_path=["Finance", "Revenue", "Q1", "Detail"], yaml_path=deep_yaml)
        deep_metric["name"] = "deep_metric"
        deep_metric["id"] = "metric:deep_metric"

        metric_storage.batch_store_metrics([top_metric, deep_metric])

        # Capture node_id BEFORE rename (stable across rename)
        q1_node = metric_storage.subject_tree.get_node_by_path(["Finance", "Revenue", "Q1"])
        assert q1_node is not None
        root_id = q1_node["node_id"]

        # Rename Q1 -> Quarter1 (in place)
        metric_storage.subject_tree.rename(
            ["Finance", "Revenue", "Q1"],
            ["Finance", "Revenue", "Quarter1"],
        )

        # Sync YAML files for the renamed subtree
        metric_storage.sync_yaml_subject_tree_for_subtree(root_id)

        # Both YAMLs should reflect the new path
        top_tags = self._read_subject_tree_tags(top_yaml, "top_metric")
        assert "subject_tree: Finance/Revenue/Quarter1" in top_tags
        assert "subject_tree: Finance/Revenue/Q1" not in top_tags
        assert "keep_me" in top_tags

        deep_tags = self._read_subject_tree_tags(deep_yaml, "deep_metric")
        assert "subject_tree: Finance/Revenue/Quarter1/Detail" in deep_tags
        assert "subject_tree: Finance/Revenue/Q1/Detail" not in deep_tags
        assert "keep_me" in deep_tags

        try:
            pass
        finally:
            for p in (top_yaml, deep_yaml):
                if os.path.exists(p):
                    os.remove(p)

    def test_move_node_syncs_yaml_for_descendant_metrics(self, metric_storage: MetricStorage, tmp_path):
        """Moving a subject_node to a different parent should update YAML subject_tree
        for all descendant metrics."""
        tmp_path_str = str(tmp_path)

        metric_storage.subject_tree.find_or_create_path(["Finance", "Revenue", "Q1"])
        metric_storage.subject_tree.find_or_create_path(["Finance", "Costs"])

        yaml_path = self._write_metric_yaml(tmp_path_str, "movable", "Finance/Revenue/Q1")

        metric = _make_metric(12, subject_path=["Finance", "Revenue", "Q1"], yaml_path=yaml_path)
        metric["name"] = "movable"
        metric["id"] = "metric:movable"
        metric_storage.batch_store_metrics([metric])

        q1_node = metric_storage.subject_tree.get_node_by_path(["Finance", "Revenue", "Q1"])
        assert q1_node is not None
        root_id = q1_node["node_id"]

        # Move Q1 from Finance/Revenue to Finance/Costs
        metric_storage.subject_tree.rename(
            ["Finance", "Revenue", "Q1"],
            ["Finance", "Costs", "Q1"],
        )

        metric_storage.sync_yaml_subject_tree_for_subtree(root_id)

        tags = self._read_subject_tree_tags(yaml_path, "movable")
        assert "subject_tree: Finance/Costs/Q1" in tags
        assert "subject_tree: Finance/Revenue/Q1" not in tags

        try:
            pass
        finally:
            if os.path.exists(yaml_path):
                os.remove(yaml_path)

    def test_sync_skips_entries_without_yaml_path(self, metric_storage: MetricStorage, tmp_path):
        """Metrics with an empty yaml_path should be silently skipped."""
        metric_storage.subject_tree.find_or_create_path(["Finance", "Revenue", "Q1"])

        metric = _make_metric(13, subject_path=["Finance", "Revenue", "Q1"])
        metric["name"] = "no_yaml"
        metric["id"] = "metric:no_yaml"
        metric["yaml_path"] = ""
        metric_storage.batch_store_metrics([metric])

        q1_node = metric_storage.subject_tree.get_node_by_path(["Finance", "Revenue", "Q1"])
        root_id = q1_node["node_id"]

        # Should not raise
        metric_storage.sync_yaml_subject_tree_for_subtree(root_id)

    def test_sync_handles_subtree_with_no_metrics(self, metric_storage: MetricStorage):
        """A subtree with no metric entries should be a no-op (no errors)."""
        metric_storage.subject_tree.find_or_create_path(["Empty", "Branch"])
        node = metric_storage.subject_tree.get_node_by_path(["Empty", "Branch"])
        assert node is not None

        # Should not raise
        metric_storage.sync_yaml_subject_tree_for_subtree(node["node_id"])
