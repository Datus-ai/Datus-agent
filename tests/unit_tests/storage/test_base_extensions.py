# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for BaseEmbeddingStore extensions: table_prefix, extra_fields, default_values."""

from unittest.mock import MagicMock

import pyarrow as pa

from datus.storage.base import BaseEmbeddingStore


class _FakeEmbeddingModel:
    dim_size = 384
    batch_size = 32
    model_name = "fake"
    is_model_failed = False
    model_error_message = ""
    device = None

    @property
    def model(self):
        return MagicMock()


def _base_schema():
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=384)),
        ]
    )


class TestTablePrefix:
    """Tests for table_prefix parameter."""

    def test_no_prefix_keeps_original_name(self):
        store = BaseEmbeddingStore(
            table_name="metrics",
            embedding_model=_FakeEmbeddingModel(),
            schema=_base_schema(),
        )
        assert store.table_name == "metrics"

    def test_empty_prefix_keeps_original_name(self):
        store = BaseEmbeddingStore(
            table_name="metrics",
            embedding_model=_FakeEmbeddingModel(),
            schema=_base_schema(),
            table_prefix="",
        )
        assert store.table_name == "metrics"

    def test_prefix_is_prepended(self):
        store = BaseEmbeddingStore(
            table_name="metrics",
            embedding_model=_FakeEmbeddingModel(),
            schema=_base_schema(),
            table_prefix="tb_",
        )
        assert store.table_name == "tb_metrics"

    def test_prefix_with_different_tables(self):
        for name, expected in [
            ("schema_metadata", "tb_schema_metadata"),
            ("semantic_model", "tb_semantic_model"),
            ("ext_knowledge", "tb_ext_knowledge"),
        ]:
            store = BaseEmbeddingStore(
                table_name=name,
                embedding_model=_FakeEmbeddingModel(),
                schema=_base_schema(),
                table_prefix="tb_",
            )
            assert store.table_name == expected


class TestExtraFields:
    """Tests for extra_fields parameter."""

    def test_no_extra_fields_keeps_original_schema(self):
        schema = _base_schema()
        store = BaseEmbeddingStore(
            table_name="test",
            embedding_model=_FakeEmbeddingModel(),
            schema=schema,
        )
        assert store._schema is schema

    def test_extra_fields_are_appended(self):
        schema = _base_schema()
        extra = [
            pa.field("workspace_id", pa.string()),
            pa.field("created_by", pa.string()),
        ]
        store = BaseEmbeddingStore(
            table_name="test",
            embedding_model=_FakeEmbeddingModel(),
            schema=schema,
            extra_fields=extra,
        )
        field_names = [f.name for f in store._schema]
        assert "workspace_id" in field_names
        assert "created_by" in field_names
        # Original fields still present
        assert "id" in field_names
        assert "text" in field_names

    def test_extra_fields_with_none_schema(self):
        """extra_fields with schema=None should not crash."""
        store = BaseEmbeddingStore(
            table_name="test",
            embedding_model=_FakeEmbeddingModel(),
            schema=None,
            extra_fields=[pa.field("workspace_id", pa.string())],
        )
        # schema stays None since base schema is None
        assert store._schema is None

    def test_extra_fields_empty_list(self):
        schema = _base_schema()
        store = BaseEmbeddingStore(
            table_name="test",
            embedding_model=_FakeEmbeddingModel(),
            schema=schema,
            extra_fields=[],
        )
        # Empty list should not modify schema
        assert store._schema is schema


class TestDefaultValues:
    """Tests for default_values parameter."""

    def test_no_default_values(self):
        store = BaseEmbeddingStore(
            table_name="test",
            embedding_model=_FakeEmbeddingModel(),
            schema=_base_schema(),
        )
        assert store._default_values == {}

    def test_default_values_stored(self):
        defaults = {"workspace_id": "ws_123", "created_by": "user_1"}
        store = BaseEmbeddingStore(
            table_name="test",
            embedding_model=_FakeEmbeddingModel(),
            schema=_base_schema(),
            default_values=defaults,
        )
        assert store._default_values == defaults

    def test_apply_default_values_fills_missing(self):
        store = BaseEmbeddingStore(
            table_name="test",
            embedding_model=_FakeEmbeddingModel(),
            schema=_base_schema(),
            default_values={"workspace_id": "ws_123"},
        )
        data = [{"id": "1", "text": "hello"}]
        result = store._apply_default_values(data)
        assert result[0]["workspace_id"] == "ws_123"

    def test_apply_default_values_does_not_overwrite_existing(self):
        store = BaseEmbeddingStore(
            table_name="test",
            embedding_model=_FakeEmbeddingModel(),
            schema=_base_schema(),
            default_values={"workspace_id": "ws_default"},
        )
        data = [{"id": "1", "workspace_id": "ws_explicit"}]
        result = store._apply_default_values(data)
        assert result[0]["workspace_id"] == "ws_explicit"

    def test_apply_default_values_empty_defaults_is_noop(self):
        store = BaseEmbeddingStore(
            table_name="test",
            embedding_model=_FakeEmbeddingModel(),
            schema=_base_schema(),
        )
        data = [{"id": "1"}]
        result = store._apply_default_values(data)
        assert result == [{"id": "1"}]

    def test_apply_default_values_multiple_rows(self):
        store = BaseEmbeddingStore(
            table_name="test",
            embedding_model=_FakeEmbeddingModel(),
            schema=_base_schema(),
            default_values={"workspace_id": "ws_123", "tenant": "t1"},
        )
        data = [
            {"id": "1"},
            {"id": "2", "workspace_id": "ws_override"},
        ]
        result = store._apply_default_values(data)
        assert result[0]["workspace_id"] == "ws_123"
        assert result[0]["tenant"] == "t1"
        assert result[1]["workspace_id"] == "ws_override"
        assert result[1]["tenant"] == "t1"


class TestCombinedFeatures:
    """Test that table_prefix, extra_fields, and default_values work together."""

    def test_all_three_features_combined(self):
        extra = [pa.field("workspace_id", pa.string())]
        store = BaseEmbeddingStore(
            table_name="metrics",
            embedding_model=_FakeEmbeddingModel(),
            schema=_base_schema(),
            table_prefix="tb_",
            extra_fields=extra,
            default_values={"workspace_id": "ws_123"},
        )
        assert store.table_name == "tb_metrics"
        assert "workspace_id" in [f.name for f in store._schema]
        assert store._default_values == {"workspace_id": "ws_123"}
