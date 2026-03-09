# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for BaseEmbeddingStore._apply_scope_filter method."""

from datus.storage.conditions import Node, build_where, eq
from datus.storage.schema_metadata import SchemaStorage


class TestApplyScopeFilter:
    """Tests for _apply_scope_filter combining scope filter with where expressions."""

    def _make_store(self, tmp_path) -> SchemaStorage:
        """Create a minimal SchemaStorage instance for testing."""
        from datus.storage.embedding_models import get_db_embedding_model

        return SchemaStorage(get_db_embedding_model())

    def test_no_scope_filter_returns_where_as_is(self, tmp_path):
        """With no scope filter, the original where is returned unchanged."""
        store = self._make_store(tmp_path)
        assert store._scope_filter is None
        where = eq("table_name", "users")
        result = store._apply_scope_filter(where)
        assert result is where

    def test_scope_filter_with_none_where(self, tmp_path):
        """When where is None but scope_filter exists, return the scope filter."""
        store = self._make_store(tmp_path)
        scope = eq("table_name", "users")
        store._scope_filter = scope

        result = store._apply_scope_filter(None)
        assert result is scope

    def test_scope_filter_with_string_where(self, tmp_path):
        """When where is a string, combine via string AND."""
        store = self._make_store(tmp_path)
        scope = eq("schema_name", "public")
        store._scope_filter = scope

        result = store._apply_scope_filter("table_name = 'users'")
        assert isinstance(result, str)
        assert "table_name = 'users'" in result
        assert "AND" in result
        assert "schema_name" in result

    def test_scope_filter_with_node_where(self, tmp_path):
        """When both are Node objects, combine via and_()."""
        store = self._make_store(tmp_path)
        scope = eq("schema_name", "public")
        store._scope_filter = scope

        where = eq("table_name", "users")
        result = store._apply_scope_filter(where)
        assert isinstance(result, Node)
        clause = build_where(result)
        assert "schema_name = 'public'" in clause
        assert "table_name = 'users'" in clause
        assert "AND" in clause

    def test_no_scope_filter_none_where_returns_none(self, tmp_path):
        """With no scope filter and None where, returns None."""
        store = self._make_store(tmp_path)
        result = store._apply_scope_filter(None)
        assert result is None
