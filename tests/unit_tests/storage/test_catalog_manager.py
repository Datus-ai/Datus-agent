# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/storage/catalog_manager.py — CatalogUpdater pure logic methods."""

import json

from datus.storage.catalog_manager import CatalogUpdater


class TestParseJsonField:
    """Tests for CatalogUpdater._parse_json_field static behaviour."""

    def _make_updater_class(self):
        """Return a bare CatalogUpdater instance skipping __init__ for pure method tests."""
        # _parse_json_field is an instance method but uses no instance state,
        # so we can call it on an object created via __new__.
        obj = object.__new__(CatalogUpdater)
        return obj

    def test_parse_json_field_none_returns_none(self):
        """None input should return None."""
        updater = self._make_updater_class()
        assert updater._parse_json_field(None) is None

    def test_parse_json_field_valid_list_passthrough(self):
        """A Python list should be returned as-is."""
        updater = self._make_updater_class()
        data = [{"name": "col1"}, {"name": "col2"}]
        result = updater._parse_json_field(data)
        assert result == data
        assert result is data  # exact same object reference

    def test_parse_json_field_empty_list_passthrough(self):
        """An empty Python list should be returned as-is."""
        updater = self._make_updater_class()
        result = updater._parse_json_field([])
        assert result == []

    def test_parse_json_field_valid_json_string_list(self):
        """A JSON string encoding a list should be parsed correctly."""
        updater = self._make_updater_class()
        json_str = json.dumps([{"name": "dim1", "type": "string"}])
        result = updater._parse_json_field(json_str)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "dim1"

    def test_parse_json_field_valid_json_string_empty_list(self):
        """A JSON string encoding an empty list should return an empty list."""
        updater = self._make_updater_class()
        result = updater._parse_json_field("[]")
        assert result == []

    def test_parse_json_field_invalid_json_returns_none(self):
        """Malformed JSON string should return None."""
        updater = self._make_updater_class()
        result = updater._parse_json_field("{not valid json")
        assert result is None

    def test_parse_json_field_json_dict_returns_none(self):
        """A JSON string encoding a dict (not a list) should return None."""
        updater = self._make_updater_class()
        result = updater._parse_json_field(json.dumps({"key": "value"}))
        assert result is None

    def test_parse_json_field_json_string_scalar_returns_none(self):
        """A JSON string encoding a scalar should return None."""
        updater = self._make_updater_class()
        result = updater._parse_json_field('"just a string"')
        assert result is None

    def test_parse_json_field_json_number_returns_none(self):
        """A JSON string encoding a number should return None (not a list)."""
        updater = self._make_updater_class()
        result = updater._parse_json_field("42")
        assert result is None

    def test_parse_json_field_integer_returns_none(self):
        """An integer value should return None."""
        updater = self._make_updater_class()
        result = updater._parse_json_field(42)
        assert result is None

    def test_parse_json_field_float_returns_none(self):
        """A float value should return None."""
        updater = self._make_updater_class()
        result = updater._parse_json_field(3.14)
        assert result is None

    def test_parse_json_field_dict_returns_none(self):
        """A Python dict should return None (not a list)."""
        updater = self._make_updater_class()
        result = updater._parse_json_field({"key": "val"})
        assert result is None

    def test_parse_json_field_bool_returns_none(self):
        """A boolean value should return None."""
        updater = self._make_updater_class()
        result = updater._parse_json_field(True)
        assert result is None


class TestUpdateColumnsFieldMapping:
    """Tests for _update_columns column_type / type field mapping constants.

    The full change-detection loop is tested via TestUpdateColumnsMethod below.
    These tests verify the field mapping constants (allowed_fields sets) that
    are passed to _update_columns in update_semantic_model.
    """

    def _make_updater(self):
        """Create a bare CatalogUpdater for pure method tests."""
        obj = object.__new__(CatalogUpdater)
        obj.datasource_id = "test_datasource"
        return obj

    def test_type_to_column_type_mapping_detects_change(self):
        """column_type field maps from 'type' key in source data — verified via real method."""
        updater = self._make_updater()
        calls = []

        class FakeStorage:
            def update(self, where, values, unique_filter=None):
                calls.append(values)

        old = [{"name": "col1", "type": "string", "description": "A column"}]
        new = [{"name": "col1", "type": "integer", "description": "A column"}]

        updater._update_columns(
            storages=[FakeStorage()],
            catalog_name="cat",
            database_name="db",
            schema_name="sch",
            table_name="t",
            old_columns=old,
            new_columns=new,
            kind_field="is_dimension",
            allowed_fields={"description", "expr", "column_type"},
        )
        assert len(calls) == 1
        assert calls[0].get("column_type") == "integer"
        assert "description" not in calls[0]  # unchanged

    def test_no_changes_when_values_identical(self):
        """When old and new columns are identical, _update_columns makes no updates."""
        updater = self._make_updater()
        calls = []

        class FakeStorage:
            def update(self, where, values, unique_filter=None):
                calls.append(values)

        item = [{"name": "col1", "type": "string", "description": "desc", "expr": "x"}]

        updater._update_columns(
            storages=[FakeStorage()],
            catalog_name="cat",
            database_name="db",
            schema_name="sch",
            table_name="t",
            old_columns=item,
            new_columns=item,
            kind_field="is_dimension",
            allowed_fields={"description", "expr", "column_type"},
        )
        assert calls == []

    def test_description_change_detected(self):
        """A changed description field triggers an update with only the changed value."""
        updater = self._make_updater()
        calls = []

        class FakeStorage:
            def update(self, where, values, unique_filter=None):
                calls.append(values)

        old = [{"name": "col1", "type": "string", "description": "old desc"}]
        new = [{"name": "col1", "type": "string", "description": "new desc"}]

        updater._update_columns(
            storages=[FakeStorage()],
            catalog_name="cat",
            database_name="db",
            schema_name="sch",
            table_name="t",
            old_columns=old,
            new_columns=new,
            kind_field="is_dimension",
            allowed_fields={"description", "expr", "column_type"},
        )
        assert len(calls) == 1
        assert calls[0] == {"description": "new desc"}

    def test_missing_old_field_detected_as_change(self):
        """If old_item lacks a field that new_item has, _update_columns detects it as a change."""
        updater = self._make_updater()
        calls = []

        class FakeStorage:
            def update(self, where, values, unique_filter=None):
                calls.append(values)

        old = [{"name": "col1"}]
        new = [{"name": "col1", "description": "new desc", "type": "string"}]

        updater._update_columns(
            storages=[FakeStorage()],
            catalog_name="cat",
            database_name="db",
            schema_name="sch",
            table_name="t",
            old_columns=old,
            new_columns=new,
            kind_field="is_dimension",
            allowed_fields={"description", "expr", "column_type"},
        )
        assert len(calls) == 1
        assert "description" in calls[0]
        assert "column_type" in calls[0]


# ---------------------------------------------------------------------------
# _update_columns method tests
# ---------------------------------------------------------------------------


class TestUpdateColumnsMethod:
    """Tests for CatalogUpdater._update_columns via direct method invocation."""

    def _make_updater(self):
        """Create a bare CatalogUpdater for pure method tests."""
        obj = object.__new__(CatalogUpdater)
        obj.datasource_id = "test_datasource"
        return obj

    def test_update_columns_skips_items_without_name(self):
        """Columns without a 'name' key are skipped."""
        updater = self._make_updater()
        calls = []

        class FakeStorage:
            def update(self, where, values, unique_filter=None):
                calls.append((where, values))

        updater._update_columns(
            storages=[FakeStorage()],
            catalog_name="cat",
            database_name="db",
            schema_name="sch",
            table_name="t",
            old_columns=[{"name": "col1", "description": "old"}],
            new_columns=[{"description": "new"}],  # no 'name'
            kind_field="is_dimension",
            allowed_fields={"description"},
        )
        assert len(calls) == 0

    def test_update_columns_with_matching_names(self):
        """Matching columns with changed fields trigger storage.update."""
        updater = self._make_updater()
        calls = []

        class FakeStorage:
            def update(self, where, values, unique_filter=None):
                calls.append((where, values))

        old = [{"name": "col1", "description": "old desc"}]
        new = [{"name": "col1", "description": "new desc"}]

        updater._update_columns(
            storages=[FakeStorage()],
            catalog_name="cat",
            database_name="db",
            schema_name="sch",
            table_name="t",
            old_columns=old,
            new_columns=new,
            kind_field="is_dimension",
            allowed_fields={"description"},
        )
        assert len(calls) == 1
        assert calls[0][1] == {"description": "new desc"}

    def test_update_columns_no_changes_means_no_update(self):
        """When old and new columns are identical, no update is called."""
        updater = self._make_updater()
        calls = []

        class FakeStorage:
            def update(self, where, values, unique_filter=None):
                calls.append((where, values))

        identical = [{"name": "col1", "description": "same", "type": "string"}]

        updater._update_columns(
            storages=[FakeStorage()],
            catalog_name="cat",
            database_name="db",
            schema_name="sch",
            table_name="t",
            old_columns=identical,
            new_columns=identical,
            kind_field="is_measure",
            allowed_fields={"description", "column_type"},
        )
        assert len(calls) == 0

    def test_update_columns_multiple_storages(self):
        """Update is called on each storage in the list."""
        updater = self._make_updater()
        calls_1 = []
        calls_2 = []

        class FakeStorage1:
            def update(self, where, values, unique_filter=None):
                calls_1.append(values)

        class FakeStorage2:
            def update(self, where, values, unique_filter=None):
                calls_2.append(values)

        old = [{"name": "col1", "description": "old"}]
        new = [{"name": "col1", "description": "new"}]

        updater._update_columns(
            storages=[FakeStorage1(), FakeStorage2()],
            catalog_name="cat",
            database_name="db",
            schema_name="sch",
            table_name="t",
            old_columns=old,
            new_columns=new,
            kind_field="is_dimension",
            allowed_fields={"description"},
        )
        assert len(calls_1) == 1
        assert len(calls_2) == 1

    def test_update_columns_none_inputs_handled(self):
        """None old_columns and new_columns are handled gracefully."""
        updater = self._make_updater()
        calls = []

        class FakeStorage:
            def update(self, where, values, unique_filter=None):
                calls.append(values)

        updater._update_columns(
            storages=[FakeStorage()],
            catalog_name="cat",
            database_name="db",
            schema_name="sch",
            table_name="t",
            old_columns=None,
            new_columns=None,
            kind_field="is_dimension",
            allowed_fields={"description"},
        )
        assert len(calls) == 0

    def test_update_columns_json_string_inputs(self):
        """JSON string columns are parsed correctly."""
        updater = self._make_updater()
        calls = []

        class FakeStorage:
            def update(self, where, values, unique_filter=None):
                calls.append(values)

        old_json = json.dumps([{"name": "col1", "description": "old"}])
        new_json = json.dumps([{"name": "col1", "description": "new"}])

        updater._update_columns(
            storages=[FakeStorage()],
            catalog_name="cat",
            database_name="db",
            schema_name="sch",
            table_name="t",
            old_columns=old_json,
            new_columns=new_json,
            kind_field="is_entity_key",
            allowed_fields={"description"},
        )
        assert len(calls) == 1
        assert calls[0] == {"description": "new"}


# ---------------------------------------------------------------------------
# update_semantic_model
# ---------------------------------------------------------------------------


class TestUpdateSemanticModel:
    """Tests for update_semantic_model logic."""

    def _make_updater(self):
        """Create a bare CatalogUpdater for tests."""
        obj = object.__new__(CatalogUpdater)
        obj.datasource_id = "test_datasource"
        return obj

    def test_update_semantic_model_description(self):
        """update_semantic_model updates table description across storages."""
        updater = self._make_updater()
        calls = []

        class FakeStorage:
            def update(self, where, values, unique_filter=None):
                calls.append(("update", values))

        updater._get_all_storages = lambda: [FakeStorage()]

        old_values = {
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": "orders",
            "semantic_model_name": "orders_model",
        }
        update_values = {"description": "Updated description"}

        updater.update_semantic_model(old_values, update_values)

        # Should have at least one update call for the description
        assert any(v[1].get("description") == "Updated description" for v in calls)

    def test_update_semantic_model_dimensions(self):
        """update_semantic_model updates dimension columns."""
        updater = self._make_updater()
        calls = []

        class FakeStorage:
            def update(self, where, values, unique_filter=None):
                calls.append(values)

        updater._get_all_storages = lambda: [FakeStorage()]

        old_values = {
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": "orders",
            "semantic_model_name": "orders_model",
            "dimensions": [{"name": "region", "description": "old region desc"}],
        }
        update_values = {
            "dimensions": [{"name": "region", "description": "new region desc"}],
        }

        updater.update_semantic_model(old_values, update_values)

        assert any("description" in v and v["description"] == "new region desc" for v in calls)

    def test_update_semantic_model_measures(self):
        """update_semantic_model updates measure columns."""
        updater = self._make_updater()
        calls = []

        class FakeStorage:
            def update(self, where, values, unique_filter=None):
                calls.append(values)

        updater._get_all_storages = lambda: [FakeStorage()]

        old_values = {
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": "orders",
            "semantic_model_name": "orders_model",
            "measures": [{"name": "total_amount", "agg": "SUM"}],
        }
        update_values = {
            "measures": [{"name": "total_amount", "agg": "AVERAGE"}],
        }

        updater.update_semantic_model(old_values, update_values)

        assert any("agg" in v and v["agg"] == "AVERAGE" for v in calls)

    def test_update_semantic_model_identifiers(self):
        """update_semantic_model updates identifier columns."""
        updater = self._make_updater()
        calls = []

        class FakeStorage:
            def update(self, where, values, unique_filter=None):
                calls.append(values)

        updater._get_all_storages = lambda: [FakeStorage()]

        old_values = {
            "catalog_name": "cat",
            "database_name": "db",
            "schema_name": "sch",
            "table_name": "orders",
            "semantic_model_name": "orders_model",
            "identifiers": [{"name": "order_id", "entity": "order"}],
        }
        update_values = {
            "identifiers": [{"name": "order_id", "entity": "transaction"}],
        }

        updater.update_semantic_model(old_values, update_values)

        assert any("entity" in v and v["entity"] == "transaction" for v in calls)
