# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.datasource_scope."""

from unittest.mock import MagicMock

import pytest

from datus.storage.datasource_scope import (
    DATASOURCE_ID_COLUMN,
    LEGACY_STORAGE_KEY_PREFIX,
    STORAGE_KEY_COLUMN,
    add_datasource_scope_to_rows,
    build_storage_key,
    combine_conditions,
    datasource_condition,
    resolve_datasource_id,
)
from datus.utils.exceptions import DatusException


class TestResolveDatasourceId:
    def test_explicit_datasource_id_returned(self):
        cfg = MagicMock()
        cfg.current_datasource = "ignored"
        assert resolve_datasource_id(cfg, datasource_id="my_ds") == "my_ds"

    def test_falls_back_to_agent_config_current_datasource(self):
        cfg = MagicMock()
        cfg.current_datasource = "cfg_ds"
        assert resolve_datasource_id(cfg) == "cfg_ds"

    def test_strips_whitespace(self):
        cfg = MagicMock()
        cfg.current_datasource = "  ds  "
        assert resolve_datasource_id(cfg) == "ds"

    def test_raises_when_empty_after_strip(self):
        cfg = MagicMock()
        cfg.current_datasource = ""
        with pytest.raises(DatusException):
            resolve_datasource_id(cfg)

    def test_raises_when_none_datasource_and_empty_config(self):
        cfg = MagicMock()
        cfg.current_datasource = None
        with pytest.raises(DatusException):
            resolve_datasource_id(cfg, datasource_id=None)

    def test_raises_when_whitespace_only_explicit(self):
        cfg = MagicMock()
        cfg.current_datasource = "anything"
        with pytest.raises(DatusException):
            resolve_datasource_id(cfg, datasource_id="   ")


class TestDatasourceCondition:
    def test_returns_node(self):
        node = datasource_condition("ds1")
        assert node is not None

    def test_different_ids_produce_different_conditions(self):
        node_a = datasource_condition("ds_a")
        node_b = datasource_condition("ds_b")
        assert node_a != node_b


class TestCombineConditions:
    def test_empty_returns_none(self):
        assert combine_conditions([]) is None

    def test_all_none_returns_none(self):
        assert combine_conditions([None, None]) is None

    def test_single_non_none_returned_as_is(self):
        cond = datasource_condition("ds1")
        result = combine_conditions([None, cond, None])
        assert result is cond

    def test_two_conditions_returns_and_node(self):
        cond_a = datasource_condition("ds_a")
        cond_b = datasource_condition("ds_b")
        result = combine_conditions([cond_a, cond_b])
        assert result is not None
        assert result is not cond_a
        assert result is not cond_b

    def test_three_conditions_combines(self):
        conds = [datasource_condition(f"ds_{i}") for i in range(3)]
        result = combine_conditions(conds)
        assert result is not None


class TestBuildStorageKey:
    def test_with_datasource_and_business_id(self):
        key = build_storage_key("my_ds", "row_123")
        assert key == "my_ds:row_123"

    def test_empty_datasource_uses_legacy_prefix(self):
        key = build_storage_key("", "row_456")
        assert key == f"{LEGACY_STORAGE_KEY_PREFIX}row_456"

    def test_none_datasource_uses_legacy_prefix(self):
        key = build_storage_key(None, "row_789")
        assert key == f"{LEGACY_STORAGE_KEY_PREFIX}row_789"

    def test_empty_business_id_raises(self):
        with pytest.raises(Exception, match="business id is required"):
            build_storage_key("ds", "")

    def test_none_business_id_raises(self):
        with pytest.raises(Exception, match="business id is required"):
            build_storage_key("ds", None)

    def test_whitespace_only_business_id_raises(self):
        with pytest.raises(Exception, match="business id is required"):
            build_storage_key("ds", "   ")

    def test_integer_business_id_stringified(self):
        key = build_storage_key("ds", 42)
        assert key == "ds:42"


class TestAddDatasourceScopeToRows:
    def test_adds_datasource_id_column(self):
        rows = [{"id": "r1", "name": "orders"}]
        result = add_datasource_scope_to_rows(rows, "my_ds")
        assert result[0][DATASOURCE_ID_COLUMN] == "my_ds"

    def test_adds_storage_key_when_id_present(self):
        rows = [{"id": "r1", "name": "orders"}]
        result = add_datasource_scope_to_rows(rows, "my_ds")
        assert result[0][STORAGE_KEY_COLUMN] == "my_ds:r1"

    def test_no_storage_key_when_id_missing(self):
        rows = [{"name": "orders"}]
        result = add_datasource_scope_to_rows(rows, "my_ds")
        assert STORAGE_KEY_COLUMN not in result[0]

    def test_no_storage_key_when_id_is_empty_string(self):
        rows = [{"id": "", "name": "orders"}]
        result = add_datasource_scope_to_rows(rows, "my_ds")
        assert STORAGE_KEY_COLUMN not in result[0]

    def test_custom_id_field(self):
        rows = [{"metric_id": "m1", "name": "revenue"}]
        result = add_datasource_scope_to_rows(rows, "ds1", id_field="metric_id")
        assert result[0][STORAGE_KEY_COLUMN] == "ds1:m1"

    def test_original_rows_not_mutated(self):
        rows = [{"id": "r1", "name": "orders"}]
        add_datasource_scope_to_rows(rows, "my_ds")
        assert DATASOURCE_ID_COLUMN not in rows[0]

    def test_empty_rows_returns_empty(self):
        assert add_datasource_scope_to_rows([], "ds") == []

    def test_multiple_rows_all_scoped(self):
        rows = [{"id": f"r{i}"} for i in range(3)]
        result = add_datasource_scope_to_rows(rows, "ds")
        assert all(r[DATASOURCE_ID_COLUMN] == "ds" for r in result)
        assert [r[STORAGE_KEY_COLUMN] for r in result] == ["ds:r0", "ds:r1", "ds:r2"]
