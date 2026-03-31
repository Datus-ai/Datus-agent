# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ToolRegistry."""

from unittest.mock import MagicMock

import pytest

from datus.tools.registry.tool_registry import ToolRegistry


@pytest.mark.ci
class TestToolRegistryInit:
    def test_empty(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        assert reg.to_dict() == {}

    def test_from_initial_dict(self):
        reg = ToolRegistry({"execute_sql": "db_tools", "read_file": "filesystem_tools"})
        assert len(reg) == 2
        assert reg.get("execute_sql") == "db_tools"

    def test_initial_dict_is_copied(self):
        original = {"a": "cat_a"}
        reg = ToolRegistry(original)
        original["b"] = "cat_b"
        assert "b" not in reg


@pytest.mark.ci
class TestRegisterTools:
    def test_register_tools_with_name_attr(self):
        reg = ToolRegistry()
        tool1 = MagicMock()
        tool1.name = "execute_sql"
        tool2 = MagicMock()
        tool2.name = "list_tables"

        reg.register_tools("db_tools", [tool1, tool2])

        assert reg.get("execute_sql") == "db_tools"
        assert reg.get("list_tables") == "db_tools"
        assert len(reg) == 2

    def test_register_tools_overwrites(self):
        reg = ToolRegistry({"execute_sql": "old_category"})
        tool = MagicMock()
        tool.name = "execute_sql"
        reg.register_tools("db_tools", [tool])
        assert reg.get("execute_sql") == "db_tools"

    def test_register_tools_stringifiable(self):
        reg = ToolRegistry()
        reg.register_tools("misc", ["plain_string_tool"])
        assert reg.get("plain_string_tool") == "misc"


@pytest.mark.ci
class TestReadAccess:
    def test_get_existing(self):
        reg = ToolRegistry({"a": "cat"})
        assert reg.get("a") == "cat"

    def test_get_missing_returns_none(self):
        reg = ToolRegistry()
        assert reg.get("missing") is None

    def test_get_missing_returns_default(self):
        reg = ToolRegistry()
        assert reg.get("missing", "fallback") == "fallback"

    def test_contains(self):
        reg = ToolRegistry({"a": "cat"})
        assert "a" in reg
        assert "b" not in reg

    def test_iter(self):
        reg = ToolRegistry({"a": "x", "b": "y"})
        assert sorted(reg) == ["a", "b"]

    def test_items(self):
        reg = ToolRegistry({"a": "x", "b": "y"})
        assert sorted(reg.items()) == [("a", "x"), ("b", "y")]

    def test_to_dict_returns_copy(self):
        reg = ToolRegistry({"a": "x"})
        d = reg.to_dict()
        d["b"] = "y"
        assert "b" not in reg


@pytest.mark.ci
class TestDunderHelpers:
    def test_repr(self):
        reg = ToolRegistry({"a": "cat"})
        assert "ToolRegistry" in repr(reg)
        assert "'a'" in repr(reg)

    def test_eq_with_tool_registry(self):
        reg1 = ToolRegistry({"a": "x"})
        reg2 = ToolRegistry({"a": "x"})
        assert reg1 == reg2

    def test_eq_with_dict(self):
        reg = ToolRegistry({"a": "x"})
        assert reg == {"a": "x"}

    def test_neq(self):
        reg1 = ToolRegistry({"a": "x"})
        reg2 = ToolRegistry({"a": "y"})
        assert reg1 != reg2

    def test_eq_with_unrelated_type(self):
        reg = ToolRegistry()
        assert reg != 42
