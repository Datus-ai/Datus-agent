# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/utils/rich_util.py — CI tier, zero external deps."""

import pytest
from rich.console import Console
from rich.tree import Tree

from datus.utils.rich_util import dict_to_tree


class TestDictToTreeBasic:
    """Basic tests for dict_to_tree."""

    def test_returns_tree_instance(self):
        """dict_to_tree always returns a Tree."""
        result = dict_to_tree({"key": "value"})
        assert isinstance(result, Tree)

    def test_empty_dict_returns_root_tree(self):
        """Empty dict still returns the tree root."""
        result = dict_to_tree({})
        assert isinstance(result, Tree)

    def test_uses_provided_tree(self):
        """When a tree is provided it is returned (not a new one)."""
        existing = Tree("root")
        result = dict_to_tree({"a": "b"}, tree=existing)
        assert result is existing

    def test_creates_default_tree_when_none(self):
        """When tree=None a new Tree with '--' label is created."""
        result = dict_to_tree({"x": 1})
        assert result.label == "--"

    def test_flat_string_values(self):
        """Flat dict with string values adds leaf nodes."""
        result = dict_to_tree({"name": "alice", "city": "NY"})
        # Two children added
        assert len(result.children) == 2

    def test_nested_dict_value(self):
        """Nested dict creates a branch."""
        result = dict_to_tree({"outer": {"inner": "val"}})
        assert len(result.children) == 1

    def test_list_value_adds_branch(self):
        """Non-empty list value adds a branch with indexed items."""
        result = dict_to_tree({"items": ["a", "b", "c"]})
        assert len(result.children) == 1

    def test_list_of_dicts(self):
        """List containing dicts recurses correctly."""
        result = dict_to_tree({"rows": [{"id": 1}, {"id": 2}]})
        assert len(result.children) == 1  # one branch for 'rows'

    def test_sql_query_key_renders_syntax(self):
        """Keys named 'sql_query' trigger Syntax rendering (no exception)."""
        result = dict_to_tree({"sql_query": "SELECT 1"})
        assert isinstance(result, Tree)

    def test_sql_query_non_string_value(self):
        """sql_query with a non-string value is converted to str."""
        result = dict_to_tree({"sql_query": 42})
        assert isinstance(result, Tree)

    def test_long_value_truncated(self):
        """Values longer than max_length are truncated with '...'."""
        console = Console(width=10)
        long_value = "x" * 10000
        result = dict_to_tree({"desc": long_value}, console=console)
        assert isinstance(result, Tree)
        # The leaf label should contain '...'
        leaf_label = str(result.children[0].label)
        assert "..." in leaf_label

    def test_with_console_uses_width(self):
        """Providing a console affects the max_length calculation."""
        console = Console(width=80)
        result = dict_to_tree({"key": "value"}, console=console)
        assert isinstance(result, Tree)

    def test_empty_list_value(self):
        """Empty list falls through to else branch (rendered as string)."""
        result = dict_to_tree({"empty_list": []})
        assert len(result.children) == 1

    def test_empty_dict_value(self):
        """Empty nested dict falls through to else branch."""
        result = dict_to_tree({"empty_dict": {}})
        assert len(result.children) == 1

    def test_none_value(self):
        """None value is handled without error."""
        result = dict_to_tree({"key": None})
        assert isinstance(result, Tree)

    def test_integer_value(self):
        """Integer value is converted to string and added as leaf."""
        result = dict_to_tree({"count": 42})
        assert isinstance(result, Tree)
        assert len(result.children) == 1

    @pytest.mark.parametrize(
        "data",
        [
            {"a": "simple"},
            {"nested": {"b": "deep"}},
            {"list": [1, 2, 3]},
            {"sql_query": "SELECT * FROM t"},
            {"mixed": {"x": [{"id": 1}]}},
        ],
    )
    def test_various_structures_no_exception(self, data):
        """Variety of dict structures do not raise."""
        result = dict_to_tree(data)
        assert isinstance(result, Tree)
