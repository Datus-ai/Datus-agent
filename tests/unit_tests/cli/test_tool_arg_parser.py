# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/tool_arg_parser.py.

The shared positional + ``--key=value`` parser backing both ``/<service>.<method>``
and ``!<tool>``. Covers: type coercion (int/number/bool/array/object incl. Python
literal + CSV fallbacks), ``anyOf``/``oneOf`` type flattening, positional/named
binding, and the error paths (extra positional, unknown flag, unmatched quotes).
"""

import io

import pytest
from rich.console import Console

from datus.cli import tool_arg_parser as tap


class TestIsHelpRequest:
    @pytest.mark.parametrize("args", ["--help", "-h", "foo --help", "42 -h"])
    def test_help_tokens_detected(self, args):
        assert tap.is_help_request(args) is True

    @pytest.mark.parametrize("args", ["", "foo", "--helpme", "--limit=1"])
    def test_non_help_returns_false(self, args):
        assert tap.is_help_request(args) is False

    def test_unmatched_quotes_is_not_help(self):
        assert tap.is_help_request("'unclosed --help") is False


class TestParseArgs:
    def test_positional_bind_in_schema_order(self):
        schema = {"properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
        parsed, err = tap.parse_args("foo 42", schema)
        assert err is None
        assert parsed == {"a": "foo", "b": 42}

    def test_named_override(self):
        schema = {"properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
        parsed, err = tap.parse_args("--b=99 --a=hi", schema)
        assert err is None
        assert parsed == {"a": "hi", "b": 99}

    def test_positional_then_named(self):
        schema = {"properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
        parsed, err = tap.parse_args("first --b=7", schema)
        assert err is None
        assert parsed == {"a": "first", "b": 7}

    def test_bare_flag_is_true(self):
        schema = {"properties": {"flag": {"type": "boolean"}}}
        assert tap.parse_args("--flag", schema)[0] == {"flag": True}
        assert tap.parse_args("--flag=true", schema)[0] == {"flag": True}
        assert tap.parse_args("--flag=no", schema)[0] == {"flag": False}

    def test_unknown_flag_reports_valid_names(self):
        schema = {"properties": {"a": {"type": "string"}, "limit": {"type": "integer"}}}
        parsed, err = tap.parse_args("--bogus=x --a=ok", schema)
        assert parsed is None
        assert err == "Unknown parameter '--bogus'. Valid parameters: a, limit."

    def test_too_many_positional(self):
        schema = {"properties": {"a": {"type": "string"}}}
        parsed, err = tap.parse_args("first second", schema)
        assert parsed is None
        assert err == "Too many positional arguments. Method accepts 1 (got extra: 'second')."

    def test_unmatched_quotes(self):
        parsed, err = tap.parse_args("'unclosed", {"properties": {}})
        assert parsed is None
        assert err == "Malformed arguments: unmatched quotes."

    def test_empty_flag(self):
        parsed, err = tap.parse_args("--", {"properties": {"a": {"type": "string"}}})
        assert parsed is None
        assert "Empty flag" in err

    def test_empty_args_returns_empty_dict(self):
        parsed, err = tap.parse_args("", {"properties": {"a": {"type": "string"}}})
        assert err is None
        assert parsed == {}

    def test_self_property_is_skipped_for_positionals(self):
        schema = {"properties": {"self": {"type": "string"}, "a": {"type": "string"}}}
        parsed, err = tap.parse_args("val", schema)
        assert err is None
        assert parsed == {"a": "val"}


class TestCoerce:
    def test_integer(self):
        assert tap.coerce("42", {"type": "integer"}) == 42
        assert tap.coerce("nope", {"type": "integer"}) == "nope"

    def test_number(self):
        assert tap.coerce("3.5", {"type": "number"}) == 3.5

    @pytest.mark.parametrize("raw,expected", [("1", True), ("true", True), ("yes", True), ("0", False), ("no", False)])
    def test_boolean(self, raw, expected):
        assert tap.coerce(raw, {"type": "boolean"}) is expected

    def test_array_json(self):
        assert tap.coerce('["a","b"]', {"type": "array"}) == ["a", "b"]

    def test_array_python_literal(self):
        assert tap.coerce("['a','b']", {"type": "array"}) == ["a", "b"]

    def test_array_csv_fallback(self):
        assert tap.coerce("a,b,c", {"type": "array"}) == ["a", "b", "c"]

    def test_object_json(self):
        assert tap.coerce('{"k": 1}', {"type": "object"}) == {"k": 1}

    def test_object_python_literal(self):
        assert tap.coerce("{'k': 1}", {"type": "object"}) == {"k": 1}

    def test_object_unparseable_returns_raw(self):
        assert tap.coerce("not-a-dict", {"type": "object"}) == "not-a-dict"

    def test_optional_array_via_anyof_coerces_list(self):
        """``Optional[List[str]]`` (``anyOf`` with null) must still coerce a CSV."""
        schema = {"properties": {"items": {"anyOf": [{"type": "array"}, {"type": "null"}]}}}
        parsed, err = tap.parse_args("--items=a,b,c", schema)
        assert err is None
        assert parsed == {"items": ["a", "b", "c"]}


class TestPrimaryType:
    def test_scalar(self):
        assert tap.primary_type({"type": "integer"}) == "integer"

    def test_type_list_skips_null(self):
        assert tap.primary_type({"type": ["string", "null"]}) == "string"

    def test_anyof_skips_null(self):
        assert tap.primary_type({"anyOf": [{"type": "null"}, {"type": "array"}]}) == "array"

    def test_oneof(self):
        assert tap.primary_type({"oneOf": [{"type": "integer"}]}) == "integer"

    def test_empty_and_none(self):
        assert tap.primary_type({}) == ""
        assert tap.primary_type(None) == ""


class TestMissingRequired:
    def test_reports_all_missing(self):
        def target(a, b, c):
            return None

        assert sorted(tap.missing_required(target, {"a": 1})) == ["b", "c"]

    def test_skips_optional_with_none_default(self):
        def target(a, metrics=None):
            return None

        assert tap.missing_required(target, {"metrics": ["x"]}) == ["a"]

    def test_none_method_returns_empty(self):
        assert tap.missing_required(None, {}) == []

    def test_skips_self_and_varargs(self):
        def target(self, *args, **kwargs):
            return None

        assert tap.missing_required(target, {}) == []


class TestPrintSchema:
    def test_renders_name_type_required_description(self):
        console = Console(file=io.StringIO(), no_color=True, width=200)
        tool = type(
            "T",
            (),
            {
                "name": "search_table",
                "params_json_schema": {
                    "properties": {
                        "self": {"type": "string"},
                        "query_text": {"type": "string", "description": "the query"},
                        "top_n": {"type": "integer"},
                    },
                    "required": ["query_text"],
                },
            },
        )()
        tap.print_schema(console, tool, hint="bad flag")
        out = console.file.getvalue()
        assert "search_table" in out
        assert "query_text" in out
        assert "the query" in out
        assert "top_n" in out
        # ``self`` is filtered out of the rendered rows.
        assert "bad flag" in out
