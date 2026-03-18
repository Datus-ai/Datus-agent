# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.schemas.base — covers parse_table_type_by_db and base model methods."""

import pytest
from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult, CommonData, parse_table_type_by_db

pytestmark = pytest.mark.ci


# ---------------------------------------------------------------------------
# parse_table_type_by_db (lines 13-18)
# ---------------------------------------------------------------------------


class TestParseTableTypeByDb:
    """Tests for parse_table_type_by_db function."""

    def test_table_literal(self):
        assert parse_table_type_by_db("TABLE") == "table"

    def test_base_table(self):
        assert parse_table_type_by_db("BASE TABLE") == "table"

    def test_table_lowercase(self):
        assert parse_table_type_by_db("table") == "table"

    def test_base_table_mixed_case(self):
        assert parse_table_type_by_db("Base Table") == "table"

    def test_view(self):
        assert parse_table_type_by_db("VIEW") == "view"

    def test_view_lowercase(self):
        assert parse_table_type_by_db("view") == "view"

    def test_materialized_view_returns_mv(self):
        assert parse_table_type_by_db("MATERIALIZED VIEW") == "mv"

    def test_unknown_type_returns_mv(self):
        """Unknown types fall through to the default 'mv' return."""
        assert parse_table_type_by_db("EXTERNAL TABLE") == "mv"

    def test_empty_string_returns_mv(self):
        assert parse_table_type_by_db("") == "mv"


# ---------------------------------------------------------------------------
# Concrete subclasses for testing abstract base models
# ---------------------------------------------------------------------------


class SampleInput(BaseInput):
    name: str = Field(default="test")
    value: int = Field(default=42)


class SampleResult(BaseResult):
    data: str = Field(default="")


class SampleCommon(CommonData):
    label: str = Field(default="hello")


# ---------------------------------------------------------------------------
# BaseInput (lines 29-51)
# ---------------------------------------------------------------------------


class TestBaseInput:
    """Tests for BaseInput methods."""

    def test_get_existing_field(self):
        obj = SampleInput(name="alice", value=10)
        assert obj.get("name") == "alice"

    def test_get_missing_field_returns_default(self):
        obj = SampleInput()
        assert obj.get("nonexistent", "fallback") == "fallback"

    def test_get_missing_field_returns_none_by_default(self):
        obj = SampleInput()
        assert obj.get("nonexistent") is None

    def test_getitem_existing_field(self):
        obj = SampleInput(name="bob")
        assert obj["name"] == "bob"

    def test_to_str_returns_json_string(self):
        obj = SampleInput(name="charlie", value=99)
        result = obj.to_str()
        assert isinstance(result, str)
        assert "charlie" in result
        assert "99" in result

    def test_from_str_roundtrip(self):
        original = SampleInput(name="dave", value=7)
        serialized = original.to_str()
        restored = SampleInput.from_str(serialized)
        assert restored.name == "dave"
        assert restored.value == 7

    def test_to_dict_returns_dict(self):
        obj = SampleInput(name="eve", value=5)
        d = obj.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "eve"
        assert d["value"] == 5

    def test_from_dict_roundtrip(self):
        original = SampleInput(name="frank", value=3)
        d = original.to_dict()
        restored = SampleInput.from_dict(d)
        assert restored.name == "frank"
        assert restored.value == 3


# ---------------------------------------------------------------------------
# BaseResult (lines 73-96)
# ---------------------------------------------------------------------------


class TestBaseResult:
    """Tests for BaseResult methods."""

    def test_get_existing_field(self):
        obj = SampleResult(success=True, data="ok")
        assert obj.get("success") is True

    def test_get_missing_field_returns_default(self):
        obj = SampleResult(success=False)
        assert obj.get("nonexistent", 42) == 42

    def test_getitem_existing_field(self):
        obj = SampleResult(success=True, data="result")
        assert obj["data"] == "result"

    def test_to_str_returns_json_string(self):
        obj = SampleResult(success=True, data="output")
        result = obj.to_str()
        assert isinstance(result, str)
        assert "output" in result

    def test_to_dict_returns_dict(self):
        obj = SampleResult(success=True, data="x")
        d = obj.to_dict()
        assert isinstance(d, dict)
        assert d["success"] is True

    def test_from_str_roundtrip(self):
        original = SampleResult(success=False, error="bad", data="")
        serialized = original.to_str()
        restored = SampleResult.from_str(serialized)
        assert restored.success is False
        assert restored.error == "bad"

    def test_from_dict_roundtrip(self):
        original = SampleResult(success=True, data="hi")
        d = original.to_dict()
        restored = SampleResult.from_dict(d)
        assert restored.success is True
        assert restored.data == "hi"


# ---------------------------------------------------------------------------
# CommonData (lines 107-130)
# ---------------------------------------------------------------------------


class TestCommonData:
    """Tests for CommonData methods."""

    def test_get_existing_field(self):
        obj = SampleCommon(label="world")
        assert obj.get("label") == "world"

    def test_get_missing_field_returns_default(self):
        obj = SampleCommon()
        assert obj.get("missing", "default") == "default"

    def test_getitem_existing_field(self):
        obj = SampleCommon(label="foo")
        assert obj["label"] == "foo"

    def test_to_str_returns_json_string(self):
        obj = SampleCommon(label="bar")
        result = obj.to_str()
        assert isinstance(result, str)
        assert "bar" in result

    def test_from_str_roundtrip(self):
        original = SampleCommon(label="baz")
        serialized = original.to_str()
        restored = SampleCommon.from_str(serialized)
        assert restored.label == "baz"

    def test_to_dict_returns_dict(self):
        obj = SampleCommon(label="qux")
        d = obj.to_dict()
        assert isinstance(d, dict)
        assert d["label"] == "qux"

    def test_from_dict_roundtrip(self):
        original = SampleCommon(label="quux")
        d = original.to_dict()
        restored = SampleCommon.from_dict(d)
        assert restored.label == "quux"
