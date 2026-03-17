import json
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from datus.utils.json_utils import (
    extract_code_block_content,
    extract_json_array,
    extract_json_object,
    extract_json_str,
    find_matching_bracket,
    json2csv,
    json_list2markdown_table,
    llm_result2json,
    llm_result2sql,
    load_jsonl,
    load_jsonl_dict,
    load_jsonl_iterator,
    strip_json_str,
    to_pretty_str,
    to_str,
)


def test_to_str_serializes_pydantic_model():
    pydantic = pytest.importorskip("pydantic")

    class SampleModel(pydantic.BaseModel):
        created_at: datetime
        amount: Decimal
        tags: set[str]

    model = SampleModel(created_at=datetime(2025, 1, 1, 12, 30), amount=Decimal("12.34"), tags={"alpha", "beta"})
    payload = json.loads(to_str(model))
    assert payload["created_at"] == "2025-01-01T12:30:00"
    assert payload["amount"] == "12.34"
    assert set(payload["tags"]) == {"alpha", "beta"}


def test_to_str_serializes_dataclass_and_uuid():
    @dataclass
    class Example:
        name: str
        identifier: UUID
        location: Path

    instance = Example(name="demo", identifier=uuid4(), location=Path("/tmp/example"))
    payload = json.loads(to_str(instance))
    assert payload["name"] == "demo"
    assert payload["identifier"] == str(instance.identifier)
    assert payload["location"] == "/tmp/example"


def test_to_pretty_str_from_json_text_roundtrip():
    pretty = to_pretty_str('{"foo": 1, "bar": 2}')
    assert "\n" in pretty
    assert json.loads(pretty) == {"foo": 1, "bar": 2}


def test_to_str_returns_raw_for_non_json_bytes():
    assert to_str(b"plain text payload") == "plain text payload"


def test_to_str_coerces_mapping_keys_to_strings():
    payload = json.loads(to_str({1: "value", Path("home"): 2}))
    assert payload == {"1": "value", "home": 2}


def test_to_str_normalizes_pandas_and_numpy_objects():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")

    frame = pd.DataFrame([{"value": Decimal("1.5"), "ts": datetime(2024, 1, 1, 8, 30)}])
    array = np.array([1, 2, 3])
    payload = json.loads(to_str({"frame": frame, "array": array}))

    assert payload["array"] == [1, 2, 3]
    assert payload["frame"][0]["value"] == "1.5"
    assert payload["frame"][0]["ts"] == "2024-01-01T08:30:00"


# ---------------------------------------------------------------------------
# json2csv
# ---------------------------------------------------------------------------
class TestJson2Csv:
    def test_empty_returns_empty_string(self):
        assert json2csv(None) == ""
        assert json2csv([]) == ""

    def test_list_of_dicts_to_csv(self):
        result = json2csv([{"a": 1, "b": 2}])
        assert "a" in result
        assert "1" in result

    def test_dict_wrapped_in_list(self):
        result = json2csv({"x": 10})
        assert "x" in result

    def test_json_string_list(self):
        result = json2csv('[{"name": "alice"}]')
        assert "alice" in result

    def test_json_string_object(self):
        result = json2csv('{"key": "value"}')
        assert "key" in result

    def test_plain_string_returned_as_is(self):
        result = json2csv("just a plain string")
        assert result == "just a plain string"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            json2csv(12345)

    def test_columns_filter(self):
        data = [{"a": 1, "b": 2, "c": 3}]
        result = json2csv(data, columns=["a", "b"])
        assert "a" in result
        assert "b" in result
        assert "c" not in result


# ---------------------------------------------------------------------------
# find_matching_bracket
# ---------------------------------------------------------------------------
class TestFindMatchingBracket:
    def test_simple_array(self):
        assert find_matching_bracket("[1,2,3]", 0) == 6

    def test_nested_brackets(self):
        assert find_matching_bracket("[[1,2],[3,4]]", 0) == 12

    def test_no_match_returns_negative_one(self):
        assert find_matching_bracket("[unclosed", 0) == -1

    def test_curly_braces(self):
        assert find_matching_bracket('{"a":1}', 0, "{", "}") == 6

    def test_mismatched_close_first(self):
        # Closing bracket before any opening - returns -1
        result = find_matching_bracket("]something", 0)
        assert result == -1


# ---------------------------------------------------------------------------
# extract_json_object
# ---------------------------------------------------------------------------
class TestExtractJsonObject:
    def test_simple_object(self):
        result = extract_json_object('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_no_object_returns_empty(self):
        result = extract_json_object("no json here")
        assert result == ""

    def test_object_with_prefix(self):
        result = extract_json_object('prefix {"key": 1} suffix')
        assert '{"key": 1}' == result

    def test_unclosed_object_returns_empty(self):
        result = extract_json_object('{"unclosed": ')
        assert result == ""


# ---------------------------------------------------------------------------
# extract_json_array
# ---------------------------------------------------------------------------
class TestExtractJsonArray:
    def test_simple_array(self):
        result = extract_json_array("[1, 2, 3]")
        assert result == "[1, 2, 3]"

    def test_no_array_returns_empty(self):
        result = extract_json_array("no array here")
        assert result == ""

    def test_array_with_prefix(self):
        result = extract_json_array("prefix [1, 2] suffix")
        assert "[1, 2]" == result

    def test_unclosed_array_returns_empty(self):
        result = extract_json_array("[unclosed")
        assert result == ""


# ---------------------------------------------------------------------------
# extract_code_block_content
# ---------------------------------------------------------------------------
class TestExtractCodeBlockContent:
    def test_json_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_code_block_content(text)
        assert result == '{"key": "value"}'

    def test_plain_code_block(self):
        text = "```\nsome content\n```"
        result = extract_code_block_content(text)
        assert result == "some content"

    def test_no_code_block_returns_empty(self):
        result = extract_code_block_content("no code block here")
        assert result == ""

    def test_unclosed_code_block_returns_empty(self):
        result = extract_code_block_content("```json\n{unclosed")
        assert result == ""


# ---------------------------------------------------------------------------
# llm_result2json
# ---------------------------------------------------------------------------
class TestLlmResult2Json:
    def test_plain_json_dict(self):
        result = llm_result2json('{"sql": "SELECT 1"}')
        assert result is not None
        assert result["sql"] == "SELECT 1"

    def test_json_in_code_block(self):
        result = llm_result2json('```json\n{"sql": "SELECT 1"}\n```')
        assert result is not None

    def test_returns_none_for_empty_string(self):
        result = llm_result2json("")
        assert result is None

    def test_returns_none_for_invalid_json(self):
        # json_repair may fix some things, but truly invalid should fail
        result = llm_result2json("not json at all without any brackets")
        assert result is None

    def test_returns_list_result(self):
        result = llm_result2json('[{"a": 1}]', expected_type=list)
        assert result is not None
        assert isinstance(result, list)

    def test_returns_none_when_both_sql_and_output_empty(self):
        result = llm_result2json('{"sql": "", "output": ""}')
        assert result is None

    def test_returns_dict_when_output_present(self):
        result = llm_result2json('{"sql": "", "output": "some result"}')
        assert result is not None
        assert result["output"] == "some result"

    def test_returns_none_for_dict_without_sql_or_output(self):
        # llm_result2json returns None when dict lacks both 'sql' and 'output' keys
        result = llm_result2json('{"plan": "step1", "action": "do_something"}')
        assert result is None

    def test_returns_dict_with_sql_key(self):
        result = llm_result2json('{"sql": "SELECT 1", "action": "query"}')
        assert result is not None
        assert result["sql"] == "SELECT 1"


# ---------------------------------------------------------------------------
# llm_result2sql
# ---------------------------------------------------------------------------
class TestLlmResult2Sql:
    def test_extracts_sql_from_code_block(self):
        text = "```sql\nSELECT * FROM orders\n```"
        result = llm_result2sql(text)
        assert result == "SELECT * FROM orders"

    def test_extracts_sql_case_insensitive(self):
        text = "```SQL\nSELECT 1\n```"
        result = llm_result2sql(text)
        assert result == "SELECT 1"

    def test_returns_none_for_empty(self):
        assert llm_result2sql("") is None
        assert llm_result2sql(None) is None

    def test_returns_none_for_non_string(self):
        assert llm_result2sql(42) is None

    def test_fallback_generic_code_block_with_sql_keywords(self):
        text = "```\nSELECT id FROM users WHERE active = 1\n```"
        result = llm_result2sql(text)
        assert result is not None
        assert "SELECT" in result

    def test_returns_none_when_no_sql_found(self):
        result = llm_result2sql("just some plain text with no code blocks")
        assert result is None

    def test_empty_sql_block_returns_none(self):
        result = llm_result2sql("```sql\n\n```")
        assert result is None


# ---------------------------------------------------------------------------
# json_list2markdown_table
# ---------------------------------------------------------------------------
class TestJsonList2MarkdownTable:
    def test_empty_list_returns_empty(self):
        result = json_list2markdown_table([])
        assert result == ""

    def test_single_row(self):
        result = json_list2markdown_table([{"name": "alice", "age": 30}])
        assert result is not None
        assert "alice" in result

    def test_multiple_rows(self):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = json_list2markdown_table(data)
        assert "|" in result


# ---------------------------------------------------------------------------
# strip_json_str
# ---------------------------------------------------------------------------
class TestStripJsonStr:
    def test_none_returns_empty(self):
        assert strip_json_str(None) == ""

    def test_empty_returns_empty(self):
        assert strip_json_str("") == ""

    def test_strips_json_code_block(self):
        text = '```json\n{"key": 1}\n```'
        result = strip_json_str(text)
        assert result.startswith("{")

    def test_strips_plain_code_block(self):
        text = '```\n{"key": 1}\n```'
        result = strip_json_str(text)
        assert result.startswith("{")

    def test_extracts_embedded_object(self):
        text = 'Here is the result: {"key": "value"} end'
        result = strip_json_str(text)
        assert result.startswith("{")

    def test_extracts_embedded_array(self):
        text = "Result is: [1, 2, 3] done"
        result = strip_json_str(text)
        assert result.startswith("[")

    def test_completes_truncated_object(self):
        text = '{"key": "value"'
        result = strip_json_str(text)
        assert result.endswith("}")

    def test_completes_truncated_array(self):
        text = "[1, 2, 3"
        result = strip_json_str(text)
        assert result.endswith("]")

    def test_already_valid_json_unchanged(self):
        text = '{"key": "value"}'
        result = strip_json_str(text)
        assert result == text


# ---------------------------------------------------------------------------
# extract_json_str
# ---------------------------------------------------------------------------
class TestExtractJsonStr:
    def test_extracts_from_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_json_str(text)
        assert "key" in result

    def test_plain_json_unchanged(self):
        text = '{"key": "value"}'
        result = extract_json_str(text)
        assert result == text

    def test_empty_returns_empty(self):
        result = extract_json_str("")
        assert result == ""


# ---------------------------------------------------------------------------
# load_jsonl / load_jsonl_iterator / load_jsonl_dict
# ---------------------------------------------------------------------------
class TestLoadJsonl:
    def test_load_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
        result = load_jsonl(str(f))
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_load_jsonl_iterator(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"b": 10}\n{"b": 20}\n', encoding="utf-8")
        items = list(load_jsonl_iterator(str(f)))
        assert len(items) == 2
        assert items[1]["b"] == 20

    def test_load_jsonl_dict(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"instance_id": "x1", "val": 1}\n{"instance_id": "x2", "val": 2}\n', encoding="utf-8")
        result = load_jsonl_dict(str(f))
        assert "x1" in result
        assert result["x2"]["val"] == 2

    def test_load_jsonl_dict_custom_key(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"id": "abc", "name": "test"}\n', encoding="utf-8")
        result = load_jsonl_dict(str(f), key_field="id")
        assert "abc" in result


# ---------------------------------------------------------------------------
# _normalize_for_json (via to_str / to_pretty_str)
# ---------------------------------------------------------------------------
class TestNormalizeForJson:
    def test_datetime_serialized_as_isoformat(self):
        dt = datetime(2025, 1, 15, 10, 30, 0)
        result = json.loads(to_str({"ts": dt}))
        assert result["ts"] == "2025-01-15T10:30:00"

    def test_date_serialized_as_isoformat(self):
        d = date(2025, 6, 1)
        result = json.loads(to_str({"d": d}))
        assert result["d"] == "2025-06-01"

    def test_time_serialized_as_isoformat(self):
        t = time(12, 30, 45)
        result = json.loads(to_str({"t": t}))
        assert result["t"] == "12:30:45"

    def test_decimal_serialized_as_string(self):
        result = json.loads(to_str({"amount": Decimal("99.99")}))
        assert result["amount"] == "99.99"

    def test_uuid_serialized_as_string(self):
        uid = UUID("12345678-1234-5678-1234-567812345678")
        result = json.loads(to_str({"id": uid}))
        assert result["id"] == "12345678-1234-5678-1234-567812345678"

    def test_path_serialized_as_string(self):
        p = Path("/tmp/test")
        result = json.loads(to_str({"p": p}))
        assert result["p"] == "/tmp/test"

    def test_bytes_decoded_to_string(self):
        result = json.loads(to_str({"data": b"hello"}))
        assert result["data"] == "hello"

    def test_enum_value_serialized(self):
        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        result = json.loads(to_str({"color": Color.RED}))
        assert result["color"] == "red"

    def test_enum_with_non_primitive_value(self):
        class Status(Enum):
            ACTIVE = 1
            INACTIVE = 2

        result = json.loads(to_str({"s": Status.ACTIVE}))
        assert result["s"] == 1

    def test_set_serialized_as_list(self):
        result = json.loads(to_str({"items": {1, 2, 3}}))
        assert sorted(result["items"]) == [1, 2, 3]

    def test_tuple_serialized_as_list(self):
        result = json.loads(to_str({"t": (1, 2, 3)}))
        assert result["t"] == [1, 2, 3]

    def test_frozenset_serialized_as_list(self):
        result = json.loads(to_str({"fs": frozenset([4, 5])}))
        assert sorted(result["fs"]) == [4, 5]

    def test_nested_dict(self):
        result = json.loads(to_str({"outer": {"inner": 42}}))
        assert result["outer"]["inner"] == 42

    def test_none_returns_none(self):
        assert to_str(None) is None
        assert to_pretty_str(None) is None

    def test_object_with_dict_attr(self):
        class Obj:
            def __init__(self):
                self.x = 1
                self.y = "hello"

        result = json.loads(to_str(Obj()))
        assert result["x"] == 1
        assert result["y"] == "hello"

    def test_dataclass_serialized(self):
        @dataclass
        class Point:
            x: int
            y: int

        result = json.loads(to_str(Point(x=3, y=4)))
        assert result["x"] == 3
        assert result["y"] == 4

    def test_to_pretty_str_has_indentation(self):
        result = to_pretty_str({"key": "value"})
        assert "\n" in result

    def test_to_str_compact(self):
        result = to_str({"key": "value"})
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_bytes_payload_non_json_returned_as_is(self):
        result = to_str(b"plain text")
        assert result == "plain text"

    def test_bytes_json_reparsed(self):
        result = to_str(b'{"x": 1}')
        parsed = json.loads(result)
        assert parsed["x"] == 1

    def test_empty_bytes_returned_as_is(self):
        result = to_str(b"")
        assert result == ""

    def test_string_json_reparsed(self):
        result = to_str('{"a": 2}')
        parsed = json.loads(result)
        assert parsed["a"] == 2

    def test_string_non_json_returned_as_is(self):
        result = to_str("hello world")
        assert result == "hello world"

    def test_pydantic_model(self):
        pydantic = pytest.importorskip("pydantic")

        class MyModel(pydantic.BaseModel):
            name: str
            count: int

        m = MyModel(name="test", count=5)
        result = json.loads(to_str(m))
        assert result["name"] == "test"
        assert result["count"] == 5
