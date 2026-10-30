import copy
import json
from unittest.mock import patch

import pytest

from datus.models.request_capacity import ContextCapacityError
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.tool_archive import ToolArchive
from datus.utils.tool_result_store import ACTIVE_RESULT_STORE, ToolResultStore, json_text
from datus.utils.tool_result_view import project_tool_results


@pytest.fixture
def store(tmp_path):
    return ToolResultStore(ToolArchive("project", "session", base_dir=tmp_path))


def collect(store, page, section=""):
    rows = []
    while True:
        assert len(json_text(page).encode()) <= 8192
        rows.extend(page["result"]["items"])
        if not page["result"]["has_more"]:
            return rows
        extra = page["result"]["extra"]
        page = store.read(extra["result_id"], section, extra["next_cursor"])


def test_large_list_round_trips_all_records_across_restart(store):
    original = [{"qualified_name": f"warehouse.schema.table_{i}"} for i in range(30000)]
    raw = {"success": 1, "error": None, "result": original}
    before = copy.deepcopy(raw)
    page = store.project("call_1", raw)
    assert raw == before
    rid = page["result"]["extra"]["result_id"]
    restarted = ToolResultStore(ToolArchive("project", "session", base_dir=store.root.parent))
    assert collect(restarted, restarted.read(rid)) == original
    assert restarted.read(rid)["result"]["extra"]["captured_count"] == 30000
    assert page["result"]["total"] is None
    count = len(list(store.root.iterdir()))
    assert store.project("call_1", raw) == page
    assert len(list(store.root.iterdir())) == count


def test_small_result_is_unchanged(store):
    raw = {"success": 1, "result": [{"name": "one"}]}
    assert store.project("call", raw) is raw
    assert not store.root.exists()


def test_external_result_can_use_extra_as_an_ordinary_scalar_field(store):
    raw = {"extra": "description", "content": "x" * 30000}
    page = store.project("external", raw)
    rid = page["result"]["extra"]["result_id"]
    assert store.read(rid, "fields", limit=1)["result"]["items"] == [{"key": "extra", "value": "description"}]


def test_external_result_id_is_not_mistaken_for_an_archive_reference(store):
    raw = {"success": 1, "result": {"items": [{"value": 1}], "extra": {"result_id": "upstream-query-id"}}}
    page = store.project("external-id", raw, force=True)
    assert page["result"]["items"] == [{"value": 1}]
    assert page["result"]["extra"]["result_id"] != "upstream-query-id"


def test_forced_page_reduction_keeps_the_current_cursor(store):
    first = store.project("tables", [{"name": f"table_{i}"} for i in range(5000)])
    extra = first["result"]["extra"]
    second = store.read(extra["result_id"], cursor=extra["next_cursor"])
    reduced = store.project("read_page", json_text(second), force=True)
    assert reduced["result"]["items"] == second["result"]["items"][:1]
    assert reduced["result"]["extra"]["cursor"] == extra["next_cursor"]


@pytest.mark.parametrize("serialize", [json_text, str])
def test_serialized_preview_keeps_the_original_archive_and_remaining_records(store, serialize):
    original = [{"qualified_name": f"warehouse.schema.table_{i:05d}"} for i in range(5000)]
    first = store.project("catalog", {"success": 1, "result": original})
    extra = first["result"]["extra"]
    second = store.read(extra["result_id"], cursor=extra["next_cursor"], limit=200, max_tokens=4096, max_bytes=16384)
    wire = serialize(second)
    assert not store._fits(wire, 4096, 16384)
    archives = set(store.root.glob("*.raw"))
    for attempt in range(3):
        projected = store.project(f"sdk_view_{attempt}", wire)
        assert projected["result"]["extra"]["result_id"] == extra["result_id"]
        assert projected["result"]["extra"]["cursor"] == extra["next_cursor"]
        assert projected["result"]["extra"]["captured_count"] == len(original)
        assert projected["result"]["has_more"]
        wire = serialize(projected)
    assert set(store.root.glob("*.raw")) == archives
    assert collect(store, projected) == original[len(first["result"]["items"]) :]


def test_source_pagination_is_independent_of_archive_pagination(store):
    page = store.project(
        "partial",
        {"success": 1, "result": {"items": [{"name": "one"}], "has_more": True, "total": 100000}},
        force=True,
    )
    assert page["result"]["has_more"] is False
    assert page["result"]["total"] == 100000
    assert page["result"]["extra"]["source_has_more"] is True


def test_grep_pages_keep_source_truncation_and_recover_long_lines(store):
    match = {"file": "etl.sql", "line": 7, "content": "SELECT " + "column, " * 1000}
    page = store.project("grep", {"success": 1, "result": {"matches": [match], "truncated": True}}, force=True)
    assert page["result"]["extra"]["section"] == "matches"
    assert page["result"]["extra"]["source_truncated"] is True
    row = page["result"]["items"][0]
    assert len(row["preview"]) <= 512
    fragments = collect(store, store.read(**row["record_ref"]), row["record_ref"]["section"])
    assert json.loads("".join(part["text"] for part in fragments)) == match


def test_describe_columns_are_indexed_separately_from_table_metadata(store):
    columns = [{"name": f"column_{i}", "type": "text"} for i in range(3000)]
    page = store.project("describe", {"success": 1, "result": {"columns": columns, "table": {"name": "t"}}})
    rid = page["result"]["extra"]["result_id"]
    assert collect(store, store.read(rid, "columns"), "columns") == columns
    assert store.read(rid, "fields")["result"]["items"] == [{"key": "table", "value": {"name": "t"}}]


def test_missing_archive_stops_only_results_that_need_preserving():
    small = [{"type": "function_call_output", "call_id": "c", "output": "small"}]
    assert project_tool_results(small, None) is small
    large = [{**small[0], "output": "x" * 20000}]
    with pytest.raises(ContextCapacityError, match="RESULT_ARCHIVE_FAILED"):
        project_tool_results(large, None)
    assert large[0]["output"] == "x" * 20000


def test_sdk_text_blocks_are_paged_and_images_stay_intact(store):
    image = {"type": "input_image", "image_url": "data:image/png;base64,abc"}
    items = [
        {
            "type": "function_call_output",
            "call_id": "c",
            "output": [
                {"type": "input_text", "text": "x" * 30000},
                image,
            ],
        }
    ]
    store.register("c", items[0]["output"])
    projected = project_tool_results(items, store)
    assert projected[0]["output"][1] is image
    assert len(projected[0]["output"][0]["text"]) < 16384
    assert items[0]["output"][0]["text"] == "x" * 30000
    assert "c" not in store._registered


def test_lineage_sections_and_source_completeness(store):
    statements = [{"target": f"target_{i}", "sources": [f"source_{i}"], "evidence": ["f1:1"]} for i in range(7000)]
    graph = {
        "success": 1,
        "result": {
            "complete": False,
            "lineage": statements,
            "nodes": {f"table_{i}": {"role": "leaf"} for i in range(7000)},
            "files": {"f1": "etl.sql"},
            "components": [],
        },
    }
    page = store.project("lineage", graph)
    extra = page["result"]["extra"]
    rid = extra["result_id"]
    assert extra["source_complete"] is False
    assert collect(store, store.read(rid, "lineage"), "lineage") == statements
    assert len(collect(store, store.read(rid, "nodes"), "nodes")) == 7000
    assert store.read(rid, "files")["result"]["items"] == [{"key": "f1", "value": "etl.sql"}]


def test_oversized_record_has_a_recoverable_fragment(store):
    sql = "SELECT '" + "汉字" * 20000 + "'"
    original = {"sql": sql}
    page = store.project("grep", {"success": 1, "result": [original]})
    row = page["result"]["items"][0]
    assert row["preview_complete"] is False
    ref = row["record_ref"]
    fragments = collect(store, store.read(**ref), ref["section"])
    assert json.loads("".join(part["text"] for part in fragments)) == original


def test_cursor_and_result_are_session_scoped(store, tmp_path):
    page = store.project("first", [{"value": str(i)} for i in range(5000)])
    extra = page["result"]["extra"]
    other = ToolResultStore(ToolArchive("other", "other", base_dir=tmp_path / "other"))
    with pytest.raises(ValueError, match="expired"):
        other.read(extra["result_id"])
    with pytest.raises(ValueError, match="Invalid"):
        store.read(extra["result_id"], cursor=store._cursor("0" * 64, "items", 1))
    with pytest.raises(ValueError):
        store.read("../../etc/passwd")
    (store.root / f"{extra['result_id']}.json").unlink()
    with pytest.raises(ValueError, match="expired"):
        store.read(extra["result_id"])


def test_archive_failure_preserves_original_without_publishing_reference(store):
    raw = {"success": 1, "result": "x" * 30000}
    before = copy.deepcopy(raw)
    with patch("datus.utils.tool_result_store.atomic_write_text", side_effect=OSError("disk full")):
        with pytest.raises(ContextCapacityError, match="RESULT_ARCHIVE_FAILED"):
            store.project("call", raw)
    assert raw == before
    assert not list(store.root.glob("*.json"))


def test_large_error_preserves_status_and_complete_raw_content(store):
    raw = {"success": 0, "error": "数据库错误" * 5000, "result": None}
    page = store.project("failed", raw)
    assert page["success"] == 0
    assert len(page["error"].encode("utf-8")) <= 256
    assert len(page["result"]["items"][0]["text"]) <= 1000
    rid = page["result"]["extra"]["result_id"]
    fragments = collect(store, store.read(rid, "raw"), "raw")
    assert json.loads("".join(part["text"] for part in fragments)) == raw


@pytest.mark.parametrize("native", [False, True])
def test_small_block_results_preserve_identity(store, native):
    blocks = [{"type": "text", "text": "small result"}]
    items = (
        [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": blocks}]}]
        if native
        else [{"type": "function_call_output", "call_id": "c", "output": blocks}]
    )
    assert project_tool_results(items, store) is items
    assert project_tool_results(items, None) is items


@pytest.mark.parametrize("native", [False, True])
def test_projection_does_not_mutate_action_or_call_arguments(store, native):
    raw = str({"success": 1, "result": [{"name": f"table_{i}"} for i in range(5000)]})
    items = [
        {"type": "function_call", "call_id": "c", "name": "list_tables", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c", "output": raw},
    ]
    if native:
        items = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "list_tables", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": raw}]},
        ]
    before = copy.deepcopy(items)
    projected = project_tool_results(items, store)
    assert items == before
    assert projected[0] is items[0]
    assert projected[1] is not items[1]
    assert len(json_text(projected)) < len(json_text(items))


@pytest.mark.asyncio
async def test_function_tool_keeps_public_result_and_registers_it(store):
    from agents.tool_context import ToolContext
    from agents.usage import Usage

    def list_tables():
        return FuncToolResult(result=[{"name": "t"}])

    context = ToolContext(
        context=None, usage=Usage(), tool_name="list_tables", tool_call_id="call", tool_arguments="{}"
    )
    token = ACTIVE_RESULT_STORE.set(store)
    try:
        raw = await trans_to_function_tool(list_tables).on_invoke_tool(context, "{}")
    finally:
        ACTIVE_RESULT_STORE.reset(token)
    assert raw == {"success": 1, "error": None, "result": [{"name": "t"}]}
    assert store._registered["call"] is raw


@pytest.mark.parametrize("native", [False, True])
def test_exact_references_survive_repeated_lossy_summaries_and_resume(store, native):
    from datus.agent.node.compact_prompts import build_continuation_message

    original = [{"name": f"field_{i}", "comment": "a documented attribute"} for i in range(4000)]
    first = store.project("columns", {"success": 1, "result": {"columns": original}})
    initial = first["result"]["extra"]
    page = store.read(initial["result_id"], "columns", cursor=initial["next_cursor"])
    extra = page["result"]["extra"]
    if native:
        output = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": json_text(page)}]}
    else:
        output = {"type": "function_call_output", "call_id": "c", "output": json_text(page)}
    expected = {key: extra[key] for key in ("result_id", "section", "cursor", "next_cursor")}
    refs = store.recovery_references([output])
    assert refs == [expected]
    for _ in range(3):
        summary = build_continuation_message(
            "Read remaining columns from " + extra["result_id"][:8], "history.jsonl", refs
        )
        items = [{"role": "assistant", "content": [{"type": "output_text", "text": summary}]}]
        refs = store.recovery_references(items)
        assert refs == [expected]
    resumed = store.read(refs[0]["result_id"], refs[0]["section"], refs[0]["next_cursor"])
    start = len(first["result"]["items"]) + len(page["result"]["items"])
    assert resumed["result"]["items"] == original[start : start + len(resumed["result"]["items"])]


def test_recovery_prefers_latest_cursor_is_bounded_and_keeps_record_fragments(store):
    pages = [store.project(f"c{i}", [{"name": f"{i}.{j}"} for j in range(4000)]) for i in range(12)]
    items = [{"type": "function_call_output", "output": json_text(p)} for p in pages]
    refs = store.recovery_references(items)
    assert len(refs) == 8 and refs[0]["result_id"] == pages[-1]["result"]["extra"]["result_id"]
    extra = pages[-1]["result"]["extra"]
    latest = store.read(extra["result_id"], cursor=extra["next_cursor"])
    refs = store.recovery_references([*items, {"type": "function_call_output", "output": latest}])
    assert refs[0]["cursor"] == extra["next_cursor"]
    assert store.counter(store.recovery_references(items, max_tokens=500)) <= 500
    long_row = store.project("long", [{"sql": "SELECT x " * 10000}])
    refs = store.recovery_references([{"type": "function_call_output", "output": long_row}])
    record_ref = long_row["result"]["items"][0]["record_ref"]
    assert record_ref in refs
    assert store.recovery_references([{"role": "user", "content": json_text(long_row)}]) == []


def test_larger_window_pages_reduce_round_trips_within_budgets(tmp_path):
    archive = ToolArchive("project", "session", base_dir=tmp_path)
    small = ToolResultStore(archive, page_tokens=2048)
    large = ToolResultStore(archive, page_tokens=8192)
    original = [{"name": f"field_{i}", "comment": "a documented attribute"} for i in range(5000)]
    preview = large.project("catalog", original)
    rid = preview["result"]["extra"]["result_id"]
    assert large.counter(preview) <= 16384
    assert len(large.read(rid)["result"]["items"]) > len(small.read(rid)["result"]["items"])
    rows, cursor = [], ""
    while True:
        page = large.read(rid, cursor=cursor)
        assert large.counter(page) <= 8192 and len(json_text(page).encode()) <= 32768
        rows.extend(page["result"]["items"])
        if not page["result"]["has_more"]:
            break
        cursor = page["result"]["extra"]["next_cursor"]
    assert rows == original
