"""Compatibility recovery only; new outputs are owned by their source tools."""

import copy
import hashlib
import json

import pytest

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.registry.contracts import tool_contract
from datus.utils.tool_archive import ToolArchive
from datus.utils.tool_result_store import ACTIVE_RESULT_STORE, ToolResultStore, json_text
from datus.utils.tool_result_view import restore_legacy_tool_results


@pytest.fixture
def store(tmp_path):
    return ToolResultStore(ToolArchive("project", "session", base_dir=tmp_path))


def legacy_result(store, raw):
    wire = json_text(raw)
    rid = hashlib.sha256(wire.encode()).hexdigest()
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / f"{rid}.raw").write_text(wire)
    (store.root / f"{rid}.json").write_text(json_text({"sha256": hashlib.sha256(wire.encode()).hexdigest()}))
    return {
        "success": 1,
        "result": {
            "items": [],
            "extra": {
                "read_with": "read_tool_result",
                "result_id": rid,
                "section": "columns",
                "cursor": "old",
                "next_cursor": "next",
            },
        },
    }


@pytest.mark.parametrize("native", [False, True])
def test_legacy_recovery_preserves_history_and_media(store, native):
    raw = {"success": 1, "result": {"columns": [{"name": "x", "sql": "SELECT " * 10000}]}}
    page = legacy_result(store, raw)
    image = {"type": "image", "source": {"data": "image"}}
    if native:
        items = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "c",
                        "content": [{"type": "text", "text": json_text(page)}, image],
                    }
                ],
            }
        ]
    else:
        items = [
            {
                "type": "function_call_output",
                "call_id": "c",
                "output": [{"type": "text", "text": json_text(page)}, image],
            }
        ]
    before = copy.deepcopy(items)
    restored = restore_legacy_tool_results(items, store)
    assert items == before
    assert "archive_path" in json_text(restored)
    assert "read_tool_result" not in json_text(restored)
    content = restored[0]["content"][0]["content"] if native else restored[0]["output"]
    assert content[1] is image


def test_current_large_output_is_never_generically_projected(store):
    items = [{"type": "function_call_output", "output": {"result": ["x" * 200000]}}]
    assert restore_legacy_tool_results(items, store) is items
    assert restore_legacy_tool_results(items, None) is items
    assert not store.root.exists()


@pytest.mark.parametrize(
    "references",
    ["{bad json", '{"result_id":"old"}', '[{"tool":"query_metrics","result_id":"native:123","offset":200}]'],
)
def test_summary_without_legacy_references_is_unchanged(store, references):
    """Native and unparseable summaries retain their original text and identity."""
    text = "Summary\n<tool_result_references>" + references + "</tool_result_references>"
    items = [{"role": "assistant", "content": text}]
    assert restore_legacy_tool_results(items, store) is items


def test_expired_saved_summary_reference_requests_original_tool_retry(store):
    """An expired old reference reports recovery failure during ordinary resume."""
    items = [
        {
            "role": "assistant",
            "content": '<tool_result_references>[{"result_id":"../outside"}]</tool_result_references>',
        }
    ]
    restored = restore_legacy_tool_results(items, store)
    text = restored[0]["content"]
    refs = json.loads(text.split("<tool_result_references>")[1].split("</tool_result_references>")[0])
    assert "rerun the original tool" in refs[0]["error"]
    assert not store.root.exists()


def test_unavailable_or_corrupt_legacy_result_is_reported(store):
    page = legacy_result(store, {"result": [1, 2]})
    rid = page["result"]["extra"]["result_id"]
    (store.root / f"{rid}.raw").write_text("changed")
    assert store.recover_legacy(page)["success"] == 0
    page["result"]["extra"]["result_id"] = "../outside"
    assert store.recover_legacy(page)["success"] == 0
    page["result"]["extra"]["result_id"] = "0" * 64
    assert "rerun the original tool" in store.recover_legacy(page)["error"]


@pytest.mark.asyncio
async def test_function_tool_does_not_register_raw_output(store):
    from agents.tool_context import ToolContext
    from agents.usage import Usage

    @tool_contract(access="read", direct_call=True, category="db_tools")
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
    assert not store.root.exists()


@pytest.mark.parametrize("native", [False, True])
def test_legacy_references_survive_summaries_as_readable_files(store, native):
    from pathlib import Path

    from datus.agent.node.compact_prompts import build_continuation_message

    original = [{"name": f"field_{i}", "comment": "a documented attribute"} for i in range(4000)]
    first = legacy_result(store, {"success": 1, "result": {"columns": original}})
    extra = first["result"]["extra"]
    if native:
        output = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": json_text(first)}]}
    else:
        output = {"type": "function_call_output", "call_id": "c", "output": json_text(first)}
    refs = store.recovery_references([output])
    assert len(refs) == 1 and refs[0]["read_with"] == "read_file"
    expected = refs[:]
    for _ in range(3):
        summary = build_continuation_message("Read remaining columns", "history.jsonl", refs)
        refs = store.recovery_references([{"role": "assistant", "content": summary}])
        assert refs == expected
    fragments = [json.loads(line)["text"] for line in Path(refs[0]["archive_path"]).read_text().splitlines()]
    assert json.loads("".join(fragments))["result"]["columns"] == original
    # A pre-upgrade summary carries only the opaque identifier.
    old_ref = {key: extra[key] for key in ("result_id", "section", "cursor", "next_cursor")}
    summary = build_continuation_message("Continue", "history.jsonl", [old_ref])
    assert store.recovery_references([{"role": "assistant", "content": summary}]) == expected


def test_recovery_prefers_latest_native_offset_and_preserves_query_ids(store):
    items = []
    for i in range(12):
        items.extend(
            [
                {"type": "function_call", "call_id": str(i), "name": "list_tables"},
                {
                    "type": "function_call_output",
                    "call_id": str(i),
                    "output": {
                        "result": {"has_more": True, "extra": {"snapshot_id": f"snapshot:{i}", "next_offset": 100}}
                    },
                },
            ]
        )
    refs = store.recovery_references(items)
    assert len(refs) == 8 and refs[0] == {"tool": "list_tables", "snapshot_id": "snapshot:11", "offset": 100}
    items.append(
        {
            "type": "function_call_output",
            "call_id": "11",
            "output": {"result": {"has_more": True, "extra": {"snapshot_id": "snapshot:11", "next_offset": 200}}},
        }
    )
    assert store.recovery_references(items)[0]["offset"] == 200
    assert store.counter(store.recovery_references(items, max_tokens=500)) <= 500
    metric = (
        "result_id: query_metrics:abc\nrows: 250\npage: "
        + json_text({"has_more": True, "offset": 0, "next_offset": 100})
        + "\n\nm\n1\n"
    )
    refs = store.recovery_references(
        [
            {"type": "function_call", "call_id": "q", "name": "query_metrics"},
            {"type": "function_call_output", "call_id": "q", "output": {"success": 1, "result": metric}},
        ]
    )
    assert refs == [{"tool": "query_metrics", "result_id": "query_metrics:abc", "offset": 100}]
    assert store.recovery_references([{"role": "user", "content": metric}]) == []


def test_missing_legacy_reference_does_not_break_compaction(store):
    from datus.agent.node.compact_prompts import build_continuation_message

    summary = build_continuation_message("Continue", "history.jsonl", [{"result_id": "0" * 64}])
    assert store.recovery_references([{"role": "assistant", "content": summary}]) == []


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("extra", ["ordinary description", ["ordinary value"], 7, None])
def test_non_mapping_extra_preserves_other_result_references(store, native, extra):
    snapshot_id = "snapshot:" + "1" * 32
    outputs = [
        (
            "catalog",
            "list_tables",
            {"result": {"has_more": True, "extra": {"snapshot_id": snapshot_id, "next_offset": 100}}},
        ),
        ("external", "external_tool", {"result": {"extra": extra, "items": []}}),
    ]
    items = []
    for call_id, name, output in outputs:
        if native:
            items.extend(
                [
                    {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": name, "input": {}}]},
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": call_id, "content": json_text(output)}],
                    },
                ]
            )
        else:
            items.extend(
                [
                    {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"},
                    {"type": "function_call_output", "call_id": call_id, "output": json_text(output)},
                ]
            )
    before = copy.deepcopy(items)
    assert store.recovery_references(items) == [{"tool": "list_tables", "snapshot_id": snapshot_id, "offset": 100}]
    assert items == before
