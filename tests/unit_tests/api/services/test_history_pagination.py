"""Real SQLite contracts for bounded history and stateless large-result references."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from datus.api.services.chat_service import ChatService
from datus.api.services.history_pagination import (
    HistoryChangedError,
    get_history_content,
    get_history_page,
    subagent_run_offsets,
)
from datus.models.session_manager import SessionManager
from datus.utils.tool_archive import ToolArchive


@pytest.fixture
def history(tmp_path):
    service = ChatService(SimpleNamespace(session_dir=str(tmp_path)))

    def write(session_id, messages, user="alice", parent=None):
        manager = SessionManager(session_dir=str(tmp_path), scope=user)
        root = manager.session_dir
        if parent:
            root = str(Path(root) / parent)
            Path(root).mkdir(parents=True, exist_ok=True)
        path = Path(root) / f"{session_id}.db"
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_messages "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, message_data TEXT, created_at TEXT)"
            )
            for message in messages:
                conn.execute(
                    "INSERT INTO agent_messages(session_id,message_data,created_at) VALUES(?,?,?)",
                    (session_id, json.dumps(message), "2026-10-30 01:00:00"),
                )
        return path

    return service, write


def turn(number):
    return [
        {"role": "user", "content": f"question {number}"},
        {"role": "assistant", "content": [{"type": "output_text", "text": f"answer {number}"}]},
    ]


def questions(page):
    return [message.content[0].payload["content"] for message in page.messages if message.role == "user"]


def tool(call_id, name, result, *, native=False):
    output = json.dumps({"success": 1, "result": result})
    if native:
        return [
            {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": name, "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": output}]},
        ]
    return [
        {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"},
        {"type": "function_call_output", "call_id": call_id, "output": output},
    ]


def offload_result(result, archive, index, *, native=False):
    if native:
        block = result["content"][0]
        block["content"] = archive.archive(block["content"], index, "output")
    else:
        result["output"] = archive.archive(result["output"], index, "output")


def child_tool_calls(data):
    return [
        part.payload["callToolId"]
        for message in data.messages
        if message.depth == 1
        for part in message.content
        if part.type == "call-tool"
    ]


def test_recent_ten_turns_page_back_without_gaps_or_duplicates_while_new_turns_arrive(history):
    service, write = history
    write("chat_page", [message for n in range(25) for message in turn(n)])
    first = get_history_page(service, "chat_page", "alice", None, None)
    assert questions(first) == [f"question {n}" for n in range(15, 25)]
    assert first.has_more and first.next_cursor
    original_ids = [m.message_id for m in first.messages]
    write("chat_page", turn(25))
    repeated = get_history_page(service, "chat_page", "alice", 10, None)
    assert [m.message_id for m in repeated.messages if m.role == "user"][:9] == [
        m.message_id for m in first.messages if m.role == "user"
    ][1:]
    second = get_history_page(service, "chat_page", "alice", 10, first.next_cursor)
    third = get_history_page(service, "chat_page", "alice", 10, second.next_cursor)
    assert questions(second) == [f"question {n}" for n in range(5, 15)]
    assert questions(third) == [f"question {n}" for n in range(5)]
    assert not third.has_more and third.next_cursor is None
    assert not set(original_ids) & {m.message_id for m in second.messages + third.messages}


@pytest.mark.parametrize("failed", [False, True])
def test_large_outputs_are_replaced_and_downloaded_without_mutating_storage(history, failed):
    service, write = history
    sql = "INSERT INTO t VALUES (1);\n" * 170000
    output = {"success": 0, "error": sql, "result": sql} if failed else {"success": 1, "result": sql}
    raw = [
        {"role": "user", "content": "transfer"},
        {"type": "function_call", "call_id": "transfer", "name": "transfer_query_result", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "transfer", "output": json.dumps(output)},
    ]
    path = write("chat_large", raw)
    before = path.read_bytes()
    page = get_history_page(service, "chat_large", "alice", None, None)
    encoded = page.model_dump_json()
    assert len(encoded) < 40000
    parts = [part for m in page.messages for part in m.content]
    result = next(p.payload["result"] for p in parts if p.type == "call-tool-result")
    ref = result["historyContent"]["ref"]
    # The existing converter strips surrounding whitespace from error messages.
    expected = sql.strip() if failed else sql
    assert result["historyContent"]["size_bytes"] == len(expected.encode())
    assert get_history_content(service, "chat_large", "alice", ref) == expected
    assert path.read_bytes() == before
    with pytest.raises(FileNotFoundError):
        get_history_content(service, "chat_large", "bob", ref)
    with pytest.raises(ValueError):
        get_history_content(service, "other_session", "alice", ref)


def test_complete_turn_preserves_native_tool_results_and_ignores_synthetic_resume_turns(history):
    service, write = history
    native = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "native", "name": "execute_sql", "input": {}}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "native", "content": '{"success":1,"result":"ok"}'}],
        },
        {"role": "user", "content": "[DATUS_COMPACT_RESUME] Continue from the summary."},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]
    write("chat_native", native)
    page = get_history_page(service, "chat_native", "alice", 1, None)
    assert questions(page) == ["question"]
    assert not page.has_more
    assert any(part.type == "call-tool-result" for message in page.messages for part in message.content)


def test_cursor_is_scoped_and_rejects_rewritten_anchor(history):
    service, write = history
    path = write("chat_cursor", [message for n in range(12) for message in turn(n)])
    first = get_history_page(service, "chat_cursor", "alice", 10, None)
    write("other", turn(1))
    with pytest.raises(ValueError):
        get_history_page(service, "other", "alice", 10, first.next_cursor)
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM agent_messages")
    with pytest.raises(HistoryChangedError):
        get_history_page(service, "chat_cursor", "alice", 10, first.next_cursor)
    with pytest.raises(ValueError):
        get_history_page(service, "chat_cursor", "alice", 10, "not-a-cursor")


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("storage", ["inline", "archived", "relative", "missing", "reused_call_id"])
def test_subagent_reused_on_an_older_page_does_not_replay_the_wrong_run(
    history, tmp_path, monkeypatch, native, storage
):
    service, write = history
    parent = "chat_parent"
    child = "gen_job_session_child"
    data_dir = tmp_path / "alice" / parent / "data"
    if storage == "relative":
        monkeypatch.chdir(tmp_path.parent)
        service = ChatService(SimpleNamespace(session_dir=tmp_path.name))
        data_dir = data_dir.relative_to(tmp_path.parent)
    write(
        child,
        [
            *turn(0),
            *tool("child_old", "execute_sql", "old run"),
            *turn(1),
            *tool("child_new", "execute_sql", "new run"),
        ],
        parent=parent,
    )

    def parent_task(call_id, index):
        call, result = tool(call_id, "task", {"session_id": child})
        if storage in ("archived", "relative", "missing"):
            archive = ToolArchive("", parent, base_dir=data_dir, preview_chars=12)
            result["output"] = archive.archive(result["output"], index, "output")
            if index == 0 and storage == "missing":
                next(data_dir.glob("000000_output_*.txt")).unlink()
        if not native:
            return [call, result]
        return [
            {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": "task", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": [{"type": "text", "text": result["output"]}],
                    }
                ],
            },
        ]

    second_call = "task_old" if storage == "reused_call_id" else "task_new"
    write(parent, [*turn(0), *parent_task("task_old", 0), *turn(1), *parent_task(second_call, 1)])

    def child_calls(messages):
        return [p.payload["callToolId"] for m in messages if m.depth == 1 for p in m.content if p.type == "call-tool"]

    page = get_history_page(service, parent, "alice", 1, None)
    assert child_calls(page.messages) == ([] if storage == "missing" else ["child_new"])
    assert questions(page) == ["question 1"]
    assert any(
        p.payload["callToolId"] == second_call for m in page.messages for p in m.content if p.type == "call-tool"
    )

    older = get_history_page(service, parent, "alice", 1, page.next_cursor)
    assert child_calls(older.messages) == ([] if storage == "missing" else ["child_old"])
    expected = [] if storage == "missing" else ["child_old", "child_new"]
    assert child_calls(get_history_page(service, parent, "alice", 2, None).messages) == expected
    assert child_calls(service.get_history(parent, "alice").data.messages) == expected


@pytest.mark.parametrize("invalid", ["missing", "outside", "symlink", "changed"])
def test_unavailable_archived_task_result_leaves_parent_tool_card(history, tmp_path, invalid):
    service, write = history
    parent = "chat_parent"
    child = "gen_job_session_child"
    write(child, turn(0) + tool("child_step", "execute_sql", "child output"), parent=parent)
    call, result = tool("task_parent", "task", {"session_id": child})
    data_dir = tmp_path / "alice" / parent / "data"
    if invalid == "outside":
        data_dir = tmp_path / "bob" / parent / "data"
    archive = ToolArchive("", parent, base_dir=data_dir)
    result["output"] = archive.archive(result["output"], 1, "output")
    path = next(data_dir.iterdir())
    if invalid == "changed":
        path.write_text(path.read_text() + " ")
    elif invalid in ("missing", "symlink"):
        if invalid == "symlink":
            external = tmp_path / "bob" / path.name
            external.parent.mkdir()
            path.rename(external)
            path.symlink_to(external)
        else:
            path.unlink()
    write(parent, [*turn(0), call, result])

    page = get_history_page(service, parent, "alice", 1, None)
    assert [p.payload["callToolId"] for m in page.messages for p in m.content if p.type == "call-tool"] == [
        "task_parent"
    ]


def test_large_parameters_and_nested_results_have_on_demand_references(history):
    service, write = history
    content = "large file " * 3000
    raw = [
        *turn(0),
        {
            "type": "function_call",
            "call_id": "write",
            "name": "write_file",
            "arguments": json.dumps({"path": "a.txt", "content": content}),
        },
        {
            "type": "function_call_output",
            "call_id": "write",
            "output": json.dumps({"success": 1, "result": {"nested": {"output": content}}}),
        },
    ]
    write("chat_params", raw)
    page = get_history_page(service, "chat_params", "alice", 10, None)
    parts = [p for m in page.messages for p in m.content]
    call = next(p.payload for p in parts if p.type == "call-tool")
    assert call["historyOnly"] is True
    assert (
        json.loads(get_history_content(service, "chat_params", "alice", call["toolParamsHistoryContent"]["ref"]))[
            "content"
        ]
        == content
    )
    result = next(p.payload["result"] for p in parts if p.type == "call-tool-result")
    assert json.loads(get_history_content(service, "chat_params", "alice", result["historyContent"]["ref"])) == {
        "nested": {"output": content}
    }


def test_http_history_page_and_content_are_scoped_validated_and_uncached(history):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from datus.api.deps import get_app_context, get_datus_service
    from datus.api.routes.chat_routes import router

    service, write = history
    write("chat_http", [*turn(0), *tool("http_call", "execute_sql", "x" * 10000)])
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_datus_service] = lambda: SimpleNamespace(chat=service)
    app.dependency_overrides[get_app_context] = lambda: SimpleNamespace(user_id="alice")
    with TestClient(app) as client:
        response = client.get("/api/v1/chat/history/page", params={"session_id": "chat_http"})
        assert response.status_code == 200
        page = response.json()["data"]
        result = next(
            p["payload"]["result"] for m in page["messages"] for p in m["content"] if p["type"] == "call-tool-result"
        )
        params = {"session_id": "chat_http", "reference": result["historyContent"]["ref"]}
        download = client.get("/api/v1/chat/history/content", params=params)
        assert download.status_code == 200
        assert download.text == "x" * 10000
        assert download.headers["cache-control"] == "private, no-store"
        assert download.headers["content-disposition"].startswith("attachment;")
        assert (
            client.get("/api/v1/chat/history/page", params={"session_id": "chat_http", "limit": 0}).status_code == 422
        )
        assert (
            client.get("/api/v1/chat/history/page", params={"session_id": "chat_http", "limit": 101}).status_code == 422
        )
        custom = client.get(
            "/api/v1/chat/history/page", params={"session_id": "chat_http", "limit": 1, "inline_chars": 64}
        )
        assert custom.status_code == 200
        preview = next(
            p["payload"]["result"]["historyContent"]["preview"]
            for m in custom.json()["data"]["messages"]
            for p in m["content"]
            if p["type"] == "call-tool-result"
        )
        assert preview == "x" * 64
        for size in (0, 16001):
            assert (
                client.get(
                    "/api/v1/chat/history/page", params={"session_id": "chat_http", "inline_chars": size}
                ).status_code
                == 422
            )
        assert client.get("/api/v1/chat/history/content", params={**params, "session_id": "other"}).status_code == 400
        app.dependency_overrides[get_app_context] = lambda: SimpleNamespace(user_id="bob")
        assert client.get("/api/v1/chat/history/content", params=params).status_code == 404


def test_off_page_tool_bodies_are_not_deserialized(history, monkeypatch):
    import datus.models.session_manager as storage

    service, write = history
    write(
        "chat_bounded",
        [
            *turn(0),
            *tool("old", "execute_sql", "POISON_OFF_PAGE" * 10000),
            *[message for n in range(1, 21) for message in turn(n)],
        ],
    )
    real_loads = storage.json.loads

    def guarded_loads(raw, *args, **kwargs):
        if isinstance(raw, str) and "POISON_OFF_PAGE" in raw:
            pytest.fail("An old tool body outside the requested turns was deserialized")
        return real_loads(raw, *args, **kwargs)

    monkeypatch.setattr(storage.json, "loads", guarded_loads)
    page = get_history_page(service, "chat_bounded", "alice", 10, None)
    assert questions(page) == [f"question {n}" for n in range(11, 21)]


def test_target_turn_loads_one_window_with_global_indices(history):
    service, write = history
    write("chat_jump", [message for n in range(35) for message in turn(n)])
    latest = get_history_page(service, "chat_jump", "alice", 10, None)
    assert latest.total_turns == 35
    assert [anchor.index for anchor in latest.turns] == list(range(26, 36))
    target = get_history_page(service, "chat_jump", "alice", 10, None, turn=7)
    assert questions(target) == [f"question {n}" for n in range(10)]
    assert [anchor.index for anchor in target.turns] == list(range(1, 11))
    assert target.total_turns == 35
    assert not target.has_more
    middle = get_history_page(service, "chat_jump", "alice", 10, None, turn=12)
    assert [anchor.index for anchor in middle.turns] == list(range(11, 21))
    with pytest.raises(ValueError):
        get_history_page(service, "chat_jump", "alice", 10, latest.next_cursor, turn=3)


def test_shared_provider_response_ids_still_produce_distinct_stable_history_ids(history):
    service, write = history
    raw = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "provider_data": {"response_id": "shared"},
            "content": [{"type": "text", "text": "first"}],
        },
        {
            "role": "assistant",
            "provider_data": {"response_id": "shared"},
            "content": [{"type": "text", "text": "second"}],
        },
    ]
    write("chat_ids", raw)
    first = get_history_page(service, "chat_ids", "alice", None, None)
    second = get_history_page(service, "chat_ids", "alice", None, None)
    ids = [message.message_id for message in first.messages]
    assert len(ids) == len(set(ids))
    assert ids == [message.message_id for message in second.messages]


def test_subagent_offsets_count_each_output_once_and_only_use_prior_calls_in_the_same_session(history):
    _service, write = history
    task_call = {"type": "function_call", "call_id": "prior", "name": "task", "arguments": "{}"}
    path = write(
        "chat_offsets",
        [
            task_call,
            task_call,
            tool("prior", "task", {"session_id": "gen_job_session_child"})[1],
            *tool("sql", "execute_sql", {"session_id": "gen_job_session_ignored"}),
            tool("future", "task", {"session_id": "gen_job_session_future"})[1],
            tool("foreign", "task", {"session_id": "gen_job_session_foreign"})[1],
        ],
    )
    with sqlite3.connect(path) as conn:
        for row in tool("foreign", "task", {"session_id": "gen_job_session_foreign"}):
            conn.execute(
                "INSERT INTO agent_messages(session_id,message_data,created_at) VALUES(?,?,?)",
                ("other_session", json.dumps(row), "2026-10-30 01:00:00"),
            )
        start_id = conn.execute("SELECT MAX(id)+1 FROM agent_messages").fetchone()[0]
    write(
        "chat_offsets",
        [
            {**task_call, "call_id": "future"},
            tool("prior", "task", {"session_id": "gen_job_session_child"})[1],
        ],
    )
    manager = SessionManager(session_dir=_service._session_dir, scope="alice")
    assert subagent_run_offsets(manager, "chat_offsets", start_id) == {"gen_job_session_child": 1}


def test_client_requested_page_size_can_exceed_fifty_turns(history):
    service, write = history
    write("chat_custom_size", [message for number in range(61) for message in turn(number)])
    page = get_history_page(service, "chat_custom_size", "alice", 60, None)
    assert questions(page) == [f"question {number}" for number in range(1, 61)]
    assert page.has_more


@pytest.mark.parametrize("body_kind", ["text", "nested"])
def test_failed_output_with_short_error_and_large_body_is_bounded_and_downloadable(history, body_kind):
    service, write = history
    text = "FAILED_BODY_" * 370000
    body = text if body_kind == "text" else {"nested": {"rows": [{"output": text}]}}
    path = write(
        "chat_failed_body",
        [
            *turn(0),
            {"type": "function_call", "call_id": "failed", "name": "execute_sql", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "failed",
                "output": json.dumps({"success": 0, "error": "query failed", "result": body}),
            },
        ],
    )
    before = path.read_bytes()
    page = get_history_page(service, "chat_failed_body", "alice", 10, None)
    assert len(page.model_dump_json().encode()) < 40000
    result = next(p.payload["result"] for m in page.messages for p in m.content if p.type == "call-tool-result")
    assert result["error"] == "query failed"
    assert json.loads(get_history_content(service, "chat_failed_body", "alice", result["historyContent"]["ref"])) == {
        "error": "query failed",
        "result": body,
    }
    assert path.read_bytes() == before


def test_current_page_does_not_read_unrelated_older_subagents(history, monkeypatch):
    service, write = history
    parent, old_child, current_child = "chat_children", "gen_job_session_old", "gen_job_session_current"
    write(old_child, [*turn(0), *tool("old_output", "execute_sql", "x" * 4000000)], parent=parent)
    write(current_child, [*turn(1), *tool("current_output", "execute_sql", "current result")], parent=parent)
    write(
        parent,
        [
            *turn(0),
            *tool("old_task", "task", {"session_id": old_child}),
            *turn(1),
            *tool("current_task", "task", {"session_id": current_child}),
        ],
    )
    reads = []
    original = SessionManager.get_session_messages

    def track_reads(manager, session_id, **bounds):
        reads.append(session_id)
        return original(manager, session_id, **bounds)

    monkeypatch.setattr(SessionManager, "get_session_messages", track_reads)
    page = get_history_page(service, parent, "alice", 1, None)
    assert old_child not in reads
    assert reads.count(current_child) == 1
    calls = [p.payload["callToolId"] for m in page.messages for p in m.content if p.type == "call-tool"]
    assert "current_output" in calls


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("runs_on_page", [1, 2])
def test_reused_subagent_reads_only_runs_referenced_by_the_selected_page(history, monkeypatch, native, runs_on_page):
    import datus.models.session_manager as storage

    service, write = history
    parent, child = "chat_run_window", "gen_job_session_reused"

    def child_tool(call_id, result):
        if not native:
            return tool(call_id, "execute_sql", result)
        return [
            {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": "execute_sql", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": json.dumps({"success": 1, "result": result}),
                    }
                ],
            },
        ]

    write(
        child,
        [
            *turn(0),
            *child_tool("old_run", "OFF_PAGE_CHILD" * 310000),
            {"role": "user", "content": "[DATUS_COMPACT_RESUME] Continue from the summary."},
            *[
                item
                for number in range(1, runs_on_page + 1)
                for item in [
                    *turn(number),
                    *child_tool(f"current_run_{number}", "current result"),
                ]
            ],
            *turn(runs_on_page + 1),
            *child_tool("later_run", "OFF_PAGE_CHILD" * 310000),
        ],
        parent=parent,
    )
    write(
        parent,
        [
            *turn(0),
            *tool("task_0", "task", {"session_id": child}),
            *turn(1),
            *[
                item
                for number in range(1, runs_on_page + 1)
                for item in tool(f"task_{number}", "task", {"session_id": child})
            ],
            *turn(2),
            *tool("task_later", "task", {"session_id": child}),
        ],
    )
    real_loads = storage.json.loads

    def guarded_loads(raw, *args, **kwargs):
        if isinstance(raw, str) and "OFF_PAGE_CHILD" in raw:
            pytest.fail("A tool body outside the selected Subagent run was deserialized")
        return real_loads(raw, *args, **kwargs)

    monkeypatch.setattr(storage.json, "loads", guarded_loads)
    page = get_history_page(service, parent, "alice", 1, None, turn=2)
    calls = [p.payload["callToolId"] for m in page.messages for p in m.content if p.type == "call-tool"]
    assert {f"current_run_{number}" for number in range(1, runs_on_page + 1)} <= set(calls)
    assert "old_run" not in calls and "later_run" not in calls
    assert questions(page) == ["question 1"]


@pytest.mark.parametrize("target", [1, 2])
def test_bounded_subagent_runs_preserve_an_implicit_initial_assistant_run(history, target):
    service, write = history
    parent, child = "chat_implicit", "gen_job_session_implicit"
    write(
        child,
        [*tool("implicit", "execute_sql", "first"), *turn(1), *tool("prompted", "execute_sql", "second")],
        parent=parent,
    )
    write(
        parent,
        [
            *turn(0),
            *tool("task_0", "task", {"session_id": child}),
            *turn(1),
            *tool("task_1", "task", {"session_id": child}),
        ],
    )
    page = get_history_page(service, parent, "alice", 1, None, turn=target)
    calls = {p.payload["callToolId"] for m in page.messages for p in m.content if p.type == "call-tool"}
    assert ("implicit" if target == 1 else "prompted") in calls
    assert ("prompted" if target == 1 else "implicit") not in calls


@pytest.mark.parametrize("has_tool_result", [False, True])
@pytest.mark.parametrize("subagent", [False, True])
def test_history_turn_boundaries_ignore_non_object_content_blocks(history, has_tool_result, subagent):
    service, write = history
    content = [
        "plain provider text",
        7,
        True,
        None,
        ["nested", "array"],
        {"type": "input_text", "text": "mixed question"},
    ]
    if has_tool_result:
        content.insert(0, {"type": "tool_result", "tool_use_id": "other", "content": "ignored"})
    child = "gen_job_session_mixed" if subagent else None
    if subagent:
        write("chat_mixed_blocks", [*turn(0), *tool("task", "task", {"session_id": child})])
        write(
            child,
            [{"role": "user", "content": content}, *tool("mixed_child_tool", "execute_sql", "done")],
            parent="chat_mixed_blocks",
        )
    else:
        write("chat_mixed_blocks", [{"role": "user", "content": content}])
    page = get_history_page(service, "chat_mixed_blocks", "alice", 1, None)
    assert page.total_turns == 1
    if subagent:
        calls = {p.payload["callToolId"] for m in page.messages for p in m.content if p.type == "call-tool"}
        assert "mixed_child_tool" in calls
    else:
        assert questions(page) == ["mixed question"]


def test_scoped_session_directory_returns_history_page_for_authenticated_user(history):
    service, write = history
    service.agent_config.session_dir_scoped = True
    write("chat_scoped", turn(1), user=None)
    page = get_history_page(service, "chat_scoped", "alice", 10, None)
    assert questions(page) == ["question 1"]
    assert page.total_turns == 1


def test_scoped_session_directory_serves_deferred_tool_content(history):
    service, write = history
    service.agent_config.session_dir_scoped = True
    content = "large result " * 4000
    write("chat_scoped_content", turn(1)[:1] + tool("lookup", "execute_sql", content), user=None)
    page = get_history_page(service, "chat_scoped_content", "alice", 10, None)
    result = next(p.payload["result"] for m in page.messages for p in m.content if p.type == "call-tool-result")
    assert get_history_content(service, "chat_scoped_content", "alice", result["historyContent"]["ref"]) == content


@pytest.mark.asyncio
async def test_paging_and_downloads_survive_multiple_context_rewrites(history, tmp_path):
    service, _ = history
    session_id = "chat_archive"
    manager = SessionManager(session_dir=str(tmp_path), scope="alice")
    session = manager.get_session(session_id)
    large_output = "original tool output " * 80
    await session.add_items(turn(1) + turn(2) + turn(3)[:1] + tool("archive", "lookup", large_output))
    first = get_history_page(service, session_id, "alice", 1, None, inline_chars=120)
    download = next(
        part.payload["result"]["historyContent"]["ref"]
        for message in first.messages
        for part in message.content
        if part.type == "call-tool-result"
    )

    await session.replace_items([turn(3)[0], {"role": "assistant", "content": "model-only summary"}])
    await session.add_items([turn(3)[1]] + turn(4))
    await session.replace_items([{"role": "assistant", "content": "second model-only summary"}])
    await session.add_items(turn(5))

    latest = get_history_page(service, session_id, "alice", 2, None)
    assert latest.total_turns == 5
    assert [msg["content"] for msg in manager.get_session_messages(session_id) if msg["role"] == "user"] == [
        "question 5"
    ]
    assert questions(latest) == ["question 4", "question 5"]
    third = get_history_page(service, session_id, "alice", 1, None, turn=3)
    assert questions(third) == ["question 3"]
    assert third.turns[0].message_id == first.turns[0].message_id
    assert any(part.payload.get("content") == "answer 3" for message in third.messages for part in message.content)
    assert "model-only summary" not in third.model_dump_json()
    older = get_history_page(service, session_id, "alice", 2, first.next_cursor)
    assert questions(older) == ["question 1", "question 2"]
    assert get_history_content(service, session_id, "alice", download) == large_output


@pytest.mark.asyncio
async def test_existing_database_stays_sqlite_only_after_reopen_and_compact(history, tmp_path):
    service, write = history
    session_id = "chat_legacy"
    db = write(session_id, turn(1))
    old_archive = db.parent / session_id / "data" / "history_before_upgrade.jsonl"
    old_archive.parent.mkdir(parents=True)
    old_archive.write_text(json.dumps(turn(99)[0]) + "\n")
    manager = SessionManager(session_dir=str(tmp_path), scope="alice")
    session = manager.get_session(session_id)
    await session.add_items(turn(2))
    await session.replace_items(turn(3))
    assert questions(get_history_page(service, session_id, "alice", 10, None)) == ["question 3"]
    assert manager.load_session_meta(session_id, "chat_history_version") is None
    assert not (old_archive.parent.parent / "chat-history").exists()


@pytest.mark.asyncio
async def test_pending_native_input_is_archived_once_and_survives_reopen(history, tmp_path):
    service, _ = history
    session_id = "chat_native_archive"
    manager = SessionManager(session_dir=str(tmp_path), scope="alice")
    session = manager.get_session(session_id)
    request = {"role": "user", "content": [{"type": "text", "text": "repeat this"}]}
    await session.add_items([request, *turn(1)[1:]])
    await session.replace_items(
        [request, {"role": "assistant", "content": "summary"}],
        pending_user_turns=1,
        history_pending=[request],
    )
    await session.add_items([{"role": "assistant", "content": "second reply"}])
    session.close()
    page = get_history_page(service, session_id, "alice", 10, None)
    assert questions(page) == ["repeat this", "repeat this"]
    assert len({turn.message_id for turn in page.turns}) == 2
    SessionManager(session_dir=str(tmp_path), scope="alice").clear_session(session_id)
    assert questions(get_history_page(service, session_id, "alice", 10, None)) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("offloaded", [False, True])
async def test_archived_tool_display_keeps_subagent_run_alignment(history, tmp_path, monkeypatch, offloaded):
    service, _ = history
    parent = "chat_parent_archive"
    child = "gen_job_session_archive"
    manager = SessionManager(session_dir=str(tmp_path), scope="alice")
    parent_session = manager.get_session(parent)
    child_session = SessionManager(session_dir=str(tmp_path / "alice" / parent)).get_session(child)
    for index in (1, 2):
        await child_session.add_items(turn(index) + tool(f"child_{index}", "lookup", str(index)))
        await child_session.replace_items([{"role": "assistant", "content": "child summary"}])
        call, result = tool(f"task_{index}", "task", {"session_id": child})
        if offloaded:
            archive = ToolArchive("", parent, base_dir=tmp_path / "alice" / parent / "data", preview_chars=12)
            result["output"] = archive.archive(result["output"], index, "output")
        await parent_session.add_items(turn(index) + [call, result])
        if index == 1:
            await parent_session.add_items(tool("unrelated", "execute_sql", "large unrelated output " * 3000))
        await parent_session.replace_items([{"role": "assistant", "content": "parent summary"}])
        if index == 1:
            old_segment = next((tmp_path / "alice" / parent / "chat-history").glob("*.jsonl"))

    original_open = Path.open

    def skip_off_page_archive(path, *args, **kwargs):
        assert path != old_segment, "off-page tool results must stay on disk"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", skip_off_page_archive)
    page = get_history_page(service, parent, "alice", 1, None)
    children = [
        part.payload["callToolId"]
        for msg in page.messages
        for part in msg.content
        if msg.depth == 1 and part.type == "call-tool"
    ]
    assert questions(page) == ["question 2"]
    assert children == ["child_2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_indexed_child_remains_available_when_its_tool_result_file_is_missing(history, tmp_path, native):
    service, write = history
    parent, child = "chat_indexed_child", "gen_job_session_child"
    write(
        child,
        turn(1) + tool("child_1", "lookup", "first") + turn(2) + tool("child_2", "lookup", "second"),
        parent=parent,
    )
    session = SessionManager(session_dir=str(tmp_path), scope="alice").get_session(parent)
    archive = ToolArchive("", parent, base_dir=tmp_path / "alice" / parent / "data", preview_chars=12)
    call, result = tool("task_1", "task", {"session_id": child}, native=native)
    offload_result(result, archive, 1, native=native)
    await session.add_items(turn(1) + [call, result])
    await session.replace_items([{"role": "assistant", "content": "summary"}])
    next(archive.dir.glob("000001_output_*.txt")).unlink()
    await session.add_items(turn(2) + tool("task_2", "task", {"session_id": child}, native=native))
    session.close()

    assert child_tool_calls(get_history_page(service, parent, "alice", 1, None)) == ["child_2"]
    assert child_tool_calls(get_history_page(service, parent, "alice", 2, None)) == ["child_1", "child_2"]
    assert child_tool_calls(service.get_history(parent, "alice").data) == ["child_1", "child_2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("storage", ["live", "archived", "split"])
async def test_non_task_result_with_reused_id_does_not_shift_subagent_runs(history, tmp_path, native, storage):
    service, write = history
    parent, child = "chat_reused_tool_id", "gen_job_session_child"
    write(
        child,
        turn(1) + tool("child_1", "lookup", "first") + turn(2) + tool("child_2", "lookup", "second"),
        parent=parent,
    )
    session = SessionManager(session_dir=str(tmp_path), scope="alice").get_session(parent)
    archive = ToolArchive("", parent, base_dir=tmp_path / "alice" / parent / "data", preview_chars=12)
    summary = [{"role": "assistant", "content": "summary"}]
    await session.add_items(turn(1) + tool("shared", "task", {"session_id": child}, native=native))
    call, result = tool("shared", "execute_sql", "unrelated", native=native)
    offload_result(result, archive, 1, native=native)
    next(archive.dir.glob("000001_output_*.txt")).unlink()
    await session.add_items(turn(2) + [call])
    if storage == "split":
        await session.replace_items(summary)
    await session.add_items([result])
    if storage == "archived":
        await session.replace_items(summary)
    await session.add_items(turn(3) + tool("last", "task", {"session_id": child}, native=native))
    session.close()

    assert child_tool_calls(service.get_history(parent, "alice").data) == ["child_1", "child_2"]
    assert child_tool_calls(get_history_page(service, parent, "alice", 1, None)) == ["child_2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("ordinary_tool", ["execute_sql", "lookup"])
async def test_compaction_matches_live_tool_results_to_legacy_calls(history, tmp_path, native, ordinary_tool):
    service, write = history
    parent, child = "chat_legacy_call_id", "gen_job_session_child"
    write(
        child,
        turn(1) + tool("child_1", "lookup", "first") + turn(3) + tool("child_2", "lookup", "second"),
        parent=parent,
    )
    manager = SessionManager(session_dir=str(tmp_path), scope="alice")
    session = manager.get_session(parent)
    archive = ToolArchive("", parent, base_dir=tmp_path / "alice" / parent / "data", preview_chars=12)
    summary = [{"role": "assistant", "content": "summary"}]
    first_call, first_result = tool("shared", "task", {"session_id": child}, native=native)
    offload_result(first_result, archive, 1, native=native)
    await session.add_items(turn(1) + [first_call, first_result])
    await session.replace_items(summary)
    ordinary_call, ordinary_result = tool("shared", ordinary_tool, "unrelated", native=native)
    offload_result(ordinary_result, archive, 2, native=native)
    if ordinary_tool == "lookup":
        next(archive.dir.glob("000002_output_*.txt")).unlink()
    await session.add_items(turn(2) + [ordinary_call])
    await session.replace_items(summary)

    # The baseline writer only indexed the task call. Its offloaded result and
    # the ordinary call with the same ID were present in the segments.
    db_path = tmp_path / "alice" / f"{parent}.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM chat_history_tasks WHERE child_id IS NOT NULL")
        assert conn.execute("SELECT call_id,child_id FROM chat_history_tasks").fetchall() == [("shared", None)]

    await session.add_items([ordinary_result])
    ordinary_result_id = manager.get_max_message_id(parent)
    next_call_id = "shared" if ordinary_tool == "lookup" else "last"
    await session.add_items(turn(3) + tool(next_call_id, "task", {"session_id": child}, native=native))
    assert subagent_run_offsets(manager, parent, manager.get_max_message_id(parent) + 1) == {child: 2}
    assert child_tool_calls(get_history_page(service, parent, "alice", 1, None)) == ["child_2"]
    await session.replace_items(summary)
    session.close()

    assert child_tool_calls(service.get_history(parent, "alice").data) == ["child_1", "child_2"]
    assert child_tool_calls(get_history_page(service, parent, "alice", 1, None)) == ["child_2"]
    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT child_id FROM chat_history_tasks WHERE session_id=? AND message_id=? AND call_id=?",
                (parent, ordinary_result_id, "shared"),
            ).fetchone()
            is None
        )
        # Tolerate an empty association persisted by the previous writer too.
        conn.execute("INSERT INTO chat_history_tasks VALUES (?,?,?,'')", (parent, ordinary_result_id, "shared"))
    assert child_tool_calls(get_history_page(service, parent, "alice", 1, None)) == ["child_2"]


@pytest.mark.asyncio
async def test_unfinished_task_in_indexed_segment_does_not_open_off_page_archives(history, tmp_path, monkeypatch):
    service, write = history
    parent, child = "chat_unfinished_task", "gen_job_session_child"
    write(child, turn(1) + tool("child_1", "lookup", "first"), parent=parent)
    session = SessionManager(session_dir=str(tmp_path), scope="alice").get_session(parent)
    summary = [{"role": "assistant", "content": "summary"}]
    await session.add_items(turn(0) + tool("stopped", "task", None)[:1])
    await session.replace_items(summary)
    for index in range(1, 4):
        await session.add_items(turn(index) + tool(f"sql_{index}", "execute_sql", "off-page output " * 1000))
        await session.replace_items(summary)
    await session.add_items(turn(4) + tool("latest", "task", {"session_id": child}))
    session.close()

    original_open = Path.open
    off_page = set((tmp_path / "alice" / parent / "chat-history").glob("*.jsonl"))
    assert len(off_page) == 4

    def open_latest_only(path, *args, **kwargs):
        assert path not in off_page, "a new index already covers these off-page segments"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_latest_only)
    assert child_tool_calls(get_history_page(service, parent, "alice", 1, None)) == ["child_1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("index", ["legacy", "legacy_missing", "indexed_missing", "indexed_known"])
async def test_legacy_task_index_gaps_do_not_shift_the_next_subagent_run(history, tmp_path, index):
    service, write = history
    parent, child = "chat_legacy_task_index", "gen_job_session_child"
    write(
        child,
        turn(1) + tool("child_1", "lookup", "old run") + turn(2) + tool("child_2", "lookup", "new run"),
        parent=parent,
    )
    manager = SessionManager(session_dir=str(tmp_path), scope="alice")
    session = manager.get_session(parent)
    archive = ToolArchive("", parent, base_dir=tmp_path / "alice" / parent / "data", preview_chars=12)
    call, result = tool("task_1", "task", {"session_id": child})
    result["output"] = archive.archive(result["output"], 1, "output")
    offloaded = next(archive.dir.glob("000001_output_*.txt"))
    parked = offloaded.with_suffix(".parked")
    if index == "indexed_missing":
        offloaded.rename(parked)
    await session.add_items(turn(1) + [call, result])
    await session.replace_items([{"role": "assistant", "content": "summary"}])

    if index.startswith("legacy"):
        # The old writer indexed the call but could not index a result behind a tool archive marker.
        with sqlite3.connect(tmp_path / "alice" / f"{parent}.db") as conn:
            conn.execute("DELETE FROM chat_history_tasks WHERE child_id IS NOT NULL")
            assert conn.execute("SELECT call_id,child_id FROM chat_history_tasks").fetchall() == [("task_1", None)]
    if index in ("legacy_missing", "indexed_known"):
        offloaded.rename(parked)

    await session.add_items(turn(2) + tool("task_2", "task", {"session_id": child}))
    session.close()

    def latest_child_calls():
        page = get_history_page(service, parent, "alice", 1, None)
        return [
            part.payload["callToolId"]
            for msg in page.messages
            if msg.depth == 1
            for part in msg.content
            if part.type == "call-tool"
        ]

    assert latest_child_calls() == ([] if index in ("legacy_missing", "indexed_missing") else ["child_2"])
    if parked.exists():
        parked.rename(offloaded)
    assert latest_child_calls() == ["child_2"]


@pytest.mark.asyncio
async def test_compaction_defers_task_attribution_when_legacy_segment_is_unreadable(history, tmp_path):
    service, write = history
    parent, child = "chat_unreadable_task_index", "gen_job_session_child"
    write(
        child,
        turn(1) + tool("child_1", "lookup", "first") + turn(2) + tool("child_2", "lookup", "second"),
        parent=parent,
    )
    session = SessionManager(session_dir=str(tmp_path), scope="alice").get_session(parent)
    call, result = tool("first", "task", {"session_id": child})
    summary = [{"role": "assistant", "content": "summary"}]
    await session.add_items([turn(1)[0], call, turn(1)[1]])
    await session.replace_items(summary)
    db_path = tmp_path / "alice" / f"{parent}.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM chat_history_tasks WHERE call_id=''")
    segment = next((tmp_path / "alice" / parent / "chat-history").glob("*.jsonl"))
    parked = segment.with_suffix(".parked")
    segment.rename(parked)

    await session.add_items([result] + turn(2) + tool("last", "task", {"session_id": child}))
    await session.replace_items(summary)
    assert await session.get_items() == summary
    session.close()

    latest = get_history_page(service, parent, "alice", 1, None)
    assert questions(latest) == ["question 2"]
    assert child_tool_calls(latest) == []
    assert any(
        part.payload["callToolId"] == "last"
        for message in latest.messages
        for part in message.content
        if part.type == "call-tool"
    )

    parked.rename(segment)
    assert child_tool_calls(get_history_page(service, parent, "alice", 1, None)) == ["child_2"]
    assert child_tool_calls(service.get_history(parent, "alice").data) == ["child_1", "child_2"]


@pytest.mark.asyncio
async def test_archive_write_failure_preserves_model_rows_and_existing_cursor(history, tmp_path, monkeypatch):
    from datus.storage.semantic_model import artifact_file

    service, _ = history
    session_id = "chat_archive_failure"
    session = SessionManager(session_dir=str(tmp_path), scope="alice").get_session(session_id)
    await session.add_items(turn(1) + turn(2))
    cursor = get_history_page(service, session_id, "alice", 1, None).next_cursor

    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(artifact_file, "atomic_write_bytes", disk_full)
    with pytest.raises(OSError, match="disk full"):
        await session.replace_items([{"role": "assistant", "content": "summary"}])
    assert await session.get_items() == turn(1) + turn(2)
    assert questions(get_history_page(service, session_id, "alice", 1, cursor)) == ["question 1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("derive", ["copy", "rewind"])
async def test_derived_sessions_archive_their_own_transcript(history, tmp_path, derive):
    service, _ = history
    manager = SessionManager(session_dir=str(tmp_path), scope="alice")
    source_id = "chat_source"
    source = manager.get_session(source_id)
    await source.add_items(turn(1))
    await source.replace_items([turn(1)[0], {"role": "assistant", "content": "source summary"}])
    await source.add_items(turn(2))
    if derive == "copy":
        derived_id = manager.copy_session(source_id, "chat")
    else:
        derived_id = manager.rewind_session(source_id, 2)
    derived = manager.get_session(derived_id)
    await derived.add_items(turn(3))
    await derived.replace_items([{"role": "assistant", "content": "derived summary"}])
    page = get_history_page(service, derived_id, "alice", 10, None)
    assert questions(page) == ["question 2", "question 3"]
    assert "source summary" not in page.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "truncated", "corrupt", "same_size_prompt", "same_size_result"])
async def test_unreadable_registered_archive_returns_service_unavailable(history, tmp_path, damage):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from datus.api.deps import get_app_context, get_datus_service
    from datus.api.routes.chat_routes import router

    service, _ = history
    session_id = "chat_unreadable_archive"
    session = SessionManager(session_dir=str(tmp_path), scope="alice").get_session(session_id)
    await session.add_items(turn(1) + tool("archive", "lookup", "long result " * 1000))
    page = get_history_page(service, session_id, "alice", 1, None)
    reference = next(
        part.payload["result"]["historyContent"]["ref"]
        for msg in page.messages
        for part in msg.content
        if part.type == "call-tool-result"
    )
    await session.replace_items([{"role": "assistant", "content": "summary"}])
    await session.add_items(turn(2))
    segment = next((tmp_path / "alice" / session_id / "chat-history").glob("*.jsonl"))
    original = segment.read_bytes()
    if damage == "missing":
        segment.unlink()
    elif damage == "truncated":
        segment.write_bytes(b"")
    elif damage == "corrupt":
        segment.write_bytes(b"[" * len(original))
    else:
        old, new = (
            (b"question 1", b"question 9") if damage == "same_size_prompt" else (b"long result ", b"fake result ")
        )
        changed = original.replace(old, new, 1)
        assert changed != original and len(changed) == len(original)
        segment.write_bytes(changed)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_datus_service] = lambda: SimpleNamespace(chat=service)
    app.dependency_overrides[get_app_context] = lambda: SimpleNamespace(user_id="alice")
    with TestClient(app) as client:
        latest = client.get("/api/v1/chat/history/page", params={"session_id": session_id, "limit": 1})
        assert latest.status_code == 200
        older_params = {"session_id": session_id, "limit": 1, "cursor": latest.json()["data"]["next_cursor"]}
        download_params = {"session_id": session_id, "reference": reference}
        assert client.get("/api/v1/chat/history/page", params=older_params).status_code == 503
        assert client.get("/api/v1/chat/history/content", params=download_params).status_code == 503

        segment.write_bytes(original)
        older = client.get("/api/v1/chat/history/page", params=older_params)
        assert older.status_code == 200
        assert older.json()["data"]["turns"][0]["text"] == "question 1"
        download = client.get("/api/v1/chat/history/content", params=download_params)
        assert download.status_code == 200 and download.text == "long result " * 1000


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
async def test_subagent_result_is_counted_when_call_and_result_cross_archive_segments(tmp_path, native, legacy):
    manager = SessionManager(session_dir=str(tmp_path), scope="alice")
    session_id, child = "chat_split_task", "dashboard_session_split"
    session = manager.get_session(session_id)
    call, result = tool("split", "task", {"session_id": child}, native=native)
    summary = [{"role": "assistant", "content": "summary"}]
    await session.add_items([turn(1)[0], call])
    await session.replace_items(summary)
    if legacy:
        with sqlite3.connect(tmp_path / "alice" / f"{session_id}.db") as conn:
            conn.execute("DELETE FROM chat_history_tasks WHERE call_id=''")
    assert subagent_run_offsets(manager, session_id, manager.get_max_message_id(session_id) + 1) == {}

    await session.add_items([result])
    assert subagent_run_offsets(manager, session_id, manager.get_max_message_id(session_id) + 1) == {child: 1}
    await session.replace_items(summary)
    assert subagent_run_offsets(manager, session_id, manager.get_max_message_id(session_id) + 1) == {child: 1}
    session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_latest_subagent_run_is_available_without_its_implicit_first_run_archive(history, tmp_path, native):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from datus.api.deps import get_app_context, get_datus_service
    from datus.api.routes.chat_routes import router

    service, _ = history
    parent, child = "chat_archived_implicit", "dashboard_session_implicit"
    root = tmp_path / "alice" / parent
    child_session = SessionManager(session_dir=str(root)).get_session(child)
    first_run = tool("implicit", "lookup", "earlier run")
    if native:
        first_run = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "implicit", "name": "lookup", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "implicit", "content": first_run[1]["output"]}],
            },
        ]
    await child_session.add_items(first_run)
    await child_session.replace_items([{"role": "assistant", "content": "summary"}])
    await child_session.add_items(turn(2) + tool("prompted", "lookup", "latest run"))

    parent_session = SessionManager(session_dir=str(tmp_path), scope="alice").get_session(parent)
    await parent_session.add_items(
        turn(1)
        + tool("task_old", "task", {"session_id": child})
        + turn(2)
        + tool("task_new", "task", {"session_id": child})
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_datus_service] = lambda: SimpleNamespace(chat=service)
    app.dependency_overrides[get_app_context] = lambda: SimpleNamespace(user_id="alice")
    with TestClient(app) as client:
        path = "/api/v1/chat/history/page"
        params = {"session_id": parent, "limit": 1}
        assert client.get(path, params=params).status_code == 200
        segment = next((root / child / "chat-history").glob("*.jsonl"))
        segment.unlink()

        latest = client.get(path, params=params)
        assert latest.status_code == 200
        data = latest.json()["data"]
        calls = [
            p["payload"]["callToolId"]
            for msg in data["messages"]
            for p in msg["content"]
            if msg["depth"] == 1 and p["type"] == "call-tool"
        ]
        assert calls == ["prompted"]
        assert client.get(path, params={**params, "cursor": data["next_cursor"]}).status_code == 503


@pytest.mark.asyncio
async def test_latest_page_checks_only_requested_records_of_the_same_archive(history, tmp_path):
    from datus.models.history_archive import HistoryArchiveReadError

    service, _ = history
    session_id = "chat_partial_integrity"
    session = SessionManager(session_dir=str(tmp_path), scope="alice").get_session(session_id)
    await session.add_items(turn(1) + turn(2))
    await session.replace_items([{"role": "assistant", "content": "summary"}])
    segment = next((tmp_path / "alice" / session_id / "chat-history").glob("*.jsonl"))
    segment.write_bytes(segment.read_bytes().replace(b"question 1", b"question 9", 1))

    latest = get_history_page(service, session_id, "alice", 1, None)
    assert questions(latest) == ["question 2"]
    with pytest.raises(HistoryArchiveReadError):
        get_history_page(service, session_id, "alice", 1, latest.next_cursor)
