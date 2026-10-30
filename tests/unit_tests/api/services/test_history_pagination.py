"""Real SQLite contracts for bounded history and stateless large-result references."""

import json
import sqlite3
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


@pytest.fixture
def history(tmp_path):
    service = ChatService(SimpleNamespace(session_dir=str(tmp_path)))

    def write(session_id, messages, user="alice", parent=None):
        manager = SessionManager(session_dir=str(tmp_path), scope=user)
        root = manager.session_dir
        if parent:
            from pathlib import Path

            root = str(Path(root) / parent)
            Path(root).mkdir(parents=True, exist_ok=True)
        from pathlib import Path

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


def tool(call_id, name, result):
    return [
        {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"},
        {"type": "function_call_output", "call_id": call_id, "output": json.dumps({"success": 1, "result": result})},
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
def test_subagent_reused_on_an_older_page_does_not_replay_the_wrong_run(history, native):
    service, write = history
    parent = "chat_parent"
    child = "gen_job_session_child"
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

    def parent_task(call_id):
        if not native:
            return tool(call_id, "task", {"session_id": child})
        return [
            {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": "task", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": json.dumps({"success": 1, "result": {"session_id": child}}),
                    }
                ],
            },
        ]

    write(parent, [*turn(0), *parent_task("task_old"), *turn(1), *parent_task("task_new")])
    page = get_history_page(service, parent, "alice", 1, None)
    parts = [p for m in page.messages for p in m.content]
    call_ids = [p.payload["callToolId"] for p in parts if p.type == "call-tool"]
    assert "child_new" in call_ids
    assert "child_old" not in call_ids
    assert questions(page) == ["question 1"]


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
