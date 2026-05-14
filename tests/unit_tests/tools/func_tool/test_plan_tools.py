# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for plan_tools module - CI level, zero external dependencies."""

import json
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from datus.tools.func_tool.plan_tools import (
    TITLE_WORD_LIMIT,
    PlanTool,
    SessionTodoStorage,
    TodoItem,
    TodoList,
    TodoStatus,
)


class TestTodoItem:
    def test_default_status_is_pending(self):
        item = TodoItem(id=1, title="Do something", content="task body")
        assert item.status == TodoStatus.PENDING

    def test_custom_status(self):
        item = TodoItem(id=1, title="Done", content="body", status=TodoStatus.COMPLETED)
        assert item.status == TodoStatus.COMPLETED

    def test_title_word_limit_accepts_exactly_eight(self):
        title = " ".join(f"w{i}" for i in range(TITLE_WORD_LIMIT))
        item = TodoItem(id=1, title=title, content="body")
        assert item.title == title

    def test_title_word_limit_rejects_nine_words(self):
        title = " ".join(f"w{i}" for i in range(TITLE_WORD_LIMIT + 1))
        with pytest.raises(ValidationError, match="8 words or fewer"):
            TodoItem(id=1, title=title, content="body")

    def test_title_empty_rejected(self):
        with pytest.raises(ValidationError):
            TodoItem(id=1, title="   ", content="body")

    def test_content_empty_rejected(self):
        with pytest.raises(ValidationError):
            TodoItem(id=1, title="Title", content="   ")

    def test_title_whitespace_stripped(self):
        item = TodoItem(id=1, title="  Short title  ", content="body")
        assert item.title == "Short title"


class TestTodoList:
    def test_add_item_assigns_incrementing_int_ids_from_one(self):
        todo_list = TodoList()
        a = todo_list.add_item("Title A", "Content A")
        b = todo_list.add_item("Title B", "Content B")
        c = todo_list.add_item("Title C", "Content C")
        assert [a.id, b.id, c.id] == [1, 2, 3]
        assert todo_list.next_id == 4

    def test_add_item_returns_pending_item(self):
        todo_list = TodoList()
        item = todo_list.add_item("Title", "Content")
        assert item.title == "Title"
        assert item.content == "Content"
        assert item.status == TodoStatus.PENDING

    def test_get_item_found(self):
        todo_list = TodoList()
        item = todo_list.add_item("Title", "Content")
        assert todo_list.get_item(item.id) is item

    def test_get_item_not_found_returns_none(self):
        todo_list = TodoList()
        todo_list.add_item("Title", "Content")
        assert todo_list.get_item(999) is None

    def test_update_item_status_success(self):
        todo_list = TodoList()
        item = todo_list.add_item("Title", "Content")
        assert todo_list.update_item_status(item.id, TodoStatus.IN_PROGRESS) is True
        assert item.status == TodoStatus.IN_PROGRESS

    def test_update_item_status_not_found(self):
        todo_list = TodoList()
        todo_list.add_item("Title", "Content")
        assert todo_list.update_item_status(999, TodoStatus.COMPLETED) is False

    def test_get_completed_items(self):
        todo_list = TodoList()
        a = todo_list.add_item("A", "ca")
        todo_list.add_item("B", "cb")
        todo_list.update_item_status(a.id, TodoStatus.COMPLETED)
        completed = todo_list.get_completed_items()
        assert len(completed) == 1
        assert completed[0] is a

    def test_next_id_reconciled_when_loaded_without_counter(self):
        """A persisted list with items but stale next_id must reconcile to max(id)+1."""
        raw = {
            "items": [
                {"id": 1, "title": "A", "content": "ca", "status": "completed"},
                {"id": 5, "title": "B", "content": "cb", "status": "pending"},
            ],
            "next_id": 1,
        }
        todo_list = TodoList(**raw)
        assert todo_list.next_id == 6
        # New add lands on 6, not 2.
        new_item = todo_list.add_item("C", "cc")
        assert new_item.id == 6

    def test_next_id_left_alone_when_already_correct(self):
        raw = {
            "items": [{"id": 1, "title": "A", "content": "ca"}],
            "next_id": 7,
        }
        todo_list = TodoList(**raw)
        assert todo_list.next_id == 7


class TestSessionTodoStorageBasic:
    @pytest.fixture
    def storage(self):
        return SessionTodoStorage(session=Mock())

    def test_initial_state_no_list(self, storage):
        assert storage.get_todo_list() is None
        assert storage.has_todo_list() is False

    def test_save_and_get_list(self, storage):
        todo_list = TodoList()
        todo_list.add_item("Title", "Content")
        assert storage.save_list(todo_list) is True
        retrieved = storage.get_todo_list()
        assert retrieved is todo_list
        assert storage.has_todo_list() is True

    def test_clear_all_in_memory(self, storage):
        storage.save_list(TodoList())
        storage.clear_all()
        assert storage.get_todo_list() is None
        assert storage.has_todo_list() is False


class TestSessionTodoStoragePersistence:
    """Persistence path: ``session_id`` -> ``project_data_dir/todos/{session_id}.json``."""

    @pytest.fixture
    def path_manager(self, tmp_path):
        from datus.utils.path_manager import DatusPathManager, reset_path_manager, set_current_path_manager

        reset_path_manager()
        pm = DatusPathManager(
            datus_home=str(tmp_path / "datus"),
            project_name="proj",
            project_root=str(tmp_path / "project"),
        )
        set_current_path_manager(pm)
        yield pm
        reset_path_manager()

    def test_save_list_writes_to_disk(self, path_manager):
        storage = SessionTodoStorage(session=Mock(), session_id="chat_session_aaaa")
        todo_list = TodoList()
        todo_list.add_item("Title A", "Content A")

        assert storage.save_list(todo_list) is True

        path = path_manager.todo_list_path("chat_session_aaaa")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["items"]) == 1
        assert data["items"][0]["title"] == "Title A"
        assert data["items"][0]["content"] == "Content A"
        assert data["next_id"] == 2

    def test_save_list_keeps_cjk_readable(self, path_manager):
        """Regression: non-ASCII content must be saved as raw UTF-8, not ``\\uXXXX`` escapes."""
        storage = SessionTodoStorage(session=Mock(), session_id="chat_session_cjk")
        todo_list = TodoList()
        todo_list.add_item("生成报表脚本", "执行 SQL 并落库")

        storage.save_list(todo_list)

        raw = path_manager.todo_list_path("chat_session_cjk").read_text(encoding="utf-8")
        assert "生成报表脚本" in raw
        assert "执行 SQL 并落库" in raw
        assert "\\u" not in raw

    def test_new_instance_lazy_loads_from_disk(self, path_manager):
        # First instance persists with id=1.
        s1 = SessionTodoStorage(session=Mock(), session_id="chat_session_bbbb")
        list1 = TodoList()
        list1.add_item("Existing", "Existing body")
        s1.save_list(list1)

        # Fresh instance recovers items AND next_id.
        s2 = SessionTodoStorage(session=Mock(), session_id="chat_session_bbbb")
        restored = s2.get_todo_list()
        assert isinstance(restored, TodoList)
        assert len(restored.items) == 1
        assert restored.items[0].title == "Existing"
        assert restored.items[0].content == "Existing body"
        assert restored.next_id == 2
        # Adding a new item after reload continues monotonic ids.
        new_item = restored.add_item("Second", "Second body")
        assert new_item.id == 2

    def test_clear_all_removes_disk_file(self, path_manager):
        storage = SessionTodoStorage(session=Mock(), session_id="chat_session_cccc")
        storage.save_list(TodoList())
        path = path_manager.todo_list_path("chat_session_cccc")
        assert path.exists()

        storage.clear_all()
        assert not path.exists()
        assert storage.get_todo_list() is None

    def test_no_session_id_falls_back_to_memory(self, path_manager):
        storage = SessionTodoStorage(session=Mock(), session_id=None)
        storage.save_list(TodoList())
        todos_dir = path_manager.project_data_dir / "todos"
        assert not todos_dir.exists() or not any(todos_dir.iterdir())

    def test_session_id_resolver_callable_defers_until_save(self, path_manager):
        sid_holder = {"value": None}
        storage = SessionTodoStorage(
            session=Mock(),
            session_id=lambda: sid_holder["value"],
        )

        # Before id allocation: no disk write.
        storage.save_list(TodoList())
        todos_dir = path_manager.project_data_dir / "todos"
        assert not todos_dir.exists() or not any(todos_dir.iterdir())

        # Agent allocates the session id; next save_list must hit disk.
        sid_holder["value"] = "chat_session_late"
        todo_list = TodoList()
        todo_list.add_item("Persisted late", "Late body")
        assert storage.save_list(todo_list) is True

        path = path_manager.todo_list_path("chat_session_late")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["items"][0]["title"] == "Persisted late"

    def test_session_id_change_reloads_from_disk(self, path_manager):
        a_path = path_manager.todo_list_path("session_a")
        b_path = path_manager.todo_list_path("session_b")
        a_path.parent.mkdir(parents=True, exist_ok=True)
        a_path.write_text(json.dumps(TodoList(items=[]).model_dump()))
        b = TodoList()
        b.add_item("From B", "Body B")
        b_path.write_text(json.dumps(b.model_dump()))

        sid_holder = {"value": "session_a"}
        storage = SessionTodoStorage(
            session=Mock(),
            session_id=lambda: sid_holder["value"],
        )
        # Initial load reads session_a (empty).
        assert storage.get_todo_list().items == []

        sid_holder["value"] = "session_b"
        loaded = storage.get_todo_list()
        assert isinstance(loaded, TodoList)
        assert len(loaded.items) == 1
        assert loaded.items[0].title == "From B"

    def test_legacy_uuid_payload_is_discarded(self, path_manager):
        """Pre-refactor todo files used uuid string ids and lacked ``title``.
        They must NOT crash the agent on resume — they are dropped with a
        warning and the storage starts fresh."""
        sid = "chat_session_legacy"
        path = path_manager.todo_list_path(sid)
        path.parent.mkdir(parents=True, exist_ok=True)
        legacy = {
            "items": [
                {"id": "abc-uuid", "content": "Old task", "status": "pending"},
            ]
        }
        path.write_text(json.dumps(legacy), encoding="utf-8")

        storage = SessionTodoStorage(session=Mock(), session_id=sid)
        assert storage.get_todo_list() is None
        assert storage.has_todo_list() is False


class TestPlanToolAvailableTools:
    def test_available_tools_returns_four(self):
        plan_tool = PlanTool(session=Mock())
        with patch("datus.tools.func_tool.plan_tools.trans_to_function_tool") as mock_trans:
            mock_trans.side_effect = lambda f: Mock(name=f.__name__)
            tools = plan_tool.available_tools()
        assert len(tools) == 4


class TestPlanToolTodoList:
    @pytest.fixture
    def plan_tool(self):
        return PlanTool(session=Mock())

    def test_empty_returns_zero_items(self, plan_tool):
        result = plan_tool.todo_list()
        assert result.success == 1
        assert result.result["items"] == []
        assert result.result["total"] == 0
        assert result.result["completed"] == 0

    def test_returns_overview_without_content(self, plan_tool):
        todo_list = TodoList()
        todo_list.add_item("Title A", "Long content A")
        todo_list.add_item("Title B", "Long content B")
        plan_tool.storage.save_list(todo_list)

        result = plan_tool.todo_list()
        assert result.success == 1
        items = result.result["items"]
        assert len(items) == 2
        assert items[0] == {"id": 1, "title": "Title A", "status": "pending"}
        assert items[1] == {"id": 2, "title": "Title B", "status": "pending"}
        # Critical: content must NOT leak into the overview.
        assert "content" not in items[0]
        assert "content" not in items[1]

    def test_completed_count_reflects_status(self, plan_tool):
        todo_list = TodoList()
        a = todo_list.add_item("A", "ca")
        todo_list.add_item("B", "cb")
        todo_list.update_item_status(a.id, TodoStatus.COMPLETED)
        plan_tool.storage.save_list(todo_list)

        result = plan_tool.todo_list()
        assert result.result["total"] == 2
        assert result.result["completed"] == 1


class TestPlanToolTodoRead:
    @pytest.fixture
    def plan_tool(self):
        pt = PlanTool(session=Mock())
        tl = TodoList()
        tl.add_item("First", "First content")
        tl.add_item("Second", "Second content")
        pt.storage.save_list(tl)
        return pt

    def test_read_by_id_returns_full_detail(self, plan_tool):
        result = plan_tool.todo_read(2)
        assert result.success == 1
        assert result.result == {
            "id": 2,
            "title": "Second",
            "status": "pending",
            "content": "Second content",
        }

    def test_read_unknown_id_returns_error(self, plan_tool):
        result = plan_tool.todo_read(999)
        assert result.success == 0
        assert "not found" in result.error

    def test_read_with_no_list_returns_error(self):
        plan_tool = PlanTool(session=Mock())
        result = plan_tool.todo_read(1)
        assert result.success == 0
        assert "No todo list" in result.error


class TestPlanToolTodoWrite:
    @pytest.fixture
    def plan_tool(self):
        return PlanTool(session=Mock())

    def test_valid_payload_returns_overview(self, plan_tool):
        payload = json.dumps(
            [
                {"title": "Step one", "content": "Do A"},
                {"title": "Step two", "content": "Do B"},
            ]
        )
        result = plan_tool.todo_write(payload)
        assert result.success == 1
        items = result.result["items"]
        assert [it["id"] for it in items] == [1, 2]
        assert [it["title"] for it in items] == ["Step one", "Step two"]
        # All new items default to pending.
        assert all(it["status"] == "pending" for it in items)
        # Overview only — no content field.
        assert all("content" not in it for it in items)

    def test_invalid_json_rejected(self, plan_tool):
        result = plan_tool.todo_write("not valid json{{{")
        assert result.success == 0
        assert "Invalid JSON" in result.error

    def test_none_argument_rejected(self, plan_tool):
        result = plan_tool.todo_write(None)
        assert result.success == 0
        assert "Invalid JSON" in result.error

    def test_empty_list_rejected(self, plan_tool):
        result = plan_tool.todo_write("[]")
        assert result.success == 0
        assert "non-empty" in result.error.lower()

    def test_non_array_payload_rejected(self, plan_tool):
        result = plan_tool.todo_write(json.dumps({"title": "x", "content": "y"}))
        assert result.success == 0

    def test_missing_title_rejected_atomically(self, plan_tool):
        """If any item is missing title, the entire batch must be rejected — no partial append."""
        payload = json.dumps(
            [
                {"title": "OK", "content": "fine"},
                {"content": "no title"},
            ]
        )
        result = plan_tool.todo_write(payload)
        assert result.success == 0
        assert "title" in result.error.lower()
        # Atomicity: storage must remain empty.
        assert plan_tool.storage.get_todo_list() is None

    def test_missing_content_rejected_atomically(self, plan_tool):
        payload = json.dumps([{"title": "Only title"}])
        result = plan_tool.todo_write(payload)
        assert result.success == 0
        assert "content" in result.error.lower()
        assert plan_tool.storage.get_todo_list() is None

    def test_title_too_long_rejected(self, plan_tool):
        long_title = " ".join(f"w{i}" for i in range(TITLE_WORD_LIMIT + 1))
        payload = json.dumps([{"title": long_title, "content": "body"}])
        result = plan_tool.todo_write(payload)
        assert result.success == 0
        assert "8 words" in result.error

    def test_title_at_limit_accepted(self, plan_tool):
        title = " ".join(f"w{i}" for i in range(TITLE_WORD_LIMIT))
        payload = json.dumps([{"title": title, "content": "body"}])
        result = plan_tool.todo_write(payload)
        assert result.success == 1

    def test_non_object_item_rejected(self, plan_tool):
        result = plan_tool.todo_write(json.dumps([["not", "an", "object"]]))
        assert result.success == 0

    def test_appends_to_existing_list_with_monotonic_ids(self, plan_tool):
        plan_tool.todo_write(json.dumps([{"title": "A", "content": "ca"}]))
        result = plan_tool.todo_write(json.dumps([{"title": "B", "content": "cb"}]))

        items = result.result["items"]
        assert [it["id"] for it in items] == [1, 2]
        assert [it["title"] for it in items] == ["A", "B"]
        assert "list now has 2 item(s)" in result.result["message"]


class TestPlanToolTodoUpdate:
    @pytest.fixture
    def plan_tool(self):
        pt = PlanTool(session=Mock())
        pt.todo_write(json.dumps([{"title": "Step", "content": "Do step"}]))
        return pt

    def test_update_to_in_progress(self, plan_tool):
        result = plan_tool.todo_update(1, "in_progress")
        assert result.success == 1
        item = result.result["updated_item"]
        assert item["status"] == "in_progress"
        assert item["title"] == "Step"
        assert item["content"] == "Do step"

    def test_update_to_completed(self, plan_tool):
        result = plan_tool.todo_update(1, "completed")
        assert result.success == 1
        assert result.result["updated_item"]["status"] == "completed"

    def test_update_to_failed(self, plan_tool):
        result = plan_tool.todo_update(1, "failed")
        assert result.success == 1
        assert result.result["updated_item"]["status"] == "failed"

    def test_full_flow_pending_inprogress_completed(self, plan_tool):
        """The recommended flow must work step by step on the same item."""
        for status in ("in_progress", "completed"):
            result = plan_tool.todo_update(1, status)
            assert result.success == 1
            assert result.result["updated_item"]["status"] == status

    def test_status_is_case_insensitive(self, plan_tool):
        result = plan_tool.todo_update(1, "IN_PROGRESS")
        assert result.success == 1
        assert result.result["updated_item"]["status"] == "in_progress"

    def test_invalid_status_rejected(self, plan_tool):
        result = plan_tool.todo_update(1, "weird_status")
        assert result.success == 0
        assert "Invalid status" in result.error

    def test_no_list_returns_error(self):
        plan_tool = PlanTool(session=Mock())
        result = plan_tool.todo_update(1, "completed")
        assert result.success == 0
        assert "No todo list" in result.error

    def test_unknown_id_returns_error(self, plan_tool):
        result = plan_tool.todo_update(999, "completed")
        assert result.success == 0
        assert "not found" in result.error
