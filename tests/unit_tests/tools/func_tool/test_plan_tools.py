# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for plan_tools module - CI level, zero external dependencies."""

import json
from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.plan_tools import PlanTool, SessionTodoStorage, TodoItem, TodoList, TodoStatus


class TestTodoItem:
    def test_default_status_is_pending(self):
        item = TodoItem(content="Do something")
        assert item.status == TodoStatus.PENDING

    def test_id_auto_generated(self):
        item1 = TodoItem(content="Task 1")
        item2 = TodoItem(content="Task 2")
        assert item1.id != item2.id

    def test_custom_status(self):
        item = TodoItem(content="Done task", status=TodoStatus.COMPLETED)
        assert item.status == TodoStatus.COMPLETED


class TestTodoList:
    def test_add_item(self):
        todo_list = TodoList()
        item = todo_list.add_item("First task")
        assert len(todo_list.items) == 1
        assert item.content == "First task"
        assert item.status == TodoStatus.PENDING

    def test_get_item_found(self):
        todo_list = TodoList()
        item = todo_list.add_item("Task")
        found = todo_list.get_item(item.id)
        assert found is item

    def test_get_item_not_found(self):
        todo_list = TodoList()
        result = todo_list.get_item("nonexistent-id")
        assert result is None

    def test_update_item_status_success(self):
        todo_list = TodoList()
        item = todo_list.add_item("Task")
        result = todo_list.update_item_status(item.id, TodoStatus.COMPLETED)
        assert result is True
        assert item.status == TodoStatus.COMPLETED

    def test_update_item_status_not_found(self):
        todo_list = TodoList()
        result = todo_list.update_item_status("bad-id", TodoStatus.COMPLETED)
        assert result is False

    def test_get_completed_items(self):
        todo_list = TodoList()
        item1 = todo_list.add_item("Task 1")
        todo_list.add_item("Task 2")
        todo_list.update_item_status(item1.id, TodoStatus.COMPLETED)
        completed = todo_list.get_completed_items()
        assert len(completed) == 1
        assert completed[0] is item1

    def test_get_completed_items_empty(self):
        todo_list = TodoList()
        todo_list.add_item("Pending task")
        assert todo_list.get_completed_items() == []


class TestSessionTodoStorage:
    @pytest.fixture
    def storage(self):
        mock_session = Mock()
        return SessionTodoStorage(session=mock_session)

    def test_initial_state_no_list(self, storage):
        assert storage.get_todo_list() is None
        assert storage.has_todo_list() is False

    def test_save_and_get_list(self, storage):
        todo_list = TodoList()
        todo_list.add_item("Task A")
        result = storage.save_list(todo_list)
        assert result is True
        retrieved = storage.get_todo_list()
        assert retrieved is todo_list
        assert storage.has_todo_list() is True

    def test_clear_all(self, storage):
        todo_list = TodoList()
        storage.save_list(todo_list)
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
        todo_list.add_item("Task")

        assert storage.save_list(todo_list) is True

        path = path_manager.todo_list_path("chat_session_aaaa")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["items"]) == 1
        assert data["items"][0]["content"] == "Task"

    def test_save_list_keeps_cjk_readable(self, path_manager):
        """Regression: non-ASCII content must be saved as raw UTF-8, not ``\\uXXXX`` escapes."""
        storage = SessionTodoStorage(session=Mock(), session_id="chat_session_cjk")
        todo_list = TodoList()
        todo_list.add_item("生成报表脚本")

        storage.save_list(todo_list)

        raw = path_manager.todo_list_path("chat_session_cjk").read_text(encoding="utf-8")
        assert "生成报表脚本" in raw
        assert "\\u" not in raw  # No JSON unicode escapes for CJK characters.

    def test_new_instance_lazy_loads_from_disk(self, path_manager):
        # First instance persists.
        s1 = SessionTodoStorage(session=Mock(), session_id="chat_session_bbbb")
        list1 = TodoList()
        list1.add_item("Existing")
        s1.save_list(list1)

        # Fresh instance with same session_id reconstructs the list from disk.
        s2 = SessionTodoStorage(session=Mock(), session_id="chat_session_bbbb")
        restored = s2.get_todo_list()
        assert restored is not None
        assert restored.items[0].content == "Existing"
        assert s2.has_todo_list() is True

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
        # Disk dir has no per-session file written.
        todos_dir = path_manager.project_data_dir / "todos"
        assert not todos_dir.exists() or not any(todos_dir.iterdir())

    def test_session_id_resolver_callable_defers_until_save(self, path_manager):
        """Regression: storage constructed during setup_tools (session_id=None
        at that moment) must still persist to disk once the agent has
        allocated a real session_id later. We pass a callable so each call
        re-resolves instead of snapshotting ``None`` forever."""
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
        todo_list.add_item("Persisted late")
        assert storage.save_list(todo_list) is True

        path = path_manager.todo_list_path("chat_session_late")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["items"][0]["content"] == "Persisted late"

    def test_session_id_change_reloads_from_disk(self, path_manager):
        """When the resolver returns a different session_id (e.g. cmd_resume
        swaps the active session), the next ``get_todo_list`` must re-read
        disk for the new session rather than serve cached state from the
        previous one."""
        # Seed disk for two distinct sessions.
        a_path = path_manager.todo_list_path("session_a")
        b_path = path_manager.todo_list_path("session_b")
        a_path.write_text(json.dumps(TodoList(items=[]).model_dump()))
        b = TodoList()
        b.add_item("From B")
        b_path.write_text(json.dumps(b.model_dump()))

        sid_holder = {"value": "session_a"}
        storage = SessionTodoStorage(
            session=Mock(),
            session_id=lambda: sid_holder["value"],
        )
        # Initial load reads session_a (empty).
        assert storage.get_todo_list().items == []

        # Resolver flips to session_b; storage must re-load.
        sid_holder["value"] = "session_b"
        loaded = storage.get_todo_list()
        assert loaded is not None
        assert loaded.items[0].content == "From B"


class TestPlanTool:
    @pytest.fixture
    def plan_tool(self):
        mock_session = Mock()
        return PlanTool(session=mock_session)

    def test_available_tools_returns_three(self, plan_tool):
        with patch("datus.tools.func_tool.plan_tools.trans_to_function_tool") as mock_trans:
            mock_trans.side_effect = lambda f: Mock(name=f.__name__)
            tools = plan_tool.available_tools()
        assert len(tools) == 3

    def test_todo_read_empty(self, plan_tool):
        result = plan_tool.todo_read()
        assert result.success == 1
        assert result.result["total_lists"] == 0
        assert result.result["lists"] == []

    def test_todo_read_with_list(self, plan_tool):
        todo_list = TodoList()
        todo_list.add_item("Task X")
        plan_tool.storage.save_list(todo_list)

        result = plan_tool.todo_read()
        assert result.success == 1
        assert result.result["total_lists"] == 1
        assert len(result.result["lists"]) == 1

    def test_todo_write_valid_json(self, plan_tool):
        todos_json = json.dumps(
            [
                {"content": "Step 1", "status": "pending"},
                {"content": "Step 2", "status": "completed"},
            ]
        )
        result = plan_tool.todo_write(todos_json)
        assert result.success == 1
        assert "todo_list" in result.result
        items = result.result["todo_list"]["items"]
        assert len(items) == 2

    def test_todo_write_invalid_json(self, plan_tool):
        result = plan_tool.todo_write("not valid json{{{")
        assert result.success == 0
        assert "Invalid JSON" in result.error

    def test_todo_write_empty_list(self, plan_tool):
        result = plan_tool.todo_write("[]")
        assert result.success == 0
        assert "no todo items" in result.error.lower()

    def test_todo_write_skips_empty_content(self, plan_tool):
        todos_json = json.dumps(
            [
                {"content": "", "status": "pending"},
                {"content": "  ", "status": "pending"},
                {"content": "Valid task", "status": "pending"},
            ]
        )
        result = plan_tool.todo_write(todos_json)
        assert result.success == 1
        items = result.result["todo_list"]["items"]
        assert len(items) == 1
        assert items[0]["content"] == "Valid task"

    def test_todo_write_completed_count(self, plan_tool):
        todos_json = json.dumps(
            [
                {"content": "Done step", "status": "completed"},
                {"content": "Pending step", "status": "pending"},
            ]
        )
        result = plan_tool.todo_write(todos_json)
        assert result.success == 1
        # Append-mode reports per-call totals.
        assert "1 completed, 1 pending" in result.result["message"]

    def test_todo_write_appends_to_existing_list(self, plan_tool):
        """Regression: ``todo_write`` is incremental — second call must not wipe first batch."""
        first = json.dumps([{"content": "Step A", "status": "pending"}])
        plan_tool.todo_write(first)

        second = json.dumps([{"content": "Step B", "status": "completed"}])
        result = plan_tool.todo_write(second)

        items = result.result["todo_list"]["items"]
        assert [i["content"] for i in items] == ["Step A", "Step B"]
        assert items[0]["status"] == "pending"
        assert items[1]["status"] == "completed"
        # Message reflects per-call counts, not list totals.
        assert "Appended 1 item" in result.result["message"]
        assert "list now has 2 item(s)" in result.result["message"]

    def test_todo_update_to_completed(self, plan_tool):
        todos_json = json.dumps([{"content": "Task A", "status": "pending"}])
        plan_tool.todo_write(todos_json)
        todo_list = plan_tool.storage.get_todo_list()
        item_id = todo_list.items[0].id

        result = plan_tool.todo_update(item_id, "completed")
        assert result.success == 1
        assert result.result["updated_item"]["status"] == "completed"

    def test_todo_update_to_failed(self, plan_tool):
        todos_json = json.dumps([{"content": "Failing task", "status": "pending"}])
        plan_tool.todo_write(todos_json)
        todo_list = plan_tool.storage.get_todo_list()
        item_id = todo_list.items[0].id

        result = plan_tool.todo_update(item_id, "failed")
        assert result.success == 1
        assert result.result["updated_item"]["status"] == "failed"

    def test_todo_update_invalid_status(self, plan_tool):
        todos_json = json.dumps([{"content": "Task", "status": "pending"}])
        plan_tool.todo_write(todos_json)
        todo_list = plan_tool.storage.get_todo_list()
        item_id = todo_list.items[0].id

        result = plan_tool.todo_update(item_id, "invalid_status")
        assert result.success == 0
        assert "Invalid status" in result.error

    def test_todo_update_no_todo_list(self, plan_tool):
        result = plan_tool.todo_update("some-id", "completed")
        assert result.success == 0
        assert "No todo list found" in result.error

    def test_todo_update_item_not_found(self, plan_tool):
        todos_json = json.dumps([{"content": "Task", "status": "pending"}])
        plan_tool.todo_write(todos_json)

        result = plan_tool.todo_update("non-existent-id", "completed")
        assert result.success == 0
        assert "not found" in result.error

    def test_todo_write_none_value_raises_error(self, plan_tool):
        result = plan_tool.todo_write(None)
        assert result.success == 0
        assert "Invalid JSON" in result.error
