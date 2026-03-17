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
        assert "1 already completed" in result.result["message"]

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
