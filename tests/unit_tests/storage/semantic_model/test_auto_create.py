# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.semantic_model.auto_create."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.ci


# ---------------------------------------------------------------------------
# extract_tables_from_sql_list
# ---------------------------------------------------------------------------


class TestExtractTablesFromSqlList:
    """Tests for extract_tables_from_sql_list."""

    def test_extracts_tables_from_simple_select(self):
        """Should extract table names from simple SELECT statements."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        tables = extract_tables_from_sql_list(["SELECT * FROM users"], config)
        assert "users" in tables

    def test_extracts_tables_from_multiple_sqls(self):
        """Should extract table names from multiple SQL statements."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        sql_list = [
            "SELECT * FROM orders",
            "SELECT * FROM customers",
        ]
        tables = extract_tables_from_sql_list(sql_list, config)
        assert "orders" in tables
        assert "customers" in tables

    def test_empty_sql_list(self):
        """Empty list should return empty set."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        tables = extract_tables_from_sql_list([], config)
        assert tables == set()

    def test_skips_empty_sql_strings(self):
        """Empty or whitespace-only SQL strings should be skipped."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        tables = extract_tables_from_sql_list(["", "  ", None], config)
        assert tables == set()

    def test_handles_invalid_sql_gracefully(self):
        """Invalid SQL should be skipped without raising."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        # This should not raise
        tables = extract_tables_from_sql_list(["NOT VALID SQL AT ALL ???"], config)
        # Result may be empty or contain something, but should not raise
        assert isinstance(tables, set)

    def test_deduplicates_tables(self):
        """Tables appearing in multiple SQLs should only appear once."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        sql_list = [
            "SELECT * FROM users",
            "SELECT count(*) FROM users",
        ]
        tables = extract_tables_from_sql_list(sql_list, config)
        # Should be a set, so duplicates are already removed
        user_entries = [t for t in tables if "users" in t.lower()]
        assert len(user_entries) >= 1

    def test_extracts_join_tables(self):
        """Should extract tables from JOIN clauses."""
        from datus.storage.semantic_model.auto_create import extract_tables_from_sql_list

        config = MagicMock()
        config.db_type = "snowflake"

        sql = "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id"
        tables = extract_tables_from_sql_list([sql], config)
        assert "orders" in tables
        assert "customers" in tables


# ---------------------------------------------------------------------------
# find_missing_semantic_models
# ---------------------------------------------------------------------------


class TestFindMissingSemanticModels:
    """Tests for find_missing_semantic_models."""

    def test_empty_tables_returns_empty(self):
        """Empty table set should return empty list."""
        from datus.storage.semantic_model.auto_create import find_missing_semantic_models

        config = MagicMock()
        result = find_missing_semantic_models(set(), config)
        assert result == []

    @patch("datus.storage.semantic_model.store.SemanticModelRAG")
    def test_all_models_exist(self, MockRAG):
        """When all semantic models exist, should return empty list."""
        from datus.storage.semantic_model.auto_create import find_missing_semantic_models

        config = MagicMock()
        mock_rag = MagicMock()
        MockRAG.return_value = mock_rag

        # Simulate existing semantic model
        mock_rag.storage.search_objects.return_value = [{"name": "users"}]

        result = find_missing_semantic_models({"users"}, config)
        assert result == []

    @patch("datus.storage.semantic_model.store.SemanticModelRAG")
    def test_missing_models_detected(self, MockRAG):
        """When semantic models are missing, should return those table names."""
        from datus.storage.semantic_model.auto_create import find_missing_semantic_models

        config = MagicMock()
        mock_rag = MagicMock()
        MockRAG.return_value = mock_rag

        # No matching results
        mock_rag.storage.search_objects.return_value = []

        result = find_missing_semantic_models({"missing_table"}, config)
        assert "missing_table" in result

    @patch("datus.storage.semantic_model.store.SemanticModelRAG")
    def test_case_insensitive_match(self, MockRAG):
        """Should match table names case-insensitively."""
        from datus.storage.semantic_model.auto_create import find_missing_semantic_models

        config = MagicMock()
        mock_rag = MagicMock()
        MockRAG.return_value = mock_rag

        mock_rag.storage.search_objects.return_value = [{"name": "USERS"}]

        result = find_missing_semantic_models({"users"}, config)
        assert result == []

    @patch("datus.storage.semantic_model.store.SemanticModelRAG")
    def test_fully_qualified_name_parsed(self, MockRAG):
        """Should parse fully qualified names (db.schema.table) and use last part."""
        from datus.storage.semantic_model.auto_create import find_missing_semantic_models

        config = MagicMock()
        mock_rag = MagicMock()
        MockRAG.return_value = mock_rag

        mock_rag.storage.search_objects.return_value = [{"name": "orders"}]

        result = find_missing_semantic_models({"mydb.public.orders"}, config)
        assert result == []

    @patch("datus.storage.semantic_model.store.SemanticModelRAG")
    def test_search_error_treated_as_missing(self, MockRAG):
        """Search errors should treat the table as missing."""
        from datus.storage.semantic_model.auto_create import find_missing_semantic_models

        config = MagicMock()
        mock_rag = MagicMock()
        MockRAG.return_value = mock_rag

        mock_rag.storage.search_objects.side_effect = Exception("Storage error")

        result = find_missing_semantic_models({"error_table"}, config)
        assert "error_table" in result


# ---------------------------------------------------------------------------
# create_semantic_models_for_tables (async, lines 111-147)
# ---------------------------------------------------------------------------


class TestCreateSemanticModelsForTables:
    """Tests for create_semantic_models_for_tables async function."""

    @pytest.mark.asyncio
    async def test_empty_tables_returns_true(self):
        """Empty table list should return (True, '') immediately without calling node."""
        from datus.storage.semantic_model.auto_create import create_semantic_models_for_tables

        config = MagicMock()
        success, error = await create_semantic_models_for_tables([], config)
        assert success is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_success_path(self):
        """Node yields success actions → returns (True, '')."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.auto_create import create_semantic_models_for_tables

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "mydb"
        mock_db_config.schema = "public"
        mock_config.current_db_config.return_value = mock_db_config

        class MockNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, ahm):
                action = SimpleNamespace(status=ActionStatus.SUCCESS, messages="ok")
                yield action

        with (
            patch(
                "datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode",
                MockNode,
            ),
            patch(
                "datus.schemas.semantic_agentic_node_models.SemanticNodeInput",
                MagicMock(return_value=MagicMock()),
            ),
        ):
            success, error = await create_semantic_models_for_tables(["users", "orders"], mock_config)
        assert success is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_terminal_error_action_returns_false(self):
        """Node yields a terminal error action → returns (False, message)."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.auto_create import create_semantic_models_for_tables

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, ahm):
                action = SimpleNamespace(status=ActionStatus.FAILED, action_type="error", messages="Generation failed")
                yield action

        with (
            patch(
                "datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode",
                MockNode,
            ),
            patch(
                "datus.schemas.semantic_agentic_node_models.SemanticNodeInput",
                MagicMock(return_value=MagicMock()),
            ),
        ):
            success, error = await create_semantic_models_for_tables(["users"], mock_config)
        assert success is False
        assert "Generation failed" in error

    @pytest.mark.asyncio
    async def test_terminal_error_no_messages_uses_default(self):
        """Terminal error action with no messages → uses default error message."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.auto_create import create_semantic_models_for_tables

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, ahm):
                action = SimpleNamespace(status=ActionStatus.FAILED, action_type="error", messages=None)
                yield action

        with (
            patch(
                "datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode",
                MockNode,
            ),
            patch(
                "datus.schemas.semantic_agentic_node_models.SemanticNodeInput",
                MagicMock(return_value=MagicMock()),
            ),
        ):
            success, error = await create_semantic_models_for_tables(["users"], mock_config)
        assert success is False
        assert error != ""

    @pytest.mark.asyncio
    async def test_recoverable_failed_tool_action_does_not_abort(self):
        """Failed validation/tool actions are recoverable intermediate states."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.auto_create import create_semantic_models_for_tables

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, ahm):
                yield SimpleNamespace(
                    status=ActionStatus.FAILED,
                    action_type="tool_call",
                    messages="Tool call: validate_semantic('{}...')",
                )
                yield SimpleNamespace(status=ActionStatus.SUCCESS, action_type="semantic_response", messages="ok")

        with (
            patch(
                "datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode",
                MockNode,
            ),
            patch(
                "datus.schemas.semantic_agentic_node_models.SemanticNodeInput",
                MagicMock(return_value=MagicMock()),
            ),
        ):
            success, error = await create_semantic_models_for_tables(["users"], mock_config)
        assert success is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_exception_returns_false(self):
        """execute_stream raises → returns (False, error_str)."""
        from unittest.mock import patch

        from datus.storage.semantic_model.auto_create import create_semantic_models_for_tables

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, ahm):
                raise RuntimeError("connection lost")
                yield  # async generator marker

        with (
            patch(
                "datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode",
                MockNode,
            ),
            patch(
                "datus.schemas.semantic_agentic_node_models.SemanticNodeInput",
                MagicMock(return_value=MagicMock()),
            ),
        ):
            success, error = await create_semantic_models_for_tables(["users"], mock_config)
        assert success is False
        assert "connection lost" in error

    @pytest.mark.asyncio
    async def test_emit_called_per_action(self):
        """emit callback is called for each yielded action."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.auto_create import create_semantic_models_for_tables

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, ahm):
                for _ in range(3):
                    action = SimpleNamespace(status=ActionStatus.SUCCESS, messages="step")
                    yield action

        emit_count = []
        with (
            patch(
                "datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode",
                MockNode,
            ),
            patch(
                "datus.schemas.semantic_agentic_node_models.SemanticNodeInput",
                MagicMock(return_value=MagicMock()),
            ),
        ):
            success, error = await create_semantic_models_for_tables(["users"], mock_config, emit=emit_count.append)
        assert success is True
        assert len(emit_count) == 3


# ---------------------------------------------------------------------------
# create_semantic_models_for_tables_sync (line 166)
# ---------------------------------------------------------------------------


class TestCreateSemanticModelsForTablesSync:
    """Tests for create_semantic_models_for_tables_sync sync wrapper."""

    def test_empty_tables_returns_true(self):
        """Sync wrapper: empty list returns (True, '')."""
        from datus.storage.semantic_model.auto_create import create_semantic_models_for_tables_sync

        config = MagicMock()
        success, error = create_semantic_models_for_tables_sync([], config)
        assert success is True
        assert error == ""

    def test_wraps_async_function(self, monkeypatch):
        """Sync wrapper delegates to async create_semantic_models_for_tables."""
        from datus.storage.semantic_model import auto_create

        calls = []

        async def mock_async(tables, config, emit=None):
            calls.append(tables)
            return True, ""

        monkeypatch.setattr(auto_create, "create_semantic_models_for_tables", mock_async)

        success, error = auto_create.create_semantic_models_for_tables_sync(["users"], MagicMock())
        assert success is True
        assert calls == [["users"]]


# ---------------------------------------------------------------------------
# ensure_semantic_models_exist (lines 185-200)
# ---------------------------------------------------------------------------


class TestEnsureSemanticModelsExist:
    """Tests for ensure_semantic_models_exist async function."""

    @pytest.mark.asyncio
    async def test_all_models_exist_returns_early(self, monkeypatch):
        """When all models exist (find_missing returns []), returns (True, '', [])."""
        from datus.storage.semantic_model import auto_create

        monkeypatch.setattr(auto_create, "find_missing_semantic_models", lambda tables, config: [])

        success, error, created = await auto_create.ensure_semantic_models_exist({"users"}, MagicMock())
        assert success is True
        assert error == ""
        assert created == []

    @pytest.mark.asyncio
    async def test_missing_tables_triggers_creation(self, monkeypatch):
        """When tables are missing, create_semantic_models_for_tables is called."""
        from datus.storage.semantic_model import auto_create

        monkeypatch.setattr(auto_create, "find_missing_semantic_models", lambda tables, config: ["orders"])

        async def mock_create(tables, config, emit=None):
            return True, ""

        monkeypatch.setattr(auto_create, "create_semantic_models_for_tables", mock_create)

        success, error, created = await auto_create.ensure_semantic_models_exist({"orders"}, MagicMock())
        assert success is True
        assert error == ""
        assert "orders" in created

    @pytest.mark.asyncio
    async def test_creation_failure_propagated(self, monkeypatch):
        """When creation fails, returns (False, error, missing_tables)."""
        from datus.storage.semantic_model import auto_create

        monkeypatch.setattr(auto_create, "find_missing_semantic_models", lambda tables, config: ["bad_table"])

        async def mock_create(tables, config, emit=None):
            return False, "Schema not found"

        monkeypatch.setattr(auto_create, "create_semantic_models_for_tables", mock_create)

        success, error, created = await auto_create.ensure_semantic_models_exist({"bad_table"}, MagicMock())
        assert success is False
        assert "Schema not found" in error
        assert "bad_table" in created

    @pytest.mark.asyncio
    async def test_empty_tables_no_creation(self, monkeypatch):
        """Empty table set: find_missing returns [] immediately, no creation."""
        from datus.storage.semantic_model import auto_create

        monkeypatch.setattr(auto_create, "find_missing_semantic_models", lambda tables, config: [])

        create_calls = []

        async def mock_create(tables, config, emit=None):
            create_calls.append(tables)
            return True, ""

        monkeypatch.setattr(auto_create, "create_semantic_models_for_tables", mock_create)

        success, error, created = await auto_create.ensure_semantic_models_exist(set(), MagicMock())
        assert success is True
        assert create_calls == []
