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


class TestExtractTableSqlEvidence:
    """Tests for table-scoped SQL evidence extraction."""

    def test_maps_evidence_by_qualified_and_unqualified_table(self):
        from datus.storage.semantic_model.auto_create import extract_table_sql_evidence

        config = MagicMock()
        config.db_type = "snowflake"
        records = [
            {
                "question": "Monthly catalog sales",
                "sql": (
                    "SELECT d.d_date, SUM(cs.cs_net_paid) "
                    "FROM SNOWFLAKE_SAMPLE_DATA.TPCDS_SF10TCL.CATALOG_SALES cs "
                    "JOIN SNOWFLAKE_SAMPLE_DATA.TPCDS_SF10TCL.DATE_DIM d "
                    "ON cs.cs_sold_date_sk = d.d_date_sk "
                    "GROUP BY d.d_date"
                ),
            }
        ]

        evidence = extract_table_sql_evidence(records, config)

        assert "catalog_sales" in evidence
        assert any("DATE_DIM" in item for item in evidence["catalog_sales"])
        assert "snowflake_sample_data.tpcds_sf10tcl.catalog_sales" in evidence

    def test_limits_evidence_per_table(self):
        from datus.storage.semantic_model.auto_create import extract_table_sql_evidence

        config = MagicMock()
        config.db_type = "snowflake"
        records = [{"question": f"q{i}", "sql": "SELECT * FROM users"} for i in range(3)]

        evidence = extract_table_sql_evidence(records, config, max_records_per_table=2)

        assert len(evidence["users"]) == 2


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
    def test_store_hit_but_yaml_missing_is_missing(self, MockRAG, tmp_path):
        """A stale store row should not suppress YAML regeneration."""
        from datus.storage.semantic_model.auto_create import find_missing_semantic_models

        semantic_dir = tmp_path / "subject" / "semantic_models" / "ac_manage"
        semantic_dir.mkdir(parents=True)
        config = MagicMock()
        config.db_type = "starrocks"
        config.current_datasource = "ac_manage"
        config.path_manager.semantic_model_path.return_value = semantic_dir
        config.current_db_config.return_value.catalog = "default_catalog"
        config.current_db_config.return_value.database = "ac_manage"
        config.current_db_config.return_value.schema = ""
        mock_rag = MagicMock()
        MockRAG.return_value = mock_rag
        mock_rag.storage.search_objects.return_value = [{"name": "users"}]

        result = find_missing_semantic_models({"users"}, config)

        assert result == ["users"]
        mock_rag.delete_semantic_model_for_table.assert_called_once_with(
            table_name="users",
            catalog_name="default_catalog",
            database_name="ac_manage",
            schema_name="",
        )

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
# create_semantic_model_for_table (single-table async)
# ---------------------------------------------------------------------------


class TestCreateSemanticModelForTable:
    """Tests for create_semantic_model_for_table async function."""

    @pytest.mark.asyncio
    async def test_success_path(self):
        """Node yields success actions → returns (True, '')."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.auto_create import create_semantic_model_for_table

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
                yield SimpleNamespace(status=ActionStatus.SUCCESS, messages="ok")

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
            success, error = await create_semantic_model_for_table("users", mock_config)
        assert success is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_terminal_error_action_returns_false(self):
        """Node yields a terminal error action → returns (False, message)."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.auto_create import create_semantic_model_for_table

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
                yield SimpleNamespace(status=ActionStatus.FAILED, action_type="error", messages="Generation failed")

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
            success, error = await create_semantic_model_for_table("users", mock_config)
        assert success is False
        assert "Generation failed" in error

    @pytest.mark.asyncio
    async def test_terminal_error_no_messages_uses_default(self):
        """Terminal error action with no messages → uses default error message."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.auto_create import create_semantic_model_for_table

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
                yield SimpleNamespace(status=ActionStatus.FAILED, action_type="error", messages=None)

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
            success, error = await create_semantic_model_for_table("users", mock_config)
        assert success is False
        assert error != ""

    @pytest.mark.asyncio
    async def test_recoverable_failed_tool_action_does_not_abort(self):
        """Failed validation/tool actions are recoverable intermediate states."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.auto_create import create_semantic_model_for_table

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
                yield SimpleNamespace(
                    status=ActionStatus.SUCCESS, action_type="gen_semantic_model_response", messages="ok"
                )

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
            success, error = await create_semantic_model_for_table("users", mock_config)
        assert success is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_exception_returns_false(self):
        """execute_stream raises → returns (False, error_str)."""
        from unittest.mock import patch

        from datus.storage.semantic_model.auto_create import create_semantic_model_for_table

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
            success, error = await create_semantic_model_for_table("users", mock_config)
        assert success is False
        assert "connection lost" in error

    @pytest.mark.asyncio
    async def test_emit_called_per_action(self):
        """emit callback is called for each yielded action."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.auto_create import create_semantic_model_for_table

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
                    yield SimpleNamespace(status=ActionStatus.SUCCESS, messages="step")

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
            success, error = await create_semantic_model_for_table("users", mock_config, emit=emit_count.append)
        assert success is True
        assert len(emit_count) == 3

    @pytest.mark.asyncio
    async def test_sql_evidence_is_added_to_user_message(self):
        """Success-story SQL evidence is passed into the semantic model prompt."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.auto_create import create_semantic_model_for_table

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = "public"
        mock_config.current_db_config.return_value = mock_db_config
        semantic_input_cls = MagicMock(return_value=MagicMock())

        class MockNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, ahm):
                yield SimpleNamespace(status=ActionStatus.SUCCESS, messages="ok")

        with (
            patch(
                "datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode",
                MockNode,
            ),
            patch(
                "datus.schemas.semantic_agentic_node_models.SemanticNodeInput",
                semantic_input_cls,
            ),
        ):
            success, error = await create_semantic_model_for_table(
                "catalog_sales",
                mock_config,
                related_tables=["catalog_sales", "date_dim"],
                sql_evidence=[
                    "Query 1:\nQuestion: q\nSQL:\nSELECT d.d_date FROM catalog_sales cs JOIN date_dim d ON cs.date_sk = d.d_date_sk"
                ],
            )

        assert success is True
        assert error == ""
        user_message = semantic_input_cls.call_args.kwargs["user_message"]
        assert "Success-story SQL evidence" in user_message
        assert "JOIN date_dim" in user_message

    @pytest.mark.asyncio
    async def test_fully_qualified_snowflake_table_sets_target_namespace(self) -> None:
        """Snowflake database.schema.table names are passed as separate DB tool coordinates."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.auto_create import create_semantic_model_for_table

        mock_config = MagicMock()
        mock_config.db_type = "snowflake"
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "SNOWFLAKE_SAMPLE_DATA"
        mock_db_config.schema = "TPCH_SF1"
        mock_config.current_db_config.return_value = mock_db_config
        semantic_input_cls = MagicMock(return_value=MagicMock())

        class MockNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, ahm):
                yield SimpleNamespace(status=ActionStatus.SUCCESS, messages="ok")

        with (
            patch(
                "datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode",
                MockNode,
            ),
            patch(
                "datus.schemas.semantic_agentic_node_models.SemanticNodeInput",
                semantic_input_cls,
            ),
        ):
            success, error = await create_semantic_model_for_table(
                "SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS",
                mock_config,
            )

        assert success is True
        assert error == ""
        kwargs = semantic_input_cls.call_args.kwargs
        assert kwargs["catalog"] == ""
        assert kwargs["database"] == "SNOWFLAKE_SAMPLE_DATA"
        assert kwargs["db_schema"] == "TPCH_SF1"
        assert 'table_name="ORDERS"' in kwargs["user_message"]
        assert 'database="SNOWFLAKE_SAMPLE_DATA"' in kwargs["user_message"]
        assert 'schema_name="TPCH_SF1"' in kwargs["user_message"]

    @pytest.mark.asyncio
    async def test_target_preparation_exception_returns_table_scoped_failure(self) -> None:
        """Target-resolution failures are isolated to the table being generated."""
        from datus.storage.semantic_model.auto_create import create_semantic_model_for_table

        mock_config = MagicMock()
        mock_config.current_db_config.side_effect = RuntimeError("missing datasource")

        success, error = await create_semantic_model_for_table("orders", mock_config)

        assert success is False
        assert error == "Error preparing semantic model input for table orders: missing datasource"


# ---------------------------------------------------------------------------
# create_semantic_models_for_tables (batch with per-table isolation)
# ---------------------------------------------------------------------------


class TestCreateSemanticModelsForTables:
    """Tests for create_semantic_models_for_tables async function."""

    @pytest.mark.asyncio
    async def test_empty_tables_returns_empty(self):
        """Empty table list should return ([], []) immediately."""
        from datus.storage.semantic_model.auto_create import create_semantic_models_for_tables

        config = MagicMock()
        succeeded, failed = await create_semantic_models_for_tables([], config)
        assert succeeded == []
        assert failed == []

    @pytest.mark.asyncio
    async def test_all_succeed(self, monkeypatch):
        """All tables succeed → succeeded=[all], failed=[]."""
        from datus.storage.semantic_model import auto_create

        async def mock_single(table, config, emit=None, related_tables=None):
            return True, ""

        monkeypatch.setattr(auto_create, "create_semantic_model_for_table", mock_single)

        succeeded, failed = await auto_create.create_semantic_models_for_tables(["users", "orders"], MagicMock())
        assert succeeded == ["users", "orders"]
        assert failed == []

    @pytest.mark.asyncio
    async def test_all_fail(self, monkeypatch):
        """All tables fail → succeeded=[], failed=[all]."""
        from datus.storage.semantic_model import auto_create

        async def mock_single(table, config, emit=None, related_tables=None):
            return False, f"{table} error"

        monkeypatch.setattr(auto_create, "create_semantic_model_for_table", mock_single)

        succeeded, failed = await auto_create.create_semantic_models_for_tables(["t1", "t2"], MagicMock())
        assert succeeded == []
        assert len(failed) == 2
        assert failed[0] == ("t1", "t1 error")
        assert failed[1] == ("t2", "t2 error")

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self, monkeypatch):
        """One table fails, others succeed — failure does not block the rest."""
        from datus.storage.semantic_model import auto_create

        async def mock_single(table, config, emit=None, related_tables=None):
            if table == "bad_table":
                return False, "Max turns exceeded"
            return True, ""

        monkeypatch.setattr(auto_create, "create_semantic_model_for_table", mock_single)

        succeeded, failed = await auto_create.create_semantic_models_for_tables(
            ["good1", "bad_table", "good2"], MagicMock()
        )
        assert succeeded == ["good1", "good2"]
        assert len(failed) == 1
        assert failed[0] == ("bad_table", "Max turns exceeded")

    @pytest.mark.asyncio
    async def test_passes_table_scoped_sql_evidence(self, monkeypatch):
        """Batch creation passes matching SQL evidence to each table."""
        from datus.storage.semantic_model import auto_create

        calls = {}

        async def mock_single(table, config, emit=None, related_tables=None, sql_evidence=None):
            calls[table] = sql_evidence
            return True, ""

        monkeypatch.setattr(auto_create, "create_semantic_model_for_table", mock_single)

        succeeded, failed = await auto_create.create_semantic_models_for_tables(
            ["catalog_sales", "date_dim"],
            MagicMock(),
            sql_evidence_by_table={
                "catalog_sales": ["catalog SQL"],
                "date_dim": ["date SQL"],
            },
        )

        assert succeeded == ["catalog_sales", "date_dim"]
        assert failed == []
        assert calls == {"catalog_sales": ["catalog SQL"], "date_dim": ["date SQL"]}

    @pytest.mark.asyncio
    async def test_batch_mode_uses_one_agent_when_all_models_created(self, monkeypatch):
        """Batch mode returns after one multi-table generation when all tables now exist."""
        from datus.storage.semantic_model import auto_create

        batch_calls = []

        async def mock_batch(tables, config, emit=None, sql_evidence_by_table=None):
            batch_calls.append(list(tables))
            return True, ""

        async def mock_single(table, config, emit=None, related_tables=None, sql_evidence=None):
            raise AssertionError("single-table fallback should not run")

        monkeypatch.setattr(auto_create, "create_semantic_models_for_tables_batch", mock_batch)
        monkeypatch.setattr(auto_create, "find_missing_semantic_models", lambda tables, config: [])
        monkeypatch.setattr(auto_create, "create_semantic_model_for_table", mock_single)

        succeeded, failed = await auto_create.create_semantic_models_for_tables(
            ["users", "orders"],
            MagicMock(),
            batch_mode=True,
        )

        assert batch_calls == [["users", "orders"]]
        assert succeeded == ["users", "orders"]
        assert failed == []

    @pytest.mark.asyncio
    async def test_batch_mode_falls_back_only_for_remaining_missing_models(self, monkeypatch):
        """Batch mode keeps created models and falls back per table only for misses."""
        from datus.storage.semantic_model import auto_create

        single_calls = []

        async def mock_batch(tables, config, emit=None, sql_evidence_by_table=None):
            return True, ""

        async def mock_single(table, config, emit=None, related_tables=None, sql_evidence=None):
            single_calls.append(table)
            return True, ""

        monkeypatch.setattr(auto_create, "create_semantic_models_for_tables_batch", mock_batch)
        monkeypatch.setattr(auto_create, "find_missing_semantic_models", lambda tables, config: ["orders"])
        monkeypatch.setattr(auto_create, "create_semantic_model_for_table", mock_single)

        succeeded, failed = await auto_create.create_semantic_models_for_tables(
            ["users", "orders"],
            MagicMock(),
            batch_mode=True,
        )

        assert single_calls == ["orders"]
        assert succeeded == ["users", "orders"]
        assert failed == []


# ---------------------------------------------------------------------------
# create_semantic_models_for_tables_sync
# ---------------------------------------------------------------------------


class TestCreateSemanticModelsForTablesSync:
    """Tests for create_semantic_models_for_tables_sync sync wrapper."""

    def test_empty_tables_returns_empty(self):
        """Sync wrapper: empty list returns ([], [])."""
        from datus.storage.semantic_model.auto_create import create_semantic_models_for_tables_sync

        config = MagicMock()
        succeeded, failed = create_semantic_models_for_tables_sync([], config)
        assert succeeded == []
        assert failed == []

    def test_wraps_async_function(self, monkeypatch):
        """Sync wrapper delegates to async create_semantic_models_for_tables."""
        from datus.storage.semantic_model import auto_create

        calls = []

        async def mock_async(tables, config, emit=None):
            calls.append(tables)
            return ["users"], []

        monkeypatch.setattr(auto_create, "create_semantic_models_for_tables", mock_async)

        succeeded, failed = auto_create.create_semantic_models_for_tables_sync(["users"], MagicMock())
        assert succeeded == ["users"]
        assert calls == [["users"]]


# ---------------------------------------------------------------------------
# ensure_semantic_models_exist
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
            return ["orders"], []

        monkeypatch.setattr(auto_create, "create_semantic_models_for_tables", mock_create)

        success, error, created = await auto_create.ensure_semantic_models_exist({"orders"}, MagicMock())
        assert success is True
        assert error == ""
        assert "orders" in created

    @pytest.mark.asyncio
    async def test_all_creation_fails(self, monkeypatch):
        """When all tables fail, returns (False, error_summary, [])."""
        from datus.storage.semantic_model import auto_create

        monkeypatch.setattr(auto_create, "find_missing_semantic_models", lambda tables, config: ["bad_table"])

        async def mock_create(tables, config, emit=None):
            return [], [("bad_table", "Schema not found")]

        monkeypatch.setattr(auto_create, "create_semantic_models_for_tables", mock_create)

        success, error, created = await auto_create.ensure_semantic_models_exist({"bad_table"}, MagicMock())
        assert success is False
        assert "Schema not found" in error
        assert created == []

    @pytest.mark.asyncio
    async def test_partial_failure_returns_success_with_warning(self, monkeypatch):
        """Some tables succeed, some fail → success=True, error has partial info."""
        from datus.storage.semantic_model import auto_create

        monkeypatch.setattr(auto_create, "find_missing_semantic_models", lambda tables, config: ["good", "bad"])

        async def mock_create(tables, config, emit=None):
            return ["good"], [("bad", "Max turns exceeded")]

        monkeypatch.setattr(auto_create, "create_semantic_models_for_tables", mock_create)

        success, error, created = await auto_create.ensure_semantic_models_exist({"good", "bad"}, MagicMock())
        assert success is True
        assert "Partial failures" in error
        assert "good" in created
        assert "bad" not in created

    @pytest.mark.asyncio
    async def test_empty_tables_no_creation(self, monkeypatch):
        """Empty table set: find_missing returns [] immediately, no creation."""
        from datus.storage.semantic_model import auto_create

        monkeypatch.setattr(auto_create, "find_missing_semantic_models", lambda tables, config: [])

        create_calls = []

        async def mock_create(tables, config, emit=None):
            create_calls.append(tables)
            return tables, []

        monkeypatch.setattr(auto_create, "create_semantic_models_for_tables", mock_create)

        success, error, created = await auto_create.ensure_semantic_models_exist(set(), MagicMock())
        assert success is True
        assert create_calls == []
