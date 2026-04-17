# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SchemaLinkingNode.

CI-level: zero external deps, zero network, zero API keys.
All external calls (RAG storage, DB connectors, SchemaLineageTool) are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node.schema_linking_node import SchemaLinkingNode
from datus.schemas.action_history import ActionStatus
from datus.schemas.node_models import TableSchema, TableValue
from datus.schemas.schema_linking_node_models import SchemaLinkingInput, SchemaLinkingResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_config(rag_path="/tmp/nonexistent_rag"):
    cfg = MagicMock()
    cfg.schema_linking_rate = "fast"
    cfg.current_database = "test_ns"
    cfg.namespaces = {"test_ns": {}}
    cfg.rag_storage_path.return_value = rag_path
    cfg.agentic_nodes = {}
    cfg.permissions_config = None
    cfg.skills_config = None
    cfg.prompt_version = None
    cfg.project_root = "."
    return cfg


def _make_workflow(table_schemas=None, table_values=None, reflection_round=0):
    wf = MagicMock()
    wf.task.task = "Show total sales"
    wf.task.database_type = "sqlite"
    wf.task.database_name = "test_db"
    wf.task.schema_linking_type = "table"
    wf.task.subject_path = []
    wf.task.external_knowledge = ""
    wf.context.table_schemas = table_schemas or []
    wf.context.table_values = table_values or []
    wf.reflection_round = reflection_round
    return wf


def _make_node(agent_config=None):
    cfg = agent_config or _make_agent_config()
    node = SchemaLinkingNode(
        node_id="schema_linking_1",
        description="Schema linking",
        node_type="schema_linking",
        agent_config=cfg,
    )
    # Set a minimal input so execute doesn't blow up
    node.input = SchemaLinkingInput(
        input_text="Show total sales",
        database_type="sqlite",
        database_name="test_db",
        matching_rate="fast",
    )
    return node


def _make_table_schema(name="orders"):
    ts = MagicMock(spec=TableSchema)
    ts.table_name = name
    return ts


def _make_table_value(name="orders"):
    tv = MagicMock(spec=TableValue)
    tv.table_name = name
    return tv


# ---------------------------------------------------------------------------
# TestSchemaLinkingNodeInit
# ---------------------------------------------------------------------------


class TestSchemaLinkingNodeInit:
    def test_node_creates_with_defaults(self):
        node = _make_node()
        assert node.id == "schema_linking_1"
        assert node._table_schemas == []
        assert node._table_values == []

    def test_node_description(self):
        node = _make_node()
        assert node.description == "Schema linking"


# ---------------------------------------------------------------------------
# TestSetupInput
# ---------------------------------------------------------------------------


class TestSetupInputSchemaLinking:
    def test_setup_input_empty_context(self):
        """When workflow has no existing table_schemas, build SchemaLinkingInput from scratch."""
        node = _make_node()
        wf = _make_workflow()
        with patch.object(node, "_search_external_knowledge", return_value=""):
            result = node.setup_input(wf)

        assert result["success"] is True
        assert isinstance(node.input, SchemaLinkingInput)
        assert node.input.input_text == "Show total sales"

    def test_setup_input_with_existing_schemas(self):
        """When workflow already has table schemas, short-circuit SchemaLinkingInput setup."""
        node = _make_node()
        existing_schema = _make_table_schema("orders")
        existing_value = _make_table_value("orders")
        wf = _make_workflow(table_schemas=[existing_schema], table_values=[existing_value])
        with patch.object(node, "_search_external_knowledge", return_value=""):
            result = node.setup_input(wf)

        assert result["success"] is True
        # table schemas should be cached
        assert node._table_schemas == [existing_schema]
        assert node._table_values == [existing_value]

    def test_setup_input_reflection_escalates_rate(self):
        """reflection_round escalates matching_rate."""
        cfg = _make_agent_config()
        cfg.schema_linking_rate = "fast"
        node = _make_node(agent_config=cfg)
        wf = _make_workflow(reflection_round=1)  # fast -> medium
        with patch.object(node, "_search_external_knowledge", return_value=""):
            node.setup_input(wf)

        assert node.input.matching_rate == "medium"

    def test_setup_input_reflection_caps_at_from_llm(self):
        """reflection_round beyond bounds caps at 'from_llm'."""
        cfg = _make_agent_config()
        cfg.schema_linking_rate = "slow"
        node = _make_node(agent_config=cfg)
        wf = _make_workflow(reflection_round=5)  # slow + 5 -> from_llm (capped)
        with patch.object(node, "_search_external_knowledge", return_value=""):
            node.setup_input(wf)

        assert node.input.matching_rate == "from_llm"

    def test_setup_input_combines_external_knowledge(self):
        """External knowledge is combined into workflow task when found."""
        node = _make_node()
        wf = _make_workflow()
        wf.task.external_knowledge = "original knowledge"
        with patch.object(node, "_search_external_knowledge", return_value="extra knowledge"):
            node.setup_input(wf)

        assert "extra knowledge" in wf.task.external_knowledge

    def test_setup_input_no_combine_when_empty_search(self):
        """When search returns empty string, workflow knowledge is not changed."""
        node = _make_node()
        wf = _make_workflow()
        wf.task.external_knowledge = "original knowledge"
        with patch.object(node, "_search_external_knowledge", return_value=""):
            node.setup_input(wf)

        assert wf.task.external_knowledge == "original knowledge"


# ---------------------------------------------------------------------------
# TestExecuteSchemaLinking
# ---------------------------------------------------------------------------


class TestExecuteSchemaLinking:
    def test_execute_uses_cached_table_schemas(self):
        """If _table_schemas is populated, return them directly without touching RAG."""
        node = _make_node()
        schemas = [_make_table_schema("orders")]
        values = [_make_table_value("orders")]
        node._table_schemas = schemas
        node._table_values = values

        result = node._execute_schema_linking()

        assert result.success is True
        assert result.table_schemas == schemas
        assert result.table_values == values
        assert result.schema_count == 1

    def test_execute_fallback_when_no_rag_path(self):
        """When RAG storage path doesn't exist, fallback is called."""
        cfg = _make_agent_config(rag_path="/nonexistent/path")
        node = _make_node(agent_config=cfg)

        fallback_result = SchemaLinkingResult(
            success=True,
            table_schemas=[_make_table_schema("orders")],
            schema_count=1,
            table_values=[],
            value_count=0,
        )
        with patch.object(node, "_execute_schema_linking_fallback", return_value=fallback_result) as mock_fallback:
            result = node._execute_schema_linking()

        mock_fallback.assert_called_once()
        assert result.success is True

    def test_execute_uses_tool_when_rag_path_exists(self, tmp_path):
        """When RAG path exists and tool succeeds, return tool result."""
        rag_path = str(tmp_path)
        cfg = _make_agent_config(rag_path=rag_path)
        node = _make_node(agent_config=cfg)

        good_result = SchemaLinkingResult(
            success=True,
            table_schemas=[_make_table_schema("orders")],
            schema_count=1,
            table_values=[],
            value_count=0,
        )

        with patch("datus.agent.node.schema_linking_node.SchemaLineageTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = good_result
            result = node._execute_schema_linking()

        assert result.success is True
        assert result.schema_count == 1

    def test_execute_fallback_when_tool_returns_no_tables(self, tmp_path):
        """When tool returns empty table_schemas, fallback is called."""
        rag_path = str(tmp_path)
        cfg = _make_agent_config(rag_path=rag_path)
        node = _make_node(agent_config=cfg)

        empty_result = SchemaLinkingResult(
            success=True,
            table_schemas=[],
            schema_count=0,
            table_values=[],
            value_count=0,
        )
        fallback_result = SchemaLinkingResult(
            success=True,
            table_schemas=[_make_table_schema("orders")],
            schema_count=1,
            table_values=[],
            value_count=0,
        )

        with patch("datus.agent.node.schema_linking_node.SchemaLineageTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = empty_result
            with patch.object(node, "_execute_schema_linking_fallback", return_value=fallback_result):
                result = node._execute_schema_linking()

        assert result.schema_count == 1

    def test_execute_fallback_when_tool_fails(self, tmp_path):
        """When tool.execute raises, fallback is called."""
        rag_path = str(tmp_path)
        cfg = _make_agent_config(rag_path=rag_path)
        node = _make_node(agent_config=cfg)

        fallback_result = SchemaLinkingResult(
            success=False,
            error="fallback error",
            table_schemas=[],
            schema_count=0,
            table_values=[],
            value_count=0,
        )

        with patch("datus.agent.node.schema_linking_node.SchemaLineageTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.side_effect = RuntimeError("tool error")
            with patch.object(node, "_execute_schema_linking_fallback", return_value=fallback_result):
                result = node._execute_schema_linking()

        assert result.success is False


# ---------------------------------------------------------------------------
# TestExecuteSchemaLinkingFallback
# ---------------------------------------------------------------------------


class TestExecuteSchemaLinkingFallback:
    def test_fallback_calls_db_manager(self):
        """Fallback gets connector from db_manager and calls tool.get_schems_by_db."""
        node = _make_node()
        mock_tool = MagicMock()
        mock_connector = MagicMock()
        mock_db_manager = MagicMock()
        mock_db_manager.get_conn.return_value = mock_connector

        expected_result = SchemaLinkingResult(
            success=True,
            table_schemas=[],
            schema_count=0,
            table_values=[],
            value_count=0,
        )
        mock_tool.get_schems_by_db.return_value = expected_result

        # db_manager_instance is imported locally inside the method body
        with patch(
            "datus.tools.db_tools.db_manager.db_manager_instance",
            return_value=mock_db_manager,
        ):
            result = node._execute_schema_linking_fallback(mock_tool)

        mock_tool.get_schems_by_db.assert_called_once()
        assert result.success is True

    def test_fallback_returns_error_result_on_exception(self):
        """When db_manager raises, fallback returns SchemaLinkingResult with success=False."""
        node = _make_node()
        mock_tool = MagicMock()

        # db_manager_instance is imported locally inside the method, patch at the source
        with patch(
            "datus.tools.db_tools.db_manager.db_manager_instance",
            side_effect=RuntimeError("db error"),
        ):
            result = node._execute_schema_linking_fallback(mock_tool)

        assert result.success is False
        assert "db error" in result.error


# ---------------------------------------------------------------------------
# TestCombineKnowledge
# ---------------------------------------------------------------------------


class TestCombineKnowledge:
    def test_both_parts_combined(self):
        node = _make_node()
        result = node._combine_knowledge("original", "enhanced")
        assert "original" in result
        assert "Relevant Business Knowledge" in result
        assert "enhanced" in result

    def test_only_original(self):
        node = _make_node()
        result = node._combine_knowledge("original", "")
        assert result == "original"

    def test_only_enhanced(self):
        node = _make_node()
        result = node._combine_knowledge("", "enhanced")
        assert "Relevant Business Knowledge" in result
        assert "enhanced" in result

    def test_both_empty(self):
        node = _make_node()
        result = node._combine_knowledge("", "")
        assert result == ""


# ---------------------------------------------------------------------------
# TestSearchExternalKnowledge
# ---------------------------------------------------------------------------


class TestSearchExternalKnowledge:
    def test_returns_empty_when_knowledge_size_zero(self):
        node = _make_node()
        with patch("datus.agent.node.schema_linking_node.ExtKnowledgeRAG") as mock_rag_class:
            mock_rag = mock_rag_class.return_value
            mock_rag.get_knowledge_size.return_value = 0
            result = node._search_external_knowledge("test query")

        assert result == ""

    def test_returns_formatted_knowledge(self):
        node = _make_node()
        search_results = [
            {"search_text": "revenue", "explanation": "Total revenue KPI"},
            {"search_text": "sales", "explanation": "Total sales metric"},
        ]
        with patch("datus.agent.node.schema_linking_node.ExtKnowledgeRAG") as mock_rag_class:
            mock_rag = mock_rag_class.return_value
            mock_rag.get_knowledge_size.return_value = 2
            mock_rag.query_knowledge.return_value = search_results
            result = node._search_external_knowledge("revenue query")

        assert "revenue" in result
        assert "Total revenue KPI" in result
        assert "sales" in result

    def test_returns_empty_on_exception(self):
        node = _make_node()
        with patch("datus.agent.node.schema_linking_node.ExtKnowledgeRAG", side_effect=RuntimeError("rag error")):
            result = node._search_external_knowledge("test query")

        assert result == ""

    def test_returns_empty_when_no_search_results(self):
        node = _make_node()
        with patch("datus.agent.node.schema_linking_node.ExtKnowledgeRAG") as mock_rag_class:
            mock_rag = mock_rag_class.return_value
            mock_rag.get_knowledge_size.return_value = 5
            mock_rag.query_knowledge.return_value = []
            result = node._search_external_knowledge("test query")

        assert result == ""


# ---------------------------------------------------------------------------
# TestUpdateContext
# ---------------------------------------------------------------------------


class TestUpdateContextSchemaLinking:
    def test_update_context_sets_schemas_when_empty(self):
        node = _make_node()
        schemas = [_make_table_schema("orders")]
        values = [_make_table_value("orders")]
        result = SchemaLinkingResult(
            success=True,
            table_schemas=schemas,
            schema_count=1,
            table_values=values,
            value_count=1,
        )
        node.result = result

        wf = _make_workflow()
        update_result = node.update_context(wf)

        assert update_result["success"] is True
        assert wf.context.table_schemas == schemas
        assert wf.context.table_values == values

    def test_update_context_skips_when_schemas_already_set(self):
        node = _make_node()
        existing_schemas = [_make_table_schema("existing")]
        schemas = [_make_table_schema("orders")]
        result = SchemaLinkingResult(
            success=True,
            table_schemas=schemas,
            schema_count=1,
            table_values=[],
            value_count=0,
        )
        node.result = result

        wf = _make_workflow(table_schemas=existing_schemas)
        node.update_context(wf)

        # Should not overwrite existing schemas
        assert wf.context.table_schemas == existing_schemas


# ---------------------------------------------------------------------------
# TestExecute
# ---------------------------------------------------------------------------


class TestExecuteSchemaLinkingNode:
    def test_execute_sets_result(self):
        node = _make_node()
        expected_result = SchemaLinkingResult(
            success=True,
            table_schemas=[],
            schema_count=0,
            table_values=[],
            value_count=0,
        )
        with patch.object(node, "_execute_schema_linking", return_value=expected_result):
            node.execute()

        assert node.result == expected_result


# ---------------------------------------------------------------------------
# TestExecuteStream (async)
# ---------------------------------------------------------------------------


class TestExecuteStreamSchemaLinking:
    @pytest.mark.asyncio
    async def test_execute_stream_yields_actions(self):
        """execute_stream yields knowledge and schema actions."""
        node = _make_node()
        node.input = SchemaLinkingInput(
            input_text="Show total sales",
            database_type="sqlite",
            database_name="test_db",
            matching_rate="fast",
        )
        good_result = SchemaLinkingResult(
            success=True,
            table_schemas=[],
            schema_count=0,
            table_values=[],
            value_count=0,
        )

        with patch.object(node, "_search_external_knowledge", return_value=""):
            with patch.object(node, "_execute_schema_linking", return_value=good_result):
                actions = []
                async for action in node.execute_stream():
                    actions.append(action)

        assert len(actions) >= 2
        action_ids = [a.action_id for a in actions]
        assert "external_knowledge_search" in action_ids
        assert "schema_linking" in action_ids

    @pytest.mark.asyncio
    async def test_execute_stream_marks_success(self):
        """execute_stream marks schema_linking action as SUCCESS on success."""
        node = _make_node()
        node.input = SchemaLinkingInput(
            input_text="sales",
            database_type="sqlite",
            database_name="test_db",
            matching_rate="fast",
        )
        good_result = SchemaLinkingResult(
            success=True,
            table_schemas=[_make_table_schema("orders")],
            schema_count=1,
            table_values=[],
            value_count=0,
        )

        with patch.object(node, "_search_external_knowledge", return_value="knowledge"):
            with patch.object(node, "_execute_schema_linking", return_value=good_result):
                actions = []
                async for action in node.execute_stream():
                    actions.append(action)

        schema_actions = [a for a in actions if a.action_id == "schema_linking"]
        assert schema_actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_stream_stores_result(self):
        """execute_stream stores result on self.result."""
        node = _make_node()
        node.input = SchemaLinkingInput(
            input_text="sales",
            database_type="sqlite",
            database_name="test_db",
            matching_rate="fast",
        )
        good_result = SchemaLinkingResult(
            success=True,
            table_schemas=[],
            schema_count=0,
            table_values=[],
            value_count=0,
        )

        with patch.object(node, "_search_external_knowledge", return_value=""):
            with patch.object(node, "_execute_schema_linking", return_value=good_result):
                async for _ in node.execute_stream():
                    pass

        assert node.result == good_result
