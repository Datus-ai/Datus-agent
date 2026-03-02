import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from datus.schemas.agent_models import ScopedContext, ScopedContextLists, SubAgentConfig
from datus.storage.conditions import build_where
from datus.storage.sub_agent_kb_bootstrap import SUPPORTED_COMPONENTS, SubAgentBootstrapper


class DummyAgentConfig:
    def __init__(self):
        self.current_namespace = "demo"
        self.current_database = "warehouse"
        self.db_type = "sqlite"
        self.agentic_nodes = {}

    def rag_storage_path(self) -> str:
        return "/tmp/data"

    def sub_agent_storage_path(self, sub_agent_name: str):
        return os.path.join(self.rag_storage_path(), "sub_agents", sub_agent_name)

    def sub_agent_config(self, sub_agent_name: str) -> Dict[str, Any]:
        return self.agentic_nodes.get(sub_agent_name, {})


class DummyDBManager:
    def __init__(self, db_config):
        self._config = {"demo": db_config}

    def current_db_configs(self, namespace: str):
        return {"logic": self._config[namespace]}


@pytest.fixture
def bootstrapper():
    agent_config = DummyAgentConfig()
    sub_agent = SubAgentConfig(system_prompt="tester", scoped_context=ScopedContext())
    agent_config.agentic_nodes["tester"] = sub_agent.model_dump()
    return SubAgentBootstrapper(sub_agent=sub_agent, agent_config=agent_config)


def test_scoped_context_as_lists_normalizes_entries():
    context = ScopedContext(
        tables="orders, customers\norders ",
        metrics="revenue, revenue\n profit ",
        sqls="daily_sales\nmonthly_sales, daily_sales",
        ext_knowledge="Finance/Revenue, Finance/Revenue\nSales/Marketing",
    )
    lists = context.as_lists()
    assert lists.tables == ["orders", "customers"]
    assert lists.metrics == ["revenue", "profit"]
    assert lists.sqls == ["daily_sales", "monthly_sales"]
    assert lists.ext_knowledge == ["Finance/Revenue", "Sales/Marketing"]
    assert lists.any()


def test_scoped_context_lists_any_returns_false_when_empty():
    lists = ScopedContextLists()
    assert not lists.any()


def test_scoped_context_ext_knowledge_is_empty():
    assert ScopedContext(ext_knowledge="Finance.*").is_empty is False
    assert ScopedContext().is_empty is True


def test_scoped_context_ext_knowledge_as_lists():
    context = ScopedContext(ext_knowledge="Finance.*")
    lists = context.as_lists()
    assert lists.ext_knowledge == ["Finance.*"]


def test_scoped_context_lists_ext_knowledge_any():
    assert ScopedContextLists(ext_knowledge=["a"]).any() is True
    assert ScopedContextLists().any() is False


def test_supported_components_includes_ext_knowledge():
    assert "ext_knowledge" in SUPPORTED_COMPONENTS


def test_handle_ext_knowledge_empty_tokens(bootstrapper):
    result = bootstrapper._handle_ext_knowledge([])
    assert result.status == "skipped"
    assert result.component == "ext_knowledge"


@patch("datus.storage.sub_agent_kb_bootstrap.ExtKnowledgeRAG")
def test_handle_ext_knowledge_with_matches(mock_rag_cls, bootstrapper, tmp_path):
    # Make _ensure_source_ready return True by patching rag_storage_path to a real dir
    bootstrapper.agent_config.rag_storage_path = lambda: str(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)

    mock_store = MagicMock()
    mock_store.search_all_knowledge.return_value = [
        {"subject_path": ["Finance", "Revenue"], "name": "Q1_report"},
    ]
    mock_rag_instance = MagicMock()
    mock_rag_instance.store = mock_store
    mock_rag_cls.return_value = mock_rag_instance

    result = bootstrapper._handle_ext_knowledge(["Finance/Revenue"])
    assert result.status == "plan"
    assert result.component == "ext_knowledge"
    assert result.details["match_count"] == 1
    assert result.details["missing"] == []
    assert result.details["invalid"] == []


@patch("datus.storage.sub_agent_kb_bootstrap.ExtKnowledgeRAG")
def test_handle_ext_knowledge_with_missing(mock_rag_cls, bootstrapper, tmp_path):
    bootstrapper.agent_config.rag_storage_path = lambda: str(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)

    mock_store = MagicMock()
    mock_store.search_all_knowledge.return_value = []
    mock_rag_instance = MagicMock()
    mock_rag_instance.store = mock_store
    mock_rag_cls.return_value = mock_rag_instance

    result = bootstrapper._handle_ext_knowledge(["NonExistent/Path"])
    assert result.status == "plan"
    assert result.details["match_count"] == 0
    assert result.details["missing"] == ["NonExistent/Path"]


@patch("datus.storage.sub_agent_kb_bootstrap.ExtKnowledgeRAG")
def test_run_with_ext_knowledge_component(mock_rag_cls, tmp_path):
    agent_config = DummyAgentConfig()
    agent_config.rag_storage_path = lambda: str(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)

    sub_agent = SubAgentConfig(
        system_prompt="knowledge_agent",
        scoped_context=ScopedContext(ext_knowledge="Finance/Revenue"),
    )
    agent_config.agentic_nodes["knowledge_agent"] = sub_agent.model_dump()

    mock_store = MagicMock()
    mock_store.search_all_knowledge.return_value = [
        {"subject_path": ["Finance", "Revenue"], "name": "Q1"},
    ]
    mock_rag_instance = MagicMock()
    mock_rag_instance.store = mock_store
    mock_rag_cls.return_value = mock_rag_instance

    bs = SubAgentBootstrapper(sub_agent=sub_agent, agent_config=agent_config)
    result = bs.run(selected_components=["ext_knowledge"])

    assert result.should_bootstrap is True
    assert len(result.results) == 1
    assert result.results[0].component == "ext_knowledge"
    assert result.results[0].status == "plan"


def test_metadata_condition_applies_defaults_and_wildcards(bootstrapper):
    # Single-part token maps to table_name (rightmost field)
    condition = bootstrapper._metadata_condition_for_token("sales")
    clause = build_where(condition)
    assert "table_name = 'sales'" in clause

    # Two-part token: database_name.table_name (with wildcard)
    condition = bootstrapper._metadata_condition_for_token("sales.orders*")
    clause = build_where(condition)
    assert "database_name = 'sales'" in clause
    assert "table_name LIKE 'orders%'" in clause


def test_run_with_unsupported_strategy(bootstrapper):
    """run() raises DatusException for unsupported strategy."""
    from datus.utils.exceptions import DatusException

    # Give the sub_agent a non-empty scoped_context so it doesn't return early
    bootstrapper.sub_agent = SubAgentConfig(
        system_prompt="tester",
        scoped_context=ScopedContext(tables="users"),
    )
    with pytest.raises(DatusException, match="Unsupported strategy"):
        bootstrapper.run(strategy="invalid_strategy")


def test_ensure_source_ready_nonexistent_path(bootstrapper):
    """_ensure_source_ready returns False for non-existent path."""
    result = bootstrapper._ensure_source_ready("/nonexistent/path/xyz", "metadata")
    assert result is False


def test_ensure_source_ready_existing_path(bootstrapper, tmp_path):
    """_ensure_source_ready returns True for existing directory."""
    result = bootstrapper._ensure_source_ready(str(tmp_path), "metadata")
    assert result is True


def test_metadata_conditions_with_valid_and_invalid_tokens(bootstrapper):
    """_metadata_conditions returns mapped conditions and invalid tokens."""
    mapped, invalid = bootstrapper._metadata_conditions(["users", "", "   "])
    assert len(mapped) == 1
    assert mapped[0][0] == "users"
    assert len(invalid) == 0  # empty strings are filtered by strip()


def test_metadata_conditions_empty_list(bootstrapper):
    """_metadata_conditions returns empty lists for no tokens."""
    mapped, invalid = bootstrapper._metadata_conditions([])
    assert mapped == []
    assert invalid == []


def test_metadata_condition_for_token_empty_returns_none(bootstrapper):
    """_metadata_condition_for_token returns None for empty token."""
    assert bootstrapper._metadata_condition_for_token("") is None
    assert bootstrapper._metadata_condition_for_token("  ") is None


def test_metadata_condition_for_token_with_postgres_dialect():
    """_metadata_condition_for_token uses schema_name for postgres."""
    agent_config = DummyAgentConfig()
    agent_config.db_type = "postgresql"
    sub_agent = SubAgentConfig(system_prompt="tester", scoped_context=ScopedContext())
    agent_config.agentic_nodes["tester"] = sub_agent.model_dump()
    bs = SubAgentBootstrapper(sub_agent=sub_agent, agent_config=agent_config)

    condition = bs._metadata_condition_for_token("public.users")
    clause = build_where(condition)
    assert "schema_name = 'public'" in clause
    assert "table_name = 'users'" in clause


def test_metadata_condition_for_token_with_snowflake_dialect():
    """_metadata_condition_for_token handles catalog for snowflake."""
    agent_config = DummyAgentConfig()
    agent_config.db_type = "snowflake"
    sub_agent = SubAgentConfig(system_prompt="tester", scoped_context=ScopedContext())
    agent_config.agentic_nodes["tester"] = sub_agent.model_dump()
    bs = SubAgentBootstrapper(sub_agent=sub_agent, agent_config=agent_config)

    condition = bs._metadata_condition_for_token("mydb.public.users")
    clause = build_where(condition)
    assert "table_name = 'users'" in clause
    assert "schema_name = 'public'" in clause


def test_combine_conditions_empty(bootstrapper):
    """_combine_conditions returns None for empty list."""
    assert bootstrapper._combine_conditions([]) is None


def test_combine_conditions_single(bootstrapper):
    """_combine_conditions returns the single node directly."""
    from datus.storage.conditions import eq

    node = eq("table_name", "users")
    result = bootstrapper._combine_conditions([("users", node)])
    clause = build_where(result)
    assert "table_name = 'users'" in clause


def test_combine_conditions_multiple(bootstrapper):
    """_combine_conditions returns OR of multiple nodes."""
    from datus.storage.conditions import eq

    node1 = eq("table_name", "users")
    node2 = eq("table_name", "orders")
    result = bootstrapper._combine_conditions([("users", node1), ("orders", node2)])
    clause = build_where(result)
    assert "users" in clause
    assert "orders" in clause
    assert "OR" in clause


def test_format_table_identifier():
    """_format_table_identifier joins non-None fields with dots."""
    row = {"catalog_name": None, "database_name": "db1", "schema_name": "public", "table_name": "users"}
    result = SubAgentBootstrapper._format_table_identifier(row)
    assert result == "db1.public.users"


def test_format_table_identifier_minimal():
    """_format_table_identifier works with just table_name."""
    row = {"table_name": "users"}
    result = SubAgentBootstrapper._format_table_identifier(row)
    assert result == "users"


def test_format_subject_identifier():
    """_format_subject_identifier formats subject path with name."""
    row = {"subject_path": ["Finance", "Revenue"], "name": "Q1_report"}
    result = SubAgentBootstrapper._format_subject_identifier(row)
    assert result == "Finance/Revenue/Q1_report"


@patch("datus.storage.sub_agent_kb_bootstrap.MetricRAG")
def test_handle_metrics_with_matches(mock_rag_cls, bootstrapper, tmp_path):
    """_handle_metrics returns plan with matched metrics."""
    bootstrapper.agent_config.rag_storage_path = lambda: str(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)

    mock_rag_instance = MagicMock()
    mock_rag_instance.search_all_metrics.return_value = [
        {"subject_path": ["Finance"], "name": "revenue"},
    ]
    mock_rag_cls.return_value = mock_rag_instance

    result = bootstrapper._handle_metrics(["Finance.revenue"])
    assert result.status == "plan"
    assert result.component == "metrics"
    assert result.details["match_count"] == 1


@patch("datus.storage.sub_agent_kb_bootstrap.ReferenceSqlRAG")
def test_handle_reference_sql_with_matches(mock_rag_cls, bootstrapper, tmp_path):
    """_handle_reference_sql returns plan with matched entries."""
    bootstrapper.agent_config.rag_storage_path = lambda: str(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)

    mock_rag_instance = MagicMock()
    mock_rag_instance.search_all_reference_sql.return_value = [
        {"subject_path": ["Analytics"], "name": "daily_sales"},
    ]
    mock_rag_cls.return_value = mock_rag_instance

    result = bootstrapper._handle_reference_sql(["Analytics.daily_sales"])
    assert result.status == "plan"
    assert result.component == "reference_sql"
    assert result.details["match_count"] == 1
    assert result.details["missing"] == []


def test_handle_metadata_skipped_when_empty(bootstrapper):
    """_handle_metadata returns skipped for empty tables list."""
    result = bootstrapper._handle_metadata([])
    assert result.status == "skipped"
    assert result.component == "metadata"


def test_handle_metrics_skipped_when_empty(bootstrapper):
    """_handle_metrics returns skipped for empty metrics list."""
    result = bootstrapper._handle_metrics([])
    assert result.status == "skipped"
    assert result.component == "metrics"


def test_handle_reference_sql_skipped_when_empty(bootstrapper):
    """_handle_reference_sql returns skipped for empty list."""
    result = bootstrapper._handle_reference_sql([])
    assert result.status == "skipped"
    assert result.component == "reference_sql"


def test_handle_metadata_error_when_path_missing(bootstrapper):
    """_handle_metadata returns error when global path doesn't exist."""
    bootstrapper.agent_config.rag_storage_path = lambda: "/nonexistent/path"
    result = bootstrapper._handle_metadata(["users"])
    assert result.status == "error"
    assert "not initialized" in result.message


def test_handle_metrics_error_when_path_missing(bootstrapper):
    """_handle_metrics returns error when global path doesn't exist."""
    bootstrapper.agent_config.rag_storage_path = lambda: "/nonexistent/path"
    result = bootstrapper._handle_metrics(["Finance.revenue"])
    assert result.status == "error"


def test_handle_reference_sql_error_when_path_missing(bootstrapper):
    """_handle_reference_sql returns error when global path doesn't exist."""
    bootstrapper.agent_config.rag_storage_path = lambda: "/nonexistent/path"
    result = bootstrapper._handle_reference_sql(["Analytics.daily_sales"])
    assert result.status == "error"
