import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from datus.schemas.agent_models import ScopedContext, ScopedContextLists, SubAgentConfig
from datus.storage.lancedb_conditions import build_where
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
