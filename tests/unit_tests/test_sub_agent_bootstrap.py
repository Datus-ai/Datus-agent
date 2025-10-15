import os
from typing import Any, Dict

import pytest

from datus.schemas.agent_models import ScopedContext, ScopedContextLists, SubAgentConfig
from datus.storage.lancedb_conditions import build_where
from datus.storage.sub_agent_kb_bootstrap import SubAgentBootstrapper


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
    )
    lists = context.as_lists()
    assert lists.tables == ["orders", "customers"]
    assert lists.metrics == ["revenue", "profit"]
    assert lists.sqls == ["daily_sales", "monthly_sales"]
    assert lists.any()


def test_scoped_context_lists_any_returns_false_when_empty():
    lists = ScopedContextLists()
    assert not lists.any()


def test_metadata_condition_applies_defaults_and_wildcards(bootstrapper):
    # Default database/schema applied
    condition = bootstrapper._metadata_condition_for_token("sales")
    clause = build_where(condition)
    assert "database_name = 'sales'" in clause

    # Schema/table wildcards honored
    condition = bootstrapper._metadata_condition_for_token("sales.orders*")
    clause = build_where(condition)
    assert "database_name = 'sales'" in clause
    assert "table_name LIKE 'orders%'" in clause


def test_hierarchical_condition_partial_segments_use_prefix(bootstrapper):
    tokens = ["finance", "finance.layer1", "ops.layer1.layer2.metric*"]
    conditions, invalid = bootstrapper._hierarchical_conditions(tokens, ("domain", "layer1", "layer2", "name"))
    assert not invalid
    clauses = [build_where(node) for _, node in conditions]
    assert "domain = 'finance'" in clauses[0]
    assert "layer1 = 'layer1'" in clauses[1]
    assert "name like 'metric%'" in clauses[2].lower()
