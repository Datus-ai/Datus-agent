from pathlib import Path

from datus_storage_base.conditions import build_where

from datus.configuration.agent_config import AgentConfig
from datus.storage.cache import clear_cache, create_storage_with_scope
from datus.storage.conditions import build_where
from datus.storage.ext_knowledge import ExtKnowledgeStore
from datus.storage.metric.store import MetricStorage
from datus.storage.reference_sql import ReferenceSqlStorage
from datus.storage.schema_metadata import SchemaStorage
from datus.storage.semantic_model.store import SemanticModelStorage
from datus.utils.exceptions import DatusException


class DummyAgentConfig(AgentConfig):
    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._sub_agent_configs = {}

    def rag_storage_path(self) -> str:
        return str(self._base_dir / "global")

    def sub_agent_storage_path(self, sub_agent_name: str) -> str:
        return str(self._base_dir / "sub_agents" / sub_agent_name)

    def sub_agent_config(self, sub_agent_name: str) -> dict:
        """Return sub agent config. Returns dict with scoped_context if configured."""
        return self._sub_agent_configs.get(sub_agent_name, {})

    def add_sub_agent_with_scoped_context(self, sub_agent_name: str, scoped_attrs: dict):
        """Add a sub agent with scoped context configuration."""
        self._sub_agent_configs[sub_agent_name] = {"scoped_context": scoped_attrs}


def test_global_instances_are_cached(tmp_path):
    config = DummyAgentConfig(tmp_path)

    schema_first = create_storage_with_scope(SchemaStorage, config, "database", "tables")
    schema_second = create_storage_with_scope(SchemaStorage, config, "database", "tables")
    assert schema_first is schema_second

    metrics_first = create_storage_with_scope(MetricStorage, config, "metric", "metrics")
    metrics_second = create_storage_with_scope(MetricStorage, config, "metric", "metrics")
    assert metrics_first is metrics_second

    sql_first = create_storage_with_scope(ReferenceSqlStorage, config, "reference_sql", "sqls")
    sql_second = create_storage_with_scope(ReferenceSqlStorage, config, "reference_sql", "sqls")
    assert sql_first is sql_second


def test_sub_agent_instances_are_cached_per_name(tmp_path):
    config = DummyAgentConfig(tmp_path)
    # Sub agents without scoped_context fall back to global storage
    # which uses the LRU cache, so all calls return the same instance

    first = create_storage_with_scope(SchemaStorage, config, "database", "tables", "team_a")
    second = create_storage_with_scope(SchemaStorage, config, "database", "tables", "team_a")
    third = create_storage_with_scope(SchemaStorage, config, "database", "tables", "team_b")

    # Without scoped_context, all sub agents use the same global cached instance
    assert first is second
    assert first is third


def test_invalidate_resets_scope(tmp_path):
    config = DummyAgentConfig(tmp_path)
    # Add scoped context for team_a to use global storage with scope filter
    # tables field must be a non-empty string to enable scoped storage
    config.add_sub_agent_with_scoped_context("team_a", {"tables": "orders"})

    original = create_storage_with_scope(SchemaStorage, config, "database", "tables", "team_a")
    clear_cache()

    refreshed = create_storage_with_scope(SchemaStorage, config, "database", "tables", "team_a")

    assert original is not refreshed


def test_ext_knowledge_global_instances_are_cached(tmp_path):
    config = DummyAgentConfig(tmp_path)

    first = create_storage_with_scope(ExtKnowledgeStore, config, "ext_knowledge", "ext_knowledge")
    second = create_storage_with_scope(ExtKnowledgeStore, config, "ext_knowledge", "ext_knowledge")
    assert first is second


def test_ext_knowledge_scoped_fails_close_without_subject_tree(tmp_path):
    """Scoped ext_knowledge raises when subject_tree is unavailable (fail-close)."""
    import pytest

    config = DummyAgentConfig(tmp_path)
    config.add_sub_agent_with_scoped_context("team_a", {"ext_knowledge": "Finance/*"})

    with pytest.raises(DatusException, match="Cannot build scope filter"):
        create_storage_with_scope(ExtKnowledgeStore, config, "ext_knowledge", "ext_knowledge", "team_a")


def test_scoped_instances_are_cached_across_calls(tmp_path):
    """Scoped storage instances are cached so repeated calls return the same object."""
    config = DummyAgentConfig(tmp_path)
    config.add_sub_agent_with_scoped_context("team_a", {"tables": "orders"})

    first = create_storage_with_scope(SchemaStorage, config, "database", "tables", "team_a")
    second = create_storage_with_scope(SchemaStorage, config, "database", "tables", "team_a")
    assert first is second


def test_table_scoped_storage_has_scope_filter(tmp_path):
    """Sub-agent with tables scoped context gets a scope filter on the storage."""
    config = DummyAgentConfig(tmp_path)
    config.add_sub_agent_with_scoped_context("team_a", {"tables": "public.users"})

    storage = create_storage_with_scope(SchemaStorage, config, "database", "tables", "team_a")
    # The scope filter should be set because 'tables' was specified
    assert storage._scope_filter is not None
    clause = build_where(storage._scope_filter)
    assert "users" in clause


def test_build_scope_filter_empty_value_returns_none(tmp_path):
    """_build_scope_filter returns None when scope value is empty."""
    config = DummyAgentConfig(tmp_path)
    config.add_sub_agent_with_scoped_context("team_a", {"tables": ""})

    # With empty tables value, sub-agent falls back to global cached storage
    storage = create_storage_with_scope(SchemaStorage, config, "database", "tables", "team_a")
    # No scope filter should be set since tables is empty
    assert storage._scope_filter is None


def test_semantic_scoped_storage_has_scope_filter(tmp_path):
    """Sub-agent with tables scoped context for semantic storage gets a filter."""
    config = DummyAgentConfig(tmp_path)
    config.add_sub_agent_with_scoped_context("team_a", {"tables": "orders"})

    storage = create_storage_with_scope(SemanticModelStorage, config, "semantic_model", "tables", "team_a")
    assert storage._scope_filter is not None
    clause = build_where(storage._scope_filter)
    assert "orders" in clause
