from pathlib import Path

from datus.configuration.agent_config import AgentConfig
from datus.storage.cache import StorageCache, clear_cache
from datus.storage.conditions import build_where
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


class RecordingStorage:
    def __init__(self, path: str):
        self.path = path


def _build_cache(tmp_path):
    return StorageCache(agent_config=DummyAgentConfig(tmp_path))


def test_global_instances_are_cached(tmp_path):
    cache = _build_cache(tmp_path)

    schema_first = cache.schema_storage()
    schema_second = cache.schema_storage()
    assert schema_first is schema_second

    metrics_first = cache.metric_storage()
    metrics_second = cache.metric_storage()
    assert metrics_first is metrics_second

    sql_first = cache.reference_sql_storage()
    sql_second = cache.reference_sql_storage()
    assert sql_first is sql_second


def test_sub_agent_instances_are_cached_per_name(tmp_path):
    config = DummyAgentConfig(tmp_path)
    # Sub agents without scoped_context fall back to global storage
    # which uses the LRU cache, so all calls return the same instance
    cache = StorageCache(agent_config=config)

    first = cache.schema_storage("team_a")
    second = cache.schema_storage("team_a")
    third = cache.schema_storage("team_b")

    # Without scoped_context, all sub agents use the same global cached instance
    assert first is second
    assert first is third


def test_invalidate_resets_scope(tmp_path):
    config = DummyAgentConfig(tmp_path)
    # Add scoped context for team_a to use global storage with scope filter
    # tables field must be a non-empty string to enable scoped storage
    config.add_sub_agent_with_scoped_context("team_a", {"tables": "orders"})
    cache = StorageCache(agent_config=config)

    original = cache.schema_storage("team_a")
    clear_cache()
    refreshed = cache.schema_storage("team_a")

    assert original is not refreshed


def test_ext_knowledge_global_instances_are_cached(tmp_path):
    cache = _build_cache(tmp_path)

    first = cache.ext_knowledge_storage()
    second = cache.ext_knowledge_storage()
    assert first is second


def test_ext_knowledge_scoped_fails_close_without_subject_tree(tmp_path):
    """Scoped ext_knowledge raises when subject_tree is unavailable (fail-close)."""
    import pytest

    config = DummyAgentConfig(tmp_path)
    config.add_sub_agent_with_scoped_context("team_a", {"ext_knowledge": "Finance/*"})
    cache = StorageCache(agent_config=config)

    with pytest.raises(DatusException, match="Cannot build scope filter"):
        cache.ext_knowledge_storage("team_a")


def test_scoped_instances_are_cached_across_calls(tmp_path):
    """Scoped storage instances are cached so repeated calls return the same object."""
    config = DummyAgentConfig(tmp_path)
    config.add_sub_agent_with_scoped_context("team_a", {"tables": "orders"})
    cache = StorageCache(agent_config=config)

    first = cache.schema_storage("team_a")
    second = cache.schema_storage("team_a")
    assert first is second


def test_table_scoped_storage_has_scope_filter(tmp_path):
    """Sub-agent with tables scoped context gets a scope filter on the storage."""
    config = DummyAgentConfig(tmp_path)
    config.add_sub_agent_with_scoped_context("team_a", {"tables": "public.users"})
    cache = StorageCache(agent_config=config)

    storage = cache.schema_storage("team_a")
    # The scope filter should be set because 'tables' was specified
    assert storage._scope_filter is not None
    clause = build_where(storage._scope_filter)
    assert "users" in clause


def test_build_scope_filter_empty_value_returns_none(tmp_path):
    """_build_scope_filter returns None when scope value is empty."""
    config = DummyAgentConfig(tmp_path)
    config.add_sub_agent_with_scoped_context("team_a", {"tables": ""})
    cache = StorageCache(agent_config=config)

    # With empty tables value, sub-agent falls back to global cached storage
    storage = cache.schema_storage("team_a")
    # No scope filter should be set since tables is empty
    assert storage._scope_filter is None


def test_semantic_scoped_storage_has_scope_filter(tmp_path):
    """Sub-agent with tables scoped context for semantic storage gets a filter."""
    config = DummyAgentConfig(tmp_path)
    config.add_sub_agent_with_scoped_context("team_a", {"tables": "orders"})
    cache = StorageCache(agent_config=config)

    storage = cache.semantic_storage("team_a")
    assert storage._scope_filter is not None
    clause = build_where(storage._scope_filter)
    assert "orders" in clause
