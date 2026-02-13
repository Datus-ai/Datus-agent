from pathlib import Path

from datus.configuration.agent_config import AgentConfig
from datus.storage.cache import StorageCache, clear_cache


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
    assert schema_first.db_path == str(tmp_path / "global")

    metrics_first = cache.metric_storage()
    metrics_second = cache.metric_storage()
    assert metrics_first is metrics_second
    assert metrics_first.db_path == str(tmp_path / "global")

    sql_first = cache.reference_sql_storage()
    sql_second = cache.reference_sql_storage()
    assert sql_first is sql_second
    assert sql_first.db_path == str(tmp_path / "global")


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
    assert first.db_path == str(tmp_path / "global")
    assert third.db_path == str(tmp_path / "global")


def test_invalidate_resets_scope(tmp_path):
    config = DummyAgentConfig(tmp_path)
    # Add scoped context for team_a to use separate storage path
    # metrics field must be a non-empty string to enable scoped storage
    config.add_sub_agent_with_scoped_context("team_a", {"metrics": "*"})
    cache = StorageCache(agent_config=config)

    original = cache.metric_storage("team_a")
    clear_cache()
    refreshed = cache.metric_storage("team_a")

    assert original is not refreshed
    assert refreshed.db_path == str(tmp_path / "sub_agents" / "team_a")
