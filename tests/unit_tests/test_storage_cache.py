from pathlib import Path

from datus.configuration.agent_config import AgentConfig
from datus.storage.cache import StorageCache


class DummyAgentConfig(AgentConfig):
    def __init__(self, base_dir: Path):
        self._base_dir = base_dir

    def rag_storage_path(self) -> str:
        return str(self._base_dir / "global")

    def sub_agent_storage_path(self, sub_agent_name: str) -> str:
        return str(self._base_dir / "sub_agents" / sub_agent_name)


class RecordingStorage:
    def __init__(self, path: str):
        self.path = path


def _build_cache(tmp_path):
    return StorageCache(agent_config=DummyAgentConfig(tmp_path))


def test_global_instances_are_cached(tmp_path):
    cache = _build_cache(tmp_path)

    schema_first = cache.schema_rag()
    schema_second = cache.schema_rag()
    assert schema_first is schema_second
    assert schema_first.path == str(tmp_path / "global")

    metrics_first = cache.metrics_rag()
    metrics_second = cache.metrics_rag()
    assert metrics_first is metrics_second
    assert metrics_first.path == str(tmp_path / "global")

    sql_first = cache.reference_sql_rag()
    sql_second = cache.reference_sql_rag()
    assert sql_first is sql_second
    assert sql_first.path == str(tmp_path / "global")


def test_sub_agent_instances_are_cached_per_name(tmp_path):
    cache = _build_cache(tmp_path)

    first = cache.schema_rag("team_a")
    second = cache.schema_rag("team_a")
    third = cache.schema_rag("team_b")

    assert first is second
    assert first is not third
    assert first.path == str(tmp_path / "sub_agents" / "team_a")
    assert third.path == str(tmp_path / "sub_agents" / "team_b")


def test_invalidate_resets_scope(tmp_path):
    cache = _build_cache(tmp_path)

    original = cache.metrics_rag("team_a")
    cache.invalidate("team_a")
    refreshed = cache.metrics_rag("team_a")

    assert original is not refreshed
    assert refreshed.path == str(tmp_path / "sub_agents" / "team_a")
