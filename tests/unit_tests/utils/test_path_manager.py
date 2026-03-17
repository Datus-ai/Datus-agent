# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/utils/path_manager.py — CI tier, zero external deps."""

import threading
from pathlib import Path

import pytest

from datus.utils.path_manager import DatusPathManager, get_path_manager, reset_path_manager


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the global path manager before and after every test."""
    reset_path_manager()
    yield
    reset_path_manager()


class TestDatusPathManagerInit:
    """Tests for DatusPathManager.__init__."""

    def test_default_home_is_dot_datus(self):
        pm = DatusPathManager()
        assert pm.datus_home == Path.home() / ".datus"

    def test_custom_home_is_resolved(self, tmp_path):
        pm = DatusPathManager(datus_home=str(tmp_path))
        assert pm.datus_home == tmp_path.resolve()

    def test_tilde_expansion(self):
        pm = DatusPathManager(datus_home="~/.datus_test")
        assert "~" not in str(pm.datus_home)

    def test_update_home(self, tmp_path):
        pm = DatusPathManager()
        new_home = tmp_path / "new_datus"
        pm.update_home(str(new_home))
        assert pm.datus_home == new_home.resolve()


class TestDatusPathManagerProperties:
    """Tests for DatusPathManager directory properties."""

    @pytest.fixture
    def pm(self, tmp_path):
        return DatusPathManager(datus_home=str(tmp_path / "datus"))

    def test_conf_dir(self, pm):
        assert pm.conf_dir == pm.datus_home / "conf"

    def test_data_dir(self, pm):
        assert pm.data_dir == pm.datus_home / "data"

    def test_logs_dir(self, pm):
        assert pm.logs_dir == pm.datus_home / "logs"

    def test_sessions_dir(self, pm):
        assert pm.sessions_dir == pm.datus_home / "sessions"

    def test_template_dir(self, pm):
        assert pm.template_dir == pm.datus_home / "template"

    def test_sample_dir(self, pm):
        assert pm.sample_dir == pm.datus_home / "sample"

    def test_run_dir(self, pm):
        assert pm.run_dir == pm.datus_home / "run"

    def test_benchmark_dir(self, pm):
        assert pm.benchmark_dir == pm.datus_home / "benchmark"

    def test_save_dir(self, pm):
        assert pm.save_dir == pm.datus_home / "save"

    def test_workspace_dir(self, pm):
        assert pm.workspace_dir == pm.datus_home / "workspace"

    def test_trajectory_dir(self, pm):
        assert pm.trajectory_dir == pm.datus_home / "trajectory"

    def test_semantic_models_dir(self, pm):
        assert pm.semantic_models_dir == pm.datus_home / "semantic_models"

    def test_sql_summaries_dir(self, pm):
        assert pm.sql_summaries_dir == pm.datus_home / "sql_summaries"

    def test_ext_knowledge_dir(self, pm):
        assert pm.ext_knowledge_dir == pm.datus_home / "ext_knowledge"


class TestDatusPathManagerConfigPaths:
    """Tests for configuration file paths."""

    @pytest.fixture
    def pm(self, tmp_path):
        return DatusPathManager(datus_home=str(tmp_path / "datus"))

    def test_agent_config_path(self, pm):
        assert pm.agent_config_path() == pm.conf_dir / "agent.yml"

    def test_mcp_config_path(self, pm):
        assert pm.mcp_config_path() == pm.conf_dir / ".mcp.json"

    def test_auth_config_path(self, pm):
        assert pm.auth_config_path() == pm.conf_dir / "auth_clients.yml"

    def test_history_file_path(self, pm):
        assert pm.history_file_path() == pm.datus_home / "history"

    def test_dashboard_path(self, pm):
        assert pm.dashboard_path() == pm.datus_home / "dashboard"

    def test_pid_file_path_default(self, pm):
        assert pm.pid_file_path() == pm.run_dir / "datus-agent-api.pid"

    def test_pid_file_path_custom_service(self, pm):
        assert pm.pid_file_path("my-service") == pm.run_dir / "my-service.pid"


class TestDatusPathManagerDataPaths:
    """Tests for data/storage path methods."""

    @pytest.fixture
    def pm(self, tmp_path):
        return DatusPathManager(datus_home=str(tmp_path / "datus"))

    def test_rag_storage_path_creates_dir(self, pm):
        path = pm.rag_storage_path("test_ns")
        assert path == pm.data_dir / "datus_db_test_ns"
        assert path.exists()

    def test_sub_agent_path_creates_dir(self, pm):
        path = pm.sub_agent_path("my_agent")
        assert path == pm.data_dir / "sub_agents" / "my_agent"
        assert path.exists()

    def test_session_db_path(self, pm):
        path = pm.session_db_path("session123")
        assert path == pm.sessions_dir / "session123.db"
        assert pm.sessions_dir.exists()

    def test_semantic_model_path_creates_dir(self, pm):
        path = pm.semantic_model_path("ns1")
        assert path == pm.semantic_models_dir / "ns1"
        assert path.exists()

    def test_sql_summary_path_creates_dir(self, pm):
        path = pm.sql_summary_path("ns2")
        assert path == pm.sql_summaries_dir / "ns2"
        assert path.exists()

    def test_ext_knowledge_path_creates_dir(self, pm):
        path = pm.ext_knowledge_path("ns3")
        assert path == pm.ext_knowledge_dir / "ns3"
        assert path.exists()


class TestResolveRunDir:
    """Tests for DatusPathManager.resolve_run_dir."""

    def test_without_run_id(self, tmp_path):
        base = tmp_path / "base"
        path = DatusPathManager.resolve_run_dir(base, "myns")
        assert path == base / "myns"
        assert path.exists()

    def test_with_run_id(self, tmp_path):
        base = tmp_path / "base"
        path = DatusPathManager.resolve_run_dir(base, "myns", "20250101")
        assert path == base / "myns" / "20250101"
        assert path.exists()


class TestResolveConfigPath:
    """Tests for DatusPathManager.resolve_config_path."""

    @pytest.fixture
    def pm(self, tmp_path):
        return DatusPathManager(datus_home=str(tmp_path / "datus"))

    def test_explicit_path_exists_is_returned(self, pm, tmp_path):
        explicit = tmp_path / "explicit_agent.yml"
        explicit.write_text("config: true")
        result = pm.resolve_config_path("agent.yml", local_path=str(explicit))
        assert result == explicit

    def test_explicit_path_not_exists_falls_through(self, pm, tmp_path, monkeypatch):
        # Ensure we're in a directory that has no local conf/agent.yml
        monkeypatch.chdir(tmp_path)
        missing = str(tmp_path / "missing.yml")
        result = pm.resolve_config_path("agent.yml", local_path=missing)
        # Falls through to default conf dir
        assert result == pm.conf_dir / "agent.yml"

    def test_no_local_path_returns_default(self, pm, tmp_path, monkeypatch):
        # Ensure we're in a directory that has no local conf/agent.yml
        monkeypatch.chdir(tmp_path)
        result = pm.resolve_config_path("agent.yml")
        assert result == pm.conf_dir / "agent.yml"


class TestEnsureDirs:
    """Tests for DatusPathManager.ensure_dirs."""

    @pytest.fixture
    def pm(self, tmp_path):
        return DatusPathManager(datus_home=str(tmp_path / "datus"))

    def test_ensure_all_dirs_creates_them(self, pm):
        pm.ensure_dirs()
        for attr_name in pm._VALID_DIR_NAMES.values():
            directory = getattr(pm, attr_name)
            assert directory.exists(), f"{attr_name} should exist"

    def test_ensure_specific_dir(self, pm):
        pm.ensure_dirs("conf")
        assert pm.conf_dir.exists()

    def test_ensure_multiple_dirs(self, pm):
        pm.ensure_dirs("conf", "data", "logs")
        assert pm.conf_dir.exists()
        assert pm.data_dir.exists()
        assert pm.logs_dir.exists()

    def test_invalid_dir_name_raises_value_error(self, pm):
        with pytest.raises(ValueError, match="Invalid directory name"):
            pm.ensure_dirs("nonexistent_dir")

    def test_idempotent(self, pm):
        """Calling ensure_dirs twice does not raise."""
        pm.ensure_dirs("conf")
        pm.ensure_dirs("conf")
        assert pm.conf_dir.exists()


class TestGetPathManagerSingleton:
    """Tests for the get_path_manager singleton."""

    def test_returns_instance(self):
        pm = get_path_manager()
        assert isinstance(pm, DatusPathManager)

    def test_same_instance_on_repeated_calls(self):
        pm1 = get_path_manager()
        pm2 = get_path_manager()
        assert pm1 is pm2

    def test_reset_allows_new_instance(self, tmp_path):
        pm1 = get_path_manager()
        reset_path_manager()
        pm2 = get_path_manager(datus_home=tmp_path)
        assert pm1 is not pm2

    def test_thread_safe_initialization(self):
        """Multiple threads calling get_path_manager get the same instance."""
        instances = []

        def fetch():
            instances.append(get_path_manager())

        threads = [threading.Thread(target=fetch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = instances[0]
        assert all(inst is first for inst in instances)


class TestResetPathManager:
    """Tests for reset_path_manager."""

    def test_reset_clears_singleton(self):
        get_path_manager()
        reset_path_manager()
        from datus.utils import path_manager

        assert path_manager._path_manager is None

    def test_reset_is_thread_safe(self):
        """reset_path_manager can be called from multiple threads without error."""
        errors = []

        def do_reset():
            try:
                reset_path_manager()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_reset) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
