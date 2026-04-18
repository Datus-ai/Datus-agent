# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/utils/path_manager.py — CI tier, zero external deps."""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from datus.utils.path_manager import DatusPathManager, get_path_manager, reset_path_manager, set_current_path_manager


@pytest.fixture(autouse=True)
def reset_defaults():
    """Reset path-manager defaults before and after every test."""
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

    def test_default_project_name_is_empty(self, tmp_path):
        pm = DatusPathManager(datus_home=str(tmp_path))
        assert pm.project_name == ""

    def test_project_name_and_root_preserved(self, tmp_path):
        project_root = tmp_path / "proj"
        pm = DatusPathManager(
            datus_home=str(tmp_path / "home"),
            project_name="-tmp-proj",
            project_root=str(project_root),
        )
        assert pm.project_name == "-tmp-proj"
        assert pm.project_root == project_root.resolve()

    def test_project_root_defaults_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pm = DatusPathManager(datus_home=str(tmp_path / "home"))
        assert pm.project_root == tmp_path.resolve()

    def test_knowledge_base_home_kwarg_no_longer_accepted(self, tmp_path):
        """The legacy kwarg was removed; passing it should fail loudly."""
        with pytest.raises(TypeError):
            DatusPathManager(datus_home=str(tmp_path / "datus"), knowledge_base_home="")


class TestDatusPathManagerProperties:
    """Tests for DatusPathManager directory properties."""

    @pytest.fixture
    def pm(self, tmp_path):
        # Bind to a fixed project_name so sharded dirs are deterministic in tests.
        return DatusPathManager(
            datus_home=str(tmp_path / "datus"),
            project_name="proj",
            project_root=str(tmp_path / "project"),
        )

    @pytest.mark.parametrize(
        "attr,suffix",
        [
            ("conf_dir", "conf"),
            ("logs_dir", "logs"),
            ("template_dir", "template"),
            ("sample_dir", "sample"),
            ("run_dir", "run"),
            ("benchmark_dir", "benchmark"),
            ("save_dir", "save"),
            ("workspace_dir", "workspace"),
            ("trajectory_dir", "trajectory"),
        ],
    )
    def test_shared_directory_property(self, pm, attr, suffix):
        """Global, un-sharded directories stay under datus_home."""
        assert getattr(pm, attr) == pm.datus_home / suffix

    def test_sessions_dir_sharded_by_project_name(self, pm):
        assert pm.sessions_dir == pm.datus_home / "sessions" / "proj"

    def test_data_dir_is_project_agnostic(self, pm):
        """data_dir is the storage-backend root; each backend owns its project isolation."""
        assert pm.data_dir == pm.datus_home / "data"

    def test_project_data_dir_sharded_by_project_name(self, pm):
        """project_data_dir is the project-scoped helper for non-backend callers."""
        assert pm.project_data_dir == pm.datus_home / "data" / "proj"

    def test_data_dir_is_project_agnostic_without_project(self, tmp_path):
        """data_dir is global — it does not depend on project_name being set."""
        pm = DatusPathManager(datus_home=str(tmp_path / "home"))
        assert pm.data_dir == pm.datus_home / "data"

    def test_sessions_dir_requires_project_name(self, tmp_path):
        """sessions_dir raises when project_name is not configured."""
        from datus.utils.exceptions import DatusException

        pm = DatusPathManager(datus_home=str(tmp_path / "home"))
        with pytest.raises(DatusException):
            _ = pm.sessions_dir

    def test_project_data_dir_requires_project_name(self, tmp_path):
        """project_data_dir raises when project_name is not configured."""
        from datus.utils.exceptions import DatusException

        pm = DatusPathManager(datus_home=str(tmp_path / "home"))
        with pytest.raises(DatusException):
            _ = pm.project_data_dir

    def test_subject_dir_anchored_to_project_root(self, pm, tmp_path):
        assert pm.subject_dir == (tmp_path / "project").resolve() / "subject"

    def test_kb_dirs_live_under_subject(self, pm):
        assert pm.semantic_models_dir == pm.subject_dir / "semantic_models"
        assert pm.sql_summaries_dir == pm.subject_dir / "sql_summaries"
        assert pm.ext_knowledge_dir == pm.subject_dir / "ext_knowledge"

    def test_project_skills_dir(self, pm, tmp_path):
        assert pm.project_skills_dir == (tmp_path / "project").resolve() / ".datus" / "skills"


class TestDatusPathManagerConfigPaths:
    """Tests for configuration file paths."""

    @pytest.fixture
    def pm(self, tmp_path):
        return DatusPathManager(datus_home=str(tmp_path / "datus"))

    @pytest.mark.parametrize(
        "method,args,expected_parts",
        [
            ("agent_config_path", [], ("conf_dir", "agent.yml")),
            ("mcp_config_path", [], ("conf_dir", ".mcp.json")),
            ("auth_config_path", [], ("conf_dir", "auth_clients.yml")),
            ("history_file_path", [], ("datus_home", "history")),
            ("dashboard_path", [], ("datus_home", "dashboard")),
            ("pid_file_path", [], ("run_dir", "datus-agent-api.pid")),
            ("pid_file_path", ["my-service"], ("run_dir", "my-service.pid")),
        ],
    )
    def test_config_path_method(self, pm, method, args, expected_parts):
        base_attr, filename = expected_parts
        expected = getattr(pm, base_attr) / filename
        assert getattr(pm, method)(*args) == expected


class TestDatusPathManagerDataPaths:
    """Tests for data/storage path methods."""

    @pytest.fixture
    def pm(self, tmp_path):
        return DatusPathManager(
            datus_home=str(tmp_path / "datus"),
            project_name="proj",
            project_root=str(tmp_path / "project"),
        )

    def test_rag_storage_path_creates_dir(self, pm):
        path = pm.rag_storage_path()
        # rag_storage_path is a non-backend helper; it lands under
        # project_data_dir (e.g. document/ co-located paths).
        assert path == pm.project_data_dir / "datus_db"
        assert path.exists()

    def test_session_db_path(self, pm):
        path = pm.session_db_path("session123")
        assert path == pm.sessions_dir / "session123.db"
        assert pm.sessions_dir.exists()

    def test_semantic_model_path_creates_dir(self, pm):
        path = pm.semantic_model_path()
        assert path == pm.semantic_models_dir
        assert path.exists()

    def test_sql_summary_path_creates_dir(self, pm):
        path = pm.sql_summary_path()
        assert path == pm.sql_summaries_dir
        assert path.exists()

    def test_ext_knowledge_path_creates_dir(self, pm):
        path = pm.ext_knowledge_path()
        assert path == pm.ext_knowledge_dir
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
        return DatusPathManager(
            datus_home=str(tmp_path / "datus"),
            project_name="proj",
            project_root=str(tmp_path / "project"),
        )

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

    def test_ensure_subject_tree_dirs(self, pm):
        pm.ensure_dirs("subject", "semantic_models", "sql_summaries", "ext_knowledge")
        assert pm.subject_dir.exists()
        assert pm.semantic_models_dir.exists()
        assert pm.sql_summaries_dir.exists()
        assert pm.ext_knowledge_dir.exists()

    def test_ensure_templates_creates_template_dir_and_copies_defaults(self, pm):
        with patch("datus.utils.resource_utils.copy_data_file") as mock_copy:
            pm.ensure_templates()

        assert pm.template_dir.exists()
        mock_copy.assert_called_once_with(
            resource_path="prompts/prompt_templates",
            target_dir=pm.template_dir,
            replace=False,
        )


class TestGetPathManager:
    """Tests for the get_path_manager factory."""

    def test_returns_instance(self):
        pm = get_path_manager()
        assert isinstance(pm, DatusPathManager)

    def test_repeated_calls_return_fresh_instances(self):
        pm1 = get_path_manager()
        pm2 = get_path_manager()
        assert pm1 is not pm2
        assert pm1.datus_home == pm2.datus_home

    def test_explicit_home_is_respected(self, tmp_path):
        pm = get_path_manager(datus_home=tmp_path)
        assert pm.datus_home == tmp_path.resolve()

    def test_context_local_home_is_used(self, tmp_path):
        set_current_path_manager(tmp_path)
        pm = get_path_manager()
        assert pm.datus_home == tmp_path.resolve()

    def test_set_current_path_manager_accepts_path_manager_instance(self, tmp_path):
        current = DatusPathManager(tmp_path / "tenant_home")
        set_current_path_manager(current)
        pm = get_path_manager()
        assert pm.datus_home == current.datus_home

    def test_set_current_path_manager_accepts_agent_config(self, tmp_path):
        agent_config = SimpleNamespace(path_manager=DatusPathManager(tmp_path / "agent_home"))
        set_current_path_manager(agent_config=agent_config)
        pm = get_path_manager()
        assert pm.datus_home == agent_config.path_manager.datus_home

    def test_path_manager_argument_has_highest_precedence(self, tmp_path):
        explicit_pm = DatusPathManager(tmp_path / "explicit_home")
        agent_config = SimpleNamespace(path_manager=DatusPathManager(tmp_path / "agent_home"))
        set_current_path_manager(tmp_path / "context_home")

        pm = get_path_manager(
            datus_home=tmp_path / "arg_home",
            path_manager=explicit_pm,
            agent_config=agent_config,
        )

        assert pm is explicit_pm

    def test_agent_config_has_precedence_over_explicit_home_and_context(self, tmp_path):
        agent_pm = DatusPathManager(tmp_path / "agent_home")
        agent_config = SimpleNamespace(path_manager=agent_pm)
        set_current_path_manager(tmp_path / "context_home")

        pm = get_path_manager(datus_home=tmp_path / "arg_home", agent_config=agent_config)

        assert pm is agent_pm

    def test_factory_is_safe_to_call_from_multiple_threads(self):
        """Multiple threads can resolve path managers without raising."""
        instances = []
        errors = []

        def fetch():
            try:
                instances.append(get_path_manager())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=fetch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(instances) == 10


class TestResetPathManager:
    """Tests for reset_path_manager."""

    def test_reset_clears_context_local_home(self, tmp_path):
        set_current_path_manager(tmp_path)
        reset_path_manager()
        from datus.utils import path_manager

        assert path_manager._current_path_manager.get() is None

    def test_reset_is_safe_from_multiple_threads(self):
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

    def test_reset_with_token_restores_previous_context(self, tmp_path):
        outer_token = set_current_path_manager(tmp_path / "outer_home")
        inner_token = set_current_path_manager(tmp_path / "inner_home")

        reset_path_manager(inner_token)
        assert get_path_manager().datus_home == (tmp_path / "outer_home").resolve()

        reset_path_manager(outer_token)
        assert get_path_manager().datus_home == (Path.home() / ".datus").resolve()

    def test_context_var_preserves_project_shard_round_trip(self, tmp_path):
        """Storing a DatusPathManager via ContextVar preserves project_name sharding."""
        project_root = tmp_path / "proj"
        pm = DatusPathManager(
            datus_home=str(tmp_path / "home"),
            project_name="-tmp-proj",
            project_root=str(project_root),
        )

        token = set_current_path_manager(pm)
        try:
            retrieved = get_path_manager()
            assert retrieved.project_name == "-tmp-proj"
            assert retrieved.subject_dir == project_root.resolve() / "subject"
            assert retrieved.data_dir == pm.datus_home / "data"
            assert retrieved.project_data_dir == pm.datus_home / "data" / "-tmp-proj"
            assert retrieved.sessions_dir == pm.datus_home / "sessions" / "-tmp-proj"
        finally:
            reset_path_manager(token)
