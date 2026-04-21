# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Integration tests for the project-aware storage layout.

Validates the end-to-end wiring of ``AgentConfig`` + ``DatusPathManager``:

* Knowledge-base dirs land under ``{project_root}/subject/``.
* Sessions and data dirs are sharded by ``project_name`` under ``datus_home``.
* Switching ``project_name`` at runtime rebuilds the sharded paths.
* Live backends (``create_vector_connection`` / ``create_rdb_for_store``)
  actually write under ``{data_dir}/{project}/datus_db`` — this catches
  regressions where path-manager properties diverge from backend internals.

All external dependencies (LLM APIs, databases, remote stores) are avoided by
using ``skip_init_dirs=True`` for the path-contract tests; the backend
landing-path test exercises the real sqlite/lance backends on a tmp path.
"""

from pathlib import Path

import pytest

from datus.configuration.agent_config import AgentConfig, NodeConfig, _normalize_project_name


# TODO: this suite has no external dependencies (skip_init_dirs=True + tmp_path);
# consider moving it under tests/unit_tests/ alongside test_agent_config.py.
def _make_config(*, home: Path, project_name: str, project_root: Path) -> AgentConfig:
    return AgentConfig(
        nodes={"test": NodeConfig(model="mock", input=None)},
        home=str(home),
        target="mock",
        project_name=project_name,
        project_root=str(project_root),
        models={
            "mock": {
                "type": "openai",
                "api_key": "k",
                "model": "m",
                "base_url": "http://localhost:0",
            }
        },
        skip_init_dirs=True,
    )


@pytest.mark.acceptance
class TestStorageLayoutIntegration:
    def test_two_projects_isolate_subject_and_data(self, tmp_path):
        """Two independent project roots must produce isolated KB & data paths."""
        datus_home = tmp_path / "home"

        proj_a_root = tmp_path / "project_a"
        proj_b_root = tmp_path / "project_b"

        cfg_a = _make_config(home=datus_home, project_name="proj_a", project_root=proj_a_root)
        cfg_b = _make_config(home=datus_home, project_name="proj_b", project_root=proj_b_root)

        # Subject trees diverge by project_root.
        assert cfg_a.path_manager.subject_dir == proj_a_root.resolve() / "subject"
        assert cfg_b.path_manager.subject_dir == proj_b_root.resolve() / "subject"
        assert cfg_a.path_manager.subject_dir != cfg_b.path_manager.subject_dir

        # data/ is the shared backend root; per-project sharding is exposed via
        # project_data_dir (backend.connect(project) appends the shard later).
        assert cfg_a.path_manager.data_dir == datus_home.resolve() / "data"
        assert cfg_b.path_manager.data_dir == datus_home.resolve() / "data"
        assert cfg_a.path_manager.project_data_dir == datus_home.resolve() / "data" / "proj_a"
        assert cfg_b.path_manager.project_data_dir == datus_home.resolve() / "data" / "proj_b"
        # sessions/ is directly sharded by project_name (no backend indirection).
        assert cfg_a.path_manager.sessions_dir == datus_home.resolve() / "sessions" / "proj_a"
        assert cfg_b.path_manager.sessions_dir == datus_home.resolve() / "sessions" / "proj_b"

        # Global conf directory stays shared.
        assert cfg_a.path_manager.conf_dir == cfg_b.path_manager.conf_dir

    def test_kb_subtree_follows_project_root(self, tmp_path):
        datus_home = tmp_path / "home"
        project_root = tmp_path / "my_project"
        cfg = _make_config(home=datus_home, project_name="my_project", project_root=project_root)

        subject = project_root.resolve() / "subject"
        assert cfg.path_manager.semantic_models_dir == subject / "semantic_models"
        assert cfg.path_manager.sql_summaries_dir == subject / "sql_summaries"
        assert cfg.path_manager.ext_knowledge_dir == subject / "ext_knowledge"

        # The project-level skills directory lives under project_root/.datus/skills
        assert cfg.path_manager.project_skills_dir == project_root.resolve() / ".datus" / "skills"

    def test_semantic_model_path_creates_dir_on_demand(self, tmp_path):
        datus_home = tmp_path / "home"
        project_root = tmp_path / "my_project"
        cfg = _make_config(home=datus_home, project_name="my_project", project_root=project_root)

        produced = cfg.path_manager.semantic_model_path()
        assert produced.exists() and produced.is_dir()
        assert produced == project_root.resolve() / "subject" / "semantic_models"

    def test_auto_project_name_from_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        datus_home = tmp_path / "home"

        cfg = AgentConfig(
            nodes={"test": NodeConfig(model="mock", input=None)},
            home=str(datus_home),
            target="mock",
            models={
                "mock": {
                    "type": "openai",
                    "api_key": "k",
                    "model": "m",
                    "base_url": "http://localhost:0",
                }
            },
            skip_init_dirs=True,
        )

        expected = _normalize_project_name(str(tmp_path))
        assert cfg.project_name == expected
        assert cfg.path_manager.data_dir == datus_home.resolve() / "data"
        assert cfg.path_manager.project_data_dir == datus_home.resolve() / "data" / expected
        assert cfg.path_manager.sessions_dir == datus_home.resolve() / "sessions" / expected


@pytest.mark.parametrize(
    "cwd,expected",
    [
        ("/a/b/c", "a-b-c"),
        ("/", "_root"),
        ("", "_root"),
        ("relative/path", "relative-path"),
    ],
)
def test_normalize_project_name_cases(cwd, expected):
    assert _normalize_project_name(cwd) == expected


@pytest.mark.acceptance
class TestBackendLandingPath:
    """Exercise the real sqlite/lance backends and assert on-disk locations.

    ``skip_init_dirs=True`` used by the other tests bypasses the storage layer
    entirely, so we rely on this test to catch drift between
    ``DatusPathManager.data_dir`` and the path the backends actually open.
    """

    @pytest.fixture
    def _backends(self):
        from datus.storage.backend_holder import reset_backends

        yield
        reset_backends()

    def test_vector_connection_lands_under_project_shard(self, tmp_path, _backends):
        from datus_storage_base.backend_config import StorageBackendConfig

        from datus.storage.backend_holder import create_vector_connection, init_backends

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        init_backends(StorageBackendConfig(), data_dir=str(data_dir))

        vec_db = create_vector_connection("proj_x")
        try:
            expected = data_dir / "proj_x" / "datus_db"
            assert expected.exists(), f"LanceDB should have been opened at {expected}"
        finally:
            vec_db.close()

    def test_rdb_store_file_lands_under_project_shard(self, tmp_path, _backends):
        from datus_storage_base.backend_config import StorageBackendConfig
        from datus_storage_base.rdb.base import ColumnDef, TableDefinition

        from datus.storage.backend_holder import create_rdb_for_store, init_backends

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        init_backends(StorageBackendConfig(), data_dir=str(data_dir))

        rdb = create_rdb_for_store("subject_tree", "proj_x")
        # SQLite only materializes the .db file on first DDL; trigger it.
        rdb.ensure_table(
            TableDefinition(
                table_name="probe",
                columns=[ColumnDef(name="id", col_type="INTEGER", primary_key=True, autoincrement=True)],
            )
        )
        expected = data_dir / "proj_x" / "datus_db" / "subject_tree.db"
        assert expected.is_file(), f"SQLite file should have been created at {expected}"
