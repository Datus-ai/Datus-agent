# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Integration tests for the project-aware storage layout.

Validates the end-to-end wiring of ``AgentConfig`` + ``DatusPathManager``:

* Knowledge-base dirs land under ``{project_root}/subject/``.
* Sessions and data dirs are sharded by ``project_name`` under ``datus_home``.
* Switching ``project_name`` at runtime rebuilds the sharded paths.

All external dependencies (LLM APIs, databases, remote stores) are avoided by
using ``skip_init_dirs=True``; this is a pure filesystem-layout contract test.
"""

from pathlib import Path

import pytest

from datus.configuration.agent_config import AgentConfig, NodeConfig, _normalize_project_name


# todo 不依赖外部的测试可以放到ut
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

        # data/ and sessions/ diverge by project_name under the shared home.
        assert cfg_a.path_manager.data_dir == datus_home.resolve() / "data" / "proj_a"
        assert cfg_b.path_manager.data_dir == datus_home.resolve() / "data" / "proj_b"
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

    def test_switching_project_name_rebuilds_shards(self, tmp_path):
        datus_home = tmp_path / "home"
        project_root = tmp_path / "my_project"
        cfg = _make_config(home=datus_home, project_name="initial", project_root=project_root)

        cfg.project_name = "rotated"
        assert cfg.path_manager.data_dir == datus_home.resolve() / "data" / "rotated"
        assert cfg.path_manager.sessions_dir == datus_home.resolve() / "sessions" / "rotated"
        # Subject tree stays pinned to project_root — sharding happens on datus_home.
        assert cfg.path_manager.subject_dir == project_root.resolve() / "subject"

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
        assert cfg.path_manager.data_dir == datus_home.resolve() / "data" / expected
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
