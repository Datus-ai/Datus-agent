# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from pathlib import Path
from types import SimpleNamespace

import pytest

from datus.prompts.prompt_manager import PromptManager
from datus.utils.path_manager import DatusPathManager, reset_path_manager, set_current_path_manager


def _write_template(directory: Path, template_name: str, version: str, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{template_name}_{version}.j2"
    path.write_text(content, encoding="utf-8")
    return path


def _make_manager(
    tmp_path: Path,
    *,
    path_manager: DatusPathManager | None = None,
    agent_config: object | None = None,
) -> PromptManager:
    manager = PromptManager(path_manager=path_manager, agent_config=agent_config)
    manager.default_templates_dir = tmp_path / "default_templates"
    manager.default_templates_dir.mkdir(parents=True, exist_ok=True)
    return manager


@pytest.fixture(autouse=True)
def reset_context_home():
    reset_path_manager()
    yield
    reset_path_manager()


class TestPromptManager:
    def test_user_templates_dir_uses_explicit_path_manager(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)

        assert manager.user_templates_dir == path_manager.template_dir

    def test_user_templates_dir_uses_agent_config(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        agent_config = SimpleNamespace(path_manager=path_manager)
        manager = _make_manager(tmp_path, agent_config=agent_config)

        assert manager.user_templates_dir == path_manager.template_dir

    def test_get_env_is_cached_per_home(self, tmp_path):
        manager = _make_manager(tmp_path)

        outer_token = set_current_path_manager(tmp_path / "home_a")
        env_a = manager._get_env()
        reset_path_manager(outer_token)

        inner_token = set_current_path_manager(tmp_path / "home_b")
        env_b = manager._get_env()
        reset_path_manager(inner_token)

        assert env_a is not env_b
        assert len(manager._env_cache) == 2

    def test_get_template_path_prefers_user_template(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        user_path = _write_template(manager.user_templates_dir, "greet", "1.0", "Hello from user")
        _write_template(manager.default_templates_dir, "greet", "1.0", "Hello from default")

        assert manager._get_template_path("greet", "1.0") == user_path

    def test_get_template_path_falls_back_to_default(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        default_path = _write_template(manager.default_templates_dir, "greet", "1.0", "Hello from default")

        assert manager._get_template_path("greet", "1.0") == default_path

    def test_render_template_and_get_raw_template(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "1.0", "Hello {{ name }}")

        assert manager.render_template("greet", "1.0", name="Ada") == "Hello Ada"
        assert manager.get_raw_template("greet", "1.0") == "Hello {{ name }}"

    def test_list_templates_merges_user_and_default_templates(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "default_only", "1.0", "default")
        _write_template(manager.user_templates_dir, "user_only", "1.0", "user")
        _write_template(manager.user_templates_dir, "shared", "1.0", "user shared")
        _write_template(manager.default_templates_dir, "shared", "1.0", "default shared")
        (manager.default_templates_dir / "invalid_name.j2").write_text("ignored", encoding="utf-8")

        assert manager.list_templates() == ["default_only", "shared", "user_only"]

    def test_list_template_versions_merges_and_sorts_versions(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "2.0", "default 2.0")
        _write_template(manager.default_templates_dir, "greet", "1.0", "default 1.0")
        _write_template(manager.user_templates_dir, "greet", "1.1", "user 1.1")
        _write_template(manager.user_templates_dir, "greet", "2.0", "user 2.0")

        assert manager.list_template_versions("greet") == ["1.0", "1.1", "2.0"]

    def test_get_latest_version_raises_when_missing(self, tmp_path):
        manager = _make_manager(tmp_path)

        with pytest.raises(FileNotFoundError, match="No versions found"):
            manager.get_latest_version("missing")

    def test_create_template_version_copies_from_latest_default(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "1.0", "v1")
        _write_template(manager.default_templates_dir, "greet", "1.1", "v1.1")

        manager.create_template_version("greet", "1.2")

        assert (manager.user_templates_dir / "greet_1.2.j2").read_text(encoding="utf-8") == "v1.1"

    def test_create_template_version_rejects_existing_version(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "1.0", "v1")
        _write_template(manager.user_templates_dir, "greet", "1.1", "existing")

        with pytest.raises(ValueError, match="already exists"):
            manager.create_template_version("greet", "1.1", base_version="1.0")

    def test_template_exists_handles_present_and_missing_templates(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "1.0", "hello")

        assert manager.template_exists("greet", "1.0") is True
        assert manager.template_exists("missing", "1.0") is False

    def test_get_template_info_reports_versions(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "1.0", "v1")
        _write_template(manager.default_templates_dir, "greet", "2.0", "v2")

        assert manager.get_template_info("greet") == {
            "name": "greet",
            "available_versions": ["1.0", "2.0"],
            "latest_version": "2.0",
            "total_versions": 2,
        }

    def test_copy_to_creates_user_dir_and_respects_overwrite(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "1.0", "default")

        copied_path = Path(manager.copy_to("greet", "greet_copy", "1.0"))
        copied_path.write_text("customized", encoding="utf-8")

        manager.copy_to("greet", "greet_copy", "1.0", overwrite=False)
        assert copied_path.read_text(encoding="utf-8") == "customized"

        manager.copy_to("greet", "greet_copy", "1.0", overwrite=True)
        assert copied_path.read_text(encoding="utf-8") == "default"
