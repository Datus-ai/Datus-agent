# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/configuration/project_config.py.

CI-level: zero external deps; all I/O is under tmp_path.
"""

import logging

import pytest
import yaml

from datus.configuration.project_config import (
    ALLOWED_KEYS,
    PROJECT_CONFIG_REL,
    ProjectOverride,
    load_project_override,
    project_config_path,
    save_project_override,
)


class TestProjectConfigPath:
    def test_path_uses_cwd_when_not_given(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert project_config_path() == tmp_path / PROJECT_CONFIG_REL

    def test_path_uses_explicit_cwd(self, tmp_path):
        assert project_config_path(str(tmp_path)) == tmp_path / PROJECT_CONFIG_REL


class TestLoadProjectOverride:
    def test_missing_file_returns_none(self, tmp_path):
        assert load_project_override(str(tmp_path)) is None

    def test_empty_file_returns_empty_override(self, tmp_path):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text("")
        result = load_project_override(str(tmp_path))
        assert isinstance(result, ProjectOverride)
        assert result.is_empty()

    def test_parse_all_three_fields(self, tmp_path):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(
            yaml.safe_dump(
                {
                    "target": "deepseek",
                    "default_datasource": "my_db",
                    "project_name": "proj_a",
                }
            )
        )
        result = load_project_override(str(tmp_path))
        assert result.target == "deepseek"
        assert result.default_datasource == "my_db"
        assert result.project_name == "proj_a"
        assert not result.is_empty()

    def test_partial_fields_leaves_others_none(self, tmp_path):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"target": "deepseek"}))
        result = load_project_override(str(tmp_path))
        assert result.target == "deepseek"
        assert result.default_datasource is None
        assert result.project_name is None

    def test_unknown_keys_warn_and_drop(self, tmp_path, caplog):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(
            yaml.safe_dump(
                {
                    "target": "deepseek",
                    "models": {"foo": "bar"},  # forbidden nested config
                    "random_key": 42,
                }
            )
        )
        with caplog.at_level(logging.WARNING):
            result = load_project_override(str(tmp_path))
        assert result.target == "deepseek"
        # Forbidden keys are not stored on the dataclass
        assert not hasattr(result, "models")
        assert not hasattr(result, "random_key")
        # Warning mentions the dropped keys
        warning_text = " ".join(r.message for r in caplog.records)
        assert "models" in warning_text
        assert "random_key" in warning_text

    def test_invalid_yaml_returns_none(self, tmp_path, caplog):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text("key: [unterminated")
        with caplog.at_level(logging.WARNING):
            result = load_project_override(str(tmp_path))
        assert result is None

    def test_non_mapping_top_level_returns_none(self, tmp_path, caplog):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(["just", "a", "list"]))
        with caplog.at_level(logging.WARNING):
            result = load_project_override(str(tmp_path))
        assert result is None


class TestSaveProjectOverride:
    def test_writes_and_creates_parent_dir(self, tmp_path):
        override = ProjectOverride(target="x", default_datasource="y", project_name="z")
        written = save_project_override(override, cwd=str(tmp_path))
        assert written == tmp_path / PROJECT_CONFIG_REL
        assert written.exists()
        assert written.parent.name == ".datus"

    def test_none_fields_are_omitted(self, tmp_path):
        override = ProjectOverride(target="x")  # default_datasource & project_name left as None
        written = save_project_override(override, cwd=str(tmp_path))
        loaded = yaml.safe_load(written.read_text())
        assert loaded == {"target": "x"}

    def test_round_trip(self, tmp_path):
        original = ProjectOverride(target="a", default_datasource="b", project_name="c")
        save_project_override(original, cwd=str(tmp_path))
        loaded = load_project_override(str(tmp_path))
        assert loaded == original

    def test_overwrites_existing(self, tmp_path):
        save_project_override(ProjectOverride(target="old"), cwd=str(tmp_path))
        save_project_override(ProjectOverride(target="new"), cwd=str(tmp_path))
        loaded = load_project_override(str(tmp_path))
        assert loaded.target == "new"


class TestAllowedKeys:
    def test_whitelist_contains_exactly_three_keys(self):
        assert ALLOWED_KEYS == frozenset({"target", "default_datasource", "project_name"})


class TestProjectOverrideDataclass:
    def test_is_empty_when_all_none(self):
        assert ProjectOverride().is_empty()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("target", "x"),
            ("default_datasource", "y"),
            ("project_name", "z"),
        ],
    )
    def test_is_not_empty_when_any_set(self, field, value):
        override = ProjectOverride(**{field: value})
        assert not override.is_empty()
