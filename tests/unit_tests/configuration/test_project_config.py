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
        assert result.language is None

    def test_parse_language_field(self, tmp_path):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"language": "zh"}))
        result = load_project_override(str(tmp_path))
        assert result.language == "zh"
        assert result.target is None

    @pytest.mark.parametrize("value", ["off", "minimal", "low", "medium", "high"])
    def test_parse_reasoning_effort_field(self, tmp_path, value):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"reasoning_effort": value}))
        result = load_project_override(str(tmp_path))
        assert result.reasoning_effort == value

    def test_invalid_reasoning_effort_dropped_with_warning(self, tmp_path, caplog):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"reasoning_effort": "nuclear"}))
        with caplog.at_level(logging.WARNING):
            result = load_project_override(str(tmp_path))
        assert result.reasoning_effort is None
        warning_text = " ".join(r.message for r in caplog.records)
        assert "nuclear" in warning_text

    def test_reasoning_effort_case_insensitive(self, tmp_path):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"reasoning_effort": "HIGH"}))
        result = load_project_override(str(tmp_path))
        assert result.reasoning_effort == "high"

    def test_non_string_reasoning_effort_dropped(self, tmp_path, caplog):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"reasoning_effort": 3}))
        with caplog.at_level(logging.WARNING):
            result = load_project_override(str(tmp_path))
        assert result.reasoning_effort is None

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

    def test_round_trip_with_language(self, tmp_path):
        original = ProjectOverride(target="a", language="zh")
        save_project_override(original, cwd=str(tmp_path))
        loaded = load_project_override(str(tmp_path))
        assert loaded.language == "zh"
        assert loaded.target == "a"

    def test_language_none_omitted_from_yaml(self, tmp_path):
        override = ProjectOverride(target="x", language=None)
        written = save_project_override(override, cwd=str(tmp_path))
        loaded = yaml.safe_load(written.read_text())
        assert "language" not in loaded

    def test_round_trip_with_reasoning_effort(self, tmp_path):
        original = ProjectOverride(target="a", reasoning_effort="high")
        save_project_override(original, cwd=str(tmp_path))
        loaded = load_project_override(str(tmp_path))
        assert loaded.reasoning_effort == "high"
        assert loaded.target == "a"

    def test_reasoning_effort_none_omitted_from_yaml(self, tmp_path):
        override = ProjectOverride(target="x")
        written = save_project_override(override, cwd=str(tmp_path))
        loaded = yaml.safe_load(written.read_text())
        assert "reasoning_effort" not in loaded

    def test_overwrites_existing(self, tmp_path):
        save_project_override(ProjectOverride(target="old"), cwd=str(tmp_path))
        save_project_override(ProjectOverride(target="new"), cwd=str(tmp_path))
        loaded = load_project_override(str(tmp_path))
        assert loaded.target == "new"


class TestAllowedKeys:
    def test_whitelist_contains_expected_keys(self):
        assert ALLOWED_KEYS == frozenset(
            {
                "target",
                "default_datasource",
                "dashboard",
                "scheduler",
                "semantic",
                "project_name",
                "language",
                "reasoning_effort",
                "bash_allow",
            }
        )


class TestProjectOverrideDataclass:
    def test_is_empty_when_all_none(self):
        assert ProjectOverride().is_empty()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("target", "x"),
            ("default_datasource", "y"),
            ("dashboard", "superset"),
            ("scheduler", "airflow"),
            ("semantic", "metricflow"),
            ("project_name", "z"),
            ("language", "zh"),
            ("reasoning_effort", "high"),
        ],
    )
    def test_is_not_empty_when_any_set(self, field, value):
        override = ProjectOverride(**{field: value})
        assert not override.is_empty()


class TestServiceDefaultFields:
    """``dashboard`` / ``scheduler`` / ``semantic`` overrides — project-level
    pins for the three service sections. Loaded by
    ``_apply_project_override`` and surfaced via
    ``AgentConfig.active_dashboard()`` / ``active_scheduler()`` /
    ``active_semantic()``."""

    def test_load_all_three_service_pins(self, tmp_path):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(
            yaml.safe_dump({"dashboard": "superset_prod", "scheduler": "airflow", "semantic": "metricflow"})
        )
        result = load_project_override(str(tmp_path))
        assert result.dashboard == "superset_prod"
        assert result.scheduler == "airflow"
        assert result.semantic == "metricflow"
        assert not result.is_empty()

    def test_load_dashboard_and_scheduler(self, tmp_path):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"dashboard": "superset_prod", "scheduler": "airflow"}))
        result = load_project_override(str(tmp_path))
        assert result.dashboard == "superset_prod"
        assert result.scheduler == "airflow"
        assert result.semantic is None
        assert not result.is_empty()

    def test_blank_semantic_collapses_to_none(self, tmp_path):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"semantic": "   "}))
        result = load_project_override(str(tmp_path))
        assert result.semantic is None

    def test_non_string_semantic_dropped(self, tmp_path, caplog):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"semantic": 42}))
        with caplog.at_level(logging.WARNING):
            result = load_project_override(str(tmp_path))
        assert result.semantic is None
        warning_text = " ".join(r.message for r in caplog.records)
        assert "semantic" in warning_text

    def test_save_round_trip_with_semantic(self, tmp_path):
        original = ProjectOverride(dashboard="superset", scheduler="airflow", semantic="metricflow")
        save_project_override(original, cwd=str(tmp_path))
        loaded = load_project_override(str(tmp_path))
        assert loaded == original

    def test_blank_dashboard_collapses_to_none(self, tmp_path):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"dashboard": "   "}))
        result = load_project_override(str(tmp_path))
        assert result.dashboard is None

    def test_non_string_scheduler_dropped(self, tmp_path, caplog):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"scheduler": 42}))
        with caplog.at_level(logging.WARNING):
            result = load_project_override(str(tmp_path))
        assert result.scheduler is None
        warning_text = " ".join(r.message for r in caplog.records)
        assert "scheduler" in warning_text

    def test_save_round_trip(self, tmp_path):
        original = ProjectOverride(dashboard="superset", scheduler="airflow")
        save_project_override(original, cwd=str(tmp_path))
        loaded = load_project_override(str(tmp_path))
        assert loaded == original

    def test_save_omits_none_service_fields(self, tmp_path):
        override = ProjectOverride(dashboard="only_dash")
        written = save_project_override(override, cwd=str(tmp_path))
        loaded = yaml.safe_load(written.read_text())
        assert "dashboard" in loaded
        assert "scheduler" not in loaded


class TestBashAllow:
    """Project-level bash_allow parsing and text-level appending."""

    def _write(self, tmp_path, content: str):
        path = tmp_path / PROJECT_CONFIG_REL
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_parse_bash_allow_list(self, tmp_path):
        self._write(tmp_path, yaml.safe_dump({"bash_allow": ["make:*", "git push:*"]}))
        result = load_project_override(str(tmp_path))
        assert result.bash_allow == ["make:*", "git push:*"]

    def test_non_list_bash_allow_dropped(self, tmp_path):
        self._write(tmp_path, yaml.safe_dump({"bash_allow": "make:*"}))
        result = load_project_override(str(tmp_path))
        assert result.bash_allow is None

    def test_non_string_entries_dropped(self, tmp_path):
        self._write(tmp_path, yaml.safe_dump({"bash_allow": ["make:*", 42, ""]}))
        result = load_project_override(str(tmp_path))
        assert result.bash_allow == ["make:*"]

    def test_append_creates_file(self, tmp_path):
        from datus.configuration.project_config import append_project_bash_allow

        append_project_bash_allow("make:*", str(tmp_path))
        result = load_project_override(str(tmp_path))
        assert result.bash_allow == ["make:*"]

    def test_append_to_file_without_key(self, tmp_path):
        from datus.configuration.project_config import append_project_bash_allow

        self._write(tmp_path, "# my project config\nproject_name: proj_a\n")
        append_project_bash_allow("uv run:*", str(tmp_path))
        result = load_project_override(str(tmp_path))
        assert result.bash_allow == ["uv run:*"]
        assert result.project_name == "proj_a"
        # comments preserved
        text = (tmp_path / PROJECT_CONFIG_REL).read_text()
        assert "# my project config" in text

    def test_append_to_existing_key_preserves_entries_and_comments(self, tmp_path):
        from datus.configuration.project_config import append_project_bash_allow

        self._write(
            tmp_path,
            '# header comment\nbash_allow:\n  - "make:*"\nproject_name: proj_a\n',
        )
        append_project_bash_allow("git push:*", str(tmp_path))
        result = load_project_override(str(tmp_path))
        assert sorted(result.bash_allow) == ["git push:*", "make:*"]
        assert result.project_name == "proj_a"
        assert "# header comment" in (tmp_path / PROJECT_CONFIG_REL).read_text()

    def test_append_is_idempotent(self, tmp_path):
        from datus.configuration.project_config import append_project_bash_allow

        append_project_bash_allow("make:*", str(tmp_path))
        append_project_bash_allow("make:*", str(tmp_path))
        result = load_project_override(str(tmp_path))
        assert result.bash_allow == ["make:*"]

    def test_append_empty_pattern_raises(self, tmp_path):
        from datus.configuration.project_config import append_project_bash_allow
        from datus.utils.exceptions import DatusException

        with pytest.raises(DatusException):
            append_project_bash_allow("   ", str(tmp_path))

    def test_append_pattern_with_quote_does_not_corrupt_file(self, tmp_path):
        """A pattern containing ``"`` (or a trailing backslash) must be
        escaped — an invalid YAML line would silently drop EVERY override
        in the file, not just ``bash_allow``."""
        from datus.configuration.project_config import append_project_bash_allow

        self._write(tmp_path, "project_name: proj_a\n")
        append_project_bash_allow('grep:"quoted"', str(tmp_path))
        append_project_bash_allow("find:*\\", str(tmp_path))
        result = load_project_override(str(tmp_path))
        # Later appends insert right after the key line, hence the order.
        assert result.bash_allow == ["find:*\\", 'grep:"quoted"']
        # The rest of the file still parses.
        assert result.project_name == "proj_a"

    def test_save_round_trips_bash_allow(self, tmp_path):
        override = ProjectOverride(bash_allow=["make:*"])
        save_project_override(override, str(tmp_path))
        result = load_project_override(str(tmp_path))
        assert result.bash_allow == ["make:*"]
