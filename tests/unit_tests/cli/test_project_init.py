# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/project_init.py.

CI-level: no TTY / LLM / DB dependencies; select_choice + Prompt.ask are mocked.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from datus.cli.project_init import run_project_init
from datus.configuration.project_config import load_project_override
from datus.utils.exceptions import DatusException


def _make_base_config(models, datasources, target="", default_database=""):
    """Build a minimal AgentConfig stand-in with the attributes the wizard reads."""
    cfg = MagicMock()
    cfg.models = models
    cfg.target = target
    cfg.services = SimpleNamespace(
        datasources={name: SimpleNamespace(type=t) for name, t in datasources.items()},
        default_database=default_database,
    )
    return cfg


class TestRunProjectInit:
    def test_happy_path_writes_all_three_fields(self, tmp_path):
        base = _make_base_config(
            models={"openai": {}, "deepseek": {}},
            datasources={"db1": "sqlite", "db2": "duckdb"},
            target="openai",
            default_database="db1",
        )
        with (
            patch("datus.cli.project_init.select_choice", side_effect=["deepseek", "db2"]),
            patch("datus.cli.project_init.Prompt.ask", return_value="my_proj"),
        ):
            override = run_project_init(base, cwd=str(tmp_path))
        assert override.target == "deepseek"
        assert override.default_database == "db2"
        assert override.project_name == "my_proj"
        # File persisted with those values
        reloaded = load_project_override(str(tmp_path))
        assert reloaded.target == "deepseek"
        assert reloaded.default_database == "db2"
        assert reloaded.project_name == "my_proj"

    def test_uses_base_target_as_default(self, tmp_path):
        base = _make_base_config(
            models={"openai": {}, "deepseek": {}},
            datasources={"db1": "sqlite"},
            target="deepseek",
            default_database="db1",
        )
        # select_choice returning empty string simulates Ctrl+C → fall back to default
        with (
            patch("datus.cli.project_init.select_choice", side_effect=["", ""]),
            patch("datus.cli.project_init.Prompt.ask", return_value="_my_cwd_"),
        ):
            override = run_project_init(base, cwd=str(tmp_path))
        assert override.target == "deepseek"
        assert override.default_database == "db1"

    def test_no_models_raises(self, tmp_path):
        base = _make_base_config(
            models={},
            datasources={"db1": "sqlite"},
        )
        with pytest.raises(DatusException) as exc:
            run_project_init(base, cwd=str(tmp_path))
        assert "models" in str(exc.value)

    def test_no_datasources_raises(self, tmp_path):
        base = _make_base_config(
            models={"openai": {}},
            datasources={},
        )
        with pytest.raises(DatusException) as exc:
            run_project_init(base, cwd=str(tmp_path))
        assert "datasources" in str(exc.value)

    def test_invalid_project_name_reprompts(self, tmp_path):
        base = _make_base_config(
            models={"openai": {}},
            datasources={"db1": "sqlite"},
            target="openai",
            default_database="db1",
        )
        with (
            patch("datus.cli.project_init.select_choice", side_effect=["openai", "db1"]),
            # First attempt has forbidden space; second is valid
            patch("datus.cli.project_init.Prompt.ask", side_effect=["bad name with spaces", "good_name"]),
        ):
            override = run_project_init(base, cwd=str(tmp_path))
        assert override.project_name == "good_name"

    def test_default_project_name_derived_from_cwd(self, tmp_path):
        """When the user accepts the auto-derived name, project_name is stored as None
        so future CWD-derivation still works."""
        base = _make_base_config(
            models={"openai": {}},
            datasources={"db1": "sqlite"},
            target="openai",
            default_database="db1",
        )
        from datus.configuration.agent_config import _normalize_project_name

        expected_default = _normalize_project_name(str(tmp_path))
        with (
            patch("datus.cli.project_init.select_choice", side_effect=["openai", "db1"]),
            patch("datus.cli.project_init.Prompt.ask", return_value=expected_default),
        ):
            override = run_project_init(base, cwd=str(tmp_path))
        # project_name stored as None so it stays CWD-derived if the path changes
        assert override.project_name is None
