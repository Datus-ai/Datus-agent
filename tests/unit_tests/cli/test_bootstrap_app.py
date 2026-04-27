# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.bootstrap_app` (form-only assertions).

Drives ``_collect_for`` directly rather than running the prompt_toolkit
Application — running the TUI in pytest is unreliable across CI.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from datus.cli.bootstrap_app import (
    PANEL_NAMES,
    BootstrapApp,
    BootstrapPlan,
    TaskSpec,
    _Tab,
    _ValidationError,
)


@pytest.fixture()
def console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120, log_path=False)


@pytest.fixture()
def app(console: Console) -> BootstrapApp:
    return BootstrapApp(console, datasource_default="ssb_sqlite")


# ─────────────────────────────────────────────────────────────────────
# Sanity
# ─────────────────────────────────────────────────────────────────────


def test_panel_names_match_tab_enum() -> None:
    assert PANEL_NAMES == tuple(t.value for t in _Tab)


def test_dataclasses_construct() -> None:
    spec = TaskSpec(name="metadata", options={"datasource": "x"})
    plan = BootstrapPlan(task=spec)
    assert plan.task.name == "metadata"
    assert plan.task.options == {"datasource": "x"}


# ─────────────────────────────────────────────────────────────────────
# Per-tab _collect_for shapes — defaults: overwrite checkbox unchecked
# means build_mode == "incremental"
# ─────────────────────────────────────────────────────────────────────


def test_collect_schema_defaults_incremental(app: BootstrapApp) -> None:
    opts = app._collect_for(_Tab.SCHEMA)
    assert opts == {
        "datasource": "ssb_sqlite",
        "build_mode": "incremental",
    }


def test_collect_schema_overwrite_checked(app: BootstrapApp) -> None:
    app._schema_overwrite.checked = True
    opts = app._collect_for(_Tab.SCHEMA)
    assert opts["build_mode"] == "overwrite"


def test_collect_schema_missing_datasource_raises(app: BootstrapApp) -> None:
    app._schema_datasource.text = "  "
    with pytest.raises(_ValidationError):
        app._collect_for(_Tab.SCHEMA)


def test_collect_sql_full_form(app: BootstrapApp) -> None:
    app._sql_dir.text = "/data/sql"
    app._sql_pool.text = "5"
    app._sql_subject_tree.text = "Finance, Revenue "
    app._sql_overwrite.checked = True
    opts = app._collect_for(_Tab.SQL)
    assert opts == {
        "datasource": "ssb_sqlite",
        "sql_dir": "/data/sql",
        "pool_size": 5,
        "subject_tree": "Finance, Revenue",
        "build_mode": "overwrite",
    }


def test_collect_sql_missing_dir(app: BootstrapApp) -> None:
    with pytest.raises(_ValidationError):
        app._collect_for(_Tab.SQL)


def test_collect_sql_invalid_pool(app: BootstrapApp) -> None:
    app._sql_dir.text = "/data/sql"
    app._sql_pool.text = "0"
    with pytest.raises(_ValidationError):
        app._collect_for(_Tab.SQL)


def test_collect_template_shape(app: BootstrapApp) -> None:
    app._tpl_dir.text = "/data/templates"
    opts = app._collect_for(_Tab.TEMPLATE)
    assert opts == {
        "datasource": "ssb_sqlite",
        "template_dir": "/data/templates",
        "pool_size": 3,
        "subject_tree": "",
        "build_mode": "incremental",
    }


def test_collect_semantic_only_success_story(app: BootstrapApp) -> None:
    app._sem_success_story.text = "/data/success.csv"
    opts = app._collect_for(_Tab.SEMANTIC)
    # No semantic_yaml / from_adapter / catalog / subject_path / source.
    assert opts == {
        "datasource": "ssb_sqlite",
        "success_story": "/data/success.csv",
        "build_mode": "incremental",
    }


def test_collect_semantic_overwrite(app: BootstrapApp) -> None:
    app._sem_success_story.text = "/data/success.csv"
    app._sem_overwrite.checked = True
    assert app._collect_for(_Tab.SEMANTIC)["build_mode"] == "overwrite"


def test_collect_semantic_missing_success_story(app: BootstrapApp) -> None:
    with pytest.raises(_ValidationError):
        app._collect_for(_Tab.SEMANTIC)


def test_collect_metrics_full_form(app: BootstrapApp) -> None:
    app._met_success_story.text = "/data/success.csv"
    app._met_pool.text = "2"
    app._met_subject_tree.text = "Finance"
    opts = app._collect_for(_Tab.METRICS)
    assert opts == {
        "datasource": "ssb_sqlite",
        "success_story": "/data/success.csv",
        "pool_size": 2,
        "subject_tree": "Finance",
        "build_mode": "incremental",
    }


def test_collect_knowledge_only_success_story(app: BootstrapApp) -> None:
    app._know_success_story.text = "/data/success.csv"
    opts = app._collect_for(_Tab.KNOWLEDGE)
    # No source / ext_knowledge_csv keys — they were removed in the simplification.
    assert opts == {
        "datasource": "ssb_sqlite",
        "success_story": "/data/success.csv",
        "pool_size": 4,
        "subject_tree": "",
        "build_mode": "incremental",
    }


def test_collect_knowledge_overwrite_with_subjects(app: BootstrapApp) -> None:
    app._know_success_story.text = "/data/success.csv"
    app._know_subject_tree.text = "A,B"
    app._know_overwrite.checked = True
    opts = app._collect_for(_Tab.KNOWLEDGE)
    assert opts["build_mode"] == "overwrite"
    assert opts["subject_tree"] == "A,B"


# ─────────────────────────────────────────────────────────────────────
# Removed-fields guard — every tab must produce ONLY the simplified key
# set; if anyone re-adds a stale field they'll trip these assertions.
# ─────────────────────────────────────────────────────────────────────


_EXPECTED_KEYS = {
    _Tab.SCHEMA: {"datasource", "build_mode"},
    _Tab.SQL: {"datasource", "sql_dir", "pool_size", "subject_tree", "build_mode"},
    _Tab.TEMPLATE: {"datasource", "template_dir", "pool_size", "subject_tree", "build_mode"},
    _Tab.SEMANTIC: {"datasource", "success_story", "build_mode"},
    _Tab.METRICS: {"datasource", "success_story", "pool_size", "subject_tree", "build_mode"},
    _Tab.KNOWLEDGE: {"datasource", "success_story", "pool_size", "subject_tree", "build_mode"},
}


@pytest.mark.parametrize("tab", list(_EXPECTED_KEYS.keys()))
def test_no_unexpected_keys_per_tab(app: BootstrapApp, tab: _Tab) -> None:
    """Fill every required field with a placeholder, then assert key set."""
    if tab == _Tab.SQL:
        app._sql_dir.text = "x"
    elif tab == _Tab.TEMPLATE:
        app._tpl_dir.text = "x"
    elif tab == _Tab.SEMANTIC:
        app._sem_success_story.text = "x"
    elif tab == _Tab.METRICS:
        app._met_success_story.text = "x"
    elif tab == _Tab.KNOWLEDGE:
        app._know_success_story.text = "x"
    opts = app._collect_for(tab)
    assert set(opts.keys()) == _EXPECTED_KEYS[tab]
