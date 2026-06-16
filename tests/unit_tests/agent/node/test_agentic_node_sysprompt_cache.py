# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the per-session system-prompt snapshot cache.

Covers:
- ``_get_session_system_prompt``: build-once / replay-verbatim across turns and
  node instances, rebuild on a meta change (model switch), no caching without a
  session id.
- ``_system_prompt_snapshot_meta``: identity keys exclude per-turn live values
  (datasource, profile, date, language).
- ``_build_datasource_reminder``: live per-turn datasource + dialect line for
  the user-message ``<system_reminder>`` envelope; never mentions the
  permission profile.
- ``_inject_runtime_context``: renders date-only shared runtime context
  WITHOUT requiring a DB tool.
- ``_inject_datasource_runtime_context``: renders datasource catalog +
  workspace root only for DB-tool-capable nodes and WITHOUT the current
  datasource selection.
- Template hygiene: the active node templates no longer inline the per-turn
  volatile variables.

A lightweight fake node bypasses the heavy ``AgenticNode.__init__`` and wires a
real :class:`SessionManager` over ``tmp_path`` — no LLM, no network.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.models.session_manager import SessionManager

TEMPLATE_DIR = Path(__file__).resolve().parents[4] / "datus" / "prompts" / "prompt_templates"
ACTIVE_TEMPLATES = [
    "chat_system_1.2.j2",
    "gen_sql_system_1.2.j2",
    "explore_system_1.0.j2",
    "gen_report_system_1.0.j2",
    "ask_metrics_system_1.0.j2",
]
# Deliverable templates never inlined the datasource selection; they are held
# to the same no-volatile-variables bar but don't point to <system_reminder>.
DELIVERABLE_TEMPLATES = [
    "gen_table_system_1.0.j2",
    "gen_dashboard_system_1.0.j2",
    "gen_job_system_1.0.j2",
    "scheduler_system_1.0.j2",
]


def _agent_config(*, current_datasource=None, services=None, model="gpt-4.1"):
    return SimpleNamespace(
        prompt_version="1.2",
        current_datasource=current_datasource,
        services=services,
        active_model=lambda: SimpleNamespace(type="openai", model=model),
    )


class _SnapshotNode(AgenticNode):
    """Minimal node exposing the real snapshot/reminder/runtime-context methods."""

    def __init__(self, session_manager: SessionManager, agent_config, *, db_func_tool=None):
        self.session_id = "chat_session_x"
        self._session_manager = session_manager
        self.agent_config = agent_config
        self.db_func_tool = db_func_tool
        self.build_count = 0
        self.lazy_mount_count = 0
        # Lazy-tool wiring runs on the cache-hit path too; give the ensure
        # methods the attributes they gate on (all "nothing to mount" here).
        self.skill_func_tool = None
        self.bash_tool = None
        self.memory_func_tool = None
        self._is_subagent = True
        self._node_model_name = None

    def _ensure_lazy_tools_mounted(self) -> None:
        self.lazy_mount_count += 1
        super()._ensure_lazy_tools_mounted()

    def get_node_name(self) -> str:
        return "chat"

    def _resolve_workspace_root(self) -> str:
        return "/tmp/ws"

    def _get_system_prompt(
        self,
        prompt_version: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        # Each rebuild is observable and uniquely tagged so replay vs rebuild
        # is unambiguous in assertions.
        self.build_count += 1
        return f"SYS#{self.build_count}"


@pytest.fixture
def session_manager(tmp_path):
    manager = SessionManager(session_dir=str(tmp_path))
    yield manager
    manager.close_all_sessions()


class TestGetSessionSystemPrompt:
    def test_first_turn_builds_and_persists(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config())
        prompt = node._get_session_system_prompt(prompt_version="1.2")
        assert prompt == "SYS#1"
        assert node.build_count == 1
        # Snapshot file now persists the exact prompt for replay.
        snapshot = session_manager.load_system_prompt_snapshot(node.session_id)
        assert snapshot["prompt"] == "SYS#1"
        assert snapshot["model_name"] == "openai:gpt-4.1"

    def test_second_turn_replays_verbatim(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config())
        first = node._get_session_system_prompt(prompt_version="1.2")
        second = node._get_session_system_prompt(prompt_version="1.2")
        assert first == second == "SYS#1"
        # The expensive builder ran exactly once across both turns.
        assert node.build_count == 1

    def test_new_node_instance_replays_snapshot(self, session_manager):
        """API per-request nodes and /resume reuse the snapshot of the session."""
        node1 = _SnapshotNode(session_manager, _agent_config())
        first = node1._get_session_system_prompt(prompt_version="1.2")

        node2 = _SnapshotNode(session_manager, _agent_config())
        node2.session_id = node1.session_id
        second = node2._get_session_system_prompt(prompt_version="1.2")

        assert second == first == "SYS#1"
        assert node2.build_count == 0

    def test_model_switch_rebuilds(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config(model="gpt-4.1"))
        first = node._get_session_system_prompt(prompt_version="1.2")
        assert first == "SYS#1"
        # Switch model → meta mismatch → rebuild and overwrite.
        node.agent_config = _agent_config(model="gpt-5")
        second = node._get_session_system_prompt(prompt_version="1.2")
        assert second == "SYS#2"
        assert node.build_count == 2
        # The overwritten snapshot now replays for the new model.
        assert node._get_session_system_prompt(prompt_version="1.2") == "SYS#2"
        assert node.build_count == 2

    def test_prompt_version_change_rebuilds(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config())
        assert node._get_session_system_prompt(prompt_version="1.2") == "SYS#1"
        assert node._get_session_system_prompt(prompt_version="1.3") == "SYS#2"

    def test_no_session_id_skips_cache(self, session_manager, tmp_path):
        node = _SnapshotNode(session_manager, _agent_config())
        node.session_id = ""
        node._get_session_system_prompt(prompt_version="1.2")
        node._get_session_system_prompt(prompt_version="1.2")
        # Without a session id there is no snapshot to replay → always rebuilds,
        # and nothing is written to disk.
        assert node.build_count == 2
        assert list(tmp_path.glob("*.sysprompt.json")) == []

    def test_datasource_switch_does_not_rebuild(self, session_manager):
        """The live datasource is a user-message concern, never a cache key."""
        node = _SnapshotNode(session_manager, _agent_config(current_datasource="main"))
        node._get_session_system_prompt(prompt_version="1.2")
        node.agent_config = _agent_config(current_datasource="dev")
        node._get_session_system_prompt(prompt_version="1.2")
        assert node.build_count == 1

    def test_cache_hit_mounts_lazy_tools(self, session_manager):
        """A replayed prompt advertises skill/bash/memory tools — the hit path
        must run the (skipped) build's tool-mounting side effect."""
        node1 = _SnapshotNode(session_manager, _agent_config())
        node1._get_session_system_prompt(prompt_version="1.2")

        # Fresh instance resuming the session: pure cache hit, no build.
        node2 = _SnapshotNode(session_manager, _agent_config())
        node2.session_id = node1.session_id
        node2._get_session_system_prompt(prompt_version="1.2")
        assert node2.build_count == 0
        assert node2.lazy_mount_count == 1

    def test_node_model_override_switch_rebuilds(self, session_manager):
        """A node-level model override change must invalidate the snapshot."""
        node = _SnapshotNode(session_manager, _agent_config())
        node._node_model_name = "gpt-5-mini"
        assert node._get_session_system_prompt(prompt_version="1.2") == "SYS#1"
        node._node_model_name = "gpt-5"
        assert node._get_session_system_prompt(prompt_version="1.2") == "SYS#2"
        assert node.build_count == 2


class TestSnapshotMeta:
    def test_meta_is_exactly_the_identity_keys(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config(current_datasource="main"))
        meta = node._system_prompt_snapshot_meta("1.2")
        assert meta == {"node_name": "chat", "prompt_version": "1.2", "model_name": "openai:gpt-4.1"}

    def test_meta_falls_back_to_agent_config_version(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config())
        meta = node._system_prompt_snapshot_meta(None)
        assert meta["prompt_version"] == "1.2"

    def test_meta_survives_broken_model_resolution(self, session_manager):
        def boom():
            raise RuntimeError("no model configured")

        cfg = _agent_config()
        cfg.active_model = boom
        node = _SnapshotNode(session_manager, cfg)
        meta = node._system_prompt_snapshot_meta("1.2")
        assert meta["model_name"] == ""

    def test_meta_prefers_node_model_override(self, session_manager):
        """``node_config.model`` pins the effective model — it is the identity,
        not the agent-level target."""
        node = _SnapshotNode(session_manager, _agent_config(model="gpt-4.1"))
        node._node_model_name = "gpt-5-mini"
        meta = node._system_prompt_snapshot_meta("1.2")
        assert meta["model_name"] == "node:gpt-5-mini"


class TestDatasourceReminder:
    def test_requires_db_tool(self, session_manager):
        cfg = _agent_config(current_datasource="main")
        without = _SnapshotNode(session_manager, cfg, db_func_tool=None)
        assert without._build_datasource_reminder() == ""

        with_db = _SnapshotNode(session_manager, cfg, db_func_tool=object())
        line = with_db._build_datasource_reminder()
        assert "Current datasource: main" in line
        assert "authoritative" in line

    def test_dialect_included_when_available(self, session_manager):
        services = SimpleNamespace(datasources={"main": SimpleNamespace(type="snowflake")})
        cfg = _agent_config(current_datasource="main", services=services)
        node = _SnapshotNode(session_manager, cfg, db_func_tool=object())
        line = node._build_datasource_reminder()
        assert "Current datasource: main (dialect: snowflake)" in line
        assert "generate SQL for THIS dialect" in line

    def test_empty_without_current_datasource(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config(current_datasource=None), db_func_tool=object())
        assert node._build_datasource_reminder() == ""

    def test_never_mentions_permission_profile(self, session_manager):
        """The profile is enforced by permission hooks, never prompted."""
        services = SimpleNamespace(datasources={"main": SimpleNamespace(type="duckdb")})
        cfg = _agent_config(current_datasource="main", services=services)
        cfg.active_profile_name = "dangerous"
        node = _SnapshotNode(session_manager, cfg, db_func_tool=object())
        line = node._build_datasource_reminder()
        assert "profile" not in line.lower()
        assert "dangerous" not in line

    def test_merges_catalog_database_schema_details(self, session_manager):
        """The reminder absorbs the legacy Database Context fields in one line."""
        services = SimpleNamespace(datasources={"main": SimpleNamespace(type="snowflake")})
        cfg = _agent_config(current_datasource="main", services=services)
        db_tool = SimpleNamespace(connector=SimpleNamespace(database_name="proddb"))
        node = _SnapshotNode(session_manager, cfg, db_func_tool=db_tool)

        user_input = SimpleNamespace(catalog="c1", database="", db_schema="s1")
        line = node._build_datasource_reminder(user_input)

        assert "Current datasource: main (dialect: snowflake, catalog: c1, database: proddb, schema: s1)" in line

    def test_explicit_database_overrides_connector_default(self, session_manager):
        services = SimpleNamespace(datasources={"main": SimpleNamespace(type="duckdb")})
        cfg = _agent_config(current_datasource="main", services=services)
        db_tool = SimpleNamespace(connector=SimpleNamespace(database_name="default_db"))
        node = _SnapshotNode(session_manager, cfg, db_func_tool=db_tool)

        user_input = SimpleNamespace(database="explicit_db")
        line = node._build_datasource_reminder(user_input)

        assert "database: explicit_db" in line
        assert "default_db" not in line

    def test_enhanced_message_carries_reminder_in_envelope(self, session_manager):
        """The reminder rides inside the existing <system_reminder> envelope."""
        from datus.utils.message_utils import extract_enhanced_context, extract_user_input

        services = SimpleNamespace(datasources={"main": SimpleNamespace(type="snowflake")})
        cfg = _agent_config(current_datasource="main", services=services)
        node = _SnapshotNode(session_manager, cfg, db_func_tool=object())
        node.plan_mode_active = False
        node._plan_just_confirmed = False
        node.plan_file_path = None

        user_input = SimpleNamespace(user_message="show me revenue", plan_mode=False)
        message = node._build_enhanced_message(user_input)

        enhanced = extract_enhanced_context(message)
        assert "Current datasource: main (dialect: snowflake)" in enhanced
        assert extract_user_input(message) == "show me revenue"
        # Merged design: the dialect value is stated exactly once — no
        # separate legacy "Database Context" block when the reminder is present.
        assert "Database Context" not in enhanced
        assert enhanced.lower().count("dialect:") == 1

    def test_legacy_database_context_when_reminder_absent(self, session_manager):
        """Nodes without a DB tool keep the legacy Database Context block."""
        from datus.utils.message_utils import extract_enhanced_context

        cfg = _agent_config(current_datasource="main")
        cfg.db_type = "sqlite"
        node = _SnapshotNode(session_manager, cfg, db_func_tool=None)
        node.plan_mode_active = False
        node._plan_just_confirmed = False
        node.plan_file_path = None

        user_input = SimpleNamespace(user_message="hello", plan_mode=False, database="db1")
        message = node._build_enhanced_message(user_input)

        enhanced = extract_enhanced_context(message)
        assert "Database Context" in enhanced
        assert "**Dialect**: sqlite" in enhanced
        assert "Current datasource:" not in enhanced


class TestInjectRuntimeContext:
    def test_appends_date_without_db_tool(self, session_manager):
        node = _SnapshotNode(session_manager, _agent_config(), db_func_tool=None)
        out = node._inject_runtime_context("BASE")
        assert out.startswith("BASE")
        assert "Current context:" in out
        assert re.search(r"Current date: \d{4}-\d{2}-\d{2}", out)
        assert "Available datasources:" not in out
        assert "Current sql files root directory:" not in out

    def test_runtime_context_render_failure_returns_base_prompt(self, session_manager, monkeypatch):
        from datus.agent.node import agentic_node as agentic_node_module

        calls = []

        class RaisingPromptManager:
            def render_template(self, **kwargs):
                calls.append(kwargs)
                raise RuntimeError("template unavailable")

        monkeypatch.setattr(
            agentic_node_module,
            "get_prompt_manager",
            lambda **_: RaisingPromptManager(),
        )

        node = _SnapshotNode(session_manager, _agent_config(), db_func_tool=None)
        node._runtime_context_current_date = lambda: "2024-06-15"

        assert node._inject_runtime_context("BASE") == "BASE"
        assert calls == [
            {
                "template_name": "runtime_context",
                "version": None,
                "current_date": "2024-06-15",
            }
        ]

    def test_skips_datasource_runtime_context_without_db_tool(self, session_manager):
        services = SimpleNamespace(datasources={"main": SimpleNamespace(type="snowflake")})
        cfg = _agent_config(current_datasource="main", services=services)
        node = _SnapshotNode(session_manager, cfg, db_func_tool=None)
        assert node._inject_datasource_runtime_context("BASE") == "BASE"

    def test_appends_datasource_catalog_and_workspace_for_db_tool(self, session_manager):
        services = SimpleNamespace(
            datasources={"main": SimpleNamespace(type="snowflake"), "dev": SimpleNamespace(type="duckdb")}
        )
        cfg = _agent_config(current_datasource="main", services=services)
        node = _SnapshotNode(session_manager, cfg, db_func_tool=object())
        out = node._inject_datasource_runtime_context("BASE")
        assert out.startswith("BASE")
        assert "Datasource context:" in out
        assert "Current date:" not in out
        assert "Available datasources:" in out
        assert "main (snowflake)" in out
        assert "dev (duckdb)" in out
        assert "/tmp/ws" in out

    def test_datasource_runtime_context_render_failure_returns_base_prompt(self, session_manager, monkeypatch):
        from datus.agent.node import agentic_node as agentic_node_module

        calls = []

        class RaisingPromptManager:
            def render_template(self, **kwargs):
                calls.append(kwargs)
                raise RuntimeError("template unavailable")

        monkeypatch.setattr(
            agentic_node_module,
            "get_prompt_manager",
            lambda **_: RaisingPromptManager(),
        )

        services = SimpleNamespace(datasources={"main": SimpleNamespace(type="snowflake")})
        cfg = _agent_config(current_datasource="main", services=services)
        node = _SnapshotNode(session_manager, cfg, db_func_tool=object())

        assert node._inject_datasource_runtime_context("BASE") == "BASE"
        assert calls == [
            {
                "template_name": "datasource_runtime_context",
                "version": None,
                "available_datasources": {"main": "snowflake"},
                "workspace_root": "/tmp/ws",
            }
        ]

    def test_does_not_pin_current_datasource_selection(self, session_manager):
        """The frozen prompt lists the catalog but never the live selection."""
        services = SimpleNamespace(datasources={"main": SimpleNamespace(type="snowflake")})
        cfg = _agent_config(current_datasource="main", services=services)
        node = _SnapshotNode(session_manager, cfg, db_func_tool=object())
        out = node._inject_datasource_runtime_context("BASE")
        assert "Current datasource:" not in out

    def test_runtime_context_date_hook_is_overridable(self, session_manager):
        """Subclasses (GenSQL/AskMetrics) pin a reference date into the snapshot."""
        services = SimpleNamespace(datasources={"main": SimpleNamespace(type="duckdb")})
        cfg = _agent_config(current_datasource="main", services=services)
        node = _SnapshotNode(session_manager, cfg, db_func_tool=None)
        node._runtime_context_current_date = lambda: "1999-12-31"
        out = node._inject_runtime_context("BASE")
        assert "Current date: 1999-12-31" in out


class TestReferenceDateOverrides:
    def test_gen_sql_uses_date_parsing_reference_date(self):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = GenSQLAgenticNode.__new__(GenSQLAgenticNode)
        node.date_parsing_tools = SimpleNamespace(reference_date="2024-06-15")
        assert node._runtime_context_current_date() == "2024-06-15"

    def test_ask_metrics_uses_input_reference_date(self):
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

        node = AskMetricsAgenticNode.__new__(AskMetricsAgenticNode)
        node.input = SimpleNamespace(reference_date="2024-06-15")
        assert node._runtime_context_current_date() == "2024-06-15"


class TestTemplateHygiene:
    """The active templates must not inline per-turn volatile variables."""

    @pytest.mark.parametrize("template", ACTIVE_TEMPLATES + DELIVERABLE_TEMPLATES)
    def test_no_inline_volatile_variables(self, template):
        source = (TEMPLATE_DIR / template).read_text(encoding="utf-8")
        assert "{{ current_date }}" not in source
        assert "{{ active_profile }}" not in source
        assert "Current datasource: {{ datasource }}" not in source

    @pytest.mark.parametrize("template", ACTIVE_TEMPLATES)
    def test_points_to_system_reminder(self, template):
        source = (TEMPLATE_DIR / template).read_text(encoding="utf-8")
        assert "<system_reminder>" in source

    def test_runtime_context_partial_has_no_current_selection(self):
        source = (TEMPLATE_DIR / "runtime_context_1.0.j2").read_text(encoding="utf-8")
        assert "{{ current_date }}" in source
        assert "available_datasources" not in source
        assert "workspace_root" not in source
        assert "Current datasource: {{ datasource }}" not in source

    def test_datasource_runtime_context_partial_has_no_current_selection(self):
        source = (TEMPLATE_DIR / "datasource_runtime_context_1.0.j2").read_text(encoding="utf-8")
        assert "{{ current_date }}" not in source
        assert "available_datasources" in source
        assert "workspace_root" in source
        assert "Current datasource: {{ datasource }}" not in source
