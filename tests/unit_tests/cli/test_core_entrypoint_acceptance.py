# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Deterministic acceptance coverage for user-facing core CLI entrypoints."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.bootstrap_bi_commands import BootstrapBiCommands
from datus.cli.bootstrap_bi_picker import BootstrapBiPlan, DashboardCliOptions
from datus.cli.bootstrap_streams import stream_metrics, stream_semantic_model
from datus.cli.datasource_commands import DatasourceCommands
from datus.cli.init_commands import _INIT_PROMPT, InitCommands
from datus.cli.model_commands import ModelCommands
from datus.cli.repl import CommandType, DatusCLI
from datus.cli.skill_command_utils import render_skill_prompt
from datus.cli.slash_registry import lookup
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

pytestmark = pytest.mark.acceptance


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120, log_path=False)


def _action(
    message: str,
    *,
    action_type: str = "probe",
    status: ActionStatus = ActionStatus.SUCCESS,
) -> ActionHistory:
    return ActionHistory.create_action(
        role=ActionRole.TOOL,
        action_type=action_type,
        messages=message,
        input_data={"function_name": action_type},
        status=status,
    )


def _build_core_cli() -> DatusCLI:
    cli = object.__new__(DatusCLI)
    cli.console = _console()
    cli.plan_mode_active = True
    cli.tui_app = None
    cli._prefill_input = None
    cli.default_agent = ""

    agent_config = MagicMock()
    agent_config.current_datasource = "local_db"
    agent_config.datasource_configs = {"local_db": object(), "warehouse": object()}
    agent_config.services = SimpleNamespace(datasources={})
    agent_config.db_type = "duckdb"
    agent_config.set_active_provider_model = MagicMock()
    cli.agent_config = agent_config

    connector = MagicMock()
    connector.catalog_name = "catalog"
    connector.database_name = "warehouse_db"
    connector.schema_name = "analytics"
    cli.db_manager = MagicMock()
    cli.db_manager.first_conn_with_name.return_value = ("warehouse", connector)
    cli.db_connector = MagicMock()
    cli.cli_context = MagicMock()
    cli.reset_session = MagicMock()
    cli.chat_commands = MagicMock()
    cli.chat_commands.update_chat_node_tools = MagicMock()
    cli.bg_sync = MagicMock()
    cli.service_commands = MagicMock()
    cli.service_commands.dispatch = MagicMock(return_value=False)
    cli.session_summarize_commands = MagicMock()
    cli.memory_organize_commands = MagicMock()

    cli.agent_commands = MagicMock()
    cli.context_commands = MagicMock()
    cli.metadata_commands = MagicMock()
    cli.bootstrap_bi_commands = MagicMock()
    cli.bootstrap_commands = MagicMock()
    cli.model_commands = ModelCommands(cli)
    cli.language_commands = MagicMock()
    cli.effort_commands = MagicMock()
    cli.init_commands = InitCommands(cli)
    cli.datasource_commands = DatasourceCommands(cli)
    cli._cmd_help = MagicMock()
    cli._cmd_exit = MagicMock()
    cli._cmd_agent = MagicMock()
    cli._cmd_subagent = MagicMock()
    cli._cmd_mcp = MagicMock()
    cli._cmd_skill = MagicMock()
    cli._cmd_permission = MagicMock()
    cli._cmd_profile = MagicMock()

    cli.commands = {}
    for spec_name, handler in DatusCLI._build_slash_handler_map(cli).items():
        spec = lookup(spec_name)
        if spec is None:
            raise AssertionError(f"Slash handler '{spec_name}' has no registry entry")
        cli.commands[f"/{spec.name}"] = handler
        for alias in spec.aliases:
            cli.commands[f"/{alias}"] = handler
    return cli


def test_cli_slash_commands_route_through_shared_repl_dispatch() -> None:
    cli = _build_core_cli()

    cmd_type, cmd, args = DatusCLI._parse_command(cli, "/init")
    assert (cmd_type, cmd, args) == (CommandType.SLASH, "/init", "")
    DatusCLI._execute_slash_command(cli, cmd, args)
    cli.chat_commands.execute_chat_command.assert_called_once_with(
        render_skill_prompt(_INIT_PROMPT, ""),
        plan_mode=True,
        subagent_name=None,
    )

    cmd_type, cmd, args = DatusCLI._parse_command(cli, "/model openai/gpt-5.5")
    assert (cmd_type, cmd, args) == (CommandType.SLASH, "/model", "openai/gpt-5.5")
    DatusCLI._execute_slash_command(cli, cmd, args)
    cli.agent_config.set_active_provider_model.assert_called_once_with("openai", "gpt-5.5")

    cmd_type, cmd, args = DatusCLI._parse_command(cli, "/datasource warehouse")
    assert (cmd_type, cmd, args) == (CommandType.SLASH, "/datasource", "warehouse")
    with (
        patch("datus.configuration.project_config.load_project_override", return_value=None),
        patch("datus.configuration.project_config.save_project_override") as save_override,
    ):
        DatusCLI._execute_slash_command(cli, cmd, args)

    assert cli.agent_config.current_datasource == "warehouse"
    cli.db_manager.first_conn_with_name.assert_called_once_with("warehouse")
    cli.cli_context.update_database_context.assert_called_once_with(
        catalog="catalog",
        db_name="warehouse_db",
        schema="analytics",
    )
    cli.reset_session.assert_called_once_with()
    cli.chat_commands.update_chat_node_tools.assert_called_once_with()
    save_override.assert_called_once()
    saved_override = save_override.call_args.args[0]
    assert saved_override.default_datasource == "warehouse"
    cli.bg_sync.schedule.assert_called_once_with(datasource="warehouse", reason="switch")


def test_quickstart_documented_repl_entrypoints_stay_routable() -> None:
    """Quickstart REPL commands should remain valid slash/SQL/chat entrypoints."""
    repo_root = Path(__file__).resolve().parents[3]
    quickstart = (repo_root / "docs" / "getting_started" / "Quickstart.md").read_text(encoding="utf-8")
    documented_slash_commands = ("/datasource", "/model", "/init", "/tables")

    for command in documented_slash_commands:
        assert command in quickstart
        spec = lookup(command.removeprefix("/"))
        if spec is None:
            raise AssertionError(f"Quickstart command {command} is missing from slash registry")
        assert spec.name == command.removeprefix("/")

    cli = _build_core_cli()

    parsed = [DatusCLI._parse_command(cli, command) for command in documented_slash_commands]
    assert parsed == [
        (CommandType.SLASH, "/datasource", ""),
        (CommandType.SLASH, "/model", ""),
        (CommandType.SLASH, "/init", ""),
        (CommandType.SLASH, "/tables", ""),
    ]

    assert DatusCLI._parse_command(cli, "desc gold_vs_bitcoin")[0] == CommandType.SQL
    assert DatusCLI._parse_command(cli, "Detailed analysis of gold-Bitcoin correlation.")[0] == CommandType.CHAT


@pytest.mark.asyncio
async def test_bootstrap_semantic_model_and_metrics_streams_orchestrate_fake_helpers() -> None:
    agent_config = SimpleNamespace(current_datasource="", project_name="acceptance-project")
    calls: list[tuple] = []

    async def fake_semantic(agent_config, success_story, emit, *, build_mode, action_callback):
        calls.append(("semantic", agent_config.current_datasource, success_story, build_mode))
        action_callback(_action("semantic artifact persisted", action_type="semantic_model_saved"))
        return True, ""

    async def fake_metrics(
        *,
        agent_config,
        success_story,
        subject_tree,
        emit,
        build_mode,
        action_callback,
    ):
        calls.append(
            (
                "metrics",
                agent_config.current_datasource,
                success_story,
                tuple(subject_tree or ()),
                build_mode,
            )
        )
        action_callback(_action("metrics artifact persisted", action_type="metrics_saved"))
        return True, "", {"semantic_models": ["subject/semantic_models/local/orders.yml"]}

    with (
        patch(
            "datus.storage.semantic_model.semantic_model_init.init_success_story_semantic_model_async",
            side_effect=fake_semantic,
        ),
        patch(
            "datus.storage.metric.metric_init.init_success_story_metrics_async",
            side_effect=fake_metrics,
        ),
    ):
        semantic_actions = [
            action
            async for action in stream_semantic_model(
                agent_config,
                datasource="local",
                success_story="/tmp/story.csv",
                build_mode="overwrite",
            )
        ]
        metrics_actions = [
            action
            async for action in stream_metrics(
                agent_config,
                datasource="local",
                success_story="/tmp/story.csv",
                subject_tree=["Sales", "Orders"],
                build_mode="incremental",
            )
        ]

    assert calls == [
        ("semantic", "local", "/tmp/story.csv", "overwrite"),
        ("metrics", "local", "/tmp/story.csv", ("Sales", "Orders"), "incremental"),
    ]
    assert any(action.action_type == "semantic_model_saved" for action in semantic_actions)
    assert any(action.action_type == "metrics_saved" for action in metrics_actions)


@pytest.mark.asyncio
async def test_bootstrap_bi_extracts_context_and_hands_it_to_save_stream() -> None:
    agent_config = SimpleNamespace(
        current_datasource="local",
        db_type="duckdb",
        agentic_nodes={},
        path_manager=SimpleNamespace(),
        resolve_semantic_adapter=lambda value: value,
        current_db_config=lambda *_a, **_k: SimpleNamespace(catalog="cat", database="db", schema="public"),
    )
    plan = BootstrapBiPlan(
        options=DashboardCliOptions(
            platform="superset",
            dashboard_url="http://bi.example/d/1",
            api_base_url="http://bi.example",
            auth_params=None,
            dialect="duckdb",
        ),
        adapter=MagicMock(),
        dashboard=SimpleNamespace(id=1, name="Sales Performance", description="sales dashboard"),
        dashboard_id=1,
        chart_selections_ref=[MagicMock()],
        chart_selections_metrics=[MagicMock()],
        assembled=SimpleNamespace(
            tables=["orders"],
            reference_sqls=[MagicMock()],
            metric_sqls=[MagicMock()],
        ),
        pool_size=2,
    )
    cmd = BootstrapBiCommands(agent_config, _console())
    saved: dict[str, object] = {}

    async def metadata_stream(*_args, **_kwargs):
        yield _action("metadata crawled", action_type="schema_crawl")

    async def reference_sql_stream(*_args, state, **_kwargs):
        state.ref_sqls.append("superset.sales.orders_by_month")
        yield _action("reference sql indexed", action_type="sql_summary_response")

    async def semantic_model_stream(*_args, state, **_kwargs):
        state.semantic_ok = True
        yield _action("semantic model ready", action_type="gen_semantic_model")

    async def metrics_stream(*_args, state, **_kwargs):
        state.metrics.append("superset.sales.total_orders")
        yield _action("metrics ready", action_type="gen_metrics")

    async def save_stream(*_args, sub_agent_name, scoped_context, **_kwargs):
        saved["sub_agent_name"] = sub_agent_name
        saved["scoped_context"] = scoped_context
        yield _action("subagents saved", action_type="save_subagents")

    actions: list[ActionHistory] = []
    with (
        patch("datus.cli.bootstrap_bi_commands.stream_bi_metadata", side_effect=metadata_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_reference_sql", side_effect=reference_sql_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_semantic_model", side_effect=semantic_model_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_metrics", side_effect=metrics_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_save_subagents", side_effect=save_stream),
        patch("datus.cli.bootstrap_bi_commands.qualify_table_names", return_value=["cat.db.public.orders"]),
        patch("datus.cli.bootstrap_bi_commands.SubAgentManager"),
        patch("datus.cli.bootstrap_bi_commands.configuration_manager"),
    ):
        await cmd._run_plan(plan, actions)

    scoped_context = saved["scoped_context"]
    assert saved["sub_agent_name"] == "superset_sales_performance"
    assert scoped_context.tables == "cat.db.public.orders"
    assert scoped_context.sqls == "superset.sales.orders_by_month"
    assert scoped_context.metrics == "superset.sales.total_orders"
    assert [
        action.action_type for action in actions if action.action_type in {"schema_crawl", "sql_summary_response"}
    ] == [
        "schema_crawl",
        "sql_summary_response",
    ]
    assert any(action.action_type == "gen_semantic_model" for action in actions)
    assert any(action.action_type == "gen_metrics" for action in actions)
    assert any(action.action_type == "save_subagents" for action in actions)


@pytest.mark.asyncio
async def test_bootstrap_bi_missing_table_context_fails_before_subagent_save() -> None:
    agent_config = SimpleNamespace(
        current_datasource="local",
        db_type="duckdb",
        agentic_nodes={},
        path_manager=SimpleNamespace(),
        resolve_semantic_adapter=lambda value: value,
        current_db_config=lambda *_a, **_k: SimpleNamespace(catalog="cat", database="db", schema="public"),
    )
    plan = BootstrapBiPlan(
        options=DashboardCliOptions(
            platform="superset",
            dashboard_url="http://bi.example/d/1",
            api_base_url="http://bi.example",
            auth_params=None,
            dialect="duckdb",
        ),
        adapter=MagicMock(),
        dashboard=SimpleNamespace(id=1, name="Broken Dashboard", description="missing table metadata"),
        dashboard_id=1,
        chart_selections_ref=[MagicMock()],
        chart_selections_metrics=[],
        assembled=SimpleNamespace(tables=[], reference_sqls=[], metric_sqls=[]),
        pool_size=2,
    )
    cmd = BootstrapBiCommands(agent_config, _console())
    save_calls: list[object] = []

    async def metadata_stream(*_args, table_names, **_kwargs):
        assert table_names == []
        yield _action(
            "No tables in scope; skipping metadata crawl.",
            action_type="missing_table_metadata",
            status=ActionStatus.FAILED,
        )

    async def save_stream(*_args, **_kwargs):
        save_calls.append(True)
        yield _action("unexpected save", action_type="save_subagents")

    actions: list[ActionHistory] = []
    with (
        patch("datus.cli.bootstrap_bi_commands.stream_bi_metadata", side_effect=metadata_stream),
        patch("datus.cli.bootstrap_bi_commands.stream_bi_save_subagents", side_effect=save_stream),
        patch("datus.cli.bootstrap_bi_commands.qualify_table_names", return_value=[]),
        patch("datus.cli.bootstrap_bi_commands.SubAgentManager"),
        patch("datus.cli.bootstrap_bi_commands.configuration_manager"),
    ):
        await cmd._run_plan(plan, actions)

    assert save_calls == []
    assert any(
        action.action_type == "missing_table_metadata" and action.status == ActionStatus.FAILED.value
        for action in actions
    )
    assert any("No scoped context derived" in action.messages for action in actions)
