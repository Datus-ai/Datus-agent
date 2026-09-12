# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Deterministic Datus Agent + managed SQL policy plugin regression."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from starlette.requests import Request

from datus.api.auth.header_context_provider import HeaderContextProvider
from datus.api.routes.chat_routes import _policy_context_pre_check
from datus.configuration.agent_config import AgentConfig, NodeConfig
from datus.plugins.registry import invalidate_plugin_cache
from datus.tools.db_tools.config import SQLiteConfig
from datus.tools.db_tools.sqlite_connector import SQLiteConnector
from datus.tools.func_tool.database import DBFuncTool
from datus.tools.middleware.tool_middleware import transform_tool_args
from datus.tools.permission.bash_rules import evaluate_bash_command
from datus.tools.permission.permission_config import PermissionLevel

SQL_POLICY_ID = "sp_store_scope_sql"
METRIC_POLICY_ID = "sp_store_scope_metrics"


def _policies() -> list[dict]:
    return [
        {
            "id": SQL_POLICY_ID,
            "name": "store_scope_sql",
            "type": "row_filter",
            "applies_to": {"datasources": ["warehouse"], "tables": ["orders"]},
            "condition": {
                "column": "store_id",
                "operator": "in",
                "value_from": "policy_context.row_filter.store_ids",
            },
        },
        {
            "id": METRIC_POLICY_ID,
            "name": "store_scope_metrics",
            "type": "metric_row_filter",
            "applies_to": {"datasets": ["orders"]},
            "condition": {
                "column": "orders.store_id",
                "operator": "in",
                "value_from": "policy_context.row_filter.store_ids",
            },
        },
    ]


def _scoped_context(*, sql_mode: str | None = None, metric_mode: str | None = None) -> dict:
    """Build scoped policy context, optionally overriding either policy mode."""
    row_filter = {"access_mode": "scoped", "store_ids": ["S001"]}
    policy_modes = {}
    if sql_mode is not None:
        policy_modes[SQL_POLICY_ID] = sql_mode
    if metric_mode is not None:
        policy_modes[METRIC_POLICY_ID] = metric_mode
    if policy_modes:
        row_filter["policy_modes"] = policy_modes
    return {"row_filter": row_filter}


def _transform_metric(config: AgentConfig, policy_context: dict) -> dict:
    """Apply the managed metric policy to the shared nightly query."""
    return transform_tool_args(
        "query_metrics",
        {"metrics": ["order_revenue"], "where": "orders.amount > 0"},
        category="semantic_tools",
        active_plugin_names=config.active_plugin_names(),
        context={
            "agent_config": config,
            "policy_context": policy_context,
            "metric_datasets": {"order_revenue": ["orders"]},
        },
    )


def _agent_document(home, workspace, database) -> dict:
    return {
        "agent": {
            "home": str(home),
            "project_name": "nightly_sql_policy",
            "project_root": str(workspace),
            "target": "mock",
            "models": {
                "mock": {
                    "type": "openai",
                    "api_key": "unused",
                    "model": "unused",
                    "base_url": "http://127.0.0.1:1",
                }
            },
            "nodes": {"chat": {"model": "mock"}},
            "services": {
                "datasources": {
                    "warehouse": {
                        "type": "sqlite",
                        "uri": str(database),
                        "default": True,
                    }
                }
            },
            "plugins": {
                "sql-policy": {
                    "default": {
                        "default": True,
                        "policies": _policies(),
                    }
                }
            },
            "permissions": {
                "profile": "normal",
                "bash_commands": {"deny": ["datus sql-policy status"]},
            },
        }
    }


def _agent_config(document: dict) -> AgentConfig:
    raw = document["agent"]
    nodes = {name: NodeConfig(input=None, **value) for name, value in raw["nodes"].items()}
    config = AgentConfig(nodes=nodes, **{key: value for key, value in raw.items() if key != "nodes"})
    config.current_datasource = "warehouse"
    return config


def _request(policy_context: dict) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/chat/stream",
            "headers": [
                (
                    b"x-datus-policy-context",
                    json.dumps(policy_context, separators=(",", ":")).encode(),
                )
            ],
        }
    )


@pytest.mark.nightly
@pytest.mark.timeout(300)
@pytest.mark.asyncio
async def test_managed_sql_policy_enforces_sql_semantic_and_api_reads(
    p0_external_sources,
    managed_plugin_runtime,
    tmp_path,
):
    """Exercise the managed plugin and every Agent-side enforcement seam."""
    assert importlib.util.find_spec("datus_sql_policies") is None, (
        "P0 must not have a globally installed datus-sql-policies fallback"
    )
    installed = managed_plugin_runtime.install(p0_external_sources.sql_policies)
    assert "sql-policy" in installed.stdout

    database = tmp_path / "orders.sqlite"
    connector = SQLiteConnector(SQLiteConfig(db_path=str(database)))
    try:
        assert connector.execute_query("CREATE TABLE orders (store_id TEXT NOT NULL, amount INTEGER NOT NULL)").success
        assert connector.execute_query("INSERT INTO orders VALUES ('S001', 10), ('S002', 20)").success

        datus_home = managed_plugin_runtime.home / ".datus"
        document = _agent_document(datus_home, tmp_path / "workspace", database)
        config_file = datus_home / "conf" / "agent.yml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

        listed = managed_plugin_runtime.run("plugin", "list")
        assert "sql-policy" in listed.stdout
        info = managed_plugin_runtime.run("plugin", "info", "sql-policy")
        assert "entry:   datus_sql_policies" in info.stdout
        status = managed_plugin_runtime.run("sql-policy", "status")
        assert "2 policies configured" in status.stdout
        assert "store_scope_sql" in status.stdout
        assert SQL_POLICY_ID in status.stdout
        assert METRIC_POLICY_ID in status.stdout
        checked = managed_plugin_runtime.run(
            "sql-policy",
            "check",
            "--sql",
            "SELECT store_id, amount FROM orders ORDER BY store_id",
            "--datasource",
            "warehouse",
            "--dialect",
            "sqlite",
            "--policy-context",
            json.dumps(_scoped_context()),
        )
        assert "store_scope_sql" in checked.stdout
        assert "S001" in checked.stdout

        config = _agent_config(document)
        invalidate_plugin_cache()
        assert config.active_plugin_names() is None
        plugin_spec = importlib.util.find_spec("datus_sql_policies")
        assert plugin_spec is not None and plugin_spec.origin
        managed_plugin_dir = datus_home / "plugins" / "sql-policy"
        assert Path(plugin_spec.origin).resolve().is_relative_to(managed_plugin_dir.resolve())

        scoped = _scoped_context()
        config.policy_context = scoped
        tool = DBFuncTool(connector, agent_config=config, default_datasource="warehouse")
        sql_result = tool.execute_read_enforced(
            "SELECT store_id, amount FROM orders ORDER BY store_id",
            connector,
            datasource="warehouse",
        )
        assert sql_result.success, sql_result.error
        assert sql_result.sql_return == [{"store_id": "S001", "amount": 10}]

        transformed = _transform_metric(config, scoped)
        assert "orders.amount > 0" in transformed["where"]
        assert "orders.store_id" in transformed["where"]
        assert "S001" in transformed["where"]

        sql_unrestricted = _scoped_context(sql_mode="unrestricted", metric_mode="scoped")
        config.policy_context = sql_unrestricted
        unrestricted_sql_result = tool.execute_read_enforced(
            "SELECT store_id, amount FROM orders ORDER BY store_id",
            connector,
            datasource="warehouse",
        )
        assert unrestricted_sql_result.success is True, unrestricted_sql_result.error
        assert unrestricted_sql_result.sql_return == [
            {"store_id": "S001", "amount": 10},
            {"store_id": "S002", "amount": 20},
        ]
        metric_still_scoped = _transform_metric(config, sql_unrestricted)
        assert "orders.store_id" in metric_still_scoped["where"]

        metric_unrestricted = _scoped_context(sql_mode="scoped", metric_mode="unrestricted")
        config.policy_context = metric_unrestricted
        sql_still_scoped = tool.execute_read_enforced(
            "SELECT store_id, amount FROM orders ORDER BY store_id",
            connector,
            datasource="warehouse",
        )
        assert sql_still_scoped.success is True, sql_still_scoped.error
        assert sql_still_scoped.sql_return == [{"store_id": "S001", "amount": 10}]
        unrestricted_metric_args = _transform_metric(config, metric_unrestricted)
        assert unrestricted_metric_args["where"] == "orders.amount > 0"

        provider = HeaderContextProvider()
        config.policy_context = scoped
        scoped_context = await provider.authenticate(_request(scoped))
        assert scoped_context.policy_context == scoped
        assert _policy_context_pre_check(SimpleNamespace(agent_config=config), scoped_context) is None

        unknown_policy = {
            "row_filter": {
                "access_mode": "scoped",
                "policy_modes": {"sp_unknown": "unrestricted"},
                "store_ids": ["S001"],
            }
        }
        unknown_context = await provider.authenticate(_request(unknown_policy))
        unknown_outcome = _policy_context_pre_check(SimpleNamespace(agent_config=config), unknown_context)
        assert unknown_outcome.allow is False
        assert unknown_outcome.error_type == "POLICY_CONTEXT_REJECTED"

        denied = {"row_filter": {"access_mode": "denied"}}
        denied_context = await provider.authenticate(_request(denied))
        outcome = _policy_context_pre_check(SimpleNamespace(agent_config=config), denied_context)
        assert outcome is not None
        assert outcome.allow is False
        assert outcome.error_type == "POLICY_CONTEXT_REJECTED"

        config.policy_context = denied
        denied_result = tool.execute_read_enforced(
            "SELECT store_id, amount FROM orders",
            connector,
            datasource="warehouse",
        )
        assert not denied_result.success
        assert "denies all data reads" in denied_result.error

        plugin_decision = evaluate_bash_command(
            "datus sql-policy status",
            config.plugin_bash_rules["normal"],
        )
        assert plugin_decision.level == PermissionLevel.ALLOW

        user_override_decision = evaluate_bash_command(
            "datus sql-policy status",
            config.permissions_config.bash_commands,
        )
        assert user_override_decision.level == PermissionLevel.DENY
    finally:
        connector.close()
