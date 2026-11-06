# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Deterministic Datus Agent + managed SQL policy plugin regression."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
import yaml
from starlette.requests import Request

from datus.api.auth.header_context_provider import HeaderContextProvider
from datus.api.routes.chat_routes import _policy_context_pre_check
from datus.configuration.agent_config import AgentConfig, NodeConfig
from datus.plugins.registry import invalidate_plugin_cache
from datus.tools.db_tools.config import DuckDBConfig
from datus.tools.db_tools.duckdb_connector import DuckdbConnector
from datus.tools.func_tool.database import DBFuncTool
from datus.tools.permission.bash_rules import evaluate_bash_command
from datus.tools.permission.permission_config import PermissionLevel

SQL_POLICY_ID = "sp_store_scope_sql"


def _policies() -> list[dict]:
    return [
        {
            "id": SQL_POLICY_ID,
            "name": "store_scope_sql",
            "type": "row_filter",
            "applies_to": {"datasources": ["warehouse"], "tables": ["orders.main.orders"]},
            "condition": {
                "column": "store_id",
                "operator": "in",
                "value_from": "policy_context.row_filter.store_ids",
            },
        },
    ]


def _scoped_context(*, sql_mode: str | None = None) -> dict:
    """Build scoped policy context, optionally overriding either policy mode."""
    row_filter = {"access_mode": "scoped", "store_ids": ["S001"]}
    policy_modes = {}
    if sql_mode is not None:
        policy_modes[SQL_POLICY_ID] = sql_mode
    if policy_modes:
        row_filter["policy_modes"] = policy_modes
    return {"row_filter": row_filter}


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
                        "type": "duckdb",
                        "uri": str(database),
                        "read_only": True,
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

    database = tmp_path / "orders.duckdb"
    with duckdb.connect(str(database)) as seed:
        seed.execute(
            "CREATE TABLE orders (store_id TEXT NOT NULL, amount INTEGER NOT NULL);"
            "INSERT INTO orders VALUES ('S001', 10), ('S002', 20);"
        )
    connector = DuckdbConnector(DuckDBConfig(db_path=str(database), read_only=True))
    try:
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
        assert "1 policy configured" in status.stdout
        assert "store_scope_sql" in status.stdout
        assert SQL_POLICY_ID in status.stdout
        checked = managed_plugin_runtime.run(
            "sql-policy",
            "check",
            "--sql",
            "SELECT store_id, amount FROM orders ORDER BY store_id",
            "--datasource",
            "warehouse",
            "--dialect",
            "duckdb",
            "--policy-context",
            json.dumps(_scoped_context()),
            check=False,
        )
        assert checked.returncode == 1
        assert "connection-bound native" in checked.stdout

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

        # One physical row rule protects actual Dosi execution too; no tool
        # argument transformer adds a business WHERE predicate.
        from datus.tools.policy_runtime import PolicyRuntime
        from datus.tools.semantic_tools.dosi.config import DosiConfig
        from datus.tools.semantic_tools.dosi.runtime import DosiRuntime

        model_file = tmp_path / "model.json"
        model_file.write_text(
            json.dumps(
                {
                    "version": "0.2.0.dev0",
                    "name": "orders_model",
                    "datasets": [
                        {
                            "name": "orders",
                            "source": "orders",
                            "fields": [
                                {
                                    "name": "amount",
                                    "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "amount"}]},
                                }
                            ],
                        }
                    ],
                    "metrics": [
                        {
                            "name": "order_revenue",
                            "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.amount)"}]},
                        }
                    ],
                }
            )
        )
        runtime = DosiRuntime(
            DosiConfig(
                semantic_model_path=str(model_file),
                datasource="warehouse",
                db_config={"type": "duckdb", "uri": str(database)},
            ),
            read_policy_provider=lambda: PolicyRuntime(config).bind_read(
                connector=connector, datasource="warehouse", policy_context=config.policy_context
            ),
        )
        metric = await runtime.query_metrics(["order_revenue"])
        assert metric.data == [{"order_revenue": 10}]

        sql_unrestricted = _scoped_context(sql_mode="unrestricted")
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
        metric = await runtime.query_metrics(["order_revenue"])
        assert metric.data == [{"order_revenue": 30}]

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

        # Exercise the real HTTP route, auth context, task manager and SQL tool.
        from datus.api.services.datus_service import DatusService
        from tests.helpers.product_acceptance import ModelServer, answer, isolated_connectors, tool_call
        from tests.integration.api.test_api_chat import chat_http_client, find_event, parse_sse_body

        with isolated_connectors(), ModelServer().running() as server:
            document["agent"]["models"]["mock"].update(
                model="core-test-model", base_url=server.url, context_length=262144, max_tokens=2048
            )
            config_file.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            http_config = _agent_config(document)
            service = DatusService(agent_config=http_config, project_id="policy_http")
            server.enqueue(
                tool_call("execute_sql", {"sql": "SELECT store_id, amount FROM orders ORDER BY store_id"}),
                answer("The scoped amount is 10."),
            )
            async with chat_http_client(http_config, service, config_path=str(config_file)) as client:
                response = await client.post(
                    "/api/v1/chat/stream",
                    headers={"X-Datus-Policy-Context": json.dumps(scoped)},
                    json={"message": "Read the order amounts using SQL.", "interactive": False},
                )
                assert response.status_code == 200, response.text
                events = parse_sse_body(response.text)
                assert find_event(events, "error") is None, response.text
                assert find_event(events, "end") is not None, response.text
                tool_results = [m for m in server.requests[-1]["messages"] if m.get("role") == "tool"]
                assert "S001" in json.dumps(tool_results)
                assert "S002" not in json.dumps(tool_results)
                for context in (denied, unknown_policy):
                    rejected = await client.post(
                        "/api/v1/chat/stream",
                        headers={"X-Datus-Policy-Context": json.dumps(context)},
                        json={"message": "Read all order amounts.", "interactive": False},
                    )
                    assert "POLICY_CONTEXT_REJECTED" in rejected.text, rejected.text
                assert len(server.requests) == 2

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
