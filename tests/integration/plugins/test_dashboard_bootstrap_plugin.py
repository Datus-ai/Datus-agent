# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Dashboard Bootstrap and managed Superset plugin P0 integration."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml
from rich.console import Console

from datus.cli.bootstrap_bi_picker import BootstrapBiPicker
from datus.configuration.agent_config_loader import load_agent_config
from tests.conftest import TEST_CONF_DIR
from tests.integration.tools.test_bi_dashboard import _create_adapter, _ensure_seed_dashboard


def _print_mode_tool_calls(stdout: str) -> list[dict]:
    """Extract actual tool invocations from print-mode JSONL output."""
    calls = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        calls.extend(
            item["payload"]
            for item in payload.get("content", [])
            if item.get("type") == "call-tool" and isinstance(item.get("payload"), dict)
        )
    return calls


def _plugin_agent_document(home: Path, workspace: Path, superset_url: str) -> dict:
    return {
        "agent": {
            "home": str(home),
            "project_name": "nightly_dashboard_bootstrap",
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
            "plugins": {
                "superset": {
                    "local": {
                        "default": True,
                        "api_base_url": superset_url,
                        "auth_mode": "login",
                        "username": "admin",
                        "password": "admin",
                        "provider": "db",
                        "verify_ssl": True,
                        "timeout": 30,
                        "serving_datasource": "superset",
                        "serving_database_name": "superset_examples",
                    }
                }
            },
        }
    }


def _llm_agent_document(home: Path, workspace: Path, superset_url: str) -> dict:
    document = yaml.safe_load((TEST_CONF_DIR / "agent.yml").read_text(encoding="utf-8"))
    agent = document["agent"]
    agent["home"] = str(home)
    agent["project_name"] = "nightly_dashboard_bootstrap_llm"
    agent["project_root"] = str(workspace)
    agent["plugins"] = _plugin_agent_document(home, workspace, superset_url)["agent"]["plugins"]
    for datasource in agent["services"]["datasources"].values():
        datasource.pop("default", None)
    agent["services"]["datasources"]["superset"] = {
        "type": "postgresql",
        "host": os.environ.get("SUPERSET_POSTGRES_HOST", "127.0.0.1"),
        "port": os.environ.get("SUPERSET_POSTGRES_PORT", "5433"),
        "username": "superset",
        "password": "superset",
        "database": "superset_examples",
        "schema": "public",
        "default": True,
    }
    agent["services"]["bi_platforms"]["superset"]["api_base_url"] = superset_url
    return document


def _seed_dashboard(superset_url: str) -> tuple[int, str]:
    config = load_agent_config(
        config=str(TEST_CONF_DIR / "agent.yml"),
        datasource="superset",
        reload=True,
        force=True,
        yes=True,
    )
    config.dashboard_config["superset"].api_base_url = superset_url

    database = config.services.datasources["superset"]
    database.host = os.environ.get("SUPERSET_POSTGRES_HOST", database.host)
    database.port = os.environ.get("SUPERSET_POSTGRES_PORT", "5433")
    database.username = "superset"
    database.password = "superset"
    database.database = "superset_examples"
    database.schema = "public"

    item = {
        "platform": "superset",
        "api_base_url": superset_url,
        "dashboard_url": f"{superset_url}/superset/dashboard/datus-nightly-placeholder/",
        "dialect": "postgresql",
        "seed_dashboard": True,
    }
    picker = BootstrapBiPicker(config, Console(log_path=False, force_terminal=False))
    adapter = _create_adapter(picker, config, item)
    try:
        _ensure_seed_dashboard(adapter, item)
        dashboard_id = int(adapter.parse_dashboard_id(item["dashboard_url"]))
        dashboard = adapter.get_dashboard_info(dashboard_id)
        assert dashboard is not None
        return dashboard_id, dashboard.name
    finally:
        adapter.close()


@pytest.mark.nightly
@pytest.mark.timeout(300)
def test_dashboard_bootstrap_installs_plugin_and_exports_verified_superset_queries(
    p0_external_sources,
    managed_plugin_runtime,
    tmp_path,
):
    """Exercise source install, discovery, real Superset export, and manifest integrity."""
    assert importlib.util.find_spec("datus_superset_plugin") is None, (
        "P0 must not have a globally installed datus-superset-plugin fallback"
    )
    superset_url = os.environ.get("SUPERSET_URL", "http://127.0.0.1:8088").rstrip("/")
    dashboard_id, dashboard_name = _seed_dashboard(superset_url)

    installed = managed_plugin_runtime.install(p0_external_sources.superset_plugin)
    assert "superset" in installed.stdout

    datus_home = managed_plugin_runtime.home / ".datus"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    document = _plugin_agent_document(datus_home, workspace, superset_url)
    config_file = datus_home / "conf" / "agent.yml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    listed_plugins = managed_plugin_runtime.run("plugin", "list")
    assert "superset" in listed_plugins.stdout
    info = managed_plugin_runtime.run("plugin", "info", "superset")
    assert "entry:   datus_superset_plugin" in info.stdout
    installed_export_skill = (
        datus_home / "plugins" / "superset" / "datus_superset_plugin" / "skills" / "superset-query-export" / "SKILL.md"
    )
    assert installed_export_skill.is_file()
    assert "context export-dashboard" in installed_export_skill.read_text(encoding="utf-8")

    health = managed_plugin_runtime.run("superset", "--profile", "local", "status", "health", "-o", "json")
    assert json.loads(health.stdout) == "OK"

    dashboards = managed_plugin_runtime.run(
        "superset",
        "--profile",
        "local",
        "dashboards",
        "list",
        "--param",
        f'q={{"filters":[{{"col":"id","opr":"eq","value":{dashboard_id}}}]}}',
        "-o",
        "json",
    )
    dashboard_rows = json.loads(dashboards.stdout)["result"]
    assert [int(row["id"]) for row in dashboard_rows] == [dashboard_id]

    exported = managed_plugin_runtime.run(
        "superset",
        "--profile",
        "local",
        "context",
        "export-dashboard",
        str(dashboard_id),
        "--output-root",
        "reference_sql",
        "-o",
        "json",
        cwd=workspace,
    )
    export_result = json.loads(exported.stdout)
    assert export_result["failed"] == 0
    assert export_result["succeeded"] == 2

    output_dir = Path(export_result["output_dir"])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["platform"] == "superset"
    assert manifest["profile"] == "local"
    assert int(manifest["dashboard"]["id"]) == dashboard_id
    assert manifest["dashboard"]["title"] == dashboard_name
    assert manifest["summary"] == {"total": 2, "succeeded": 2, "failed": 0}

    for query in manifest["queries"]:
        assert query["status"] == "ok"
        assert query["asset_id"]
        assert query["datasource"]
        sql_file = output_dir / query["file"]
        assert sql_file.is_file()
        sql_text = sql_file.read_text(encoding="utf-8")
        assert hashlib.sha256(sql_text.encode()).hexdigest() == query["sha256"]
        assert "SELECT" in sql_text.upper()

    source_text = "\n".join(path.read_text(encoding="utf-8") for path in (output_dir / "_source").glob("*.json"))
    assert '"password"' not in source_text.lower()
    assert '"access_token"' not in source_text.lower()


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.timeout(300)
def test_dashboard_bootstrap_real_llm_loads_skill_and_stops_at_manifest(
    p0_external_sources,
    managed_plugin_runtime,
    tmp_path,
):
    """Keep one real-model smoke at the skill's explicit confirmation boundary."""
    assert os.environ.get("DEEPSEEK_API_KEY"), "P0 Dashboard LLM smoke requires DEEPSEEK_API_KEY"
    assert importlib.util.find_spec("datus_superset_plugin") is None, (
        "P0 must not have a globally installed datus-superset-plugin fallback"
    )
    superset_url = os.environ.get("SUPERSET_URL", "http://127.0.0.1:8088").rstrip("/")
    dashboard_id, dashboard_name = _seed_dashboard(superset_url)
    managed_plugin_runtime.install(p0_external_sources.superset_plugin)

    datus_home = managed_plugin_runtime.home / ".datus"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    document = _llm_agent_document(datus_home, workspace, superset_url)
    config_file = datus_home / "conf" / "agent.yml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    prompt = (
        "Bootstrap reference SQL and metrics from the installed Superset plugin, profile local, "
        f"dashboard id {dashboard_id} ({dashboard_name}). Select every visible chart for both scopes. "
        "Do not run automatically: follow dashboard-bootstrap and stop after emitting the Generation Manifest."
    )
    result = managed_plugin_runtime.run(
        "--config",
        str(config_file),
        "--datasource",
        "superset",
        "--permission-mode",
        "dangerous",
        "--print",
        prompt,
        cwd=workspace,
        timeout=300,
    )
    tool_calls = _print_mode_tool_calls(result.stdout)
    assert any(
        call.get("toolName") == "load_skill" and call.get("toolParams", {}).get("skill_name") == "dashboard-bootstrap"
        for call in tool_calls
    )
    assert not any("export-dashboard" in json.dumps(call.get("toolParams", {})) for call in tool_calls)
    assert "dashboard-bootstrap" in result.stdout
    assert "Generation Manifest" in result.stdout
    assert dashboard_name in result.stdout
    assert not (workspace / "reference_sql").exists()
    assert not list(workspace.rglob("manifest.json"))
