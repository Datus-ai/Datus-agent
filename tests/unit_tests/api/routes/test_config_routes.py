# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/api/routes/config_routes.py."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from datus.api.routes import config_routes
from datus.api.routes.config_routes import (
    ProbeDatabaseRequest,
    ProbeModelRequest,
    UpdateDatabasesRequest,
    UpdateModelsRequest,
    get_agent_config_endpoint,
    probe_database_connectivity_endpoint,
    probe_model_connectivity_endpoint,
    update_databases_endpoint,
    update_models_endpoint,
)
from datus.utils.exceptions import DatusException


def _mock_svc(databases, *, target="deepseek", current_database="starrocks", models=None, home="~/.datus"):
    svc = MagicMock()
    svc.agent_config.target = target
    svc.agent_config.models = models if models is not None else {}
    svc.agent_config.current_namespace = current_database
    svc.agent_config.namespaces = databases
    svc.agent_config.home = home
    return svc


@pytest.mark.asyncio
async def test_get_agent_config_flattens_matching_inner_key():
    """When inner key matches the database name, that entry is returned flat."""
    starrocks_cfg = {"logic_name": "starrocks", "type": "starrocks", "host": "h1"}
    starrocks22_cfg = {"logic_name": "starrocks22", "type": "starrocks", "host": "h2"}
    svc = _mock_svc(
        databases={
            "starrocks": {"starrocks": starrocks_cfg},
            "starrocks22": {"starrocks22": starrocks22_cfg},
        },
    )

    result = await get_agent_config_endpoint(svc)

    assert result.success is True
    assert result.data["databases"] == {
        "starrocks": starrocks_cfg,
        "starrocks22": starrocks22_cfg,
    }
    assert result.data["target"] == "deepseek"
    assert result.data["current_database"] == "starrocks"
    assert result.data["home"] == "~/.datus"


@pytest.mark.asyncio
async def test_get_agent_config_falls_back_to_first_inner_value():
    """When inner key does not match database name, first inner value is used."""
    inner_cfg = {"logic_name": "db_a", "type": "duckdb"}
    svc = _mock_svc(databases={"my_db": {"db_a": inner_cfg}})

    result = await get_agent_config_endpoint(svc)

    assert result.data["databases"] == {"my_db": inner_cfg}


@pytest.mark.asyncio
async def test_get_agent_config_skips_empty_inner_dict():
    """Databases with empty inner dicts are dropped from the response."""
    real_cfg = {"logic_name": "real", "type": "duckdb"}
    svc = _mock_svc(databases={"empty": {}, "real": {"real": real_cfg}})

    result = await get_agent_config_endpoint(svc)

    assert result.data["databases"] == {"real": real_cfg}


@pytest.mark.asyncio
async def test_get_agent_config_handles_empty_databases():
    svc = _mock_svc(databases={})

    result = await get_agent_config_endpoint(svc)

    assert result.data["databases"] == {}


class _FakeConfigManager:
    """Minimal stand-in for ConfigurationManager — captures save() calls."""

    def __init__(self, initial=None):
        self.data = dict(initial) if initial else {}
        self.save_count = 0

    def save(self):
        self.save_count += 1


@pytest.fixture
def patched_cm(monkeypatch):
    """Replace the module-level configuration_manager() with a fake instance."""
    cm = _FakeConfigManager()
    monkeypatch.setattr(config_routes, "configuration_manager", lambda: cm)
    return cm


@pytest.fixture
def patched_cache(monkeypatch):
    """Replace deps._service_cache with an AsyncMock so evict() is awaitable."""
    cache = MagicMock()
    cache.evict = AsyncMock()
    monkeypatch.setattr(config_routes.deps, "_service_cache", cache)
    return cache


def _ctx(project_id="proj_a"):
    return SimpleNamespace(project_id=project_id)


@pytest.mark.asyncio
async def test_update_databases_replaces_services_databases(patched_cm, patched_cache):
    patched_cm.data = {"services": {"databases": {"old": {"type": "duckdb"}}, "other": "keep"}}
    body = UpdateDatabasesRequest(
        databases={
            "db_a": {"type": "starrocks", "host": "h1"},
            "db_b": {"type": "duckdb", "uri": "/tmp/a.db"},
        }
    )

    result = await update_databases_endpoint(body, svc=MagicMock(), ctx=_ctx("proj_a"))

    assert result.success is True
    assert result.data == {"updated": True}
    assert patched_cm.data["services"]["databases"] == {
        "db_a": {"type": "starrocks", "host": "h1"},
        "db_b": {"type": "duckdb", "uri": "/tmp/a.db"},
    }
    assert patched_cm.data["services"]["other"] == "keep"
    assert patched_cm.save_count == 1
    patched_cache.evict.assert_awaited_once_with("proj_a")


@pytest.mark.asyncio
async def test_update_databases_empty_dict_clears_block(patched_cm, patched_cache):
    patched_cm.data = {"services": {"databases": {"old": {"type": "duckdb"}}}}

    result = await update_databases_endpoint(UpdateDatabasesRequest(databases={}), svc=MagicMock(), ctx=_ctx())

    assert result.data["updated"] is True
    assert patched_cm.data["services"]["databases"] == {}


@pytest.mark.asyncio
async def test_update_databases_rejects_invalid_name(patched_cm, patched_cache):
    body = UpdateDatabasesRequest(databases={"bad name!": {"type": "duckdb"}})
    with pytest.raises(DatusException):
        await update_databases_endpoint(body, svc=MagicMock(), ctx=_ctx())
    assert patched_cm.save_count == 0
    patched_cache.evict.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_databases_initializes_services_when_missing(patched_cm, patched_cache):
    patched_cm.data = {}

    await update_databases_endpoint(
        UpdateDatabasesRequest(databases={"db_a": {"type": "duckdb"}}),
        svc=MagicMock(),
        ctx=_ctx(),
    )

    assert patched_cm.data["services"]["databases"] == {"db_a": {"type": "duckdb"}}


@pytest.mark.asyncio
async def test_update_models_replaces_models_and_target(patched_cm, patched_cache):
    patched_cm.data = {"models": {"old": {"type": "openai"}}, "target": "old"}
    body = UpdateModelsRequest(
        models={"new": {"type": "deepseek", "model": "deepseek-chat"}},
        target="new",
    )

    result = await update_models_endpoint(body, svc=MagicMock(), ctx=_ctx("proj_b"))

    assert result.data["updated"] is True
    assert patched_cm.data["models"] == {"new": {"type": "deepseek", "model": "deepseek-chat"}}
    assert patched_cm.data["target"] == "new"
    assert patched_cm.save_count == 1
    patched_cache.evict.assert_awaited_once_with("proj_b")


@pytest.mark.asyncio
async def test_update_models_target_only(patched_cm, patched_cache):
    patched_cm.data = {"models": {"m1": {"type": "openai"}, "m2": {"type": "claude"}}, "target": "m1"}

    await update_models_endpoint(UpdateModelsRequest(target="m2"), svc=MagicMock(), ctx=_ctx())

    assert patched_cm.data["target"] == "m2"
    assert patched_cm.data["models"] == {"m1": {"type": "openai"}, "m2": {"type": "claude"}}


@pytest.mark.asyncio
async def test_update_models_models_only(patched_cm, patched_cache):
    patched_cm.data = {"models": {"old": {}}, "target": "old"}

    await update_models_endpoint(
        UpdateModelsRequest(models={"old": {"type": "openai"}}),
        svc=MagicMock(),
        ctx=_ctx(),
    )

    assert patched_cm.data["models"] == {"old": {"type": "openai"}}
    assert patched_cm.data["target"] == "old"


@pytest.mark.asyncio
async def test_update_models_requires_at_least_one_field(patched_cm, patched_cache):
    with pytest.raises(DatusException):
        await update_models_endpoint(UpdateModelsRequest(), svc=MagicMock(), ctx=_ctx())
    assert patched_cm.save_count == 0


@pytest.mark.asyncio
async def test_update_models_rejects_target_not_in_models(patched_cm, patched_cache):
    patched_cm.data = {"models": {"m1": {}}, "target": "m1"}

    with pytest.raises(DatusException):
        await update_models_endpoint(UpdateModelsRequest(target="ghost"), svc=MagicMock(), ctx=_ctx())
    assert patched_cm.save_count == 0


@pytest.mark.asyncio
async def test_update_models_target_validated_against_new_models(patched_cm, patched_cache):
    """When both models and target are provided, target must exist in the NEW models."""
    patched_cm.data = {"models": {"keep_me": {}}, "target": "keep_me"}

    with pytest.raises(DatusException):
        await update_models_endpoint(
            UpdateModelsRequest(models={"only_new": {"type": "openai"}}, target="keep_me"),
            svc=MagicMock(),
            ctx=_ctx(),
        )


@pytest.mark.asyncio
async def test_update_models_rejects_invalid_model_name(patched_cm, patched_cache):
    with pytest.raises(DatusException):
        await update_models_endpoint(
            UpdateModelsRequest(models={"bad name!": {"type": "openai"}}),
            svc=MagicMock(),
            ctx=_ctx(),
        )
    assert patched_cm.save_count == 0


@pytest.mark.asyncio
async def test_update_databases_survives_missing_service_cache(monkeypatch, patched_cm):
    """No crash when the service cache hasn't been initialized yet."""
    monkeypatch.setattr(config_routes.deps, "_service_cache", None)

    result = await update_databases_endpoint(
        UpdateDatabasesRequest(databases={"db_a": {"type": "duckdb"}}),
        svc=MagicMock(),
        ctx=_ctx(),
    )

    assert result.data["updated"] is True


@pytest.mark.asyncio
async def test_update_databases_uses_default_project_when_missing(patched_cm, patched_cache):
    await update_databases_endpoint(
        UpdateDatabasesRequest(databases={"db_a": {"type": "duckdb"}}),
        svc=MagicMock(),
        ctx=_ctx(project_id=None),
    )

    patched_cache.evict.assert_awaited_once_with("default")


@pytest.mark.asyncio
async def test_test_model_connectivity_ok(monkeypatch):
    """Successful LLM probe returns ok=True and forwards the payload unchanged."""
    captured = {}

    def fake_probe(payload):
        captured["payload"] = payload

    monkeypatch.setattr(config_routes, "_probe_llm_sync", fake_probe)

    body = ProbeModelRequest(type="openai", model="gpt-4o", api_key="sk-xxx", base_url="https://api.openai.com/v1")
    result = await probe_model_connectivity_endpoint(body, svc=MagicMock())

    assert result.success is True
    assert result.data == {"ok": True}
    assert captured["payload"]["type"] == "openai"
    assert captured["payload"]["api_key"] == "sk-xxx"


@pytest.mark.asyncio
async def test_test_model_connectivity_reports_error_message(monkeypatch):
    """Probe exception is surfaced as ok=False with message; HTTP stays 200."""

    def fake_probe(payload):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(config_routes, "_probe_llm_sync", fake_probe)

    body = ProbeModelRequest(type="openai", model="gpt-4o", api_key="bad")
    result = await probe_model_connectivity_endpoint(body, svc=MagicMock())

    assert result.success is True
    assert result.data["ok"] is False
    assert "401" in result.data["message"]


@pytest.mark.asyncio
async def test_test_model_connectivity_passes_extra_fields(monkeypatch):
    """Unknown fields on the request body are forwarded to the probe (extra=allow)."""
    captured = {}

    def fake_probe(payload):
        captured["payload"] = payload

    monkeypatch.setattr(config_routes, "_probe_llm_sync", fake_probe)

    body = ProbeModelRequest.model_validate({"type": "openai", "model": "gpt-4o", "api_key": "k", "vendor": "openai"})
    await probe_model_connectivity_endpoint(body, svc=MagicMock())

    assert captured["payload"].get("vendor") == "openai"


@pytest.mark.asyncio
async def test_test_database_connectivity_ok(monkeypatch):
    """Successful DB probe returns ok=True and forwards the payload unchanged."""
    captured = {}

    def fake_probe(payload):
        captured["payload"] = payload

    monkeypatch.setattr(config_routes, "_probe_database_sync", fake_probe)

    body = ProbeDatabaseRequest.model_validate({"type": "duckdb", "uri": "/tmp/test.duckdb"})
    result = await probe_database_connectivity_endpoint(body, svc=MagicMock())

    assert result.data == {"ok": True}
    assert captured["payload"]["type"] == "duckdb"
    assert captured["payload"]["uri"] == "/tmp/test.duckdb"


@pytest.mark.asyncio
async def test_test_database_connectivity_reports_error_message(monkeypatch):
    """Probe exception is surfaced as ok=False with message; HTTP stays 200."""

    def fake_probe(payload):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(config_routes, "_probe_database_sync", fake_probe)

    body = ProbeDatabaseRequest.model_validate({"type": "starrocks", "host": "unreachable", "port": "9999"})
    result = await probe_database_connectivity_endpoint(body, svc=MagicMock())

    assert result.data["ok"] is False
    assert "connection refused" in result.data["message"]
