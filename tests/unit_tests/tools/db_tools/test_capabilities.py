# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

from datus_db_core import connector_registry

from datus.tools.db_tools import capabilities as capabilities_module
from datus.tools.db_tools.capabilities import get_dialect_operations, get_effective_capabilities, supports_namespace


class DynamicConnector:
    dialect = "flexdb"

    def __init__(self, capabilities: set[str]) -> None:
        self.capabilities = capabilities

    def get_effective_capabilities(self) -> set[str]:
        return self.capabilities


def test_instance_capabilities_override_registered_maximum():
    saved = connector_registry._capabilities.copy()
    try:
        connector_registry.register_handlers("flexdb", capabilities={"database", "schema"})
        connector = DynamicConnector({"database"})

        assert get_effective_capabilities(connector) == {"database"}
        assert supports_namespace("database", connector=connector)
        assert not supports_namespace("schema", connector=connector)
    finally:
        connector_registry._capabilities.clear()
        connector_registry._capabilities.update(saved)


def test_static_capabilities_remain_fallback_for_existing_adapters():
    saved = connector_registry._capabilities.copy()
    try:
        connector_registry.register_handlers("legacydb", capabilities={"database", "schema"})
        assert get_effective_capabilities(dialect="legacydb") == {"database", "schema"}
    finally:
        connector_registry._capabilities.clear()
        connector_registry._capabilities.update(saved)


def test_dialect_operations_are_resolved_from_connector_dialect(monkeypatch):
    operations = object()
    calls = []
    monkeypatch.setattr(
        connector_registry,
        "get_dialect_operations",
        lambda dialect: calls.append(dialect) or operations,
        raising=False,
    )

    connector = type("OracleConnector", (), {"dialect": "ORACLE"})()

    assert get_dialect_operations(connector=connector) is operations
    assert calls == ["oracle"]


def test_dialect_operations_remain_optional_for_older_core(monkeypatch):
    monkeypatch.setattr(capabilities_module, "connector_registry", object())

    assert get_dialect_operations(dialect="oracle") is None
