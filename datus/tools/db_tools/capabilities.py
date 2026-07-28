# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Helpers for connector capabilities that may depend on live configuration."""

from typing import Any, Optional, Set

from datus_db_core import connector_registry


def get_effective_capabilities(connector: Optional[Any] = None, dialect: str = "") -> Set[str]:
    """Return instance capabilities when available, otherwise registry defaults."""
    if connector is not None:
        getter = getattr(connector, "get_effective_capabilities", None)
        if callable(getter):
            capabilities = getter()
            if isinstance(capabilities, (set, frozenset, list, tuple)):
                return set(capabilities)
        dialect = dialect or getattr(connector, "dialect", "")
    getter = getattr(connector_registry, "get_capabilities", None)
    if callable(getter):
        return set(getter(dialect))
    return {
        namespace
        for namespace, supported in (
            ("catalog", connector_registry.support_catalog(dialect)),
            ("database", connector_registry.support_database(dialect)),
            ("schema", connector_registry.support_schema(dialect)),
        )
        if supported
    }


def supports_namespace(namespace: str, connector: Optional[Any] = None, dialect: str = "") -> bool:
    return namespace in get_effective_capabilities(connector=connector, dialect=dialect)
