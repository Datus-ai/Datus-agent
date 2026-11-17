# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Non-secret, immutable database coordinates resolved once for an execution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

EXECUTION_CONTEXT_META_KEY = "execution_context"


@dataclass(frozen=True)
class DatabaseExecutionContext:
    datasource: str = ""
    catalog: str = ""
    database: str = ""
    schema: str = ""

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None) -> DatabaseExecutionContext:
        values = values or {}

        def value(*keys: str) -> str:
            for key in keys:
                item = values.get(key)
                if item is not None and str(item).strip():
                    return str(item).strip()
            return ""

        return cls(
            datasource=value("datasource"),
            catalog=value("catalog", "catalog_name"),
            database=value("database", "database_name"),
            schema=value("schema", "db_schema", "schema_name"),
        )

    def to_dict(self, *, aliases: bool = False, include_empty: bool = False) -> dict[str, str]:
        result = {key: value for key, value in asdict(self).items() if value or include_empty}
        if aliases:
            for key, names in {
                "catalog": ("catalog_name",),
                "database": ("database_name",),
                "schema": ("db_schema", "schema_name"),
            }.items():
                if key in result:
                    result.update({name: result[key] for name in names})
        return result

    @property
    def identity(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def execution_context_for_node(agent_config: Any, user_input: Any = None) -> DatabaseExecutionContext:
    """Use the API's resolved context; retain CLI/workflow input compatibility."""
    getter = getattr(agent_config, "runtime_db_context", None)
    runtime = getter() if callable(getter) else {}
    if isinstance(runtime, dict) and runtime:
        return DatabaseExecutionContext.from_mapping(runtime)
    datasource = getattr(agent_config, "current_datasource", "") or ""
    try:
        config = agent_config.current_db_config(datasource)
    except Exception:
        config = None
    return DatabaseExecutionContext.from_mapping(
        {
            "datasource": datasource,
            "catalog": getattr(user_input, "catalog", None) or getattr(config, "catalog", None),
            "database": getattr(user_input, "database", None) or getattr(config, "database", None),
            "schema": getattr(user_input, "db_schema", None) or getattr(config, "schema", None),
        }
    )
