# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Resolve StarRocks physical table identities before SQL policy matching."""

from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from datus.utils.exceptions import DatusException, ErrorCode


def resolve_policy_sql(sql: str, connector: Any) -> str:
    """Give policies the same catalog/database identity the connection uses.

    StarRocks policies authored from metadata have catalog.database.table names,
    while both authored SQL and semantic compilers commonly emit short names.
    Resolve only physical sources, leaving CTE names, explicit qualifiers and
    aliases intact. Other dialects retain their existing behavior: their default
    namespace/search-path semantics require their own adapter contract.
    """
    if connector.dialect != "starrocks":
        return sql

    try:
        tree = sqlglot.parse_one(sql, dialect="starrocks")
        tables = {
            id(source): source
            for scope in traverse_scope(tree)
            for _, source in scope.selected_sources.values()
            if isinstance(source, exp.Table)
        }
        unresolved = [table for table in tables.values() if not table.catalog or not table.db]
        if not unresolved:
            return sql

        context = connector.get_current_context()
        for table in unresolved:
            for part, field in (("catalog", "catalog_name"), ("db", "database_name")):
                if table.args.get(part) is not None:
                    continue
                value = context.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Unresolved SQL namespace")
                table.set(part, exp.to_identifier(value, quoted=True))

        return tree.sql(dialect="starrocks")
    except Exception as exc:
        # SQL and connector errors can contain private query/connection values.
        raise DatusException(
            ErrorCode.POLICY_DENIED,
            message="Cannot resolve physical table names for SQL policy enforcement; qualify the table names.",
        ) from exc
