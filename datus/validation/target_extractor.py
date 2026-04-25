# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Helpers for mutating tools to populate ``FuncToolResult.result["deliverable_target"]``.

Each mutating tool calls :func:`extract_ddl_target` / :func:`extract_dml_target`
on the SQL it is about to execute and stores the returned
:class:`datus.validation.report.TableTarget` in the tool's result. The hook
later reads this field to drive validation.

When extraction fails (unusual DDL variant, parser error) the function returns
``None`` — the calling tool should just omit the ``deliverable_target`` key,
and :class:`datus.validation.hook.ValidationHook` will skip validation for that
call rather than raising.
"""

from __future__ import annotations

from typing import Optional

import sqlglot
from sqlglot import expressions
from sqlglot.errors import ParseError

from datus.utils.loggings import get_logger
from datus.utils.sql_utils import parse_dialect
from datus.validation.report import TableTarget

logger = get_logger(__name__)


def extract_ddl_target(
    sql: str, datasource: str, active_database: str = "", dialect: str = ""
) -> Optional[TableTarget]:
    """Parse a DDL statement and extract its target table, if any.

    Supports:

    - ``CREATE TABLE [IF NOT EXISTS] ...``
    - ``CREATE OR REPLACE TABLE ...``
    - ``CREATE TEMPORARY TABLE ...``
    - ``CREATE TABLE ... AS SELECT ...`` (CTAS)
    - Schema-qualified (``schema.table``) and database-qualified
      (``db.schema.table``) identifiers
    - Quoted identifiers (backticks, double quotes, brackets) — sqlglot strips
      them transparently

    Non-target DDL (``DROP TABLE``, ``ALTER TABLE``, ``CREATE/DROP SCHEMA``)
    returns ``None`` — the hook will not run target-scoped checks.

    Args:
        sql: Cleaned DDL statement (comments stripped, single statement).
        datasource: Connector key the tool executed through. Carried on
            :attr:`TableTarget.datasource` so Layer A routes ``describe_table``
            back to the same connector.
        active_database: Physical database currently selected on the
            connector (e.g. ``ac_manage`` on StarRocks, the DuckDB file
            stem). Used to populate :attr:`TableTarget.database` when the
            SQL does not qualify the table with an explicit database.
            Falls back to ``datasource`` when empty — keeps backward
            compatibility for tests / adapters that never expose a
            ``database`` attribute.
        dialect: sqlglot dialect name (optional; empty = default parsing).

    Returns:
        :class:`TableTarget` when the DDL creates a table we can validate, else
        ``None``.
    """
    if not sql or not sql.strip():
        return None

    normalized_dialect = parse_dialect(dialect) if dialect else ""
    try:
        parsed = sqlglot.parse_one(sql, dialect=normalized_dialect or None, error_level=sqlglot.ErrorLevel.IGNORE)
    except (ParseError, ValueError) as e:
        logger.debug("sqlglot failed to parse DDL for target extraction: %s", e)
        return None

    if parsed is None:
        return None

    # Only CREATE statements produce a validatable target. ALTER / DROP modify
    # or remove, and are not something we run post-write checks against (the
    # table we'd describe may no longer exist).
    if not isinstance(parsed, expressions.Create):
        return None

    # ``kind`` is "TABLE" / "VIEW" / "SCHEMA" etc. — only TABLE is a validation
    # target (views don't have row counts to gate on).
    kind = (parsed.args.get("kind") or "").upper()
    if kind != "TABLE":
        return None

    target_expr = parsed.this
    # ``parsed.this`` for CREATE TABLE is typically a Schema expression wrapping
    # the Table; CTAS may have a Table directly. Normalize to the Table.
    if isinstance(target_expr, expressions.Schema):
        target_expr = target_expr.this

    if not isinstance(target_expr, expressions.Table):
        return None

    return _table_to_target(target_expr, datasource, active_database, dialect=normalized_dialect)


def extract_dml_target(
    sql: str, datasource: str, active_database: str = "", dialect: str = ""
) -> Optional[TableTarget]:
    """Parse an INSERT / UPDATE / DELETE and extract its target table.

    Args:
        sql: Cleaned DML statement.
        datasource: Connector key the tool executed through — carried on
            :attr:`TableTarget.datasource` for Layer A routing.
        active_database: Physical database currently selected on the
            connector. Used to populate :attr:`TableTarget.database` when
            the SQL does not explicitly qualify the table. Falls back to
            ``datasource`` when empty.
        dialect: sqlglot dialect name.

    Returns:
        :class:`TableTarget` for the table being written, else ``None``.
    """
    if not sql or not sql.strip():
        return None

    normalized_dialect = parse_dialect(dialect) if dialect else ""
    try:
        parsed = sqlglot.parse_one(sql, dialect=normalized_dialect or None, error_level=sqlglot.ErrorLevel.IGNORE)
    except (ParseError, ValueError) as e:
        logger.debug("sqlglot failed to parse DML for target extraction: %s", e)
        return None

    if parsed is None:
        return None

    target_expr: Optional[expressions.Expression] = None
    if isinstance(parsed, expressions.Insert):
        target_expr = parsed.this
        # INSERT INTO schema(table) sometimes wraps in Schema
        if isinstance(target_expr, expressions.Schema):
            target_expr = target_expr.this
    elif isinstance(parsed, expressions.Update):
        target_expr = parsed.this
    elif isinstance(parsed, expressions.Delete):
        target_expr = parsed.this

    if not isinstance(target_expr, expressions.Table):
        return None

    return _table_to_target(target_expr, datasource, active_database, dialect=normalized_dialect)


def _dialect_has_catalog(dialect: str) -> bool:
    """Does this dialect have a ``catalog`` tier above ``database``?

    Delegates to :meth:`ConnectorRegistry.support_catalog`, which is the
    single source of truth — each adapter declares its capabilities
    (``{"catalog", "database", "schema"}``) at registration. Engines like
    StarRocks / BigQuery / Trino that address tables as
    ``catalog.database.table`` register with ``"catalog"``; Postgres /
    Snowflake / MySQL do not.

    When the dialect is unknown to the registry (not yet registered, or
    empty string) we fall back to ``True`` — conservative, preserves the
    historical behaviour of populating ``catalog`` so downstream validators
    still have a leading component to render. Production runs register every
    shipped adapter at startup, so this fallback mainly fires for empty
    dialect strings.
    """
    if not dialect:
        return True
    try:
        from datus.tools.db_tools import connector_registry

        return bool(connector_registry.support_catalog(dialect))
    except Exception as e:
        logger.debug("connector_registry.support_catalog(%s) failed: %s", dialect, e)
        return True


def _table_to_target(
    table_expr: expressions.Table,
    datasource: str,
    active_database: str = "",
    dialect: str = "",
) -> Optional[TableTarget]:
    """Convert a sqlglot :class:`Table` expression into :class:`TableTarget`.

    Field semantics:

    - ``TableTarget.datasource`` always holds the connector routing key
      (``datasource`` arg).
    - ``TableTarget.database`` holds the **physical** database the table
      was actually written into. Resolution order: explicit SQL-level
      qualifier > ``active_database`` (connector's current namespace) >
      fallback to ``datasource`` for backward compatibility.

    Three-part identifier semantics depend on the dialect:

    - Dialects with a real catalog tier (StarRocks, BigQuery, Trino):
      ``a.b.c`` = ``catalog.database.table``. Populate ``catalog`` so
      validators can reconstruct the fully-qualified lookup.
    - Dialects without a catalog tier (Postgres, Snowflake, MySQL, SQLite):
      ``a.b.c`` = ``database.schema.table``. sqlglot's ``catalog`` slot
      actually holds the database here, so we roll it into ``database`` and
      leave ``TableTarget.catalog`` unset — otherwise the validator prompt
      would render a spurious ``Catalog:`` line and mislead the LLM.
    """
    name = _identifier_name(table_expr.args.get("this"))
    if not name:
        return None
    schema = _identifier_name(table_expr.args.get("db")) or None
    sql_catalog_slot = _identifier_name(table_expr.args.get("catalog")) or None

    datasource_key = datasource or None
    catalog: Optional[str] = None
    # Default the *physical* database to the connector's active namespace,
    # fall back to the datasource key only when active_database is empty
    # (tests / adapters that don't expose one).
    effective_db = active_database or datasource or ""
    effective_schema = schema
    if sql_catalog_slot:
        if _dialect_has_catalog(dialect):
            # a.b.c = catalog.<namespace>.table on catalog-tier engines
            # (StarRocks: catalog.database.table with no schema tier; Trino:
            # catalog.schema.table; BigQuery: project.dataset.table). sqlglot
            # fills slots left-to-right, so the middle component lives in the
            # "db" slot. Promote it to ``database`` and drop ``schema`` — the
            # target connector decides how to interpret "database" for its
            # own hierarchy.
            catalog = sql_catalog_slot
            effective_db = schema or ""
            effective_schema = None
        else:
            # a.b.c = db.schema.table on schema-tier engines (Postgres,
            # MySQL, Snowflake). Leading component is the database; schema
            # stays in its slot.
            effective_db = sql_catalog_slot
    return TableTarget(
        catalog=catalog,
        datasource=datasource_key,
        database=effective_db,
        db_schema=effective_schema,
        table=name,
    )


def _identifier_name(expr: Optional[expressions.Expression]) -> str:
    if expr is None:
        return ""
    if isinstance(expr, expressions.Identifier):
        return expr.name or ""
    if hasattr(expr, "name"):
        return expr.name or ""
    return ""
