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
from datus.validation.report import TableTarget

logger = get_logger(__name__)


def extract_ddl_target(sql: str, database: str, dialect: str = "") -> Optional[TableTarget]:
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
        sql: Cleaned DDL statement (comments stripped, single statement)
        database: Database this DDL is executed against
        dialect: sqlglot dialect name (optional; empty = default parsing)

    Returns:
        :class:`TableTarget` when the DDL creates a table we can validate, else
        ``None``.
    """
    if not sql or not sql.strip():
        return None

    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect or None, error_level=sqlglot.ErrorLevel.IGNORE)
    except ParseError as e:
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

    return _table_to_target(target_expr, database)


def extract_dml_target(sql: str, database: str, dialect: str = "") -> Optional[TableTarget]:
    """Parse an INSERT / UPDATE / DELETE and extract its target table.

    Args:
        sql: Cleaned DML statement
        database: Database this DML is executed against
        dialect: sqlglot dialect name

    Returns:
        :class:`TableTarget` for the table being written, else ``None``.
    """
    if not sql or not sql.strip():
        return None

    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect or None, error_level=sqlglot.ErrorLevel.IGNORE)
    except ParseError as e:
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

    return _table_to_target(target_expr, database)


def _table_to_target(table_expr: expressions.Table, database: str) -> Optional[TableTarget]:
    """Convert a sqlglot :class:`Table` expression into :class:`TableTarget`."""
    name = _identifier_name(table_expr.args.get("this"))
    if not name:
        return None
    schema = _identifier_name(table_expr.args.get("db")) or None
    # If the SQL qualified with a leading DB (e.g. ``db.schema.table``) we
    # prefer the one in the SQL over the tool-level ``database`` parameter.
    sql_db = _identifier_name(table_expr.args.get("catalog")) or None
    effective_db = sql_db or database or ""
    return TableTarget(database=effective_db, db_schema=schema, table=name)


def _identifier_name(expr: Optional[expressions.Expression]) -> str:
    if expr is None:
        return ""
    if isinstance(expr, expressions.Identifier):
        return expr.name or ""
    if hasattr(expr, "name"):
        return expr.name or ""
    return ""
