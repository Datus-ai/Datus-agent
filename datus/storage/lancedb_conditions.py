# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Condition Builder for LanceDB `where` Clauses
=============================================

This module provides a small DSL for building structured query conditions
that compile into LanceDB-compatible `where` clause strings.

Why?
----
LanceDB supports SQL-like `where` strings, but does not support `IN` and
complex nested logical grouping directly. This utility lets you compose
conditions in Python objects, then compile them safely.

Main Features
-------------
- Atomic conditions with operators: =, !=, >, >=, <, <=, LIKE
- Simulated `IN` using OR chains
- Logical composition with AND, OR, NOT
- Automatic escaping of field names and values
- Pythonic factory helpers (eq, gt, in_, and_, or_, not_)
- Safe handling of NULL, booleans, dates, strings

Quick Examples
--------------
Example 1: Simple AND
    >>> expr = and_(eq("status", "active"), ge("score", 80))
    >>> build_where(expr)
    "(status = 'active' AND score >= 80)"

Example 2: Mixing AND/OR
    >>> expr = or_(
    ...     and_(eq("status", "active"), ge("score", 80)),
    ...     and_(eq("country", "US"), lt("age", 30)),
    ... )
    >>> build_where(expr)
    "((status = 'active' AND score >= 80) OR (country = 'US' AND age < 30))"

Example 3: IN expansion
    >>> expr = in_("type", ["A", "B", "C"])
    >>> build_where(expr)
    "(type = 'A' OR type = 'B' OR type = 'C')"

Example 4: Using NOT
    >>> expr = and_(
    ...     not_(eq("is_blocked", True)),
    ...     or_(like("name", "Alice%"), like("name", "%Bob%")),
    ... )
    >>> build_where(expr)
    "((NOT is_blocked = TRUE) AND (name LIKE 'Alice%' OR name LIKE '%Bob%'))"

Usage in LanceDB
----------------
    table = db.table("my_table")
    where_clause = build_where(expr)
    results = table.search("query").where(where_clause)

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Iterable, Optional, Sequence, Union

from datus.utils.exceptions import DatusException, ErrorCode


# ---------- Operators ----------
class Op(str, Enum):
    EQ = "="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    IN = "IN"  # Will be expanded into OR chain (LanceDB doesn't support native IN)
    LIKE = "LIKE"  # SQL-like semantics with % and _


# ---------- AST Nodes ----------
@dataclass(frozen=True)
class Condition:
    """
    Represents a simple condition such as `field = value` or `field > value`.
    Example:
        Condition("age", Op.GT, 18)   ->  "age > 18"
    """

    field: str
    op: Op
    value: Any


@dataclass(frozen=True)
class And:
    """
    Logical AND of multiple nodes.
    Example:
        And([Condition("status", Op.EQ, "active"), Condition("score", Op.GT, 80)])
        -> "(status = 'active' AND score > 80)"
    """

    nodes: Sequence["Node"]


@dataclass(frozen=True)
class Or:
    """
    Logical OR of multiple nodes.
    Example:
        Or([Condition("role", Op.EQ, "admin"), Condition("role", Op.EQ, "owner")])
        -> "(role = 'admin' OR role = 'owner')"
    """

    nodes: Sequence["Node"]


@dataclass(frozen=True)
class Not:
    """
    Logical negation of a node.
    Example:
        Not(Condition("is_blocked", Op.EQ, True))
        -> "(NOT is_blocked = TRUE)"
    """

    node: "Node"


Node = Union[Condition, And, Or, Not]


# ---------- Compilation Helpers ----------
def _escape_identifier(name: str) -> str:
    """
    Return a safe identifier for use in a where-clause.
    If the field contains spaces or special characters, wrap it in double quotes.
    Adjust this to your own identifier policy if needed.
    Example:
        "user name" -> "\"user name\""
    """
    safe = name.strip()
    if not safe:
        raise DatusException(ErrorCode.COMMON_FIELD_INVALID, message="Identifier cannot be empty")

    first_char_requires_quote = not (safe[0].isalpha() or safe[0] == "_")
    needs_quote = first_char_requires_quote or any(c in safe for c in ' "().-+/\\|&*[]=<>!')
    if needs_quote:
        escaped = safe.replace('"', '""')
        return f'"{escaped}"'
    return safe


def _escape_value(v: Any) -> str:
    """
    Convert a Python value into a SQL literal compatible with LanceDB where.
    - None is handled in the operator layer (IS NULL / IS NOT NULL) and returns 'NULL' here.
    - Booleans become TRUE/FALSE.
    - Numbers are unquoted.
    - date/datetime become ISO-8601 quoted strings.
    - Everything else becomes a single-quoted string with internal quotes escaped.

    Examples:
        42            -> "42"
        True          -> "TRUE"
        "O'Hara"      -> "'O''Hara'"
        datetime(...) -> "'2025-09-29T17:30:00'"
    """
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (datetime, date)):
        return f"'{v.isoformat()}'"
    s = str(v).replace("'", "''")
    return f"'{s}'"


def _compile_condition(c: Condition) -> str:
    """
    Convert a Condition object into a SQL-like string.
    Handles special cases: NULL and IN.
    """
    field = _escape_identifier(c.field)
    op = c.op

    # NULL handling for equality/inequality
    if c.value is None:
        if op == Op.EQ:
            return f"{field} IS NULL"
        if op == Op.NE:
            return f"{field} IS NOT NULL"
        raise ValueError(f"Operator {op} is invalid with NULL (field: {c.field})")

    # Emulate IN using OR chain
    if op == Op.IN:
        if not isinstance(c.value, Iterable) or isinstance(c.value, (str, bytes)):
            raise TypeError("IN expects a non-string iterable value")
        values = list(c.value)
        if not values:
            # Empty IN is always false
            return "1 = 0"
        non_null_values = [v for v in values if v is not None]
        include_null = any(v is None for v in values)

        parts = []
        if non_null_values:
            parts.extend(f"{field} = {_escape_value(v)}" for v in non_null_values)
        if include_null:
            parts.append(f"{field} IS NULL")
        return "(" + " OR ".join(parts) + ")"

    right = _escape_value(c.value)
    if op == Op.LIKE:
        return f"{field} LIKE {right}"
    if op in {Op.EQ, Op.NE, Op.GT, Op.GTE, Op.LT, Op.LTE}:
        return f"{field} {op.value} {right}"

    raise ValueError(f"Unsupported operator: {op}")


def _compile_node(node: Node) -> str:
    """Recursively compile an AST node into a string."""
    if isinstance(node, Condition):
        return _compile_condition(node)
    if isinstance(node, And):
        parts = [_compile_node(n) for n in node.nodes if n is not None]
        if not parts:
            return "1 = 1"
        return "(" + " AND ".join(parts) + ")"
    if isinstance(node, Or):
        parts = [_compile_node(n) for n in node.nodes if n is not None]
        if not parts:
            return "1 = 0"
        return "(" + " OR ".join(parts) + ")"
    if isinstance(node, Not):
        inner = _compile_node(node.node)
        return f"(NOT {inner})"
    raise TypeError(f"Unknown node type: {type(node)}")


WhereExpr = Union[str, Node, None]


def build_where(where: WhereExpr) -> Optional[str]:
    """
    Compile a structured AST into a LanceDB-compatible where clause.

    Example:
        expr = And([
            Condition("status", Op.EQ, "active"),
            Or([Condition("role", Op.EQ, "admin"), Condition("role", Op.EQ, "owner")])
        ])
        build_where(expr)
        -> "(status = 'active' AND (role = 'admin' OR role = 'owner'))"
    """
    if where is None:
        return None
    if isinstance(where, str):
        stripped = where.strip()
        return stripped or None
    return _compile_node(where)


# ---------- Convenience Constructors ----------
def eq(field: str, value: Any) -> Condition:
    return Condition(field, Op.EQ, value)


def ne(field: str, value: Any) -> Condition:
    return Condition(field, Op.NE, value)


def gte(field: str, value: Any) -> Condition:
    return Condition(field, Op.GTE, value)


def ge(field: str, value: Any) -> Condition:
    return Condition(field, Op.GTE, value)


def lt(field: str, value: Any) -> Condition:
    return Condition(field, Op.LT, value)


def lte(field: str, value: Any) -> Condition:
    return Condition(field, Op.LTE, value)


def in_(field: str, values: Iterable[Any]) -> Condition:
    return Condition(field, Op.IN, list(values))


def like(field: str, pattern: str) -> Condition:
    return Condition(field, Op.LIKE, _replace_wildcard(pattern))


def _replace_wildcard(value: str) -> str:
    return value.replace("*", "%")


# Logical helpers
def and_(*nodes: Node) -> And:
    return And(nodes)


def or_(*nodes: Node) -> Or:
    return Or(nodes)


def not_(node: Node) -> Not:
    return Not(node)
