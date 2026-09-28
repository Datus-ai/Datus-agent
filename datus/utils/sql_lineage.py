# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Deterministic table-level lineage and join-key extraction from SQL text.

Pure functions only — no LLM, no database, no filesystem policy. The func-tool
wrapper (``datus/tools/func_tool/lineage_tools.py``) owns path resolution and
result shaping; everything here takes SQL strings and returns plain data so it
can be unit-tested in isolation.

Two artifacts come out of every statement:

- **lineage edges** — ``target <- sources`` for statements that write a table
  (INSERT / INSERT OVERWRITE / CTAS / CREATE VIEW / MERGE / UPDATE ... FROM /
  DELETE ... USING). Pure ``SELECT`` statements produce no edge.
- **join edges** — equality predicates between two columns (``JOIN ... ON`` and
  implicit ``WHERE a.x = b.y``) whose sides resolve, through derived tables and
  CTEs, to physical table columns. Collected from every statement, including
  pure queries, so a validated-query corpus contributes relationship evidence.

- **rules** — constant predicates (``col = 1``, ``col NOT IN (...)``, ``col >
  '2024-12-07'``) in WHERE / ON / CASE conditions, CASE / IF value-to-label
  mappings, and ``ROW_NUMBER()`` partitions, each attributed to the physical
  column it tests.
- **comments** — ``--`` and ``/* */`` comments with the code they annotate.
  Comments are the only place a script's author explains intent in their own
  words, so prose notes are kept verbatim; commented-out code is flagged.

Join cardinality, layer inference and business meaning are deliberately out of
scope — those need the database or judgment and belong to the calling skill.
"""

from __future__ import annotations

import re
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import ParseError, TokenError
from sqlglot.optimizer.scope import Scope, build_scope

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_MAX_RESOLVE_DEPTH = 12

# Dialect used to render expressions back to SQL (transforms, window keys), so
# they read the way the author wrote them rather than in sqlglot's default dialect.
_RENDER_DIALECT: ContextVar[Optional[str]] = ContextVar("sql_lineage_render_dialect", default=None)


def _render(node: exp.Expression) -> str:
    return node.sql(dialect=_RENDER_DIALECT.get(), comments=False)


# dbt macros resolve to real table names; every other template construct is
# replaced by an identifier-safe placeholder so the statement still parses.
_DBT_REF_RE = re.compile(r"\{\{\s*ref\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}")
_DBT_SOURCE_RE = re.compile(r"\{\{\s*source\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}")
_JINJA_BLOCK_RE = re.compile(r"\{%-?.*?-?%\}", re.DOTALL)
_JINJA_COMMENT_RE = re.compile(r"\{#.*?#\}", re.DOTALL)
_JINJA_EXPR_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)
_DOLLAR_VAR_RE = re.compile(r"\$\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}")

_PY_TRIPLE_QUOTED_RE = re.compile(r"(?:[rRfFbBuU]{0,2})(\"\"\"|''')(.*?)\1", re.DOTALL)
_SQL_LEAD_RE = re.compile(r"^\s*(?:--[^\n]*\n\s*)*(insert|create|merge|with|select|update|delete)\b", re.IGNORECASE)


@dataclass
class SqlFragment:
    """A chunk of SQL text and where it came from."""

    text: str
    file: str
    line_offset: int = 0  # 0-based line of ``text``'s first line inside ``file``


@dataclass
class LineageEdge:
    target: str
    sources: List[str]
    statement: str
    file: str
    line: int
    templated: bool = False
    via_temp: List[str] = field(default_factory=list)
    # truncate_reload / overwrite / incremental / insert — how the script loads the target
    load_mode: str = "insert"
    window: List[str] = field(default_factory=list)  # templated predicates bounding an incremental load


@dataclass
class JoinEdge:
    """One join between two physical tables; composite keys stay together."""

    left_table: str
    right_table: str
    keys: List[Tuple[str, str]]  # (left_column, right_column), sorted
    join_type: str  # INNER / LEFT / RIGHT / FULL / CROSS / WHERE
    file: str
    line: int
    # "table.column" -> transform applied before the comparison, e.g. LEFT(issue_id, 6)
    transforms: Dict[str, str] = field(default_factory=dict)
    statement_target: Optional[str] = None  # table the enclosing statement writes, if any


@dataclass
class Predicate:
    """A constant test on a physical column, e.g. ``MA_billing_status = 1``."""

    table: str
    column: str
    predicate: str  # "= 1", "NOT IN ('a', 'b')", "LIKE '%x%'", ...
    clause: str  # WHERE / JOIN / CASE
    file: str
    line: int
    statement_target: Optional[str] = None
    transform: Optional[str] = None


@dataclass
class ValueMapping:
    """``CASE WHEN col = <value> THEN '<label>'`` (or ``IF``) on a physical column."""

    table: str
    column: str
    value: str
    label: str
    file: str
    line: int
    statement_target: Optional[str] = None
    transform: Optional[str] = None  # the mapping applies to this expression of the column, not the raw value


@dataclass
class Dedup:
    """``ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)`` resolved to physical columns."""

    table: str
    partition_by: List[str]  # "column" or "TRANSFORM(column)"
    order_by: List[str]
    file: str
    line: int
    statement_target: Optional[str] = None


@dataclass
class Comment:
    text: str
    code: str  # the code line the comment annotates (same line, else the next code line)
    # "note" (prose written by a person), "code" (commented-out SQL), or "header" (the comment block a
    # file opens with: its description, often a question / knowledge / owner block — kept in full)
    kind: str
    file: str
    line: int
    statement_target: Optional[str] = None


@dataclass
class Unresolved:
    file: str
    line: int
    reason: str


@dataclass
class ExtractionResult:
    lineage: List[LineageEdge] = field(default_factory=list)
    joins: List[JoinEdge] = field(default_factory=list)
    predicates: List[Predicate] = field(default_factory=list)
    mappings: List[ValueMapping] = field(default_factory=list)
    dedups: List[Dedup] = field(default_factory=list)
    comments: List[Comment] = field(default_factory=list)
    unresolved: List[Unresolved] = field(default_factory=list)
    statements: int = 0
    parsed: int = 0
    queries: int = 0  # parsed statements with no write target
    join_predicates: int = 0  # column = column equalities seen in join / where clauses
    join_predicates_unresolved: int = 0  # equalities whose sides do not trace to physical columns
    rule_predicates: int = 0  # constant predicates seen (WHERE / JOIN / CASE)
    rule_predicates_unresolved: int = 0  # constant predicates whose column does not trace to a physical column


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_template(sql: str) -> Tuple[str, bool]:
    """Neutralize templating so sqlglot can parse the statement.

    Returns ``(sql, templated)``. ``${var}`` becomes ``__tpl_var__`` (valid both
    inside string literals and as an identifier part), dbt ``ref`` / ``source``
    become the referenced table name, and any other Jinja construct is dropped
    or replaced by ``__tpl__``. Line breaks inside removed blocks are kept so
    reported line numbers stay aligned with the original file.
    """
    original = sql

    def _keep_newlines(match: re.Match) -> str:
        return "\n" * match.group(0).count("\n")

    sql = _DBT_REF_RE.sub(lambda m: m.group(1), sql)
    sql = _DBT_SOURCE_RE.sub(lambda m: f"{m.group(1)}.{m.group(2)}", sql)
    sql = _JINJA_COMMENT_RE.sub(_keep_newlines, sql)
    sql = _JINJA_BLOCK_RE.sub(_keep_newlines, sql)
    sql = _JINJA_EXPR_RE.sub("__tpl__", sql)
    sql = _DOLLAR_VAR_RE.sub(lambda m: f"__tpl_{m.group(1)}__", sql)
    return sql, sql != original


def extract_sql_from_python(source: str, file: str) -> List[SqlFragment]:
    """Pull SQL-looking triple-quoted strings out of Python source.

    Only literal triple-quoted strings whose first keyword is a SQL verb are
    taken; SQL assembled by concatenation or formatting at runtime cannot be
    recovered statically and is left out.
    """
    fragments = []
    for match in _PY_TRIPLE_QUOTED_RE.finditer(source):
        body = match.group(2)
        if not _SQL_LEAD_RE.match(body):
            continue
        body_start = match.start(2)
        fragments.append(SqlFragment(text=body, file=file, line_offset=source.count("\n", 0, body_start)))
    return fragments


def split_statements(sql: str, dialect: Optional[str]) -> List[Tuple[str, int]]:
    """Split SQL text into ``(statement, start_line)`` pairs (0-based lines).

    Splits on semicolon tokens so semicolons inside strings and comments are
    ignored. Falls back to a naive split when the text does not tokenize.
    """
    return [(text, code_line) for text, code_line, _ in _split_with_raw_lines(sql, dialect)]


def _split_with_raw_lines(sql: str, dialect: Optional[str]) -> List[Tuple[str, int, int]]:
    """``(statement, first_code_line, first_raw_line)``; the raw line includes leading comments."""
    try:
        tokens = sqlglot.tokenize(sql, read=dialect or None)
    except (TokenError, ValueError) as e:
        logger.debug("Tokenize failed, falling back to naive split: %s", e)
        return _naive_split(sql)

    pieces: List[Tuple[str, int]] = []
    start = 0
    for token in tokens:
        if token.token_type == sqlglot.TokenType.SEMICOLON:
            pieces.append((sql[start : token.start], sql.count("\n", 0, start)))
            start = token.end + 1
    pieces.append((sql[start:], sql.count("\n", 0, start)))
    return [(text, line + _leading_newlines(text), line) for text, line in pieces if _has_code(text)]


def _naive_split(sql: str) -> List[Tuple[str, int, int]]:
    pieces = []
    offset = 0
    for text in sql.split(";"):
        if _has_code(text):
            raw = sql.count("\n", 0, offset)
            pieces.append((text, raw + _leading_newlines(text), raw))
        offset += len(text) + 1
    return pieces


def _leading_newlines(text: str) -> int:
    """Lines to skip before the first line that holds code (not blank / comment)."""
    count = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            break
        count += 1
    return count


def _has_code(text: str) -> bool:
    return any(line.strip() and not line.strip().startswith("--") for line in text.split("\n"))


# ---------------------------------------------------------------------------
# Table naming
# ---------------------------------------------------------------------------


def table_full_name(table: exp.Table, default_database: str = "") -> str:
    """``catalog.db.table`` as written, filling the database from the default."""
    parts = [p for p in (table.catalog, table.db or default_database, table.name) if p]
    return ".".join(parts)


def _is_physical(table: exp.Table) -> bool:
    # Table functions (UNNEST, generate_series, ...) have no plain name.
    return bool(table.name) and isinstance(table.this, exp.Identifier)


# ---------------------------------------------------------------------------
# Statement analysis
# ---------------------------------------------------------------------------


def _write_target(stmt: exp.Expression) -> Tuple[Optional[exp.Table], str, Optional[exp.Expression]]:
    """Return ``(target_table, statement_kind, query_part)`` for a write statement.

    ``query_part`` is the expression holding the sources. For statements that
    do not write a table the target is ``None``.
    """
    if isinstance(stmt, exp.Insert):
        target = stmt.this.this if isinstance(stmt.this, exp.Schema) else stmt.this
        kind = "INSERT OVERWRITE" if stmt.args.get("overwrite") else "INSERT INTO"
        return (target if isinstance(target, exp.Table) else None), kind, stmt.expression
    if isinstance(stmt, exp.Create):
        target = stmt.this.this if isinstance(stmt.this, exp.Schema) else stmt.this
        kind = f"CREATE {(stmt.args.get('kind') or 'TABLE').upper()}"
        if stmt.expression is None:
            return None, kind, None  # plain DDL, no sources
        return (target if isinstance(target, exp.Table) else None), kind, stmt.expression
    if isinstance(stmt, exp.Merge):
        return (stmt.this if isinstance(stmt.this, exp.Table) else None), "MERGE", stmt
    if isinstance(stmt, exp.Update):
        return (stmt.this if isinstance(stmt.this, exp.Table) else None), "UPDATE", stmt
    if isinstance(stmt, exp.Delete):
        return (stmt.this if isinstance(stmt.this, exp.Table) else None), "DELETE", stmt
    return None, "QUERY", stmt


def _source_tables(query: exp.Expression, target: Optional[exp.Table], default_database: str) -> List[str]:
    """Physical tables read by ``query``, excluding CTE references and the target itself."""
    cte_names = {cte.alias_or_name.lower() for cte in query.find_all(exp.CTE)}
    seen: Dict[str, None] = {}
    for table in query.find_all(exp.Table):
        if table is target or not _is_physical(table):
            continue
        if not table.db and table.name.lower() in cte_names:
            continue
        seen.setdefault(table_full_name(table, default_database), None)
    if target is not None:
        seen.pop(table_full_name(target, default_database), None)
    return list(seen)


def _iter_scopes(query: exp.Expression) -> Iterable[Scope]:
    """Every scope under ``query``: CTEs, derived tables and WHERE subqueries."""
    roots = [query] if isinstance(query, (exp.Query, exp.Subquery)) else list(query.find_all(exp.Select))[:1]
    for root in roots:
        try:
            scope = build_scope(root)
        except Exception as e:  # sqlglot raises OptimizeError and friends on odd shapes
            logger.debug("build_scope failed: %s", e)
            continue
        if scope is None:
            continue
        yield from scope.traverse()


_MAX_ORIGINS = 8


@dataclass(frozen=True)
class _Origin:
    table: str
    column: str
    transform: Optional[str] = None  # innermost transform applied to the physical column, e.g. LEFT(issue_id, 6)


def _resolve_column(scope: Scope, column: exp.Column, default_database: str, depth: int = 0) -> List[_Origin]:
    """Trace ``column`` through derived tables / CTEs / UNIONs to physical columns.

    Pass-through projections (``x``, ``t.x AS y``) carry no transform.
    ``COALESCE`` / ``IFNULL`` follow every argument column; any other expression
    is followed only through its value columns (CASE result branches, or the
    single column a function wraps) and records that expression as the origin's
    ``transform``. An empty list means no physical origin could be attributed.
    """
    if depth > _MAX_RESOLVE_DEPTH:
        return []
    source = _column_source(scope, column)
    if isinstance(source, exp.Table):
        if not _is_physical(source):
            return []
        return [_Origin(table_full_name(source, default_database), column.name)]
    if not isinstance(source, Scope):
        return []
    branches = source.union_scopes if source.union_scopes else [source]
    index = _projection_index(branches[0].expression, column.name)
    origins: List[_Origin] = []
    for branch in branches:
        select = branch.expression
        if not isinstance(select, exp.Select):
            continue
        if index is None:
            # ``SELECT *`` pass-through: the column comes from the inner single source.
            if any(isinstance(p, exp.Star) for p in select.selects):
                origins.extend(_resolve_column(branch, exp.column(column.name), default_database, depth + 1))
            continue
        if index >= len(select.selects):
            continue
        origins.extend(_resolve_expression(branch, select.selects[index], default_database, depth + 1))
    return list(dict.fromkeys(origins))[:_MAX_ORIGINS]


def _projection_index(select: exp.Expression, name: str) -> Optional[int]:
    if not isinstance(select, exp.Select):
        return None
    for i, projection in enumerate(select.selects):
        if projection.alias_or_name == name:
            return i
    return None


def _resolve_expression(scope: Scope, projection: exp.Expression, default_database: str, depth: int) -> List[_Origin]:
    node = projection.this if isinstance(projection, exp.Alias) else projection
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Column):
        return _resolve_column(scope, node, default_database, depth)
    if isinstance(node, exp.Coalesce):
        args = [node.this, *node.expressions]
        if all(isinstance(a, exp.Column) for a in args):
            return [o for a in args for o in _resolve_column(scope, a, default_database, depth)]
    if isinstance(node, exp.Case):
        values = [branch.args.get("true") for branch in node.args.get("ifs") or []] + [node.args.get("default")]
        columns = [v for v in values if isinstance(v, exp.Column)]
        if not columns or len(columns) != len([v for v in values if v is not None and not _is_constant(v)]):
            return []
        return [_derived(o, node) for c in columns for o in _resolve_column(scope, c, default_database, depth)]
    if isinstance(node, (exp.Func, exp.Cast)):
        columns = {c.sql(): c for c in node.find_all(exp.Column) if not _inside_subquery(c, node)}
        if len(columns) == 1:
            column = next(iter(columns.values()))
            return [_derived(o, node) for o in _resolve_column(scope, column, default_database, depth)]
    return []


def _inside_subquery(column: exp.Column, root: exp.Expression) -> bool:
    """True when ``column`` sits in a scalar subquery nested under ``root``."""
    parent = column.parent
    while parent is not None and parent is not root:
        if isinstance(parent, exp.Subquery):
            return True
        parent = parent.parent
    return False


def _is_constant(node: exp.Expression) -> bool:
    return isinstance(node, (exp.Literal, exp.Null, exp.Boolean))


_MAX_TRANSFORM_LEN = 120


def _derived(origin: _Origin, node: exp.Expression) -> _Origin:
    """Mark ``origin`` as transformed; the innermost transform (closest to the table) wins."""
    if origin.transform is not None:
        return origin
    text = _render(node)
    if len(text) > _MAX_TRANSFORM_LEN:
        text = text[: _MAX_TRANSFORM_LEN - 3] + "..."
    return _Origin(origin.table, origin.column, text)


def _column_source(scope: Scope, column: exp.Column):
    if column.table:
        return scope.sources.get(column.table)
    # An unqualified column is only attributable when the scope has one source.
    if len(scope.selected_sources) == 1:
        return next(iter(scope.selected_sources.values()))[1]
    return None


def _join_edges_in_scope(scope: Scope, default_database: str, result: ExtractionResult) -> List[Tuple]:
    """``(left_table, right_table, keys, join_type, transforms)`` for every join predicate in ``scope``.

    Equalities of one predicate that link the same table pair are grouped into
    a single composite-key edge; ``left_table`` is the lexicographically smaller
    name so the same relationship aggregates regardless of written order.
    """
    select = scope.expression
    if not isinstance(select, exp.Select):
        return []
    predicates: List[Tuple[exp.Expression, str]] = []
    for join in select.args.get("joins") or []:
        join_type = (join.side or join.kind or "INNER").upper()
        on = join.args.get("on")
        if on is not None:
            predicates.append((on, join_type))
        for using in join.args.get("using") or []:
            name = using.name
            right = join.this.alias_or_name
            left_sources = [alias for alias in scope.selected_sources if alias != right]
            if len(left_sources) == 1:
                eq = exp.EQ(this=exp.column(name, table=left_sources[0]), expression=exp.column(name, table=right))
                predicates.append((eq, join_type))
    where = select.args.get("where")
    if where is not None:
        predicates.append((where.this, "WHERE"))

    edges = []
    for predicate, join_type in predicates:
        grouped: Dict[Tuple[str, str], set] = defaultdict(set)
        transforms: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
        for eq in _conjunct_equalities(predicate):
            result.join_predicates += 1
            lefts = _resolve_column(scope, eq.this, default_database)
            rights = _resolve_column(scope, eq.expression, default_database)
            if not lefts or not rights:
                result.join_predicates_unresolved += 1
                continue
            # One equality whose side traces to several origins (UNION branches, CASE
            # results) yields alternatives, not a composite key: fold them into a
            # single "a|b" key per table pair.
            alternatives: Dict[Tuple[str, str], Tuple[Dict[str, None], Dict[str, None]]] = {}
            for left in lefts:
                for right in rights:
                    if left.table == right.table:
                        continue  # self-join or row filter; not a cross-table relationship
                    lo, hi = (left, right) if left.table < right.table else (right, left)
                    lo_cols, hi_cols = alternatives.setdefault((lo.table, hi.table), ({}, {}))
                    lo_cols[lo.column] = None
                    hi_cols[hi.column] = None
                    for origin in (lo, hi):
                        if origin.transform:
                            transforms[(lo.table, hi.table)][f"{origin.table}.{origin.column}"] = origin.transform
            for pair, (lo_cols, hi_cols) in alternatives.items():
                grouped[pair].add(("|".join(lo_cols), "|".join(hi_cols)))
        for pair, keys in grouped.items():
            edges.append((pair[0], pair[1], sorted(keys), join_type, transforms[pair]))
    return edges


def _conjunct_equalities(predicate: exp.Expression) -> List[exp.EQ]:
    """``col = col`` equalities in the top-level AND chain of ``predicate``.

    OR branches are skipped: an equality under OR is not a join condition.
    """
    if isinstance(predicate, exp.Paren):
        return _conjunct_equalities(predicate.this)
    if isinstance(predicate, exp.And):
        return _conjunct_equalities(predicate.this) + _conjunct_equalities(predicate.expression)
    if isinstance(predicate, exp.EQ) and isinstance(predicate.this, exp.Column):
        if isinstance(predicate.expression, exp.Column):
            return [predicate]
    return []


# ---------------------------------------------------------------------------
# Rules: constant predicates, value mappings, dedup windows
# ---------------------------------------------------------------------------

_TEMPLATE_MARK = "__tpl"
_COMPARISONS = {
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.LT: "<",
    exp.LTE: "<=",
}


_TEMPLATE_PLACEHOLDER_RE = re.compile(r"__tpl_([A-Za-z0-9_]+?)__")


def _restore_template(text: str) -> str:
    """Show ``__tpl_x__`` placeholders as the ``${x}`` the author wrote."""
    return _TEMPLATE_PLACEHOLDER_RE.sub(lambda m: "${" + m.group(1) + "}", text)


def _literal_sql(node: Optional[exp.Expression]) -> Optional[str]:
    """SQL text of a constant operand, or ``None`` when ``node`` is not a constant."""
    if node is None:
        return None
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal):
        return f"-{node.this.this}"
    if isinstance(node, (exp.Literal, exp.Null, exp.Boolean)):
        return _render(node)
    return None


def _atoms(predicate: exp.Expression, include_or: bool) -> List[Tuple[exp.Expression, bool]]:
    """Atomic conditions of a boolean expression as ``(atom, negated)``.

    WHERE / ON only yield the top-level AND chain (an atom under OR is not a
    filter every row satisfies). CASE conditions describe branches, so OR
    operands are included there.
    """
    if isinstance(predicate, exp.Paren):
        return _atoms(predicate.this, include_or)
    if isinstance(predicate, exp.And) or (include_or and isinstance(predicate, exp.Or)):
        return _atoms(predicate.this, include_or) + _atoms(predicate.expression, include_or)
    if isinstance(predicate, exp.Not):
        return [(atom, not negated) for atom, negated in _atoms(predicate.this, include_or)]
    return [(predicate, False)]


def _constant_test(atom: exp.Expression, negated: bool) -> Optional[Tuple[exp.Column, str]]:
    """``(column, predicate_text)`` when ``atom`` compares a column with constants."""
    op = _COMPARISONS.get(type(atom))
    if op is not None:
        left, right = atom.this, atom.expression
        if isinstance(right, exp.Column) and not isinstance(left, exp.Column):
            left, right = right, left
            op = {">": "<", ">=": "<=", "<": ">", "<=": ">="}.get(op, op)
        value = _literal_sql(right)
        if isinstance(left, exp.Column) and value is not None:
            if negated:
                op = {"=": "!=", "!=": "="}.get(op, f"NOT {op}")
            return left, f"{op} {value}"
        return None
    if isinstance(atom, exp.In) and isinstance(atom.this, exp.Column) and atom.expressions:
        values = [_literal_sql(e) for e in atom.expressions]
        if all(v is not None for v in values):
            return atom.this, f"{'NOT IN' if negated else 'IN'} ({', '.join(sorted(values))})"
        return None
    if isinstance(atom, (exp.Like, exp.ILike)) and isinstance(atom.this, exp.Column):
        value = _literal_sql(atom.expression)
        if value is not None:
            word = "LIKE" if isinstance(atom, exp.Like) else "ILIKE"
            return atom.this, f"{'NOT ' if negated else ''}{word} {value}"
        return None
    if isinstance(atom, exp.Is) and isinstance(atom.this, exp.Column) and isinstance(atom.expression, exp.Null):
        return atom.this, "IS NOT NULL" if negated else "IS NULL"
    if isinstance(atom, exp.Between) and isinstance(atom.this, exp.Column):
        low, high = _literal_sql(atom.args.get("low")), _literal_sql(atom.args.get("high"))
        if low is not None and high is not None:
            return atom.this, f"{'NOT ' if negated else ''}BETWEEN {low} AND {high}"
    return None


def _single_origin(scope: Scope, column: exp.Column, default_database: str) -> Optional[_Origin]:
    origins = _resolve_column(scope, column, default_database)
    tables = {(o.table, o.column) for o in origins}
    return origins[0] if len(tables) == 1 else None


def _owned_by(node: exp.Expression, select: exp.Select) -> bool:
    """True when ``node`` belongs to ``select`` itself, not to a nested subquery."""
    return node.find_ancestor(exp.Select) is select


def _rule_clauses(select: exp.Select) -> List[Tuple[exp.Expression, str, bool]]:
    """``(condition, clause_name, include_or)`` for every condition owned by ``select``."""
    clauses: List[Tuple[exp.Expression, str, bool]] = []
    where = select.args.get("where")
    if where is not None:
        clauses.append((where.this, "WHERE", False))
    having = select.args.get("having")
    if having is not None:
        clauses.append((having.this, "HAVING", False))
    for join in select.args.get("joins") or []:
        on = join.args.get("on")
        if on is not None:
            clauses.append((on, "JOIN", False))
    for case in select.find_all(exp.Case):
        if _owned_by(case, select):
            for branch in case.args.get("ifs") or []:
                if case.this is None:
                    clauses.append((branch.this, "CASE", True))
    for cond in select.find_all(exp.If):
        if _owned_by(cond, select) and cond.parent is not None and not isinstance(cond.parent, exp.Case):
            clauses.append((cond.this, "CASE", True))
    return clauses


def _window_in_scope(scope: Scope) -> List[str]:
    """Templated constant predicates (``month >= '${month}'``) bounding an incremental load.

    Part of lineage (they decide ``load_mode``), so collected whichever sections are requested.
    """
    select = scope.expression
    if not isinstance(select, exp.Select):
        return []
    window = []
    for clause, _, include_or in _rule_clauses(select):
        for atom, negated in _atoms(clause, include_or):
            test = _constant_test(atom, negated)
            if test is not None and _TEMPLATE_MARK in test[1]:
                window.append(_restore_template(f"{test[0].name} {test[1]}"))
    return window


def _rules_in_scope(
    scope: Scope,
    default_database: str,
    result: ExtractionResult,
    file: str,
    line: int,
    target: Optional[str],
) -> None:
    select = scope.expression
    if not isinstance(select, exp.Select):
        return

    for clause, name, include_or in _rule_clauses(select):
        for atom, negated in _atoms(clause, include_or):
            test = _constant_test(atom, negated)
            if test is None:
                continue
            column, text = test
            if _TEMPLATE_MARK in text:
                continue  # an incremental-load window, reported on the lineage edge
            result.rule_predicates += 1
            origin = _single_origin(scope, column, default_database)
            if origin is None:
                result.rule_predicates_unresolved += 1
                continue
            result.predicates.append(
                Predicate(origin.table, origin.column, text, name, file, line, target, origin.transform)
            )

    for case in select.find_all(exp.Case):
        if not _owned_by(case, select):
            continue
        for branch in case.args.get("ifs") or []:
            label = branch.args.get("true")
            if case.this is not None:  # CASE col WHEN v THEN label
                column, value = case.this, _literal_sql(branch.this)
            else:
                cond = branch.this
                if not isinstance(cond, exp.EQ):
                    continue
                column, value = cond.this, _literal_sql(cond.expression)
            _add_mapping(scope, column, value, label, default_database, result, file, line, target)
    for cond in select.find_all(exp.If):
        if _owned_by(cond, select) and not isinstance(cond.parent, exp.Case) and isinstance(cond.this, exp.EQ):
            _add_mapping(
                scope,
                cond.this.this,
                _literal_sql(cond.this.expression),
                cond.args.get("true"),
                default_database,
                result,
                file,
                line,
                target,
            )

    for window_expr in select.find_all(exp.Window):
        if not (_owned_by(window_expr, select) and isinstance(window_expr.this, exp.RowNumber)):
            continue
        partition = [_origin_text(scope, e, default_database) for e in window_expr.args.get("partition_by") or []]
        order = window_expr.args.get("order")
        order_by = []
        for o in order.expressions if order is not None else []:
            _, text = _origin_text(scope, o.this, default_database)
            order_by.append(text + (" DESC" if o.args.get("desc") else ""))
        tables = {t for t, _ in partition if t}
        if len(tables) != 1:
            continue
        result.dedups.append(Dedup(tables.pop(), [c for _, c in partition], order_by, file, line, target))


def _add_mapping(scope, column, value, label, default_database, result, file, line, target) -> None:
    if not isinstance(column, exp.Column) or value is None:
        return
    if not (isinstance(label, exp.Literal) and label.is_string):
        return
    origin = _single_origin(scope, column, default_database)
    if origin is None:
        return
    result.mappings.append(
        ValueMapping(origin.table, origin.column, value.strip("'"), label.this, file, line, target, origin.transform)
    )


def _origin_text(scope: Scope, node: exp.Expression, default_database: str) -> Tuple[Optional[str], str]:
    """``(table, "column")`` for a window key, keeping the transform when there is one."""
    if isinstance(node, exp.Column):
        origin = _single_origin(scope, node, default_database)
        if origin is not None:
            return origin.table, origin.transform or origin.column
        return None, _render(node)
    columns = [c for c in node.find_all(exp.Column) if not _inside_subquery(c, node)]
    if len(columns) == 1:
        origin = _single_origin(scope, columns[0], default_database)
        if origin is not None:
            text = _render(node).replace(_render(columns[0]), origin.column)
            return origin.table, text
    return None, _render(node)


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_DECORATION_RE = re.compile(r"^[\s=\-*#~_+/]*$")
_CODE_LIKE_RE = re.compile(
    r"^\s*(select|from|where|and|or|left|right|inner|full|join|on|group|order|having|insert|into|case|when|"
    r"then|else|end|union|with|limit|as|,|\(|\))\b|^\s*[\w.`]+\s*(=|<>|!=|>=|<=|>|<|\bin\b|\bis\b|\blike\b)|"
    r"^\s*[\w.`]+\s*(,|\))\s*$|^\s*(count|sum|max|min|avg|if|ifnull|coalesce|date_format|concat)\s*\(",
    re.IGNORECASE,
)
_MAX_COMMENT_LEN = 400
_MAX_HEADER_LEN = 4000
_MAX_CODE_LEN = 160


def extract_comments(sql: str, file: str, line_offset: int = 0) -> List[Comment]:
    """Every ``--`` / ``/* */`` comment with the code line it annotates.

    A small scanner (not the SQL tokenizer) so comments survive statements
    that fail to parse, and ``--`` inside string literals is not mistaken for
    a comment. A trailing comment annotates the code before it on the same
    line; a full-line comment annotates the next line holding code.
    """
    lines = sql.split("\n")
    found: List[Tuple[int, str, str, bool]] = []  # (0-based line, text, code before it on the line, is_header)
    seen_code = False
    i, n, line_no = 0, len(sql), 0
    line_start = 0
    quote: Optional[str] = None
    while i < n:
        ch = sql[i]
        if ch == "\n":
            line_no += 1
            line_start = i + 1
            i += 1
            continue
        if quote:
            if ch == "\\" and quote != "`":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            seen_code = True
            i += 1
            continue
        if sql.startswith("--", i) or ch == "#" and _hash_comment(sql, i, line_start):
            end = sql.find("\n", i)
            end = n if end == -1 else end
            text = sql[i + (2 if ch == "-" else 1) : end]
            found.append((line_no, text, sql[line_start:i], not seen_code))
            i = end
            continue
        if sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            end = n if end == -1 else end
            text = sql[i + 2 : end]
            found.append((line_no, text, sql[line_start:i], not seen_code))
            line_no += text.count("\n")
            i = end + 2
            if "\n" in text:
                line_start = sql.rfind("\n", 0, i) + 1
            continue
        if not ch.isspace():
            seen_code = True
        i += 1

    comments = []
    header_lines: List[str] = []
    header_line_no: Optional[int] = None
    for line_idx, raw, before, is_header in found:
        if is_header:
            # Consecutive leading comments form the file header; keep its line structure.
            header_line_no = line_idx if header_line_no is None else header_line_no
            header_lines.extend(
                part.strip(" \t*").rstrip()
                for part in raw.split("\n")
                if part.strip(" \t*") and not _DECORATION_RE.match(part.strip(" \t*"))
            )
            continue
        text = " ".join(part.strip(" \t*") for part in raw.strip().split("\n")).strip()
        if not text or _DECORATION_RE.match(text):
            continue
        code = before.strip() or _next_code_line(lines, line_idx + 1 + raw.count("\n"))
        if not _MEANINGFUL_RE.search(text):
            continue
        kind = "code" if _looks_like_code(text) else "note"
        comments.append(
            Comment(
                text=_clip(text, _MAX_COMMENT_LEN),
                code=_clip(code, _MAX_CODE_LEN),
                kind=kind,
                file=file,
                line=line_offset + line_idx + 1,
            )
        )
    header = "\n".join(header_lines).strip()
    if header and _MEANINGFUL_RE.search(header):
        if len(header) > _MAX_HEADER_LEN:
            header = header[: _MAX_HEADER_LEN - 3] + "..."
        kind = "code" if _looks_like_code(header) and "\n" not in header else "header"
        comments.insert(0, Comment(text=header, code="", kind=kind, file=file, line=line_offset + header_line_no + 1))
    return comments


_STRING_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_MEANINGFUL_RE = re.compile(r"[\w\u3400-\u9fff]")


def _looks_like_code(text: str) -> bool:
    """Commented-out SQL rather than prose; CJK inside string literals does not make it prose."""
    if not _CODE_LIKE_RE.search(text):
        return False
    return not _CJK_RE.search(_STRING_LITERAL_RE.sub("''", text))


def _hash_comment(sql: str, i: int, line_start: int) -> bool:
    # MySQL-family ``#`` comments only when they start a line; ``#`` is legal inside identifiers elsewhere.
    return not sql[line_start:i].strip()


def _next_code_line(lines: List[str], start: int) -> str:
    for candidate in lines[start : start + 5]:
        stripped = candidate.strip()
        if stripped and not stripped.startswith(("--", "/*", "#")):
            return stripped.split("--", 1)[0].strip()
    return ""


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


ALL_SECTIONS = frozenset({"joins", "rules", "comments"})


def extract_from_fragments(
    fragments: Iterable[SqlFragment],
    dialect: Optional[str] = None,
    default_database: str = "",
    sections: Optional[Iterable[str]] = None,
) -> ExtractionResult:
    """Extract lineage plus the requested ``sections`` from every statement in ``fragments``.

    Lineage is always extracted. ``sections`` selects any of ``joins``,
    ``rules`` and ``comments``; ``None`` means all of them.
    """
    wanted = ALL_SECTIONS if sections is None else frozenset(sections) & ALL_SECTIONS
    result = ExtractionResult()
    token = _RENDER_DIALECT.set(dialect or None)
    try:
        for fragment in fragments:
            _extract_fragment(fragment, dialect, default_database, wanted, result)
    finally:
        _RENDER_DIALECT.reset(token)
    return result


def _extract_fragment(
    fragment: SqlFragment,
    dialect: Optional[str],
    default_database: str,
    sections: frozenset,
    result: ExtractionResult,
) -> None:
    sql, templated = preprocess_template(fragment.text)
    file_edges: List[LineageEdge] = []
    dropped: set = set()
    truncated: set = set()
    # (first raw line incl. leading comments, target) per statement, 1-based file lines
    statement_spans: List[Tuple[int, Optional[str]]] = []
    for text, start_line, raw_line in _split_with_raw_lines(sql, dialect):
        line = fragment.line_offset + start_line + 1
        span_start = fragment.line_offset + raw_line + 1
        result.statements += 1
        try:
            stmt = sqlglot.parse_one(text, read=dialect or None)
        except (ParseError, TokenError, ValueError) as e:
            result.unresolved.append(Unresolved(fragment.file, line, f"parse error: {_short_error(e)}"))
            continue
        if stmt is None:
            continue
        if isinstance(stmt, exp.Drop):
            if isinstance(stmt.this, exp.Table):
                dropped.add(table_full_name(stmt.this, default_database))
            result.parsed += 1
            continue
        if isinstance(stmt, exp.TruncateTable):
            truncated.update(table_full_name(t, default_database) for t in stmt.expressions if isinstance(t, exp.Table))
            result.parsed += 1
            continue
        if isinstance(stmt, exp.Command):
            if stmt.this.upper() == "TRUNCATE":
                match = re.search(r"table\s+([\w.`]+)", str(stmt.expression), re.IGNORECASE)
                if match:
                    name = match.group(1).replace("`", "")
                    truncated.add(name if "." in name or not default_database else f"{default_database}.{name}")
                result.parsed += 1
                continue
            result.unresolved.append(Unresolved(fragment.file, line, f"unsupported statement: {stmt.this}"))
            continue
        result.parsed += 1

        target, kind, query = _write_target(stmt)
        if query is None:
            continue
        target_name = table_full_name(target, default_database) if target is not None else None
        statement_spans.append((span_start, target_name))
        window: List[str] = []
        # A write needs its scopes for the incremental window even when no section is requested.
        walk_scopes = target is not None or bool(sections & {"joins", "rules"})
        for scope in _iter_scopes(query) if walk_scopes else ():
            if target is not None:
                window.extend(_window_in_scope(scope))
            if "joins" in sections:
                for left_table, right_table, keys, join_type, transforms in _join_edges_in_scope(
                    scope, default_database, result
                ):
                    result.joins.append(
                        JoinEdge(
                            left_table,
                            right_table,
                            keys,
                            join_type,
                            fragment.file,
                            line,
                            transforms=transforms,
                            statement_target=target_name,
                        )
                    )
            if "rules" in sections:
                _rules_in_scope(scope, default_database, result, fragment.file, line, target_name)
        if target is None:
            result.queries += 1
            continue
        edge = LineageEdge(
            target=target_name,
            sources=_source_tables(query, target, default_database),
            statement=kind,
            file=fragment.file,
            line=line,
            templated=templated,
            window=list(dict.fromkeys(window)),
        )
        if kind == "INSERT OVERWRITE" or kind.startswith("CREATE"):
            edge.load_mode = "overwrite"
        elif edge.window:
            edge.load_mode = "incremental"
        file_edges.append(edge)

    for edge in file_edges:
        if edge.target in truncated:
            edge.load_mode = "truncate_reload"
    result.lineage.extend(_fold_temp_tables(file_edges, dropped))

    if "comments" in sections:
        for comment in extract_comments(fragment.text, fragment.file, fragment.line_offset):
            comment.statement_target = _enclosing_target(statement_spans, comment.line)
            result.comments.append(comment)


def _enclosing_target(spans: List[Tuple[int, Optional[str]]], line: int) -> Optional[str]:
    """Target of the statement block holding ``line``; a block starts right after the previous ``;``,
    so comments written above a statement belong to it."""
    target = spans[0][1] if spans else None
    for start, name in spans:
        if start > line:
            break
        target = name
    return target


def _fold_temp_tables(edges: List[LineageEdge], dropped: set) -> List[LineageEdge]:
    """Collapse tables created and dropped within one file out of the lineage.

    ``A -> tmp -> B`` with ``tmp`` dropped later in the same file becomes
    ``A -> B`` carrying ``via_temp=[tmp]``, so scratch tables do not flood the
    graph.
    """
    temp_targets = {e.target for e in edges if e.target in dropped}
    if not temp_targets:
        return edges
    temp_sources: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        if edge.target in temp_targets:
            temp_sources[edge.target].extend(edge.sources)

    def expand(name: str, trail: List[str], depth: int = 0) -> List[str]:
        if name not in temp_targets or depth > _MAX_RESOLVE_DEPTH:
            return [name]
        trail.append(name)
        out: List[str] = []
        for src in temp_sources[name]:
            out.extend(expand(src, trail, depth + 1))
        return out

    folded = []
    for edge in edges:
        if edge.target in temp_targets:
            continue
        trail: List[str] = []
        sources: Dict[str, None] = {}
        for src in edge.sources:
            for real in expand(src, trail):
                sources.setdefault(real, None)
        edge.sources = list(sources)
        edge.via_temp = list(dict.fromkeys(trail))
        folded.append(edge)
    return folded


def _short_error(e: Exception) -> str:
    message = str(e).strip().split("\n")[0]
    return message[:200]
