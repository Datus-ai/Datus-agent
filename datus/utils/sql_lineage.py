# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Deterministic table-level lineage and join-key extraction from SQL text.

Pure functions only — no LLM, no database, no filesystem policy. The func-tool
wrapper (``datus/tools/func_tool/lineage_tools.py``) owns path resolution and
result shaping; everything here takes SQL strings and returns plain data so it
can be unit-tested in isolation.

The analyzer records statement read/write facts and optional observations:

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

import ast
import hashlib
import re
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import sqlglot
from sqlglot import expressions as exp
from sqlglot.dialects import Dialect
from sqlglot.errors import ParseError, TokenError
from sqlglot.optimizer.scope import Scope, ScopeType, traverse_scope
from sqlglot.tokens import TokenType

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

_PY_TRIPLE_QUOTED_RE = re.compile(r"(?<![\w\"'])([rRfFbBuU]{0,2})(\"\"\"|''')(.*?)\2", re.DOTALL)
# The verb must be followed by the clause that makes it a statement, so prose such as a
# "Create the connection." docstring is not mistaken for SQL.
_SQL_LEAD_RE = re.compile(
    r"^\s*(?:--[^\n]*\n\s*)*(?:"
    r"insert\s+(?:into|overwrite)\b"
    r"|create\s+(?:or\s+replace\s+)?(?:(?:global\s+|local\s+)?(?:temporary|temp)\s+|external\s+|materialized\s+)?"
    r"(?:table|view)\b"
    r"|merge\s+into\b"
    r"|with\s+(?:recursive\s+)?[\w`\"]+\s*(?:\([^)]*\)\s*)?as\s*\("
    r"|select\b.*?\bfrom\b"
    r"|update\s+[\w.`\"]+(?:\s+(?:as\s+)?\w+)?\s+(?:set|join|inner|left|right|from)\b"
    r"|delete\s+(?:[\w.`\"]+\s+)?from\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)


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
    load_mode: str = "unknown"
    parameterized_predicates: List[str] = field(default_factory=list)
    sequence: int = 0


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
    condition: str = ""
    aliases: List[str] = field(default_factory=list)

    statement_id: str = ""
    condition_span: Optional[Tuple[int, int]] = None
    # Source fragment template: physical tables are wrapped in TABLE_REF_MARK (see render_expression).
    expression: str = ""


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

    statement_id: str = ""


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

    statement_id: str = ""


@dataclass
class WindowPattern:
    """``ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)`` resolved to physical columns."""

    table: Optional[str]
    partition_by: List[str]  # "column" or "TRANSFORM(column)"
    order_by: List[str]
    file: str
    line: int
    statement_target: Optional[str] = None
    selection: Optional[str] = None
    selection_kind: Optional[str] = None

    statement_id: str = ""


@dataclass
class StatementFact:
    statement_id: str
    sql_hash: str
    file: str
    line: int
    operation: str
    read_tables: List[str]
    write_tables: List[str]
    sequence: int


@dataclass
class Condition:
    expression: str
    clause: str
    status: str
    file: str
    line: int
    reason: str = ""
    output_column: Optional[str] = None
    branch: Optional[int] = None

    statement_id: str = ""
    condition_span: Optional[Tuple[int, int]] = None


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
    dedups: List[WindowPattern] = field(default_factory=list)
    window_functions: List[WindowPattern] = field(default_factory=list)
    conditions: List[Condition] = field(default_factory=list)
    statement_facts: List[StatementFact] = field(default_factory=list)
    raw_lineage: List[LineageEdge] = field(default_factory=list)
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

    sql = _DBT_REF_RE.sub(lambda m: m.group(1) + _keep_newlines(m), sql)
    sql = _DBT_SOURCE_RE.sub(lambda m: f"{m.group(1)}.{m.group(2)}" + _keep_newlines(m), sql)
    sql = _JINJA_COMMENT_RE.sub(_keep_newlines, sql)
    sql = _JINJA_BLOCK_RE.sub(_keep_newlines, sql)
    sql = _JINJA_EXPR_RE.sub(lambda m: "__tpl__" + _keep_newlines(m), sql)
    sql = _DOLLAR_VAR_RE.sub(lambda m: f"__tpl_{m.group(1)}__" + _keep_newlines(m), sql)
    return sql, sql != original


def extract_sql_from_python(source: str, file: str) -> List[SqlFragment]:
    """Pull SQL-looking triple-quoted strings out of Python source.

    Only literal triple-quoted strings that open with a SQL statement are
    taken; f-strings and SQL assembled by concatenation or formatting at
    runtime cannot be recovered statically and are left out.
    """
    fragments = []
    docstrings = _docstring_positions(source)
    for match in _PY_TRIPLE_QUOTED_RE.finditer(source):
        body = match.group(3)
        if "f" in match.group(1).lower() or not _SQL_LEAD_RE.match(body):
            continue
        line_start = source.rfind("\n", 0, match.start()) + 1
        position = (source.count("\n", 0, match.start()) + 1, len(source[line_start : match.start()].encode()))
        if position in docstrings:
            continue
        body_start = match.start(3)
        fragments.append(SqlFragment(text=body, file=file, line_offset=source.count("\n", 0, body_start)))
    return fragments


def _docstring_positions(source: str) -> set:
    """``(line, utf8_column)`` of every module / class / function docstring; empty when unparsable."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    positions = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    positions.add((first.value.lineno, first.value.col_offset))
    return positions


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


def _first_code_offset(text: str) -> int:
    """Skip leading whitespace and complete SQL comments without counting them as code."""
    offset = 0
    trivia = re.compile(r"\s+|--[^\n]*|/\*.*?\*/|#[^\n]*", re.DOTALL)
    while match := trivia.match(text, offset):
        offset = match.end()
    return offset


def _leading_newlines(text: str) -> int:
    return text.count("\n", 0, _first_code_offset(text))


def _has_code(text: str) -> bool:
    return bool(text[_first_code_offset(text) :].strip())


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
        query = stmt.expression
        if query is not None and stmt.args.get("with") is not None:
            query = query.copy()
            query.set("with", stmt.args["with"].copy())
        return (target if isinstance(target, exp.Table) else None), kind, query
    if isinstance(stmt, exp.Create):
        target = stmt.this.this if isinstance(stmt.this, exp.Schema) else stmt.this
        kind = f"CREATE {(stmt.args.get('kind') or 'TABLE').upper()}"
        if stmt.expression is None:
            return None, kind, None  # plain DDL, no sources
        query = stmt.expression
        if query is not None and stmt.args.get("with") is not None:
            query = query.copy()
            query.set("with", stmt.args["with"].copy())
        return (target if isinstance(target, exp.Table) else None), kind, query
    if isinstance(stmt, exp.Merge):
        return (stmt.this if isinstance(stmt.this, exp.Table) else None), "MERGE", stmt
    if isinstance(stmt, exp.Update):
        return (stmt.this if isinstance(stmt.this, exp.Table) else None), "UPDATE", stmt
    if isinstance(stmt, exp.Delete):
        return (stmt.this if isinstance(stmt.this, exp.Table) else None), "DELETE", stmt
    if isinstance(stmt, exp.Query):
        return None, "QUERY", stmt
    # USE / ALTER / ANALYZE / SET ...: the table they name is neither read nor written as data.
    return None, stmt.key.upper(), None


def _flatten_relation(relation: exp.Expression) -> List[exp.Join]:
    """``t JOIN s ON ...`` nested under a DML target as a flat list of joins, ``t`` first."""
    relation = relation.copy()
    nested = relation.args.get("joins") or []
    relation.set("joins", None)
    return [exp.Join(this=relation), *nested]


def _dml_scan_query(stmt: exp.Expression) -> Optional[exp.Select]:
    """An equivalent ``SELECT * FROM ... JOIN ... WHERE ...`` over the relations a DML statement matches.

    MERGE ON, UPDATE ... JOIN / FROM and DELETE ... USING conditions live outside any SELECT
    scope; restating them as one lets the scope-based join and rule extraction see them.
    """
    if isinstance(stmt, exp.Merge):
        using, on = stmt.args.get("using"), stmt.args.get("on")
        if stmt.this is None or using is None or on is None:
            return None
        parts = _flatten_relation(stmt.this) + [exp.Join(this=using.copy(), on=on.copy())]
    elif isinstance(stmt, (exp.Update, exp.Delete)):
        if not isinstance(stmt.this, exp.Table):
            return None
        parts = _flatten_relation(stmt.this)
        if isinstance(stmt, exp.Update):
            from_clause = stmt.args.get("from")
            relations = [from_clause.this] if from_clause is not None else []
        else:
            relations = stmt.args.get("using") or []
        for relation in relations:
            parts += _flatten_relation(relation)
    else:
        return None
    select = exp.Select(expressions=[exp.Star()])
    select.set("from", exp.From(this=parts[0].this))
    select.set("joins", parts[1:] or None)
    if stmt.args.get("where") is not None:
        select.set("where", stmt.args["where"].copy())
    if stmt.args.get("with") is not None:
        select.set("with", stmt.args["with"].copy())
    return select


def _source_tables(query: exp.Expression, target: Optional[exp.Table], default_database: str) -> List[str]:
    """Physical tables read by ``query``, excluding CTE references and the target itself."""
    seen: Dict[str, None] = {}
    scoped_tables = set()
    for scope in _iter_scopes(query):
        for table in scope.tables:
            scoped_tables.add(id(table))
            source = scope.sources.get(table.alias_or_name)
            if isinstance(source, exp.Table) and _is_physical(source):
                seen.setdefault(table_full_name(source, default_database), None)
    # DML targets and FROM/USING tables can sit outside a SELECT scope.
    with_clause = query.args.get("with")
    cte_names = {cte.alias_or_name for cte in with_clause.expressions} if with_clause else set()
    for table in query.find_all(exp.Table):
        if id(table) not in scoped_tables and table is not target and _is_physical(table):
            if not table.db and not table.catalog and table.name in cte_names:
                continue
            seen.setdefault(table_full_name(table, default_database), None)
    return list(seen)


def _iter_scopes(query: exp.Expression) -> Iterable[Scope]:
    """Every scope under ``query``: CTEs, derived tables and WHERE subqueries."""
    try:
        # Traverse the complete DML tree so CTE scopes remain available to sibling queries.
        yield from traverse_scope(query)
    except Exception as e:  # sqlglot raises OptimizeError and friends on odd shapes
        logger.debug("traverse_scope failed: %s", e)


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
            for projection in select.selects:
                if projection.is_star:
                    alias = projection.table if isinstance(projection, exp.Column) else None
                    origins.extend(
                        _resolve_column(branch, exp.column(column.name, table=alias), default_database, depth + 1)
                    )
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
    """True when ``column`` sits in a subquery (scalar, IN or EXISTS) nested under ``root``."""
    parent = column.parent
    while parent is not None and parent is not root:
        if isinstance(parent, (exp.Subquery, exp.Query)):
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


def _alias_scope(scope: Scope, alias: str, selected: bool = False) -> Optional[Scope]:
    """The scope that binds ``alias``: this one, or an enclosing one for a correlated subquery."""
    current = scope
    while current is not None:
        if alias in (current.selected_sources if selected else current.sources):
            return current
        # Only WHERE / SELECT subqueries see outer names; CTEs and derived tables do not.
        if current.scope_type != ScopeType.SUBQUERY:
            return None
        current = current.parent
    return None


def _column_alias(scope: Scope, column: exp.Column) -> str:
    """The relation alias ``column`` belongs to; an unqualified column is attributable only with one source."""
    if column.table:
        return column.table
    if len(scope.selected_sources) == 1:
        return next(iter(scope.selected_sources))
    return ""


def _column_source(scope: Scope, column: exp.Column):
    if column.table:
        owner = _alias_scope(scope, column.table)
        return owner.sources.get(column.table) if owner is not None else None
    # An unqualified column is only attributable when the scope has one source.
    if len(scope.selected_sources) == 1:
        return next(iter(scope.selected_sources.values()))[1]
    return None


class _SourcePositionParser:
    """Capture complete predicate token spans without searching rendered SQL.

    Mixed into the selected dialect's parser, without mutating sqlglot globals.
    These two parser hooks also retain closing parentheses and keyword-only
    operands, which identifier metadata alone cannot locate reliably.
    """

    def _parse_assignment(self):
        start = self._curr
        if self._prev and self._prev.token_type in (TokenType.ON, TokenType.WHERE):
            start = self._prev
        expression = super()._parse_assignment()
        self._record_span(expression, start)
        return expression

    def _parse_using_identifiers(self):
        start = self._prev
        identifiers = super()._parse_using_identifiers()
        for identifier in identifiers:
            self._record_span(identifier, start)
        return identifiers

    def _record_span(self, expression, start):
        if expression is not None and start is not None and self._prev is not None:
            if self._prev.end >= start.start:
                expression.meta["lineage_span"] = (
                    self.sql.count("\n", 0, start.start) + 1,
                    self.sql.count("\n", 0, self._prev.end + 1) + 1,
                )


@lru_cache(maxsize=32)
def _position_parser_class(parser_class):
    return type(f"Lineage{parser_class.__name__}", (_SourcePositionParser, parser_class), {})


def _parse_with_positions(text: str, dialect: Optional[str]) -> Optional[exp.Expression]:
    selected = Dialect.get_or_raise(dialect)
    parser = _position_parser_class(selected.parser_class)(dialect=selected)
    statements = parser.parse(selected.tokenize(text), text)
    return next((statement for statement in statements if statement is not None), None)


def _condition_span(node: exp.Expression, line_offset: int) -> Optional[Tuple[int, int]]:
    span = node.meta.get("lineage_span")
    return (span[0] + line_offset, span[1] + line_offset) if span else None


# Join expressions are templates: every physical table is wrapped in this marker so the
# tool layer can substitute response-local IDs without re-parsing the fragment.
TABLE_REF_MARK = "\x1f"
_TABLE_REF = re.compile(f"{TABLE_REF_MARK}([^{TABLE_REF_MARK}]*){TABLE_REF_MARK}")


def _table_ref(name: str) -> str:
    return f"{TABLE_REF_MARK}{name}{TABLE_REF_MARK}"


def expression_tables(expression: str) -> List[str]:
    """Physical table names referenced by a join expression template, in order of appearance."""
    return list(dict.fromkeys(_TABLE_REF.findall(expression)))


def render_expression(expression: str, resolve) -> str:
    """Replace every table marker with ``resolve(full_name)``."""
    return _TABLE_REF.sub(lambda match: resolve(match.group(1)), expression)


def _scope_relation(scope: Scope, alias: str, default_database: str) -> str:
    """Keep projection aliases bound to their scope, not a replacement physical table.

    Physical relations render as ``<ref> AS alias`` (or ``<ref>`` when unaliased). CTE and
    derived relations keep their scope name followed by ``{<ref>,...}``, the physical tables
    their projection reads; their columns may be renamed or computed, so they are never
    rewritten as physical tables.
    """
    scope = _alias_scope(scope, alias, selected=True) or scope
    node, source = scope.selected_sources[alias]
    if isinstance(source, exp.Table):
        if not _is_physical(source):
            relation = source.copy()
            relation.set("joins", None)
            return _render(relation)
        ref = _table_ref(table_full_name(source, default_database))
        return f"{ref} AS {_render(exp.to_identifier(source.alias))}" if source.alias else ref

    physical = set()
    pending = [source]
    visited = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        pending.extend(current.union_scopes)
        pending.extend(current.subquery_scopes)
        for _, child in current.selected_sources.values():
            if isinstance(child, Scope):
                pending.append(child)
            elif isinstance(child, exp.Table) and _is_physical(child):
                physical.add(table_full_name(child, default_database))
    scope_alias = ""
    if isinstance(node, exp.Table):
        label, scope_alias = _render(exp.to_identifier(node.name)), node.alias
    else:
        parent = source.expression.parent
        table_alias = parent.args.get("alias") if parent is not None else None
        label = _render(table_alias.this) if table_alias else _render(exp.to_identifier(alias))
    label += "{" + ",".join(_table_ref(name) for name in sorted(physical)) + "}"
    return f"{label} AS {_render(exp.to_identifier(scope_alias))}" if scope_alias else label


def _join_expression(
    scope: Scope,
    predicate: exp.Expression,
    kind: str,
    right_alias: str,
    source_node: exp.Expression,
    default_database: str,
) -> str:
    aliases = list(dict.fromkeys(column.table for column in predicate.find_all(exp.Column) if column.table))
    aliases = [alias for alias in aliases if _alias_scope(scope, alias, selected=True) is not None]
    if right_alias:
        lefts = [alias for alias in aliases if alias != right_alias]
        if not lefts:
            return ""
        # Every relation the condition references stays visible; extra left-side aliases
        # follow the first one as a comma list, as they would in a FROM clause.
        left = ", ".join(_scope_relation(scope, alias, default_database) for alias in lefts)
        right = _scope_relation(scope, right_alias, default_database)
        if isinstance(source_node, exp.Identifier):
            columns = source_node.parent.args.get("using") or [source_node]
            clause = "USING (" + ", ".join(_render(column) for column in columns) + ")"
        else:
            clause = "ON " + _render(predicate)
        return f"{left} {kind} JOIN {right} {clause}"
    relations = [_scope_relation(scope, alias, default_database) for alias in aliases]
    return " CROSS JOIN ".join(relations) + " WHERE " + _render(predicate)


def _join_edges_in_scope(
    scope: Scope, default_database: str, result: ExtractionResult, file: str, line: int, line_offset: int = 0
) -> List[Tuple]:
    """Resolve conjunctive equalities, preserving canonical outer-join direction."""
    select = scope.expression
    if not isinstance(select, exp.Select):
        return []
    predicates = []
    from_clause = select.args.get("from")
    # USING can only bind to relations written before its JOIN.
    preceding = [from_clause.this.alias_or_name] if from_clause is not None else []
    for join in select.args.get("joins") or []:
        kind = (join.side or join.kind or "INNER").upper()
        right = join.this.alias_or_name
        if join.args.get("on") is not None:
            predicates.append((join.args["on"], kind, right, join.args["on"]))
        lefts = [alias for alias in preceding if alias in scope.selected_sources and alias != right]
        preceding.append(right)
        for using in join.args.get("using") or []:
            if len(lefts) == 1:
                predicates.append(
                    (
                        exp.EQ(
                            this=exp.column(using.name, table=lefts[0]), expression=exp.column(using.name, table=right)
                        ),
                        kind,
                        right,
                        using,
                    )
                )
            else:
                result.join_predicates += 1
                result.join_predicates_unresolved += 1
                result.conditions.append(
                    Condition(
                        f"USING ({using.name})",
                        "JOIN",
                        "unresolved",
                        file,
                        line,
                        "Multiple possible left sources",
                        condition_span=_condition_span(using, line_offset),
                    )
                )
    if select.args.get("where") is not None:
        predicates.append((select.args["where"].this, "WHERE", "", select.args["where"].this))
    edges = []
    for predicate, kind, right_alias, source_node in predicates:
        span = _condition_span(source_node, line_offset)
        grouped = defaultdict(set)
        transforms = defaultdict(dict)
        for atom, negated in _atoms(predicate, False):
            # Columns inside a nested subquery belong to that subquery's own scope.
            columns = [c for c in atom.find_all(exp.Column) if not _inside_subquery(c, atom)]
            # Constant filters are handled by the rule extractor.
            if len(columns) < 2:
                continue
            # Comparisons within one alias are row conditions, not self-joins.
            if len({_column_alias(scope, c) for c in columns}) == 1:
                continue
            result.join_predicates += 1
            lefts = rights = []
            if isinstance(atom, exp.EQ) and not negated:
                lefts = _resolve_expression(scope, atom.this, default_database, 0)
                rights = _resolve_expression(scope, atom.expression, default_database, 0)
            if not lefts or not rights:
                result.join_predicates_unresolved += 1
                result.conditions.append(
                    Condition(
                        _render(exp.Not(this=atom.copy())) if negated else _render(atom),
                        kind,
                        "unresolved",
                        file,
                        line,
                        "Relationship is not a resolvable conjunctive equality",
                        condition_span=span,
                    )
                )
                continue
            left_aliases = {
                _column_alias(scope, c) for c in atom.this.find_all(exp.Column) if not _inside_subquery(c, atom)
            }
            right_aliases = {
                _column_alias(scope, c) for c in atom.expression.find_all(exp.Column) if not _inside_subquery(c, atom)
            }
            alternatives = {}
            for left in lefts:
                for right in rights:
                    # First orient operands as written in JOIN, independent of equality order.
                    swap = right_alias and right_alias in left_aliases and right_alias not in right_aliases
                    lo, hi = (right, left) if swap else (left, right)
                    aliases = (
                        (next(iter(sorted(right_aliases)), ""), next(iter(sorted(left_aliases)), ""))
                        if swap
                        else (next(iter(sorted(left_aliases)), ""), next(iter(sorted(right_aliases)), ""))
                    )
                    canonical_kind = kind
                    if lo.table > hi.table:
                        lo, hi = hi, lo
                        aliases = aliases[::-1]
                        canonical_kind = {"LEFT": "RIGHT", "RIGHT": "LEFT"}.get(kind, kind)
                    key = (lo.table, hi.table, canonical_kind, aliases if lo.table == hi.table else ())
                    lo_cols, hi_cols = alternatives.setdefault(key, ({}, {}))
                    lo_cols[lo.column] = None
                    hi_cols[hi.column] = None
                    for origin, alias in zip((lo, hi), aliases):
                        if origin.transform:
                            source = f"{origin.table}@{alias}" if lo.table == hi.table else origin.table
                            transforms[key][f"{source}.{origin.column}"] = origin.transform
            for key, (lo_cols, hi_cols) in alternatives.items():
                grouped[key].add(("|".join(lo_cols), "|".join(hi_cols)))
        expression = (
            _join_expression(scope, predicate, kind, right_alias, source_node, default_database) if grouped else ""
        )
        for key, keys in grouped.items():
            edges.append(
                (
                    key[0],
                    key[1],
                    sorted(keys),
                    key[2],
                    transforms[key],
                    _render(predicate),
                    list(key[3]),
                    span,
                    expression,
                )
            )
    return edges


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


def _atoms(predicate: exp.Expression, include_or: bool, negated: bool = False) -> List[Tuple[exp.Expression, bool]]:
    """Only split mandatory conjunctions; respect De Morgan's law under NOT."""
    if isinstance(predicate, exp.Paren):
        return _atoms(predicate.this, include_or, negated)
    if isinstance(predicate, exp.Not):
        return _atoms(predicate.this, include_or, not negated)
    conjunction = isinstance(predicate, exp.Or if negated else exp.And)
    if conjunction or (include_or and isinstance(predicate, (exp.And, exp.Or))):
        return _atoms(predicate.this, include_or, negated) + _atoms(predicate.expression, include_or, negated)
    return [(predicate, negated)]


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
                op = {"=": "!=", "!=": "=", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}[op]
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


def _parameters_in_scope(scope: Scope) -> List[str]:
    """Parameter predicates are observations, not proof of incremental loading."""
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
        if include_or or any(isinstance(n, (exp.Or, exp.Not)) for n in clause.walk()):
            case = clause.find_ancestor(exp.Case)
            alias = case.find_ancestor(exp.Alias) if case is not None else None
            branches = case.args.get("ifs", []) if case is not None else []
            branch = next((i + 1 for i, b in enumerate(branches) if b.this is clause), None)
            result.conditions.append(
                Condition(
                    _restore_template(_render(clause)),
                    name,
                    "observed",
                    file,
                    line,
                    "Branch/compound condition; atoms are not global requirements",
                    alias.alias if alias is not None else None,
                    branch,
                )
            )
        for atom, negated in _atoms(clause, include_or):
            test = _constant_test(atom, negated)
            if test is None:
                continue
            column, text = test
            if _TEMPLATE_MARK in text:
                continue  # parameterized predicates are reported separately
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
        table = next(iter(tables)) if len(tables) == 1 else None
        selection, selection_kind = _window_selection(scope, window_expr)
        pattern = WindowPattern(
            table, [c for _, c in partition], order_by, file, line, target, selection, selection_kind
        )
        result.window_functions.append(pattern)
        if selection_kind == "dedup":
            result.dedups.append(pattern)


def _window_selection(scope: Scope, window: exp.Window) -> Tuple[Optional[str], Optional[str]]:
    """Follow a window alias through pass-through scopes; require a mandatory filter."""
    alias = window.parent.alias if isinstance(window.parent, exp.Alias) else ""
    current = scope
    previous = None
    names = {alias} if alias else set()
    for _ in range(_MAX_RESOLVE_DEPTH):
        select = current.expression
        for key in ("qualify", "where"):
            clause = select.args.get(key)
            if clause is None or (current is scope and key == "where"):
                continue
            for atom, negated in _atoms(clause.this, False):
                if negated or not isinstance(atom, (exp.EQ, exp.NEQ, exp.LTE, exp.LT, exp.GTE, exp.GT)):
                    continue
                left, right = atom.this, atom.expression
                matches = left is window or (isinstance(left, exp.Column) and left.name in names)
                if current is not scope and isinstance(left, exp.Column):
                    source = _column_source(current, left)
                    matches = matches and source is previous
                if not matches:
                    continue
                if not isinstance(right, exp.Literal) or right.is_string:
                    return _render(atom), "filtered"
                try:
                    limit = int(right.this)
                except ValueError:
                    return _render(atom), "filtered"
                kind = (
                    "dedup"
                    if (isinstance(atom, (exp.EQ, exp.LTE)) and limit == 1 or isinstance(atom, exp.LT) and limit == 2)
                    else "top_n"
                    if isinstance(atom, (exp.LT, exp.LTE))
                    else "filtered"
                )
                return _render(atom), kind if limit > 0 else "filtered"
        parent = current.parent
        if parent is None or not isinstance(parent.expression, exp.Select):
            break
        if current is not scope:
            projected = set()
            for projection in select.selects:
                if projection.is_star:
                    projected.update(names)
                node = projection.this if isinstance(projection, exp.Alias) else projection
                if isinstance(node, exp.Column) and node.name in names and _column_source(current, node) is previous:
                    projected.add(projection.alias_or_name)
            names = projected
        previous, current = current, parent
    return None, None


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
    events: List[Tuple[int, str, str]] = []
    truncated: set = set()
    statement_spans: List[Tuple[int, Optional[str]]] = []
    for local_sequence, (text, start_line, raw_line) in enumerate(_split_with_raw_lines(sql, dialect), 1):
        line = fragment.line_offset + start_line + 1
        span_start = fragment.line_offset + raw_line + 1
        result.statements += 1
        sequence = result.statements
        try:
            stmt = _parse_with_positions(text, dialect)
        except (ParseError, TokenError, ValueError) as e:
            result.unresolved.append(Unresolved(fragment.file, line, f"parse error: {_short_error(e)}"))
            continue
        if stmt is None:
            continue
        target, kind, query = _write_target(stmt)
        writes = [table_full_name(target, default_database)] if target is not None else []
        if isinstance(stmt, exp.Drop):
            kind, query = "DROP", None
            writes = [table_full_name(stmt.this, default_database)] if isinstance(stmt.this, exp.Table) else []
        elif isinstance(stmt, exp.TruncateTable):
            kind, query = "TRUNCATE", None
            writes = [table_full_name(t, default_database) for t in stmt.expressions if isinstance(t, exp.Table)]
        elif isinstance(stmt, exp.Command):
            result.unresolved.append(Unresolved(fragment.file, line, f"unsupported statement: {stmt.this}"))
            continue
        elif isinstance(stmt, exp.Create) and target is None:
            table = stmt.this.this if isinstance(stmt.this, exp.Schema) else stmt.this
            writes = [table_full_name(table, default_database)] if isinstance(table, exp.Table) else []
        reads = _source_tables(query, target, default_database) if query is not None else []
        sql_hash = hashlib.sha256(_render(stmt).encode()).hexdigest()[:16]
        fact = StatementFact(
            f"{fragment.file}:{fragment.line_offset}:{local_sequence}",
            sql_hash,
            fragment.file,
            line,
            kind,
            reads,
            writes,
            sequence,
        )
        result.statement_facts.append(fact)
        result.parsed += 1
        for name in writes:
            events.append((sequence, kind, name))
        if kind == "TRUNCATE":
            truncated.update(writes)
        elif kind == "DROP" or kind.startswith("CREATE"):
            truncated.difference_update(writes)
        if query is None:
            continue
        target_name = writes[0] if target is not None else None
        statement_spans.append((span_start, target_name))
        parameters: List[str] = []
        collections = [result.joins, result.predicates, result.mappings, result.window_functions, result.conditions]
        starts = [len(records) for records in collections]
        for scope in _iter_scopes(_dml_scan_query(stmt) or query):
            if target is not None:
                parameters.extend(_parameters_in_scope(scope))
            if "joins" in sections:
                for (
                    left,
                    right,
                    keys,
                    join_type,
                    transforms,
                    condition,
                    aliases,
                    condition_span,
                    expression,
                ) in _join_edges_in_scope(scope, default_database, result, fragment.file, line, span_start - 1):
                    result.joins.append(
                        JoinEdge(
                            left,
                            right,
                            keys,
                            join_type,
                            fragment.file,
                            line,
                            transforms,
                            target_name,
                            condition,
                            aliases,
                            condition_span=condition_span,
                            expression=expression,
                        )
                    )
            if "rules" in sections:
                _rules_in_scope(scope, default_database, result, fragment.file, line, target_name)
        for records, start in zip(collections, starts):
            for record in records[start:]:
                record.statement_id = fact.statement_id
        if target is None:
            if isinstance(query, exp.Query):
                result.queries += 1
            continue
        mode = {
            "INSERT INTO": "insert",
            "INSERT OVERWRITE": "overwrite",
            "MERGE": "merge",
            "UPDATE": "update",
            "DELETE": "delete",
            "CREATE TABLE": "create",
            "CREATE VIEW": "create_view",
        }.get(kind, "unknown")
        if kind == "INSERT INTO" and target_name in truncated:
            mode = "truncate_reload"
        edge = LineageEdge(
            target_name,
            reads,
            kind,
            fragment.file,
            line,
            templated=templated,
            load_mode=mode,
            parameterized_predicates=list(dict.fromkeys(parameters)),
            sequence=sequence,
        )
        file_edges.append(edge)
    result.raw_lineage.extend(file_edges)
    result.lineage.extend(_fold_temp_tables(file_edges, events))
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


def _fold_temp_tables(edges: List[LineageEdge], events: List[Tuple[int, str, str]]) -> List[LineageEdge]:
    """Fold only unambiguous create-use-drop lifetimes; retain all raw edges separately.

    Multiple creations of the same name remain explicit rather than mixing their sources.
    A DROP before CREATE is cleanup, never evidence that the resulting table is temporary.
    """
    creates = defaultdict(list)
    drops = defaultdict(list)
    for sequence, kind, name in events:
        if kind.startswith("CREATE"):
            creates[name].append(sequence)
        elif kind == "DROP":
            drops[name].append(sequence)
    lifetimes = {}
    for name, starts in creates.items():
        later_drops = [n for n in drops[name] if n > starts[0]]
        if len(starts) != 1 or not later_drops:
            continue
        end = later_drops[0]
        if any(e.target == name and e.sequence > end for e in edges):
            continue
        readers = [e for e in edges if name in e.sources and e.target != name]
        if readers and all(starts[0] < e.sequence < end for e in readers):
            lifetimes[name] = (starts[0], end)

    def expand(name: str, before: int, trail: List[str], visited: set) -> List[str]:
        lifetime = lifetimes.get(name)
        if lifetime is None or not lifetime[0] < before < lifetime[1] or name in visited:
            return [name]
        trail.append(name)
        sources = []
        for edge in edges:
            if edge.target == name and lifetime[0] <= edge.sequence < before:
                for src in edge.sources:
                    sources.extend(expand(src, edge.sequence, trail, visited | {name}))
        return sources

    folded = []
    for edge in edges:
        if edge.target in lifetimes:
            continue
        trail = []
        sources = [src for name in edge.sources for src in expand(name, edge.sequence, trail, set())]
        folded.append(replace(edge, sources=list(dict.fromkeys(sources)), via_temp=list(dict.fromkeys(trail))))
    return folded


def _short_error(e: Exception) -> str:
    message = str(e).strip().split("\n")[0]
    return message[:200]
