# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Deterministic table-level lineage extraction from SQL text.

Pure functions only — no LLM, no database, no filesystem policy. The lineage
store (``datus/storage/lineage/``) owns persistence and the func-tool wrapper
(``datus/tools/func_tool/lineage_tools.py``) owns path resolution; everything
here takes SQL strings and returns plain data so it can be unit-tested in
isolation.

The analyzer records:

- **statement facts** — the tables each statement reads and writes, including
  pure ``SELECT`` statements, identified by a hash of the normalized SQL.
- **lineage edges** — ``target <- sources`` for statements that write a table
  (INSERT / INSERT OVERWRITE / CTAS / CREATE VIEW / MERGE / UPDATE ... FROM /
  DELETE ... USING), with unambiguous temporary tables folded away.
- **comments** — ``--`` and ``/* */`` comments with the code they annotate.
  Comments are the only place a script's author explains intent in their own
  words, so prose notes are kept verbatim; commented-out code is flagged.

Joins, filters, cardinality and business meaning are deliberately out of scope:
read them from the source SQL and verify them against the database.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Tuple

import sqlglot
from sqlglot import expressions as exp
from sqlglot.dialects import Dialect
from sqlglot.errors import ParseError, TokenError
from sqlglot.optimizer.scope import Scope, traverse_scope

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# Bump whenever extraction output changes, so persisted lineage analyzed by an older version is
# reported stale and re-analyzed on the next upsert.
ANALYZER_VERSION = 1

# Dialect used to render statements back to SQL for their hash, so identical SQL
# hashes identically within one dialect.
_RENDER_DIALECT: ContextVar[Optional[str]] = ContextVar("sql_lineage_render_dialect", default=None)


def _render(node: exp.Expression) -> str:
    return node.sql(dialect=_RENDER_DIALECT.get(), comments=False)


# dbt macros resolve to real table names; every other template construct is
# replaced by an identifier-safe placeholder so the statement still parses.
# ``ref('model')``, ``ref('package', 'model')`` and ``ref('model', v=2)`` all resolve to the model name.
_DBT_REF_RE = re.compile(
    r"\{\{\s*ref\(\s*['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])?(?:\s*,\s*\w+\s*=\s*[^)]*)?\s*\)\s*\}\}"
)
# dbt ``config()`` calls render to nothing, so they must not leave a placeholder behind.
_DBT_CONFIG_RE = re.compile(r"\{\{-?\s*config\s*\(.*?\)\s*-?\}\}", re.DOTALL)
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
    sequence: int = 0


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
    statement_facts: List[StatementFact] = field(default_factory=list)
    raw_lineage: List[LineageEdge] = field(default_factory=list)
    comments: List[Comment] = field(default_factory=list)
    unresolved: List[Unresolved] = field(default_factory=list)
    statements: int = 0
    parsed: int = 0
    queries: int = 0  # parsed statements with no write target


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_template(sql: str) -> Tuple[str, bool]:
    """Neutralize templating so sqlglot can parse the statement.

    Returns ``(sql, templated)``. ``${var}`` becomes ``__tpl_var__`` (valid both
    inside string literals and as an identifier part), dbt ``ref`` / ``source``
    become the referenced table name, dbt ``config()`` is dropped, and any other Jinja construct is dropped
    or replaced by ``__tpl__``. Line breaks inside removed blocks are kept so
    reported line numbers stay aligned with the original file.
    """
    original = sql

    def _keep_newlines(match: re.Match) -> str:
        return "\n" * match.group(0).count("\n")

    sql = _DBT_CONFIG_RE.sub(_keep_newlines, sql)
    sql = _DBT_REF_RE.sub(lambda m: (m.group(2) or m.group(1)) + _keep_newlines(m), sql)
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


def _parse(text: str, dialect: Optional[str]) -> Optional[exp.Expression]:
    statements = Dialect.get_or_raise(dialect).parse(text)
    return next((statement for statement in statements if statement is not None), None)


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


def extract_from_fragments(
    fragments: Iterable[SqlFragment],
    dialect: Optional[str] = None,
    default_database: str = "",
    include_comments: bool = True,
) -> ExtractionResult:
    """Extract lineage and statement facts from every statement in ``fragments``, plus comments."""
    result = ExtractionResult()
    token = _RENDER_DIALECT.set(dialect or None)
    try:
        for fragment in fragments:
            _extract_fragment(fragment, dialect, default_database, include_comments, result)
    finally:
        _RENDER_DIALECT.reset(token)
    return result


def _extract_fragment(
    fragment: SqlFragment,
    dialect: Optional[str],
    default_database: str,
    include_comments: bool,
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
            stmt = _parse(text, dialect)
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
            sequence=sequence,
        )
        file_edges.append(edge)
    result.raw_lineage.extend(file_edges)
    result.lineage.extend(_fold_temp_tables(file_edges, events))
    if include_comments:
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
