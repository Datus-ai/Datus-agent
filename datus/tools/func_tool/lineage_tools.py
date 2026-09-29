# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``extract_sql_lineage`` function tool.

Reads SQL (and SQL embedded in Python) from workspace files and returns
table-level lineage, observed join keys, constant rules and author comments,
parsed deterministically with sqlglot by :mod:`datus.utils.sql_lineage`.
Read-only: it never writes files or touches the database.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.fs_path_policy import PathAllowlist, PathZone, classify_path
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.sql_lineage import (
    ALL_SECTIONS,
    ExtractionResult,
    SqlFragment,
    extract_from_fragments,
    extract_sql_from_python,
)
from datus.utils.sql_utils import parse_dialect

logger = get_logger(__name__)

_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_ITEMS = 300
_MAX_SEEN_IN = 3
_MAX_CODE_COMMENTS_PER_TARGET = 3
_READABLE_ZONES = (PathZone.INTERNAL, PathZone.WHITELIST)
_SECTION_ORDER = ("joins", "rules", "comments")

_AGGREGATE_RE = re.compile(r"\b(count|sum|max|min|avg|group_concat|percentile\w*)\s*\(|\bcase\s+when\b", re.IGNORECASE)
_PROJECTION_ALIAS_RE = re.compile(r"(?:\bas\s+)?`?([A-Za-z_][A-Za-z0-9_]*)`?\s*,?\s*$", re.IGNORECASE)
_NOT_PROJECTION_RE = re.compile(
    r"^\s*(select|from|where|and|or|on|left|right|inner|full|join|group|order|having|union|with|when|then|else|"
    r"end|\(|\)|insert|into|limit)\b",
    re.IGNORECASE,
)


_SELECT_LEAD_RE = re.compile(r"^\s*select\s+(distinct\s+)?", re.IGNORECASE)


def _projection_alias(code: str) -> Optional[str]:
    """Output column name when ``code`` is a SELECT-list line (``x``, ``t.x``, ``expr AS x``)."""
    code = _SELECT_LEAD_RE.sub("", code or "")
    if not code or _NOT_PROJECTION_RE.match(code):
        return None
    match = _PROJECTION_ALIAS_RE.search(code)
    if match is None:
        return None
    alias = match.group(1)
    if alias.lower() in {"end", "null", "then", "else", "and", "or", "as", "on"}:
        return None
    return alias


class LineageTools:
    """Function tool wrapper for static SQL lineage extraction."""

    permission_category: str = "lineage_tools"

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        root_path: Optional[str] = None,
        path_allowlist: Optional[PathAllowlist] = None,
    ):
        """
        Args:
            agent_config: Supplies the datasource dialect / default database.
            root_path: Workspace root that relative ``paths`` resolve against.
            path_allowlist: Extra readable roots, the same allowlist the filesystem tools honour.
                Anything else outside the workspace is skipped: this tool has no interactive
                prompt to ask for an external read.
        """
        self.agent_config = agent_config
        self.root_path = Path(root_path or os.getcwd()).expanduser().resolve(strict=False)
        self.path_allowlist = path_allowlist
        self._extraction_cache: Optional[tuple[tuple, ExtractionResult]] = None

    @classmethod
    def all_tools_name(cls) -> List[str]:
        return ["extract_sql_lineage"]

    def available_tools(self) -> List[Tool]:
        return [trans_to_function_tool(self.extract_sql_lineage)]

    def extract_sql_lineage(
        self,
        paths: List[str],
        sections: Optional[List[str]] = None,
        datasource: Optional[str] = None,
        dialect: Optional[str] = None,
        default_database: Optional[str] = None,
        max_items: int = 100,
        max_files: int = 500,
        offset: int = 0,
        result_path: Optional[str] = None,
        max_output_chars: int = 60000,
    ) -> FuncToolResult:
        """
        Statically analyze SQL files (no SQL is executed): table lineage, join keys, constant rules, comments.

        Use this to locate SQL evidence before reading scripts or verifying with the database.
        Frequencies describe this corpus, not mandatory business rules. Never infer source-table grain,
        join cardinality, incremental loading or ownership from syntax alone.

        Args:
            paths: Workspace-relative SQL/Python files or globs. Narrow paths for file-level details.
            sections: Optional joins/rules/comments; defaults to all. Public lineage and table facts
                are independent of this selection.
            datasource: Datasource supplying the dialect and default database; defaults to active.
            dialect: Override the datasource dialect, e.g. starrocks or hive.
            default_database: Qualify bare names with this database; its prefix is shortened in summaries.
            max_items: Page size per result list, between 1 and 300.
            max_files: Scan limit between 1 and 5000; split paths if files_truncated is true.
            offset: Start index for result lists. Use pagination[result_path].next_offset to continue.
            result_path: Return only this collection, e.g. rules.filters, comments.file_headers,
                tables, statements, raw_lineage, filter_occurrences, join_occurrences, mapping_occurrences,
                window_occurrences, comment_occurrences or unresolved. statements/raw_lineage expose uncollapsed
                per-statement evidence; they are not included in the default summary.
            max_output_chars: JSON result character budget, at least 6000. Oversized individual records
                return source references and detail_omitted; open those files for their full contents.

        Returns:
            success/error/result envelope. result schema_version=2 contains:
            lineage (write dependencies and observed operations), tables (read/write inventory including
            SELECT), roots (read but not written in this scan, not proof of external ownership), joins
            (canonical table order, outer-join direction and transforms), rules (constant filters,
            partial value mappings, observed window_functions and confirmed ROW_NUMBER selections in
            dedup), conditions (compound/branch or unresolved relationship expressions), comments,
            unresolved, stats and pagination. Parameterized predicates do not establish incremental
            loading. Evidence includes file, statement start line and statement_id; distinct_statements
            deduplicates comment-free normalized SQL, table_read_statements is its corpus denominator.
            Pagination totals describe all matching records, independent of returned page size. Request
            every required page, especially file_headers in a question-to-SQL corpus. Missing facts in
            failed/unresolved or truncated inputs are unknown, not evidence of absence.
        """
        if not paths:
            return FuncToolResult(success=0, error="paths must contain at least one file or glob pattern")
        if not 1 <= max_files <= 5000:
            return FuncToolResult(success=0, error="max_files must be between 1 and 5000")
        if not 1 <= max_items <= _MAX_ITEMS or offset < 0 or max_output_chars < 6000:
            return FuncToolResult(success=0, error="max_items must be 1..300, offset >= 0, max_output_chars >= 6000")
        wanted = ALL_SECTIONS if sections is None else {s.strip().lower() for s in sections if s}
        unknown = wanted - ALL_SECTIONS
        if unknown:
            return FuncToolResult(
                success=0, error=f"Unknown sections: {sorted(unknown)}; choose from {list(_SECTION_ORDER)}"
            )
        try:
            dialect, database = self._resolve_dialect(datasource, dialect, default_database)
            files, skipped, truncated = self._collect_files(paths, max_files)
            fragments: List[SqlFragment] = []
            unreadable: List[Dict[str, Any]] = list(skipped)
            for file_path in files:
                display = self._display(file_path)
                try:
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    unreadable.append({"file": display, "line": 0, "reason": f"unreadable: {e}"})
                    continue
                if file_path.suffix.lower() == ".py":
                    extracted = extract_sql_from_python(text, display)
                    fragments.extend(extracted)
                    if not extracted:
                        unreadable.append({"file": display, "line": 0, "reason": "no supported literal SQL fragments"})
                else:
                    fragments.append(SqlFragment(text=text, file=display))

            # A single corpus cache serves continuation requests. Read and authorize files again;
            # content hashes, not mtimes, invalidate results when scripts or comments change.
            cache_key = (
                dialect,
                database,
                frozenset(wanted),
                tuple((f.file, f.line_offset, hashlib.sha256(f.text.encode()).digest()) for f in fragments),
            )
            cached = self._extraction_cache
            if cached is not None and cached[0] == cache_key:
                extraction = cached[1]
            else:
                extraction = extract_from_fragments(
                    fragments, dialect=dialect, default_database=database, sections=wanted
                )
                # Publish key and value together; concurrent calls keep their own local extraction.
                self._extraction_cache = (cache_key, extraction)
            shaper = _Shaper(extraction, database)
            result = shaper.shape(wanted, unreadable)
            if result_path in shaper.details:
                result[result_path] = shaper.details[result_path]
            result["stats"].update(
                {
                    "files_scanned": len(files),
                    "files_truncated": truncated,
                    "dialect": dialect or "",
                    "default_database": database,
                }
            )
            result = _page_result(result, max_items, offset, result_path, max_output_chars)
            return FuncToolResult(result=result)
        except Exception as e:
            logger.error(f"extract_sql_lineage failed: {e}")
            return FuncToolResult(success=0, error=str(e))

    # ------------------------------------------------------------------ inputs

    def _resolve_dialect(self, datasource: Optional[str], dialect: Optional[str], default_database: Optional[str]):
        explicit_dialect = dialect
        dialect = ""
        database = default_database or ""
        config = self.agent_config
        if config is not None:
            name = datasource or getattr(config, "current_datasource", "") or ""
            datasources = getattr(getattr(config, "services", None), "datasources", {}) or {}
            if datasource and datasource not in datasources:
                raise DatusException(ErrorCode.TOOL_INVALID_INPUT, f"Unknown datasource: {datasource}")
            db_config = datasources.get(name)
            if db_config is not None:
                dialect = getattr(db_config, "type", "") or ""
                if not database:
                    database = getattr(db_config, "database", "") or ""
        dialect = explicit_dialect or dialect
        return (parse_dialect(dialect) if dialect else None), database

    def _collect_files(self, patterns: List[str], max_files: int):
        """Expand globs inside readable zones; hidden and out-of-workspace paths are skipped."""
        seen: Dict[Path, None] = {}
        skipped: List[Dict[str, Any]] = []
        visited = 0
        for pattern in patterns:
            expanded = os.path.expanduser(pattern)
            anchor = expanded if os.path.isabs(expanded) else str(self.root_path / expanded)
            # Reject external/hidden literal prefixes before traversing their directories.
            prefix = re.split(r"[\[*?]", anchor, maxsplit=1)[0]
            prefix_path = (
                Path(anchor)
                if not glob.has_magic(anchor)
                else (Path(prefix) if prefix.endswith(os.sep) else Path(prefix).parent)
            )
            if (
                classify_path(str(prefix_path), root_path=self.root_path, allowlist=self.path_allowlist).zone
                not in _READABLE_ZONES
            ):
                skipped.append({"file": pattern, "line": 0, "reason": "outside the readable workspace"})
                continue
            matches = glob.iglob(anchor, recursive=True) if glob.has_magic(anchor) else iter([anchor])
            matched = False
            for match in matches:
                matched = True
                visited += 1
                if visited > max_files * 20:
                    skipped.append({"file": pattern, "line": 0, "reason": "scan entry budget exceeded; narrow paths"})
                    return sorted(seen), skipped, True
                path = Path(match)
                if not path.is_file():
                    if not glob.has_magic(anchor):
                        skipped.append({"file": pattern, "line": 0, "reason": "not a readable file"})
                    continue
                if (
                    classify_path(str(path), root_path=self.root_path, allowlist=self.path_allowlist).zone
                    not in _READABLE_ZONES
                ):
                    skipped.append({"file": pattern, "line": 0, "reason": "outside the readable workspace"})
                    continue
                if path.suffix.lower() not in {".sql", ".py"}:
                    continue
                if path.stat().st_size > _MAX_FILE_BYTES:
                    skipped.append({"file": self._display(path), "line": 0, "reason": "file too large"})
                    continue
                resolved = path.resolve(strict=False)
                if resolved in seen:
                    continue
                if len(seen) == max_files:
                    return sorted(seen), skipped, True
                seen[resolved] = None
            if not matched:
                skipped.append({"file": pattern, "line": 0, "reason": "no files matched"})
        return sorted(seen), skipped, False

    def _display(self, path: Path) -> str:
        try:
            return str(path.resolve(strict=False).relative_to(self.root_path))
        except ValueError:
            return str(path)


class _Shaper:
    """Build complete summaries first; the pager alone truncates returned collections."""

    def __init__(self, extraction: ExtractionResult, database: str):
        self.extraction = extraction
        self.prefix = f"{database}." if database else ""
        self.facts = {f.statement_id: f for f in extraction.statement_facts}
        self.details = {}

    def short(self, name: Optional[str]) -> Optional[str]:
        return name[len(self.prefix) :] if name and self.prefix and name.startswith(self.prefix) else name

    def _seen_in(self, targets: Dict[str, Any]) -> List[str]:
        return list(targets)[:_MAX_SEEN_IN]

    def _support(self, records: List[Any], table: Optional[str] = None) -> Dict[str, Any]:
        evidence = {}
        hashes = set()
        for record in records:
            statement_id = getattr(record, "statement_id", "")
            fact = self.facts.get(statement_id)
            if fact is None:
                fact = next(
                    (
                        f
                        for f in self.facts.values()
                        if f.file == record.file and f.sequence == getattr(record, "sequence", -1)
                    ),
                    None,
                )
            if fact is not None:
                hashes.add(fact.sql_hash)
                statement_id = fact.statement_id
            key = (record.file, record.line, statement_id)
            evidence[key] = {"file": record.file, "line": record.line, "statement_id": statement_id}
            if getattr(record, "condition", ""):
                evidence[key]["expression"] = record.condition
        result = {
            "evidence": list(evidence.values())[:3],
            "evidence_total": len(evidence),
            "distinct_statements": len(hashes),
        }
        if table is not None:
            result["table_read_statements"] = len({f.sql_hash for f in self.facts.values() if table in f.read_tables})
        return result

    def shape(self, sections: set, unreadable: List[Dict[str, Any]]) -> Dict[str, Any]:
        ex = self.extraction
        lineage, roots = self._lineage()
        tables = self._tables()
        result = {"schema_version": 2, "lineage": lineage, "roots": roots, "tables": tables}
        if "joins" in sections:
            result["joins"] = self._joins()
        if "rules" in sections:
            result["rules"] = {
                "filters": self._filters(),
                "value_mappings": self._mappings(),
                "window_functions": self._windows(ex.window_functions),
                "dedup": self._windows(ex.dedups),
            }
        if sections & {"rules", "joins"}:
            result["conditions"] = [asdict(c) for c in ex.conditions]
        if "comments" in sections:
            result["comments"] = self._comments()
        result["unresolved"] = unreadable + [asdict(u) for u in ex.unresolved]
        databases = Counter(
            name.rsplit(".", 1)[0]
            for f in self.facts.values()
            for name in dict.fromkeys(f.read_tables + f.write_tables)
            if "." in name
        )
        written = {t for f in self.facts.values() for t in f.write_tables}
        result["stats"] = {
            "statements": ex.statements,
            "parsed": ex.parsed,
            "queries_without_target": ex.queries,
            "tables_written": len(written),
            "tables_read_only": len(roots),
            "tables_referenced": len(tables),
            "distinct_statements": len({f.sql_hash for f in self.facts.values()}),
            "databases_referenced": dict(databases.most_common()),
            "join_predicates": ex.join_predicates,
            "join_predicates_unresolved": ex.join_predicates_unresolved,
            "rule_predicates": ex.rule_predicates,
            "rule_predicates_unresolved": ex.rule_predicates_unresolved,
            "truncated_lists": {},
            "evidence_line_kind": "statement_start",
        }
        self.details = {
            "statements": [asdict(f) for f in self.facts.values()],
            "raw_lineage": [asdict(e) for e in ex.raw_lineage],
            "filter_occurrences": [asdict(p) for p in ex.predicates],
            "join_occurrences": [asdict(j) for j in ex.joins],
            "mapping_occurrences": [asdict(m) for m in ex.mappings],
            "window_occurrences": [asdict(w) for w in ex.window_functions],
            "comment_occurrences": [asdict(c) for c in ex.comments],
        }
        return result

    def _tables(self) -> List[Dict[str, Any]]:
        names = sorted({name for f in self.facts.values() for name in f.read_tables + f.write_tables})
        return [
            {
                "table": self.short(name),
                "read_statements": sum(name in f.read_tables for f in self.facts.values()),
                "write_statements": sum(name in f.write_tables for f in self.facts.values()),
                **self._support([f for f in self.facts.values() if name in f.read_tables + f.write_tables]),
            }
            for name in names
        ]

    def _lineage(self) -> tuple[List[Dict[str, Any]], List[str]]:
        grouped = {}
        for edge in self.extraction.lineage:
            grouped.setdefault(edge.target, []).append(edge)
        lineage = []
        for target, edges in grouped.items():
            item = {
                "target": self.short(target),
                "sources": sorted({self.short(s) for e in edges for s in e.sources}),
                "statements": sorted({e.statement for e in edges}),
                "load_modes": sorted({e.load_mode for e in edges}),
                "scripts": sorted({e.file for e in edges}),
                **self._support(edges),
            }
            parameters = list(dict.fromkeys(p for e in edges for p in e.parameterized_predicates))
            if parameters:
                item["parameterized_predicates"] = parameters
            temps = sorted({self.short(t) for e in edges for t in e.via_temp})
            if temps:
                item["via_temp"] = temps
            lineage.append(item)
        reads = {t for f in self.facts.values() for t in f.read_tables}
        writes = {t for f in self.facts.values() for t in f.write_tables}
        return lineage, sorted(self.short(t) for t in reads - writes)

    def _joins(self) -> List[Dict[str, Any]]:
        grouped = {}
        for join in self.extraction.joins:
            key = (
                join.left_table,
                join.right_table,
                tuple(join.keys),
                join.join_type,
                tuple(sorted(join.transforms.items())),
                tuple(join.aliases),
            )
            grouped.setdefault(key, []).append(join)
        result = []
        for key, records in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
            left, right, keys, kind, transforms, aliases = key
            item = {
                "tables": [self.short(left), self.short(right)],
                "on": [a if a == b else f"{a}={b}" for a, b in keys],
                "join_types": [kind],
                "occurrences": len(records),
                "seen_in": self._seen_in(dict.fromkeys(self.short(r.statement_target) or r.file for r in records)),
                **self._support(records),
            }
            if transforms:
                item["transforms"] = {self.short(c): expr for c, expr in transforms}
            if aliases:
                item["aliases"] = list(aliases)
            result.append(item)
        return result

    def _filters(self) -> List[Dict[str, Any]]:
        grouped = {}
        for p in self.extraction.predicates:
            key = (p.table, p.column, p.transform or "", p.predicate, p.clause)
            grouped.setdefault(key, []).append(p)
        ranked = sorted(
            grouped.items(), key=lambda pair: (-self._support(pair[1])["distinct_statements"], -len(pair[1]), pair[0])
        )
        result = []
        for (table, column, transform, predicate, clause), records in ranked:
            item = {
                "column": f"{self.short(table)}.{column}",
                "predicate": predicate,
                "files": len({r.file for r in records}),
                "occurrences": len(records),
                "clauses": {clause: len(records)},
                "seen_in": self._seen_in(dict.fromkeys(self.short(r.statement_target) or r.file for r in records)),
                **self._support(records, table),
            }
            if transform:
                item["transform"] = transform
            result.append(item)
        return result

    def _mappings(self) -> List[Dict[str, Any]]:
        grouped = {}
        for m in self.extraction.mappings:
            grouped.setdefault((m.table, m.column, m.transform or ""), []).append(m)
        result = []
        for (table, column, transform), records in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
            values = {}
            for r in records:
                values.setdefault(r.value, Counter())[r.label] += 1
            item = {
                "column": f"{self.short(table)}.{column}",
                "values": {v: labels.most_common(1)[0][0] for v, labels in values.items()},
                "occurrences": len(records),
                "partial": True,
                "seen_in": self._seen_in(dict.fromkeys(self.short(r.statement_target) or r.file for r in records)),
                **self._support(records),
            }
            if transform:
                item["transform"] = transform
            conflicts = {v: list(labels) for v, labels in values.items() if len(labels) > 1}
            if conflicts:
                item["conflicting_labels"] = conflicts
            result.append(item)
        return result

    def _windows(self, windows: List[Any]) -> List[Dict[str, Any]]:
        grouped = {}
        for w in windows:
            key = (w.table or "", tuple(w.partition_by), tuple(w.order_by), w.selection or "", w.selection_kind or "")
            grouped.setdefault(key, []).append(w)
        return [
            {
                "table": self.short(t) or None,
                "partition_by": list(p),
                "order_by": list(o),
                "selection": selection or None,
                "selection_kind": kind or None,
                "occurrences": len(records),
                "seen_in": self._seen_in(dict.fromkeys(self.short(r.statement_target) or r.file for r in records)),
                **self._support(records),
            }
            for (t, p, o, selection, kind), records in sorted(
                grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])
            )
        ]

    # --------------------------------------------------------------- comments

    def _comments(self) -> Dict[str, Any]:
        """Split author comments by what they annotate, deduplicated across versioned script copies.

        - ``column_labels``: a comment on a plain projection line (``store_code, -- 餐厅编码``) is the
          author's name for that output column; collapsed into one ``{column: label}`` glossary.
        - ``metric_notes``: a comment on an aggregate / CASE projection names a metric; the
          expression is kept so the definition can be read without opening the file.
        - ``notes``: every other prose comment (intent, caveats, join explanations) with the code it
          sits on, the table being built, and where to read it.
        - ``file_headers``: the comment block each file opens with, kept whole (a script's description,
          or in a question→SQL corpus the question and its business knowledge) — deduplicated across copies.
        - ``commented_code``: disabled SQL, counted per file with a few samples.
        """
        labels: Dict[str, Counter] = {}
        metrics: Dict[tuple, Dict[str, Any]] = {}
        notes: Dict[tuple, Dict[str, Any]] = {}
        headers: Dict[str, Dict[str, Any]] = {}
        code_by_file: Dict[str, List[str]] = {}
        for c in self.extraction.comments:
            target = self.short(c.statement_target) or c.file
            if c.kind == "header":
                entry = headers.setdefault(
                    c.text,
                    {
                        "file": c.file,
                        "line": c.line,
                        "text": c.text,
                        "copies": 0,
                        "text_truncated": c.text.endswith("..."),
                    },
                )
                entry["copies"] += 1
                continue
            if c.kind == "code":
                code_by_file.setdefault(c.file, []).append(c.text)
                continue
            alias = _projection_alias(c.code)
            if alias and _AGGREGATE_RE.search(c.code):
                entry = metrics.setdefault(
                    (alias, c.text),
                    {
                        "column": alias,
                        "label": c.text,
                        "expr": c.code,
                        "file": c.file,
                        "line": c.line,
                        "copies": 0,
                        "seen_in": {},
                    },
                )
                entry["copies"] += 1
                entry["seen_in"][target] = None
            elif alias:
                labels.setdefault(alias, Counter())[c.text] += 1
            else:
                entry = notes.setdefault(
                    (c.text, c.code),
                    {"text": c.text, "code": c.code, "target": target, "file": c.file, "line": c.line, "copies": 0},
                )
                entry["copies"] += 1

        column_labels: Dict[str, Any] = {}
        for column, counter in sorted(labels.items()):
            ranked = [text for text, _ in counter.most_common()]
            column_labels[column] = ranked[0] if len(ranked) == 1 else ranked[:3]
        metric_notes = sorted(metrics.values(), key=lambda m: (-m["copies"], m["column"]))
        for m in metric_notes:
            m["seen_in"] = self._seen_in(m["seen_in"])
        prose = sorted(notes.values(), key=lambda n: (-n["copies"], n["file"], n["line"]))
        return {
            "file_headers": [self._drop_single(h) for h in headers.values()],
            "column_labels": column_labels,
            "metric_notes": [self._drop_single(m) for m in metric_notes],
            "notes": [self._drop_single(n) for n in prose],
            "commented_code": {
                "count": sum(len(v) for v in code_by_file.values()),
                "files": len(code_by_file),
                "samples": [
                    {"file": f, "text": texts[0]}
                    for f, texts in list(code_by_file.items())[: _MAX_CODE_COMMENTS_PER_TARGET * 3]
                ],
            },
        }

    @staticmethod
    def _drop_single(item: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(item)
        if item.get("copies") == 1:
            item.pop("copies")
        if not item.get("code"):
            item.pop("code", None)
        return item


def _page_result(
    result: Dict[str, Any], limit: int, offset: int, result_path: Optional[str], budget: int
) -> Dict[str, Any]:
    """Page every collection and bound JSON size; continuation selects one collection."""
    collections = {}

    def discover(value, prefix=""):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if key == "stats":
                continue
            if isinstance(item, list) or path == "comments.column_labels":
                collections[path] = item
            elif isinstance(item, dict):
                discover(item, path)

    discover(result)
    if result_path and result_path not in collections:
        raise DatusException(
            ErrorCode.TOOL_INVALID_INPUT, f"Unknown result_path: {result_path}; available: {sorted(collections)}"
        )
    selected = {result_path: collections[result_path]} if result_path else collections
    output = {"schema_version": result["schema_version"], "stats": dict(result["stats"]), "pagination": {}}
    pages = {}

    def assign(path, value):
        dest = output
        parts = path.split(".")
        for part in parts[:-1]:
            dest = dest.setdefault(part, {})
        dest[parts[-1]] = value

    for path, items in selected.items():
        rows = list(items.items()) if isinstance(items, dict) else items
        page = rows[offset : offset + limit]
        pages[path] = page
        assign(path, dict(page) if isinstance(items, dict) else page)
        output["pagination"][path] = {
            "total": len(rows),
            "offset": offset,
            "returned": len(page),
            "next_offset": offset + len(page) if offset + len(page) < len(rows) else None,
        }
    if not result_path and "comments" in result:
        code = result["comments"].get("commented_code", {})
        output["comments"].setdefault("commented_code", {}).update({k: v for k, v in code.items() if k != "samples"})

    def refresh(path):
        page = pages[path]
        assign(path, dict(page) if isinstance(selected[path], dict) else page)
        meta = output["pagination"][path]
        meta.update(returned=len(page), next_offset=offset + len(page) if offset + len(page) < meta["total"] else None)
        output["stats"]["truncated_lists"] = {
            p: m["total"] for p, m in output["pagination"].items() if m["next_offset"] is not None or m["offset"] > 0
        }

    for path in pages:
        refresh(path)

    def size(value):
        return len(json.dumps(value, ensure_ascii=False))

    sizes = {
        path: [size(dict([row])) - 2 if isinstance(selected[path], dict) else size(row) for row in page]
        for path, page in pages.items()
    }
    current_size = size(output)
    while current_size > budget:
        candidates = [p for p in pages if pages[p]]
        if not candidates:
            # Large database inventories must not defeat the global output budget.
            databases = output["stats"].get("databases_referenced", {})
            if databases:
                databases.pop(next(reversed(databases)))
                output["stats"]["database_stats_truncated"] = True
                current_size = size(output)
                continue
            raise DatusException(
                ErrorCode.TOOL_INVALID_INPUT, "Output metadata exceeds budget; increase max_output_chars"
            )
        path = max(candidates, key=lambda p: sum(sizes[p]) + 2 * len(sizes[p]))
        if result_path and len(pages[path]) == 1:
            record = pages[path][0]
            if isinstance(record, dict):
                refs = record.get("evidence", [])[:3]
                if record.get("file"):
                    refs = [{"file": record["file"], "line": record.get("line", 0)}]
                pages[path][0] = {
                    "detail_omitted": True,
                    "reason": "Record exceeds budget; read source files",
                    "evidence": [{k: r[k] for k in ("file", "line") if k in r} for r in refs],
                }
                refresh(path)
                sizes[path] = [size(pages[path][0])]
                current_size = size(output)
                if current_size > budget:
                    pages[path].clear()
                    sizes[path].clear()
                    refresh(path)
                    current_size = size(output)
                continue
            if path == "comments.column_labels":
                key, _ = record
                pages[path][0] = (key, {"detail_omitted": True, "reason": "Read source column comments"})
                refresh(path)
                sizes[path] = [size(dict(pages[path])) - 2]
                current_size = size(output)
                if current_size > budget:
                    pages[path].clear()
                    sizes[path].clear()
                    refresh(path)
                    current_size = size(output)
                continue
        metadata_before = size(output["pagination"]) + size(output["stats"])
        current_size -= sizes[path].pop() + (2 if len(pages[path]) > 1 else 0)
        pages[path].pop()
        refresh(path)
        current_size += size(output["pagination"]) + size(output["stats"]) - metadata_before
    return output
