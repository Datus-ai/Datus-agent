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
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.fs_path_policy import PathAllowlist, PathZone, classify_path
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
    ) -> FuncToolResult:
        """
        Statically analyze SQL files (no SQL is executed): table lineage, join keys, constant rules, comments.

        Use it before reading SQL files one by one: it covers every file at once, counts how often each
        pattern recurs, and tells you which files and lines are worth reading. Frequency is the signal —
        a filter present in most scripts is a project-wide rule; one present once is a local choice.
        Join cardinality and business meaning are NOT inferred: verify with execute_sql and read the
        files the result points at.

        Args:
            paths: Files or glob patterns relative to the workspace, e.g. ["etl/**/*.sql", "dags/*.py"].
                .sql files are parsed whole; .py files contribute only literal triple-quoted SQL strings.
            sections: Any of "joins", "rules", "comments" in addition to lineage (always returned).
                Defaults to all. Request fewer sections to keep the result small.
            datasource: Datasource whose dialect and default database are used. Defaults to the active one.
            dialect: SQL dialect of the files (e.g. "hive", "starrocks"), overriding the datasource's —
                use when the ETL scripts target a different engine than the connected datasource.
            default_database: Database used to qualify unqualified table names; tables in it are shown
                by bare name. Defaults to the datasource's configured database.
            max_items: Cap per list (joins, filters, value_mappings, dedup, comment groups), most frequent
                first; the result reports truncation.
            max_files: Cap on files scanned; the result reports truncation.

        Returns:
            dict with 'success', 'error', and 'result' containing:
            - 'lineage': one entry per target table — sources (union over scripts), statement kinds,
              load mode (truncate_reload / overwrite / incremental / insert), the templated window of an
              incremental load, and the scripts that write it (several scripts usually means versioned copies).
            - 'roots': tables read but never written by the scanned files (ingested / upstream-owned).
            - 'joins': join relationships by occurrences — the two tables, key columns in the order of
              'tables' ("a=b", "a" when both sides share the name, composite keys kept together, "a=b|c"
              for alternative columns such as UNION branches), join types, transforms applied to a key
              before comparison (e.g. LEFT(issue_id, 6)), and 'seen_in' — tables whose build contains it.
            - 'rules': 'filters' — constant predicates on physical columns with occurrences, the clauses
              they appear in (WHERE / JOIN / HAVING / CASE) and how many files use them; 'value_mappings' —
              code-to-label dictionaries read from CASE / IF; 'dedup' — ROW_NUMBER partition and order keys.
            - 'comments': author comments — 'file_headers' are the full comment block each file opens with
              (script descriptions; in a question→SQL corpus the question plus its business knowledge — read
              every one); 'metric_notes' name computed columns; 'notes' are other prose (intent, caveats) with
              the code line they annotate; 'column_labels' is a glossary; 'commented_code' samples disabled SQL.
            - 'unresolved': statements or files that could not be analyzed, with a reason.
              Missing lineage for a table listed here means "unknown", not "no upstream".
            - 'stats': coverage counters, databases referenced, truncation flags.
        """
        if not paths:
            return FuncToolResult(success=0, error="paths must contain at least one file or glob pattern")
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
                    fragments.extend(extract_sql_from_python(text, display))
                else:
                    fragments.append(SqlFragment(text=text, file=display))

            extraction = extract_from_fragments(fragments, dialect=dialect, default_database=database, sections=wanted)
            shaper = _Shaper(extraction, database, max(1, min(max_items, _MAX_ITEMS)))
            result = shaper.shape(wanted, unreadable)
            result["stats"].update(
                {
                    "files_scanned": len(files),
                    "files_truncated": truncated,
                    "dialect": dialect or "",
                    "default_database": database,
                }
            )
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
                raise ValueError(f"Unknown datasource: {datasource}")
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
        for pattern in patterns:
            expanded = os.path.expanduser(pattern)
            anchor = expanded if os.path.isabs(expanded) else str(self.root_path / expanded)
            matches = sorted(glob.glob(anchor, recursive=True)) if glob.has_magic(anchor) else [anchor]
            if not matches:
                skipped.append({"file": pattern, "line": 0, "reason": "no files matched"})
            for match in matches:
                path = Path(match)
                if not path.is_file():
                    continue
                zone = classify_path(str(path), root_path=self.root_path, allowlist=self.path_allowlist).zone
                if zone not in _READABLE_ZONES:
                    skipped.append({"file": pattern, "line": 0, "reason": "outside the readable workspace"})
                    continue
                if path.stat().st_size > _MAX_FILE_BYTES:
                    skipped.append({"file": self._display(path), "line": 0, "reason": "file too large"})
                    continue
                seen.setdefault(path.resolve(strict=False), None)
        files = list(seen)
        truncated = len(files) > max_files
        return files[:max_files], skipped, truncated

    def _display(self, path: Path) -> str:
        try:
            return str(path.resolve(strict=False).relative_to(self.root_path))
        except ValueError:
            return str(path)


class _Shaper:
    """Aggregate raw extraction records into a compact, context-friendly result.

    Table names in ``database`` are shortened to the bare table name (the stats
    block carries ``default_database``): the same prefix on every name is most
    of the payload on a single-database warehouse.
    """

    def __init__(self, extraction: ExtractionResult, database: str, max_items: int):
        self.extraction = extraction
        self.prefix = f"{database}." if database else ""
        self.max_items = max_items
        self.truncated: Dict[str, int] = {}

    def short(self, name: Optional[str]) -> Optional[str]:
        if name and self.prefix and name.startswith(self.prefix):
            return name[len(self.prefix) :]
        return name

    def shape(self, sections, unreadable: List[Dict[str, Any]]) -> Dict[str, Any]:
        ex = self.extraction
        result: Dict[str, Any] = {}
        result["lineage"], result["roots"] = self._lineage()
        if "joins" in sections:
            result["joins"] = self._joins()
        if "rules" in sections:
            result["rules"] = {
                "filters": self._filters(),
                "value_mappings": self._mappings(),
                "dedup": self._dedups(),
            }
        if "comments" in sections:
            result["comments"] = self._comments()
        result["unresolved"] = unreadable + [
            {"file": u.file, "line": u.line, "reason": u.reason} for u in ex.unresolved
        ]
        databases = Counter(
            name.rsplit(".", 1)[0] for e in ex.lineage for name in [e.target, *e.sources] if "." in name
        )
        result["stats"] = {
            "statements": ex.statements,
            "parsed": ex.parsed,
            "queries_without_target": ex.queries,
            "tables_written": len(result["lineage"]),
            "tables_read_only": len(result["roots"]),
            "databases_referenced": dict(databases.most_common()),
            "join_predicates": ex.join_predicates,
            "join_predicates_unresolved": ex.join_predicates_unresolved,
            "rule_predicates": ex.rule_predicates,
            "rule_predicates_unresolved": ex.rule_predicates_unresolved,
            "truncated_lists": self.truncated,
        }
        return result

    def _cap(self, name: str, items: List[Any]) -> List[Any]:
        if len(items) > self.max_items:
            self.truncated[name] = len(items)
        return items[: self.max_items]

    def _cap_hard(self, name: str, items: List[Any]) -> List[Any]:
        if len(items) > _MAX_ITEMS:
            self.truncated[name] = len(items)
        return items[:_MAX_ITEMS]

    def _seen_in(self, targets: Dict[str, None]) -> List[str]:
        return list(targets)[:_MAX_SEEN_IN]

    # ---------------------------------------------------------------- lineage

    def _lineage(self):
        by_target: Dict[str, Dict[str, Any]] = {}
        for edge in self.extraction.lineage:
            entry = by_target.setdefault(
                edge.target,
                {"sources": {}, "statements": {}, "load_modes": {}, "window": {}, "scripts": {}, "via_temp": {}},
            )
            entry["sources"].update(dict.fromkeys(self.short(src) for src in edge.sources))
            entry["statements"][edge.statement] = None
            entry["load_modes"][edge.load_mode] = None
            entry["window"].update(dict.fromkeys(edge.window))
            entry["scripts"][edge.file] = None
            entry["via_temp"].update(dict.fromkeys(self.short(tmp) for tmp in edge.via_temp))
        lineage = []
        for target, entry in by_target.items():
            shaped = {
                "target": self.short(target),
                "sources": list(entry["sources"]),
                "statements": list(entry["statements"]),
                "load_modes": list(entry["load_modes"]),
                "scripts": list(entry["scripts"]),
            }
            if entry["window"]:
                shaped["window"] = list(entry["window"])
            if entry["via_temp"]:
                shaped["via_temp"] = list(entry["via_temp"])
            lineage.append(shaped)
        written = {entry["target"] for entry in lineage}
        roots = sorted({src for entry in lineage for src in entry["sources"]} - written)
        return lineage, roots

    # ------------------------------------------------------------------ joins

    def _joins(self) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, Dict[str, Any]] = {}
        for join in self.extraction.joins:
            key = (join.left_table, join.right_table, tuple(join.keys))
            entry = grouped.setdefault(key, {"join_types": {}, "transforms": {}, "occurrences": 0, "seen_in": {}})
            entry["occurrences"] += 1
            entry["join_types"][join.join_type] = None
            entry["transforms"].update({self.short(col): expr for col, expr in join.transforms.items()})
            entry["seen_in"][self.short(join.statement_target) or join.file] = None
        ranked = sorted(grouped.items(), key=lambda item: (-item[1]["occurrences"], item[0]))
        joins = []
        for (left_table, right_table, keys), entry in self._cap("joins", ranked):
            shaped = {
                "tables": [self.short(left_table), self.short(right_table)],
                "on": [left if left == right else f"{left}={right}" for left, right in keys],
                "join_types": list(entry["join_types"]),
                "occurrences": entry["occurrences"],
                "seen_in": self._seen_in(entry["seen_in"]),
            }
            if entry["transforms"]:
                shaped["transforms"] = entry["transforms"]
            joins.append(shaped)
        return joins

    # ------------------------------------------------------------------ rules

    def _filters(self) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, Dict[str, Any]] = {}
        for p in self.extraction.predicates:
            key = (p.table, p.column, p.transform or "", p.predicate)
            entry = grouped.setdefault(key, {"occurrences": 0, "clauses": Counter(), "files": set(), "seen_in": {}})
            entry["occurrences"] += 1
            entry["clauses"][p.clause] += 1
            entry["files"].add(p.file)
            entry["seen_in"][self.short(p.statement_target) or p.file] = None
        ranked = sorted(grouped.items(), key=lambda item: (-len(item[1]["files"]), -item[1]["occurrences"], item[0]))
        filters = []
        for (table, column, transform, predicate), entry in self._cap("filters", ranked):
            shaped = {
                "column": f"{self.short(table)}.{column}",
                "predicate": predicate,
                "files": len(entry["files"]),
                "occurrences": entry["occurrences"],
                "clauses": dict(entry["clauses"].most_common()),
                "seen_in": self._seen_in(entry["seen_in"]),
            }
            if transform:
                shaped["transform"] = transform
            filters.append(shaped)
        return filters

    def _mappings(self) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, Dict[str, Any]] = {}
        for m in self.extraction.mappings:
            entry = grouped.setdefault(
                (m.table, m.column, m.transform or ""), {"values": {}, "occurrences": 0, "seen_in": {}}
            )
            entry["occurrences"] += 1
            entry["values"].setdefault(m.value, Counter())[m.label] += 1
            entry["seen_in"][self.short(m.statement_target) or m.file] = None
        ranked = sorted(grouped.items(), key=lambda item: (-item[1]["occurrences"], item[0]))
        mappings = []
        for (table, column, transform), entry in self._cap("value_mappings", ranked):
            values = {value: labels.most_common(1)[0][0] for value, labels in entry["values"].items()}
            shaped = {
                "column": f"{self.short(table)}.{column}",
                "values": values,
                "occurrences": entry["occurrences"],
                "seen_in": self._seen_in(entry["seen_in"]),
            }
            if transform:
                shaped["transform"] = transform
            conflicts = {value: list(labels) for value, labels in entry["values"].items() if len(labels) > 1}
            if conflicts:
                shaped["conflicting_labels"] = conflicts
            mappings.append(shaped)
        return mappings

    def _dedups(self) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, Dict[str, Any]] = {}
        for d in self.extraction.dedups:
            key = (d.table, tuple(d.partition_by), tuple(d.order_by))
            entry = grouped.setdefault(key, {"occurrences": 0, "seen_in": {}})
            entry["occurrences"] += 1
            entry["seen_in"][self.short(d.statement_target) or d.file] = None
        ranked = sorted(grouped.items(), key=lambda item: (-item[1]["occurrences"], item[0]))
        return [
            {
                "table": self.short(table),
                "partition_by": list(partition),
                "order_by": list(order),
                "occurrences": entry["occurrences"],
                "seen_in": self._seen_in(entry["seen_in"]),
            }
            for (table, partition, order), entry in self._cap("dedup", ranked)
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
                entry = headers.setdefault(c.text, {"file": c.file, "text": c.text, "copies": 0})
                entry["copies"] += 1
                continue
            if c.kind == "code":
                code_by_file.setdefault(c.file, []).append(c.text)
                continue
            alias = _projection_alias(c.code)
            if alias and _AGGREGATE_RE.search(c.code):
                entry = metrics.setdefault(
                    (alias, c.text), {"column": alias, "label": c.text, "expr": c.code, "copies": 0, "seen_in": {}}
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
            # Headers are the primary human input of a corpus: bounded by the hard cap, not by max_items.
            "file_headers": [self._drop_single(h) for h in self._cap_hard("file_headers", list(headers.values()))],
            "column_labels": column_labels,
            "metric_notes": [self._drop_single(m) for m in self._cap("metric_notes", metric_notes)],
            "notes": [self._drop_single(n) for n in self._cap("notes", prose)],
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
