# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Compact, unpaginated table lineage and join evidence from workspace SQL."""

from __future__ import annotations

import glob
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.fs_path_policy import PathAllowlist, PathZone, classify_path
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.sql_lineage import (
    ExtractionResult,
    SqlFragment,
    expression_tables,
    extract_from_fragments,
    extract_sql_from_python,
    render_expression,
)
from datus.utils.sql_utils import parse_dialect

logger = get_logger(__name__)

_MAX_FILE_BYTES = 2 * 1024 * 1024
_READABLE_ZONES = (PathZone.INTERNAL, PathZone.WHITELIST)


class LineageTools:
    """Read-only static analysis: no SQL execution, database access or file writes."""

    permission_category: str = "lineage_tools"

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        root_path: Optional[str] = None,
        path_allowlist: Optional[PathAllowlist] = None,
    ):
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
        datasource: Optional[str] = None,
        dialect: Optional[str] = None,
        default_database: Optional[str] = None,
    ) -> FuncToolResult:
        """Return the entire table dependency graph and observed joins in one response.

        Statically parses SQL and supported Python SQL literals; never executes them.
        Read source files for filters, mappings, windows, metrics and author comments.
        Syntax does not establish business grain, join cardinality or incremental loading.

        Args:
            paths: Workspace-relative SQL/Python files or globs, or paths on the read allowlist.
            datasource: Datasource supplying dialect and default database; defaults to active.
            dialect: Override the datasource dialect, e.g. starrocks, mysql or hive.
            default_database: Qualify bare tables; this prefix is shortened in the result.

        Returns:
            success/error/result envelope. schema_version=5 includes lineage and joins, with no
            pagination, clipping, rules, comments or issues. lineage records contain target,
            sources, operation and evidence; SELECT has target=null. Identical SQL with the same
            dependencies and operation is grouped. Safe temporary lifetimes are folded with
            via_temp and source evidence retained. Each joins record contains expression and
            evidence. expression combines relations, aliases, join type and ON/USING/WHERE
            conditions in their original scope: a physical table appears as its tables ID,
            "#3 AS o" or "#3"; a CTE/derived relation keeps its scope name followed by the
            physical tables its projection reads, "x{#1,#2}" -- read the projection for renamed
            or computed keys and never bind them to physical columns. Relations listed before the
            JOIN keyword are all in scope for the condition. These are source fragments, not
            standalone executable SQL or column lineage. tables maps "#n" IDs to table names;
            files maps "fn" IDs to paths, or to "=fn" when the file is byte-identical to an
            already analyzed file (its evidence is cited once). Join evidence is
            "file_id:start-end" (or "file_id:line" for one line), covering the condition clause.
            A suffix "@statement" explicitly marks fallback to the statement start. lineage
            evidence remains statement starts. complete=false signals skipped/failed inputs or
            unresolved relationships; read source files to investigate, as detailed issues are
            omitted. stats reports scan counts and effective dialect/database. The graph may
            contain cycles and does not prove external ownership.
        """
        if not paths:
            return FuncToolResult(success=0, error="paths must contain at least one file or glob pattern")
        try:
            dialect, database = self._resolve_dialect(datasource, dialect, default_database)
            files, unreadable = self._collect_files(paths)
            fragments: List[SqlFragment] = []
            # Byte-identical copies are analyzed once; the response maps them to the original.
            digests: Dict[str, str] = {}
            duplicates: Dict[str, str] = {}
            for file_path in files:
                display = self._display(file_path)
                try:
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    unreadable.append({"file": display, "line": 0, "reason": f"unreadable: {e}"})
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                if digest in digests:
                    duplicates[display] = digests[digest]
                    continue
                digests[digest] = display
                if file_path.suffix.lower() == ".py":
                    extracted = extract_sql_from_python(text, display)
                    fragments.extend(extracted)
                    if not extracted:
                        unreadable.append({"file": display, "line": 0, "reason": "no supported literal SQL fragments"})
                else:
                    fragments.append(SqlFragment(text=text, file=display))

            # Recheck file permissions and contents on every call, including cache hits.
            cache_key = (
                dialect,
                database,
                tuple((f.file, f.line_offset, hashlib.sha256(f.text.encode()).digest()) for f in fragments),
            )
            cached = self._extraction_cache
            if cached is not None and cached[0] == cache_key:
                extraction = cached[1]
            else:
                extraction = extract_from_fragments(
                    fragments, dialect=dialect, default_database=database, sections={"joins"}
                )
                self._extraction_cache = (cache_key, extraction)
            result = _Shaper(extraction, database, [self._display(p) for p in files], duplicates).shape(unreadable)
            result["stats"].update(files=len(files), dialect=dialect or "", default_database=database)
            return FuncToolResult(result=result)
        except Exception as e:
            logger.error(f"extract_sql_lineage failed: {e}")
            return FuncToolResult(success=0, error=str(e))

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

    def _collect_files(self, patterns: List[str]):
        """Expand all SQL/Python matches, applying the filesystem read policy to each path."""
        seen: Dict[Path, None] = {}
        skipped: List[Dict[str, Any]] = []
        for pattern in patterns:
            expanded = os.path.expanduser(pattern)
            anchor = expanded if os.path.isabs(expanded) else str(self.root_path / expanded)
            # Reject unreadable literal prefixes before traversing their directories.
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
                seen[resolved] = None
            if not matched:
                skipped.append({"file": pattern, "line": 0, "reason": "no files matched"})
        return sorted(seen), skipped

    def _display(self, path: Path) -> str:
        try:
            return str(path.resolve(strict=False).relative_to(self.root_path))
        except ValueError:
            return str(path)


class _Shaper:
    """Compact graph with complete provenance; never slice a connected component."""

    def __init__(self, extraction: ExtractionResult, database: str, files: List[str], duplicates: Dict[str, str]):
        self.extraction = extraction
        self.prefix = f"{database}." if database else ""
        self.file_ids = {name: f"f{i}" for i, name in enumerate(sorted(files), 1)}
        self.duplicates = duplicates
        self.facts = {f.sequence: f for f in extraction.statement_facts}
        tables = sorted({name for join in extraction.joins for name in expression_tables(join.expression)})
        self.table_ids = {name: f"#{i}" for i, name in enumerate(tables, 1)}

    def short(self, name: str) -> str:
        return name[len(self.prefix) :] if self.prefix and name.startswith(self.prefix) else name

    def reference(self, file: str, line: int) -> str:
        return f"{self.file_ids[file]}:{line}"

    def condition_reference(self, record) -> str:
        if record.condition_span is None:
            return f"{self.reference(record.file, record.line)}@statement"
        start, end = record.condition_span
        reference = self.reference(record.file, start)
        return f"{reference}-{end}" if end != start else reference

    @staticmethod
    def _add(grouped: Dict[tuple, Dict[str, Any]], key: tuple, item: Dict[str, Any], refs: List[str]) -> None:
        row = grouped.setdefault(key, {**item, "evidence": []})
        row["evidence"] = list(dict.fromkeys([*row["evidence"], *refs]))

    def _lineage(self) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, Dict[str, Any]] = {}
        ex = self.extraction
        for edge in ex.lineage:
            fact = self.facts[edge.sequence]
            dependencies = [fact]
            # A folded edge must still locate the intermediate builders. Match the fragment
            # as well as the file, since Python may contain separate temporary lifetimes.
            fragment = fact.statement_id.rsplit(":", 2)[1]
            for raw in ex.raw_lineage:
                origin = self.facts[raw.sequence]
                if (
                    raw.file == edge.file
                    and raw.target in edge.via_temp
                    and raw.sequence < edge.sequence
                    and origin.statement_id.rsplit(":", 2)[1] == fragment
                ):
                    dependencies.append(origin)
            item = {
                "target": self.short(edge.target),
                "sources": sorted(self.short(s) for s in edge.sources),
                "operation": edge.load_mode,
            }
            if edge.via_temp:
                item["via_temp"] = [self.short(t) for t in edge.via_temp]
            if edge.parameterized_predicates:
                item["parameterized_predicates"] = edge.parameterized_predicates
            key = (
                edge.target,
                tuple(sorted(edge.sources)),
                edge.load_mode,
                tuple(edge.via_temp),
                tuple(edge.parameterized_predicates),
                tuple(f.sql_hash for f in dependencies),
            )
            refs = [self.reference(f.file, f.line) for f in sorted(dependencies, key=lambda f: f.sequence)]
            self._add(grouped, key, item, refs)
        for fact in ex.statement_facts:
            if fact.operation == "QUERY":
                item = {
                    "target": None,
                    "sources": sorted(self.short(t) for t in fact.read_tables),
                    "operation": "select",
                }
                self._add(grouped, (None, fact.sql_hash), item, [self.reference(fact.file, fact.line)])
        return list(grouped.values())

    def _joins(self) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, Dict[str, Any]] = {}
        for join in self.extraction.joins:
            item = {"expression": render_expression(join.expression, self.table_ids.__getitem__)}
            self._add(grouped, (join.expression,), item, [self.condition_reference(join)])
        return list(grouped.values())

    def _files(self) -> Dict[str, str]:
        return {
            file_id: f"={self.file_ids[self.duplicates[name]]}" if name in self.duplicates else name
            for name, file_id in self.file_ids.items()
        }

    def shape(self, unreadable: List[Dict[str, Any]]) -> Dict[str, Any]:
        ex = self.extraction
        incomplete = bool(unreadable or ex.unresolved or any(c.status == "unresolved" for c in ex.conditions))
        return {
            "schema_version": 5,
            "complete": not incomplete,
            "files": self._files(),
            "lineage": self._lineage(),
            "tables": {table_id: self.short(name) for name, table_id in self.table_ids.items()},
            "joins": self._joins(),
            "stats": {"statements": ex.statements, "parsed": ex.parsed},
        }
