# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Persisted project lineage: analyze SQL into ``lineage/lineage.json`` and query it by table."""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.storage.lineage.analysis import analyze_source, content_digest
from datus.storage.lineage.graph import LineageGraph, Statement
from datus.storage.lineage.models import LineageDocument, SourceRecord, is_query_node
from datus.storage.lineage.store import LINEAGE_FILE_NAME, LineageStore, graph_keys, remove_source, replace_source
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.fs_path_policy import PathAllowlist, PathZone, classify_path
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.sql_lineage import ANALYZER_VERSION
from datus.utils.sql_utils import parse_dialect

logger = get_logger(__name__)

_MAX_FILE_BYTES = 2 * 1024 * 1024
_READABLE_ZONES = (PathZone.INTERNAL, PathZone.WHITELIST)
_INLINE_PREFIX = "inline:"
_MAX_REPORTED_ISSUES = 50


@dataclass
class _Input:
    source_id: str
    kind: Literal["file", "inline"]
    text: str
    python: bool = False
    size: Optional[int] = None
    mtime: Optional[float] = None


class LineageTools:
    """Static analysis only: SQL is parsed, never executed, and no database is accessed.

    Writes go to the project lineage file alone; every read of a source file passes the
    filesystem read policy.
    """

    permission_category: str = "lineage_tools"

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        root_path: Optional[str] = None,
        path_allowlist: Optional[PathAllowlist] = None,
        lineage_path: Optional[str] = None,
    ):
        self.agent_config = agent_config
        self.root_path = Path(root_path or os.getcwd()).expanduser().resolve(strict=False)
        self.path_allowlist = path_allowlist
        self.store = LineageStore(Path(lineage_path) if lineage_path else self._default_lineage_path())

    @classmethod
    def all_tools_name(cls) -> List[str]:
        return ["upsert_lineage", "delete_lineage", "query_lineage"]

    def available_tools(self) -> List[Tool]:
        return [
            trans_to_function_tool(self.upsert_lineage),
            trans_to_function_tool(self.delete_lineage),
            trans_to_function_tool(self.query_lineage),
        ]

    def _default_lineage_path(self) -> Path:
        path_manager = getattr(self.agent_config, "path_manager", None)
        if path_manager is not None:
            return Path(path_manager.lineage_dir) / LINEAGE_FILE_NAME
        return self.root_path / "lineage" / LINEAGE_FILE_NAME

    # -- write ------------------------------------------------------------------------------

    def upsert_lineage(
        self,
        paths: Optional[List[str]] = None,
        sql: Optional[str] = None,
        source_id: Optional[str] = None,
        datasource: Optional[str] = None,
        dialect: Optional[str] = None,
        default_database: Optional[str] = None,
        prune_missing: bool = False,
    ) -> FuncToolResult:
        """Analyze SQL and save its table lineage into the project graph.

        Statically parses SQL and supported Python SQL literals; never executes them. Each file,
        or each piece of inline SQL, is one source: re-analyzing a source replaces everything it
        contributed before, so lineage from deleted statements disappears. Sources whose content,
        dialect, default database and analyzer version are unchanged are skipped, so repeated
        calls are cheap. Returns a change summary only; call query_lineage to read the graph.

        Args:
            paths: Workspace-relative SQL/Python files or globs, or paths on the read allowlist.
                Provide exactly one of paths or sql.
            sql: SQL text that does not live in a file. Save only validated SQL worth keeping,
                never ad-hoc exploration.
            source_id: Stable name for sql, stored as "inline:<source_id>". Defaults to a content
                hash, so saving the same SQL twice is idempotent.
            datasource: Datasource supplying dialect and default database; defaults to active.
            dialect: Override the datasource dialect, e.g. starrocks, mysql or hive.
            default_database: Database used to qualify bare table names.
            prune_missing: With paths, also remove saved sources matching the patterns whose files
                no longer exist.

        Returns:
            revision of the saved graph; added, updated and removed source IDs; unchanged count;
            skipped inputs with reasons; incomplete sources with the statements that could not be
            analyzed (line and reason); delta counts of nodes and edges. An incomplete source may
            miss dependencies: read its source around the reported lines.
        """
        if bool(paths) == bool(sql):
            return FuncToolResult(success=0, error="Provide exactly one of paths or sql")
        if source_id and not sql:
            return FuncToolResult(success=0, error="source_id only applies to sql")
        if prune_missing and not paths:
            return FuncToolResult(success=0, error="prune_missing only applies to paths")
        try:
            dialect, database = self._resolve_dialect(datasource, dialect, default_database)
            inputs, skipped = self._read_inputs(paths or [], sql, source_id)
            added: List[str] = []
            updated: List[str] = []
            removed: List[str] = []
            incomplete: List[Dict[str, Any]] = []
            unchanged = 0
            with self.store.edit() as document:
                nodes_before, edges_before = graph_keys(document)
                for item in inputs:
                    existing = document.sources.get(item.source_id)
                    if existing is not None and self._unchanged(existing, item, dialect or "", database):
                        unchanged += 1
                        # Same content under a new mtime: remember it, so staleness checks need not hash.
                        existing.size, existing.mtime = item.size, item.mtime
                        continue
                    contribution = analyze_source(
                        item.source_id,
                        item.text,
                        kind=item.kind,
                        python=item.python,
                        dialect=dialect,
                        default_database=database,
                        size=item.size,
                        mtime=item.mtime,
                    )
                    replace_source(document, item.source_id, contribution)
                    (added if existing is None else updated).append(item.source_id)
                    if not contribution.record.complete:
                        issues = [issue.model_dump() for issue in contribution.record.issues]
                        incomplete.append({"source": item.source_id, "issues": issues})
                if prune_missing:
                    for sid, record in list(document.sources.items()):
                        if (
                            record.kind == "file"
                            and any(_pattern_matches(sid, self._normalize_pattern(p)) for p in paths or [])
                            and not self._source_path(sid).exists()
                        ):
                            remove_source(document, sid)
                            removed.append(sid)
                nodes_after, edges_after = graph_keys(document)
            result = {
                "revision": document.revision,
                "added": added,
                "updated": updated,
                "unchanged": unchanged,
                "removed": removed,
                "skipped": skipped,
                "incomplete": _cap_issues(incomplete),
                "delta": _delta(nodes_before, edges_before, nodes_after, edges_after),
            }
            return FuncToolResult(result=result)
        except Exception as e:
            logger.error(f"upsert_lineage failed: {e}")
            return FuncToolResult(success=0, error=str(e))

    def delete_lineage(self, sources: List[str]) -> FuncToolResult:
        """Remove saved sources and every edge only they supported.

        Args:
            sources: Source IDs or globs over source IDs, e.g. "etl/old/*.sql" or "inline:*".
                Matching uses saved IDs, so sources whose files are already gone can be removed.

        Returns:
            revision, removed source IDs, patterns that matched nothing (not_found), and delta
            counts of nodes and edges.
        """
        if not sources:
            return FuncToolResult(success=0, error="sources must contain at least one source ID or pattern")
        try:
            patterns = [self._normalize_pattern(p) for p in sources]
            with self.store.edit() as document:
                nodes_before, edges_before = graph_keys(document)
                removed = [sid for sid in document.sources if any(_pattern_matches(sid, p) for p in patterns)]
                for sid in removed:
                    remove_source(document, sid)
                nodes_after, edges_after = graph_keys(document)
            not_found = [
                raw for raw, pattern in zip(sources, patterns) if not any(_pattern_matches(s, pattern) for s in removed)
            ]
            result = {
                "revision": document.revision,
                "removed": removed,
                "not_found": not_found,
                "delta": _delta(nodes_before, edges_before, nodes_after, edges_after),
            }
            return FuncToolResult(result=result)
        except Exception as e:
            logger.error(f"delete_lineage failed: {e}")
            return FuncToolResult(success=0, error=str(e))

    # -- read -------------------------------------------------------------------------------

    def query_lineage(
        self,
        tables: Optional[List[str]] = None,
        direction: str = "both",
        depth: int = 1,
        include_queries: bool = False,
    ) -> FuncToolResult:
        """Read table lineage from the project graph saved by upsert_lineage.

        With tables, returns the subgraph around them; without, the whole table graph. Results are
        never paginated or clipped: narrow tables or depth instead. The graph may contain cycles
        and only covers analyzed sources: a table without upstream here is a root of this corpus,
        not proof of external ownership.

        Args:
            tables: Table names or globs: full names, names without the default database, or bare
                table names, matched case-insensitively. A plain name matching several tables is
                returned in ambiguous instead of being guessed.
            direction: upstream, downstream or both.
            depth: Table hops to follow from tables; -1 follows to roots and leaves. Ignored
                without tables.
            include_queries: Also return saved pure queries reading the returned tables, as
                "query:<hash>" targets; otherwise nodes carry only a queried_by count.

        Returns:
            lineage records, one per statement: target, sources (every table the statement
            reads, possibly beyond depth), operation, via_temp (folded temporary tables) and
            evidence "file_id:line". The first evidence is the statement start; the rest locate
            the statements that built via_temp tables. Read the source there for filters,
            formulas and join logic. files maps file IDs to source IDs; copies lists sources
            byte-identical to a cited file, whose statements are cited once under that file.
            nodes maps each returned
            table to role (root/intermediate/leaf/isolated, ignoring queries), component (weakly
            connected subgraph) and, when non-zero, queried_by; kind appears for views and
            queries, and label for queries that carry the author's comment. components lists the
            multi-table subgraphs involved with their size, plus roots and leaves when tables is
            omitted. complete=false means some
            sources involved are stale (changed since analysis: re-run upsert_lineage) or
            incomplete (statements failed to parse): never treat a missing edge as absent.
            resolved, ambiguous and not_found report how tables were matched. Names are
            shortened by stats.default_database.
        """
        if direction not in ("upstream", "downstream", "both"):
            return FuncToolResult(success=0, error="direction must be upstream, downstream or both")
        if not isinstance(depth, int) or depth < -1:
            return FuncToolResult(success=0, error="depth must be a non-negative integer or -1")
        try:
            _, database = self._resolve_dialect(None, None, None)
            document = self.store.load()
            # Shorten by the database the sources were analyzed with when they agree: scripts may
            # qualify tables with a database other than the datasource's.
            databases = {record.default_database for record in document.sources.values()}
            if len(databases) == 1:
                database = databases.pop()
            graph = LineageGraph(document)
            result: Dict[str, Any] = {}
            if tables:
                resolution = graph.resolve(tables, database)
                nodes, edges = graph.traverse(resolution.tables, direction, depth)
                checked = None
            else:
                resolution = None
                nodes, edges = set(graph.tables), graph.all_table_edges()
                checked = list(document.sources)
            statements = graph.statements_for(edges)
            if include_queries:
                statements += graph.query_statements(nodes)
            shaper = _Shaper(database, statements, document)
            checked = checked if checked is not None else shaper.cited_sources()
            stale = [(sid, reason) for sid in checked if (reason := self._staleness(sid, document.sources.get(sid)))]
            incomplete = [sid for sid in checked if sid in document.sources and not document.sources[sid].complete]
            result["complete"] = not stale and not incomplete
            if resolution is not None:
                result["resolved"] = {k: [shaper.short(n) for n in v] for k, v in resolution.resolved.items()}
                if resolution.ambiguous:
                    result["ambiguous"] = {k: [shaper.short(n) for n in v] for k, v in resolution.ambiguous.items()}
                if resolution.not_found:
                    result["not_found"] = resolution.not_found
            result["files"] = {fid: sid for sid, fid in shaper.file_ids.items()}
            if shaper.copies:
                result["copies"] = shaper.copies
            result["nodes"] = shaper.nodes(graph, nodes, statements if include_queries else [])
            result["lineage"] = shaper.records()
            result["components"] = shaper.components(graph, nodes, detail=resolution is None)
            if stale:
                result["stale"] = [{"source": shaper.ref(sid), "reason": reason} for sid, reason in stale]
            if incomplete:
                result["incomplete"] = [shaper.ref(sid) for sid in incomplete]
            result["stats"] = {
                "default_database": database,
                "tables": sum(1 for n in nodes if not is_query_node(n)),
                "statements": len(result["lineage"]),
                "revision": document.revision,
            }
            if not document.sources:
                result["hint"] = "No lineage has been saved for this project yet; call upsert_lineage first."
            return FuncToolResult(result=result)
        except Exception as e:
            logger.error(f"query_lineage failed: {e}")
            return FuncToolResult(success=0, error=str(e))

    # -- helpers ----------------------------------------------------------------------------

    def _read_inputs(self, paths: List[str], sql: Optional[str], source_id: Optional[str]):
        if sql:
            name = (source_id or "").strip() or f"adhoc-{content_digest(sql)[:8]}"
            return [_Input(source_id=f"{_INLINE_PREFIX}{name}", kind="inline", text=sql)], []
        files, skipped = self._collect_files(paths)
        inputs = []
        for path in files:
            display = self._display(path)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                stat = path.stat()
            except OSError as e:
                skipped.append({"file": display, "reason": f"unreadable: {e}"})
                continue
            inputs.append(
                _Input(
                    source_id=display,
                    kind="file",
                    text=text,
                    python=path.suffix.lower() == ".py",
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                )
            )
        return inputs, skipped

    @staticmethod
    def _unchanged(record: SourceRecord, item: _Input, dialect: str, database: str) -> bool:
        return (
            record.kind == item.kind
            and record.sha256 == content_digest(item.text)
            and record.dialect == dialect
            and record.default_database == database
            and record.analyzer_version == ANALYZER_VERSION
        )

    def _staleness(self, source_id: str, record: Optional[SourceRecord]) -> Optional[str]:
        if record is None:
            return None
        if record.analyzer_version < ANALYZER_VERSION:
            return "analyzer_upgraded"
        if record.kind != "file":
            return None
        path = self._source_path(source_id)
        try:
            stat = path.stat()
            if stat.st_size == record.size and stat.st_mtime == record.mtime:
                return None
            digest = content_digest(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return "deleted"
        return None if digest == record.sha256 else "modified"

    def _source_path(self, source_id: str) -> Path:
        path = Path(source_id)
        return path if path.is_absolute() else self.root_path / path

    def _normalize_pattern(self, pattern: str) -> str:
        """Express a pattern the way source IDs are stored: workspace-relative when possible."""
        if pattern.startswith(_INLINE_PREFIX):
            return pattern
        expanded = os.path.expanduser(pattern)
        if os.path.isabs(expanded):
            path = Path(expanded) if glob.has_magic(expanded) else Path(expanded).resolve(strict=False)
            try:
                return str(path.relative_to(self.root_path))
            except ValueError:
                return str(path)
        return os.path.normpath(expanded) if not glob.has_magic(expanded) else expanded.removeprefix("./")

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
                skipped.append({"file": pattern, "reason": "outside the readable workspace"})
                continue
            matches = glob.iglob(anchor, recursive=True) if glob.has_magic(anchor) else iter([anchor])
            matched = False
            for match in matches:
                matched = True
                path = Path(match)
                if not path.is_file():
                    if not glob.has_magic(anchor):
                        skipped.append({"file": pattern, "reason": "not a readable file"})
                    continue
                if (
                    classify_path(str(path), root_path=self.root_path, allowlist=self.path_allowlist).zone
                    not in _READABLE_ZONES
                ):
                    skipped.append({"file": pattern, "reason": "outside the readable workspace"})
                    continue
                if path.suffix.lower() not in {".sql", ".py"}:
                    continue
                if path.stat().st_size > _MAX_FILE_BYTES:
                    skipped.append({"file": self._display(path), "reason": "file too large"})
                    continue
                resolved = path.resolve(strict=False)
                if resolved in seen:
                    continue
                seen[resolved] = None
            if not matched:
                skipped.append({"file": pattern, "reason": "no files matched"})
        return sorted(seen), skipped

    def _display(self, path: Path) -> str:
        try:
            return str(path.resolve(strict=False).relative_to(self.root_path))
        except ValueError:
            return str(path)


def _glob_regex(pattern: str) -> "re.Pattern[str]":
    """Path-aware glob: ``*`` and ``?`` stop at ``/``; ``**/`` spans any number of directories."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[" and (end := pattern.find("]", i + 2)) != -1:
            body = pattern[i + 1 : end]
            body = "^" + body[1:] if body.startswith("!") else body
            out.append(f"[{body.replace(chr(92), chr(92) * 2)}]")
            i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _pattern_matches(source_id: str, pattern: str) -> bool:
    if not glob.has_magic(pattern):
        return source_id == pattern
    return bool(_glob_regex(pattern).match(source_id))


def _delta(nodes_before, edges_before, nodes_after, edges_after) -> Dict[str, int]:
    return {
        "nodes_added": len(nodes_after - nodes_before),
        "nodes_removed": len(nodes_before - nodes_after),
        "edges_added": len(edges_after - edges_before),
        "edges_removed": len(edges_before - edges_after),
    }


def _cap_issues(incomplete: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Report at most ``_MAX_REPORTED_ISSUES`` issues; the rest stay in the saved source records."""
    budget = _MAX_REPORTED_ISSUES
    capped = []
    for entry in incomplete:
        issues = entry["issues"][: max(budget, 0)]
        budget -= len(issues)
        row = {"source": entry["source"], "issues": issues}
        if len(issues) < len(entry["issues"]):
            row["issues_omitted"] = len(entry["issues"]) - len(issues)
        capped.append(row)
    return capped


class _Shaper:
    """Compact response: file IDs for sources and table names shortened by the default database."""

    def __init__(self, database: str, statements: List[Statement], document: LineageDocument):
        self.prefix = f"{database}." if database else ""
        # Byte-identical sources analyzed the same way contribute identical statements: cite the
        # first one and list the others as its copies.
        canonical: Dict[str, str] = {}
        first: Dict[tuple, str] = {}
        for sid in sorted({st.source for st in statements}):
            record = document.sources.get(sid)
            key = (record.sha256, record.dialect, record.default_database) if record is not None else (sid,)
            canonical[sid] = first.setdefault(key, sid)
        self.statements = [st for st in statements if canonical[st.source] == st.source]
        cited = sorted({st.source for st in self.statements})
        self.file_ids = {sid: f"f{i}" for i, sid in enumerate(cited, 1)}
        self.copies: Dict[str, List[str]] = {}
        for sid, original in canonical.items():
            if sid != original:
                self.copies.setdefault(self.file_ids[original], []).append(sid)
        self._canonical = canonical

    def cited_sources(self) -> List[str]:
        return list(self._canonical)

    def short(self, name: str) -> str:
        return name[len(self.prefix) :] if self.prefix and name.startswith(self.prefix) else name

    def ref(self, source_id: str) -> str:
        return self.file_ids.get(source_id, source_id)

    def records(self) -> List[Dict[str, Any]]:
        rows = []
        for st in self.statements:
            fid = self.file_ids[st.source]
            evidence = [f"{fid}:{st.line}"]
            evidence.extend(f"{fid}:{line}" for hop in st.via_temp for line in hop.lines)
            row: Dict[str, Any] = {
                "target": self.short(st.target),
                "sources": sorted(self.short(s) for s in st.sources),
                "operation": st.operation,
            }
            if st.via_temp:
                row["via_temp"] = [self.short(hop.table) for hop in st.via_temp]
            row["evidence"] = list(dict.fromkeys(evidence))
            rows.append(row)
        return sorted(rows, key=lambda r: (r["target"], r["evidence"][0]))

    def nodes(self, graph: LineageGraph, tables, query_statements: List[Statement]) -> Dict[str, Any]:
        components = graph.components()
        document: LineageDocument = graph.document
        result: Dict[str, Any] = {}
        for table in sorted(tables):
            entry: Dict[str, Any] = {}
            kind = document.nodes[table].kind if table in document.nodes else "table"
            if kind != "table":
                entry["kind"] = kind
            entry["role"] = graph.role(table)
            entry["component"] = components[table]
            if queried := graph.queried_by(table):
                entry["queried_by"] = queried
            result[self.short(table)] = entry
        for st in query_statements:
            if is_query_node(st.target) and st.target not in result:
                node = document.nodes.get(st.target)
                entry = {"kind": "query"}
                if node is not None and node.label:
                    entry["label"] = node.label
                result[st.target] = entry
        return result

    def components(self, graph: LineageGraph, tables, detail: bool) -> List[Dict[str, Any]]:
        membership = graph.components()
        wanted = {membership[t] for t in tables if t in membership}
        groups: Dict[int, List[str]] = {}
        for table, component in membership.items():
            if component in wanted:
                groups.setdefault(component, []).append(table)
        rows = []
        for component in sorted(groups):
            members = sorted(groups[component])
            if len(members) < 2:
                continue
            row: Dict[str, Any] = {"id": component, "tables": len(members)}
            if detail:
                row["roots"] = [self.short(t) for t in members if graph.role(t) == "root"]
                row["leaves"] = [self.short(t) for t in members if graph.role(t) == "leaf"]
            rows.append(row)
        return rows
