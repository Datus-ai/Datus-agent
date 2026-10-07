# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""In-memory read model over a ``LineageDocument``: name resolution, traversal and components."""

from __future__ import annotations

import fnmatch
import glob
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple

from datus.storage.lineage.models import LineageDocument, TempHop, is_query_node

Direction = Literal["upstream", "downstream", "both"]
# A statement is the unit readers think in: one write (or query) at one place in one source.
StatementKey = Tuple[str, int, str]  # (source, line, target)


@dataclass
class Statement:
    target: str
    source: str
    line: int
    operation: str
    sources: List[str] = field(default_factory=list)
    via_temp: List[TempHop] = field(default_factory=list)


@dataclass
class Resolution:
    resolved: Dict[str, List[str]] = field(default_factory=dict)
    ambiguous: Dict[str, List[str]] = field(default_factory=dict)
    not_found: List[str] = field(default_factory=list)

    @property
    def tables(self) -> List[str]:
        return list(dict.fromkeys(name for names in self.resolved.values() for name in names))


class LineageGraph:
    """Table-level adjacency plus the queries that read each table.

    The graph may contain cycles (``INSERT INTO t SELECT ... FROM t``); every walk tracks visited nodes.
    """

    def __init__(self, document: LineageDocument):
        self.document = document
        self.tables: List[str] = sorted(n for n in document.nodes if not is_query_node(n))
        self.upstream: Dict[str, Set[str]] = defaultdict(set)
        self.downstream: Dict[str, Set[str]] = defaultdict(set)
        self.readers: Dict[str, Set[str]] = defaultdict(set)
        self.query_inputs: Dict[str, Set[str]] = defaultdict(set)
        self.statements: Dict[StatementKey, Statement] = {}
        self.statements_by_edge: Dict[Tuple[str, str], List[StatementKey]] = defaultdict(list)
        for edge in document.edges:
            if is_query_node(edge.downstream):
                self.readers[edge.upstream].add(edge.downstream)
                self.query_inputs[edge.downstream].add(edge.upstream)
            else:
                self.upstream[edge.downstream].add(edge.upstream)
                self.downstream[edge.upstream].add(edge.downstream)
            for ev in edge.evidence:
                key = (ev.source, ev.line, edge.downstream)
                statement = self.statements.get(key)
                if statement is None:
                    statement = Statement(edge.downstream, ev.source, ev.line, ev.operation or "select")
                    self.statements[key] = statement
                statement.sources.append(edge.upstream)
                for hop in ev.via_temp or []:
                    if all(existing.table != hop.table for existing in statement.via_temp):
                        statement.via_temp.append(hop)
                self.statements_by_edge[(edge.upstream, edge.downstream)].append(key)
        self._components: Optional[Dict[str, int]] = None

    # -- naming -----------------------------------------------------------------------------

    def resolve(self, names: Iterable[str], default_database: str = "") -> Resolution:
        """Exact, then case-insensitive, then default-database, then bare-table-name matching.

        A glob resolves to every match; a plain name matching several tables is ambiguous rather
        than guessed.
        """
        prefix = f"{default_database}." if default_database else ""
        by_casefold: Dict[str, List[str]] = defaultdict(list)
        by_tail: Dict[str, List[str]] = defaultdict(list)
        for table in self.tables:
            by_casefold[table.casefold()].append(table)
            by_tail[table.rsplit(".", 1)[-1].casefold()].append(table)
        result = Resolution()
        for name in names:
            if glob.has_magic(name):
                pattern = name.casefold()
                matches = [
                    t
                    for t in self.tables
                    if fnmatch.fnmatchcase(t.casefold(), pattern)
                    or (prefix and t.startswith(prefix) and fnmatch.fnmatchcase(t[len(prefix) :].casefold(), pattern))
                ]
            else:
                matches = (
                    by_casefold.get(name.casefold())
                    or by_casefold.get(f"{prefix}{name}".casefold())
                    or ([] if "." in name else by_tail.get(name.casefold(), []))
                )
                if len(matches) > 1 and name in matches:
                    matches = [name]
                if len(matches) > 1:
                    result.ambiguous[name] = sorted(matches)
                    continue
            if matches:
                result.resolved[name] = sorted(matches)
            else:
                result.not_found.append(name)
        return result

    # -- traversal --------------------------------------------------------------------------

    def traverse(self, seeds: Iterable[str], direction: Direction, depth: int) -> Tuple[Set[str], Set[Tuple[str, str]]]:
        """Tables reached from ``seeds`` and the table edges walked; ``depth < 0`` means unbounded."""
        seeds = list(seeds)
        nodes: Set[str] = set(seeds)
        edges: Set[Tuple[str, str]] = set()
        walks = []
        if direction in ("upstream", "both"):
            walks.append((self.upstream, lambda current, nxt: (nxt, current)))
        if direction in ("downstream", "both"):
            walks.append((self.downstream, lambda current, nxt: (current, nxt)))
        for adjacency, as_edge in walks:
            visited = set(seeds)
            frontier = deque((seed, 0) for seed in seeds)
            while frontier:
                current, distance = frontier.popleft()
                if 0 <= depth <= distance:
                    continue
                for nxt in sorted(adjacency.get(current, ())):
                    edges.add(as_edge(current, nxt))
                    if nxt not in visited:
                        visited.add(nxt)
                        nodes.add(nxt)
                        frontier.append((nxt, distance + 1))
        return nodes, edges

    def all_table_edges(self) -> Set[Tuple[str, str]]:
        return {(up, down) for down, ups in self.upstream.items() for up in ups}

    def statements_for(self, edges: Iterable[Tuple[str, str]]) -> List[Statement]:
        keys = dict.fromkeys(key for edge in sorted(edges) for key in self.statements_by_edge.get(edge, ()))
        return [self.statements[key] for key in keys]

    def query_statements(self, tables: Iterable[str]) -> List[Statement]:
        queries = sorted({query for table in tables for query in self.readers.get(table, ())})
        return self.statements_for((table, query) for query in queries for table in self.query_inputs[query])

    # -- derived node facts -----------------------------------------------------------------

    def role(self, table: str) -> str:
        # A self-loop (incremental ``INSERT INTO t SELECT ... FROM t``) does not make a table intermediate.
        has_up = bool(self.upstream.get(table, set()) - {table})
        has_down = bool(self.downstream.get(table, set()) - {table})
        if has_up and has_down:
            return "intermediate"
        if has_down:
            return "root"
        if has_up:
            return "leaf"
        return "isolated"

    def queried_by(self, table: str) -> int:
        return len(self.readers.get(table, ()))

    def components(self) -> Dict[str, int]:
        """Weakly connected components over table edges, numbered largest first."""
        if self._components is None:
            seen: Set[str] = set()
            groups: List[List[str]] = []
            for table in self.tables:
                if table in seen:
                    continue
                group, stack = [], [table]
                seen.add(table)
                while stack:
                    current = stack.pop()
                    group.append(current)
                    for nxt in self.upstream.get(current, set()) | self.downstream.get(current, set()):
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)
                groups.append(sorted(group))
            groups.sort(key=lambda g: (-len(g), g[0]))
            self._components = {table: i for i, group in enumerate(groups, 1) for table in group}
        return self._components
