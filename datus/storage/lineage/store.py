# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Single-file lineage store: one project graph, replaced one source at a time."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional, Set, Tuple

from pydantic import ValidationError

from datus.storage.lineage.models import (
    SCHEMA_VERSION,
    Edge,
    LineageDocument,
    Node,
    SourceContribution,
    is_query_node,
)
from datus.storage.semantic_model.artifact_file import atomic_write_text, path_mutation_lock
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl; the in-process lock still applies
    fcntl = None

logger = get_logger(__name__)

LINEAGE_FILE_NAME = "lineage.json"


class LineageStore:
    """Reads and atomically rewrites ``lineage.json`` under a cross-process lock.

    The lock is taken on the file's directory, which exists for the graph alone, so no lock file
    is left behind for version control to pick up.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._cache: Optional[Tuple[Tuple[int, int], LineageDocument]] = None
        self._cache_lock = threading.Lock()

    def load(self) -> LineageDocument:
        """Current document; an absent file is an empty graph. Callers must not mutate the result."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return LineageDocument()
        key = (stat.st_mtime_ns, stat.st_size)
        with self._cache_lock:
            if self._cache is not None and self._cache[0] == key:
                return self._cache[1]
        document = self._read()
        with self._cache_lock:
            self._cache = (key, document)
        return document

    @contextmanager
    def edit(self) -> Iterator[LineageDocument]:
        """Yield a fresh copy to mutate; it is written back only when its content changed."""
        with path_mutation_lock(self.path), self._exclusive():
            document = self._read() if self.path.exists() else LineageDocument()
            before = serialize(document)
            yield document
            normalize(document)
            if serialize(document) != before:
                document.revision += 1
                atomic_write_text(self.path, serialize(document))

    def _read(self) -> LineageDocument:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise DatusException(
                ErrorCode.COMMON_JSON_PARSE_ERROR, message_args={"file_path": str(self.path), "error_detail": str(e)}
            ) from e
        if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
            version = raw.get("schema_version") if isinstance(raw, dict) else None
            raise DatusException(
                ErrorCode.COMMON_JSON_PARSE_ERROR,
                message_args={
                    "file_path": str(self.path),
                    "error_detail": f"unsupported schema_version {version!r}, expected {SCHEMA_VERSION}",
                },
            )
        try:
            return LineageDocument.model_validate(raw)
        except ValidationError as e:
            raise DatusException(
                ErrorCode.COMMON_JSON_PARSE_ERROR, message_args={"file_path": str(self.path), "error_detail": str(e)}
            ) from e

    @contextmanager
    def _exclusive(self):
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        if fcntl is None:
            yield
            return
        # A directory always exists while it is locked, unlike a lock file that a cleanup could
        # unlink and another process recreate, letting two writers each hold "the" lock.
        fd = os.open(directory, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def remove_source(document: LineageDocument, source_id: str) -> None:
    """Withdraw every piece of evidence ``source_id`` contributed, then drop what nothing supports."""
    document.sources.pop(source_id, None)
    edges = []
    for edge in document.edges:
        edge.evidence = [ev for ev in edge.evidence if ev.source != source_id]
        if edge.evidence:
            edges.append(edge)
    document.edges = edges
    _rebuild_nodes(document, {})


def replace_source(document: LineageDocument, source_id: str, contribution: SourceContribution) -> None:
    """Upsert: the source's previous contribution is removed before the new one is merged in."""
    remove_source(document, source_id)
    document.sources[source_id] = contribution.record
    by_pair: Dict[Tuple[str, str], Edge] = {(e.upstream, e.downstream): e for e in document.edges}
    for upstream, downstream, evidence in contribution.edges:
        edge = by_pair.get((upstream, downstream))
        if edge is None:
            edge = Edge(upstream=upstream, downstream=downstream, evidence=[])
            by_pair[(upstream, downstream)] = edge
            document.edges.append(edge)
        if not any(ev.source == evidence.source and ev.line == evidence.line for ev in edge.evidence):
            edge.evidence.append(evidence)
    _rebuild_nodes(document, contribution.query_labels)


def _rebuild_nodes(document: LineageDocument, labels: Dict[str, str]) -> None:
    """Nodes are exactly the edge endpoints; a node no edge references disappears."""
    views: Set[str] = {e.downstream for e in document.edges if any(ev.operation == "create_view" for ev in e.evidence)}
    nodes: Dict[str, Node] = {}
    for edge in document.edges:
        for node_id in (edge.upstream, edge.downstream):
            if node_id in nodes:
                continue
            if is_query_node(node_id):
                previous = document.nodes.get(node_id)
                label = labels.get(node_id) or (previous.label if previous else None)
                nodes[node_id] = Node(kind="query", label=label)
            else:
                nodes[node_id] = Node(kind="view" if node_id in views else "table")
    document.nodes = nodes


def normalize(document: LineageDocument) -> None:
    """Deterministic order, so identical graphs serialize identically and diffs stay small."""
    for edge in document.edges:
        edge.evidence.sort(key=lambda ev: (ev.source, ev.line))
    document.edges.sort(key=lambda e: (e.downstream, e.upstream))
    document.sources = dict(sorted(document.sources.items()))
    document.nodes = dict(sorted(document.nodes.items()))


def serialize(document: LineageDocument) -> str:
    return json.dumps(document.model_dump(by_alias=True, exclude_none=True), ensure_ascii=False, indent=2) + "\n"


def graph_keys(document: LineageDocument) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    return set(document.nodes), {(e.upstream, e.downstream) for e in document.edges}
