# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Persisted shape of the project lineage graph (``lineage/lineage.json``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1
QUERY_NODE_PREFIX = "query:"


class Issue(BaseModel):
    """A statement the analyzer could not use, kept so readers can locate it in the source."""

    line: int
    reason: str


class SourceRecord(BaseModel):
    """One analyzed SQL source; the unit that upsert replaces and delete removes."""

    kind: Literal["file", "inline"]
    sha256: str
    size: Optional[int] = None
    mtime: Optional[float] = None
    dialect: str = ""
    default_database: str = ""
    analyzer_version: int
    analyzed_at: str
    statements: int = 0
    parsed: int = 0
    complete: bool = True
    issues: List[Issue] = Field(default_factory=list)


class TempHop(BaseModel):
    """A folded temporary table and the lines of the statements that built it."""

    table: str
    lines: List[int]


class Evidence(BaseModel):
    """The statement that produced an edge, located by ``source`` and its start ``line``."""

    source: str
    line: int
    operation: Optional[str] = None
    via_temp: Optional[List[TempHop]] = None


class Node(BaseModel):
    kind: Literal["table", "view", "query"]
    label: Optional[str] = None


class Edge(BaseModel):
    """Data flows from ``upstream`` to ``downstream``; one edge per table pair."""

    model_config = ConfigDict(populate_by_name=True)

    upstream: str = Field(alias="from")
    downstream: str = Field(alias="to")
    evidence: List[Evidence]


class LineageDocument(BaseModel):
    schema_version: int = SCHEMA_VERSION
    revision: int = 0
    sources: Dict[str, SourceRecord] = Field(default_factory=dict)
    nodes: Dict[str, Node] = Field(default_factory=dict)
    edges: List[Edge] = Field(default_factory=list)


@dataclass
class SourceContribution:
    """Everything one source adds to the graph, produced by analysis and applied by the store."""

    record: SourceRecord
    edges: List[Tuple[str, str, Evidence]] = field(default_factory=list)
    query_labels: Dict[str, str] = field(default_factory=dict)


def is_query_node(node_id: str) -> bool:
    return node_id.startswith(QUERY_NODE_PREFIX)
