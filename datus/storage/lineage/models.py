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
    """One analyzed SQL source; the unit that upsert replaces and delete removes.

    Only facts derived from the source's content are kept, so checkouts on different machines
    produce the same record for the same content.
    """

    kind: Literal["file", "inline"]
    sha256: str
    dialect: str = ""
    default_database: str = ""
    analyzer_version: int
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
    """Data flows from ``upstream`` to ``downstream``; one edge per table pair.

    An edge without ``upstream`` holds the writes that read no table (``INSERT ... VALUES``,
    ``UPDATE t SET c = 1``), so a table loaded only that way still has its loading statements.
    """

    model_config = ConfigDict(populate_by_name=True)

    upstream: Optional[str] = Field(default=None, alias="from")
    downstream: str = Field(alias="to")
    evidence: List[Evidence]


class AnalysisSettings(BaseModel):
    """How the project's sources are analyzed; saved with the graph so every call agrees."""

    dialect: str = ""
    default_database: str = ""


class LineageDocument(BaseModel):
    schema_version: int = SCHEMA_VERSION
    settings: Optional[AnalysisSettings] = None
    sources: Dict[str, SourceRecord] = Field(default_factory=dict)
    nodes: Dict[str, Node] = Field(default_factory=dict)
    edges: List[Edge] = Field(default_factory=list)


@dataclass
class SourceContribution:
    """Everything one source adds to the graph, produced by analysis and applied by the store."""

    record: SourceRecord
    edges: List[Tuple[Optional[str], str, Evidence]] = field(default_factory=list)
    query_labels: Dict[str, str] = field(default_factory=dict)


def is_query_node(node_id: str) -> bool:
    return node_id.startswith(QUERY_NODE_PREFIX)
