# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Turn the static analysis of one SQL source into its contribution to the lineage graph."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from datus.storage.lineage.models import (
    QUERY_NODE_PREFIX,
    Evidence,
    Issue,
    SourceContribution,
    SourceRecord,
    TempHop,
)
from datus.utils.sql_lineage import (
    ANALYZER_VERSION,
    Comment,
    ExtractionResult,
    SqlFragment,
    StatementFact,
    extract_from_fragments,
    extract_sql_from_python,
)

_MAX_LABEL_LEN = 200
_COMMENT_LEADS = ("--", "#", "/*", "*")


def content_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def analyze_source(
    source_id: str,
    text: str,
    *,
    kind: Literal["file", "inline"],
    python: bool = False,
    dialect: Optional[str] = None,
    default_database: str = "",
    size: Optional[int] = None,
    mtime: Optional[float] = None,
) -> SourceContribution:
    """Statically analyze ``text`` (never executed) and attribute every edge to ``source_id``."""
    # A Python file without SQL literals is simply not a SQL source, not a failed input.
    fragments = extract_sql_from_python(text, source_id) if python else [SqlFragment(text=text, file=source_id)]
    extraction = extract_from_fragments(
        fragments, dialect=dialect, default_database=default_database, sections={"comments"}
    )
    record = SourceRecord(
        kind=kind,
        sha256=content_digest(text),
        size=size,
        mtime=mtime,
        dialect=dialect or "",
        default_database=default_database,
        analyzer_version=ANALYZER_VERSION,
        analyzed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        statements=extraction.statements,
        parsed=extraction.parsed,
        complete=not extraction.unresolved,
        issues=[Issue(line=u.line, reason=u.reason) for u in extraction.unresolved],
    )
    contribution = SourceContribution(record=record)
    _add_writes(contribution, extraction, source_id)
    _add_queries(contribution, extraction, source_id, fragments)
    return contribution


def _fragment_of(fact: StatementFact) -> str:
    # statement_id is "file:line_offset:sequence"; a Python file holds one fragment per literal.
    return fact.statement_id.rsplit(":", 2)[1]


def _add_writes(contribution: SourceContribution, extraction: ExtractionResult, source_id: str) -> None:
    facts = {fact.sequence: fact for fact in extraction.statement_facts}
    for edge in extraction.lineage:
        fragment = _fragment_of(facts[edge.sequence])
        hops = []
        for temp in edge.via_temp:
            # The builders of a folded temporary table, so readers can locate the logic it hid.
            lines = sorted(
                {
                    facts[raw.sequence].line
                    for raw in extraction.raw_lineage
                    if raw.target == temp
                    and raw.sequence < edge.sequence
                    and _fragment_of(facts[raw.sequence]) == fragment
                }
            )
            hops.append(TempHop(table=temp, lines=lines))
        for upstream in edge.sources:
            evidence = Evidence(source=source_id, line=edge.line, operation=edge.load_mode, via_temp=hops or None)
            contribution.edges.append((upstream, edge.target, evidence))


def _add_queries(
    contribution: SourceContribution,
    extraction: ExtractionResult,
    source_id: str,
    fragments: List[SqlFragment],
) -> None:
    texts = {str(fragment.line_offset): fragment for fragment in fragments}
    first_in_fragment: Dict[str, int] = {}
    for fact in extraction.statement_facts:
        first_in_fragment.setdefault(_fragment_of(fact), fact.sequence)
    for fact in extraction.statement_facts:
        if fact.operation != "QUERY" or not fact.read_tables:
            continue
        node_id = f"{QUERY_NODE_PREFIX}{fact.sql_hash}"
        for table in fact.read_tables:
            contribution.edges.append((table, node_id, Evidence(source=source_id, line=fact.line)))
        fragment = texts.get(_fragment_of(fact))
        is_first = first_in_fragment[_fragment_of(fact)] == fact.sequence
        label = _label(fact, fragment, extraction.comments, is_first) if fragment is not None else None
        if label and node_id not in contribution.query_labels:
            contribution.query_labels[node_id] = label


def _label(fact: StatementFact, fragment: SqlFragment, comments: List[Comment], is_first: bool) -> Optional[str]:
    """The author's own description of a query: the comments that annotate its first line.

    A file header describes the first statement of its fragment; any other comment belongs to the
    query only when the code it annotates is the query's first line. Nothing is invented.
    """
    lines = fragment.text.split("\n")
    index = fact.line - fragment.line_offset - 1
    first_line = " ".join(lines[index].split()) if 0 <= index < len(lines) else ""
    start = fragment.line_offset + 1
    parts = []
    for comment in comments:
        # Python files hold several fragments; only this fragment's comments can describe the query.
        if comment.file != fragment.file or not start <= comment.line <= fact.line:
            continue
        if comment.kind == "header" and is_first:
            parts.append(comment.text)
        elif comment.kind == "note" and _annotates(comment, fact.line, first_line, lines, fragment.line_offset):
            parts.append(comment.text)
    label = " ".join(" ".join(parts).split())
    if not label:
        return None
    return label if len(label) <= _MAX_LABEL_LEN else label[: _MAX_LABEL_LEN - 3] + "..."


def _annotates(comment: Comment, line: int, first_line: str, lines: List[str], line_offset: int) -> bool:
    """A comment annotates the query when it trails its first line, or only blank lines and other
    comments separate it from that line; the annotated code must also read as that first line."""
    code = comment.code[:-3] if comment.code.endswith("...") else comment.code
    if not code or not first_line.startswith(code):
        return False
    between = lines[comment.line - line_offset : line - line_offset - 1]
    return all(not text.strip() or text.strip().startswith(_COMMENT_LEADS) for text in between)
