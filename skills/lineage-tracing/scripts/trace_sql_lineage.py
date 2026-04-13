#!/usr/bin/env python3
"""Trace SQL lineage in layered repositories using sqlglot AST parsing."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import sqlglot
from sqlglot import expressions

LAYER_NAMES = {"staging", "intermediate", "marts"}


def infer_output_table(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 2 and parts[-2] in LAYER_NAMES:
        return f"{parts[-2]}.{path.stem}"
    return path.stem


def normalize_table_reference(table: expressions.Table) -> str:
    db = table.db
    name = table.name
    if db:
        return f"{db}.{name}"
    return name


def extract_upstream_tables(sql: str, dialect: str | None = None) -> set[str]:
    parsed = sqlglot.parse_one(sql, read=dialect or None, error_level=sqlglot.ErrorLevel.IGNORE)
    if parsed is None:
        return set()

    cte_names = {cte.alias_or_name for cte in parsed.find_all(expressions.CTE) if cte.alias_or_name}
    refs: set[str] = set()
    for table in parsed.find_all(expressions.Table):
        ref = normalize_table_reference(table)
        if ref in cte_names or table.name in cte_names:
            continue
        refs.add(ref)
    return refs


def build_lineage(root: Path, dialect: str | None = None) -> dict:
    sql_files = sorted(root.rglob("*.sql"))
    file_entries: list[dict] = []
    table_edges: list[dict] = []
    reverse_index: dict[str, set[str]] = defaultdict(set)
    forward_index: dict[str, set[str]] = defaultdict(set)
    internal_outputs: set[str] = set()

    for path in sql_files:
        output_table = infer_output_table(path, root)
        internal_outputs.add(output_table)

    unresolved_inputs: set[str] = set()

    for path in sql_files:
        sql = path.read_text()
        output_table = infer_output_table(path, root)
        upstreams = sorted(extract_upstream_tables(sql, dialect=dialect))
        file_entries.append(
            {
                "path": str(path.relative_to(root)),
                "output_table": output_table,
                "upstream_tables": upstreams,
            }
        )
        for upstream in upstreams:
            table_edges.append({"from": upstream, "to": output_table})
            reverse_index[output_table].add(upstream)
            forward_index[upstream].add(output_table)
            if upstream not in internal_outputs:
                unresolved_inputs.add(upstream)

    return {
        "root": str(root),
        "files": file_entries,
        "table_edges": sorted(table_edges, key=lambda edge: (edge["to"], edge["from"])),
        "forward_index": {k: sorted(v) for k, v in sorted(forward_index.items())},
        "reverse_index": {k: sorted(v) for k, v in sorted(reverse_index.items())},
        "unresolved_inputs": sorted(unresolved_inputs),
    }


def walk_graph(index: dict[str, list[str]], start: str) -> list[str]:
    seen: set[str] = set()
    queue: deque[str] = deque(index.get(start, []))
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        queue.extend(index.get(node, []))
    return sorted(seen)


def build_target_summary(lineage: dict, target: str) -> dict:
    reverse_index = lineage["reverse_index"]
    forward_index = lineage["forward_index"]
    return {
        "name": target,
        "direct_upstreams": reverse_index.get(target, []),
        "all_upstreams": walk_graph(reverse_index, target),
        "direct_downstreams": forward_index.get(target, []),
        "all_downstreams": walk_graph(forward_index, target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="SQL repository root")
    parser.add_argument("--target", help="Optional target table for upstream/downstream tracing")
    parser.add_argument("--dialect", help="Optional sqlglot read dialect")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    lineage = build_lineage(root, dialect=args.dialect)
    if args.target:
        lineage["target"] = build_target_summary(lineage, args.target)

    print(json.dumps(lineage, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
