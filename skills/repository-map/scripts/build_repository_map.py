#!/usr/bin/env python3
"""Build a compact layered repository map for SQL projects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sqlglot
from sqlglot import expressions

LAYER_NAMES = {"staging", "intermediate", "marts"}


def infer_layer(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) >= 2 and rel.parts[-2] in LAYER_NAMES:
        return rel.parts[-2]
    return "other"


def infer_output_table(path: Path, root: Path) -> str:
    layer = infer_layer(path, root)
    if layer != "other":
        return f"{layer}.{path.stem}"
    return path.stem


def extract_upstream_tables(sql: str) -> list[str]:
    parsed = sqlglot.parse_one(sql, error_level=sqlglot.ErrorLevel.IGNORE)
    if parsed is None:
        return []
    cte_names = {cte.alias_or_name for cte in parsed.find_all(expressions.CTE) if cte.alias_or_name}
    refs: set[str] = set()
    for table in parsed.find_all(expressions.Table):
        if table.name in cte_names:
            continue
        if table.db:
            refs.add(f"{table.db}.{table.name}")
        else:
            refs.add(table.name)
    return sorted(refs)


def summarize(layer: str, upstreams: list[str]) -> str:
    if not upstreams:
        return (
            f"{layer} model with no detected upstreams" if layer != "other" else "sql model with no detected upstreams"
        )
    head = upstreams[:3]
    suffix = f" +{len(upstreams) - 3} more" if len(upstreams) > 3 else ""
    joined = ", ".join(head)
    if layer == "staging":
        return f"staging transform from {joined}{suffix}"
    if layer == "intermediate":
        return f"intermediate model from {joined}{suffix}"
    if layer == "marts":
        return f"mart model from {joined}{suffix}"
    return f"sql model from {joined}{suffix}"


def build_map(root: Path) -> dict:
    grouped: dict[str, list[dict]] = {"staging": [], "intermediate": [], "marts": [], "other": []}
    for path in sorted(root.rglob("*.sql")):
        layer = infer_layer(path, root)
        sql = path.read_text()
        upstreams = extract_upstream_tables(sql)
        grouped[layer].append(
            {
                "path": str(path.relative_to(root)),
                "output_table": infer_output_table(path, root),
                "upstream_tables": upstreams,
                "summary": summarize(layer, upstreams),
            }
        )
    return {
        "root": str(root),
        "layer_counts": {layer: len(entries) for layer, entries in grouped.items()},
        "layers": grouped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="SQL repository root")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    print(json.dumps(build_map(root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
