# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Pure candidate preparation. Never writes files, synchronizes KB, or executes SQL.

The arithmetic parser only expands metric references. Native Dosi remains the
authority on both the generated fallback and the complete model semantics.
Return field patches so callers can retain YAML comments and unknown metadata.
"""

from __future__ import annotations

import ast
import copy
import json
from typing import Any

import sqlglot
import yaml
from sqlglot import exp

from datus.tools.semantic_tools.dosi.authoring import dosi_validation_text_payload

_OPERATORS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%"}
_MAX_EXPANDED_SQL = 1_000_000


def _bounded_sql(text: str) -> str:
    if len(text) > _MAX_EXPANDED_SQL:
        raise ValueError("Expanded metric SQL exceeds the automatic editing limit")
    return text


def _formula(text: str) -> ast.expr:
    if not text or len(text) > 4096:
        raise ValueError("Compose formula must contain 1–4096 characters")
    tree = ast.parse(text.strip(), mode="eval").body
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError("Compose formula is too complex")

    def check(node: ast.expr, depth: int = 0) -> None:
        if depth > 32:
            raise ValueError("Compose formula nesting is too deep")
        if isinstance(node, ast.Name):
            return
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            check(node.left, depth + 1)
            check(node.right, depth + 1)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            check(node.operand, depth + 1)
            return
        raise ValueError("Automatic compose editing supports metric names, numbers, +, -, *, /, %, and parentheses")

    check(tree)
    return tree


def _ratio_sql(sql: str, dialect: str) -> str:
    """Mirror Dosi's bare-aggregate ratio shape when inlining a member.

    A ratio is CAST(numerator AS DOUBLE)/denominator. An arithmetic numerator
    is an expression, not a Ratio, and must not receive this cast. Native Dosi
    validates the result; no SQL query or compiled SELECT is used as source.
    """
    aliases = {
        "ansi_sql": None,
        "postgresql": "postgres",
        "greenplum": "postgres",
        "hologres": "postgres",
        "gaussdb": "postgres",
        "dws": "postgres",
        "doris": "starrocks",
        "tidb": "mysql",
    }
    language = aliases.get(dialect.lower(), dialect.lower())
    node = sqlglot.parse_one(sql, read=language)

    def unparen(value: exp.Expression) -> exp.Expression:
        while isinstance(value, exp.Paren):
            value = value.this
        return value

    node = unparen(node)
    if isinstance(node, exp.Div):
        left, right = unparen(node.left), unparen(node.right)
        if isinstance(left, exp.AggFunc) and isinstance(right, exp.AggFunc):
            return f"(CAST({left.sql(dialect=language)} AS DOUBLE) / {right.sql(dialect=language)})"
    return sql


def _extension(metric: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    entries = [entry for entry in metric.get("custom_extensions", []) if entry.get("vendor_name") == "DATUS"]
    if len(entries) > 1:
        raise ValueError(f"{metric['name']}: multiple DATUS extensions cannot be edited automatically")
    if not entries:
        return None, {}
    data = json.loads(entries[0]["data"])
    if not isinstance(data, dict):
        raise ValueError(f"{metric['name']}: DATUS data must encode an object")
    return entries[0], data


def prepare_metric_definitions(
    model_text: str,
    changed_metrics: list[str],
    *,
    rename_from: str | None = None,
    rename_to: str | None = None,
) -> dict[str, Any]:
    """Expand affected compose definitions atomically and validate the candidate."""
    try:
        document = yaml.safe_load(model_text)
        models = document.get("semantic_model") if isinstance(document, dict) else None
        if not isinstance(models, list) or len(models) != 1:
            raise ValueError("The editor requires exactly one semantic_model")
        metrics = models[0].get("metrics", [])
        by_name = {metric["name"]: metric for metric in metrics}
        if len(by_name) != len(metrics):
            raise ValueError("Metric names must be unique")
        if len(metrics) > 1000:
            raise ValueError("Automatic preparation supports at most 1000 metrics per model")
        if set(changed_metrics) - by_name.keys():
            raise ValueError("Changed metric is missing from the candidate")
        original = copy.deepcopy(by_name)
        if bool(rename_from) != bool(rename_to):
            raise ValueError("Rename requires both old and new names")
        if rename_from and (rename_from in by_name or rename_to not in by_name):
            raise ValueError("Rename candidate must contain the new name and remove the old name")

        trees: dict[str, ast.expr] = {}
        dependencies: dict[str, set[str]] = {}
        unsupported: set[str] = set()
        for name, metric in by_name.items():
            entry, data = _extension(metric)
            derive = data.get("derive", {})
            window = data.get("window", {})
            refs = set()
            if derive.get("type") == "compose":
                if "fill" in derive:
                    unsupported.add(name)
                tree = _formula(derive.get("expr", ""))
                if rename_from:
                    renamed = False
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name) and node.id == rename_from:
                            node.id = rename_to
                            renamed = True
                    if renamed:
                        derive["expr"] = ast.unparse(tree)
                        entry["data"] = json.dumps(data, ensure_ascii=False)
                trees[name] = tree
                refs = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            elif derive:
                unsupported.add(name)
                if derive.get("base"):
                    refs.add(derive["base"])
            if window:
                unsupported.add(name)
                if window.get("base"):
                    refs.add(window["base"])
                if window.get("frame", {}).get("second"):
                    refs.add(window["frame"]["second"])
            if rename_from in refs:
                raise ValueError(f"{name}: rename affects a filter/window; update its advanced definition first")
            dependencies[name] = refs

        affected = set(changed_metrics)
        if rename_from:
            affected.add(rename_to)
        while True:
            expanded = affected | {name for name, refs in dependencies.items() if refs & affected}
            if expanded == affected:
                break
            affected = expanded
        blocked = affected & unsupported
        if blocked:
            raise ValueError(
                f"Automatic preparation cannot rewrite filter/window metrics: {', '.join(sorted(blocked))}"
            )

        resolved: dict[tuple[str, str], str] = {}
        visiting: set[str] = set()
        done: set[str] = set()

        def expand(name: str) -> None:
            if name not in by_name:
                raise ValueError(f"Unknown metric reference: {name}")
            if name in visiting:
                raise ValueError(f"Metric reference cycle at {name}")
            if name in done or name not in trees:
                return
            if len(visiting) >= 64:
                raise ValueError("Metric dependency nesting is too deep")
            if name in unsupported:
                raise ValueError(f"{name}: automatic expansion of fill/window/filter definitions is not supported")
            visiting.add(name)
            for ref in dependencies[name]:
                if ref not in by_name:
                    raise ValueError(f"{name}: unknown metric reference {ref}")
                expand(ref)

            metric = by_name[name]
            dialects = metric.get("expression", {}).get("dialects", [])
            if not dialects:
                raise ValueError(f"{name}: choose at least one SQL dialect")
            for dialect in dialects:
                language = dialect["dialect"]

                def render(node: ast.expr, language: str = language) -> str:
                    if isinstance(node, ast.Name):
                        variants = by_name[node.id].get("expression", {}).get("dialects", [])
                        matches = [item["expression"] for item in variants if item["dialect"] == language]
                        if len(matches) != 1:
                            raise ValueError(f"{name}: {node.id} must provide exactly one {language} expression")
                        value = resolved.get((node.id, language))
                        if value is None:
                            value = _ratio_sql(matches[0], language)
                        return _bounded_sql(f"({value})")
                    if isinstance(node, ast.Constant):
                        return str(node.value)
                    if isinstance(node, ast.UnaryOp):
                        sign = "-" if isinstance(node.op, ast.USub) else "+"
                        return f"({sign}{render(node.operand)})"
                    return _bounded_sql(f"({render(node.left)} {_OPERATORS[type(node.op)]} {render(node.right)})")

                expanded = _ratio_sql(render(trees[name]), language)
                resolved[(name, language)] = expanded
                if name in affected:
                    dialect["expression"] = expanded
            visiting.remove(name)
            done.add(name)

        for name in sorted(affected):
            expand(name)

        candidate = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
        validation = dosi_validation_text_payload(candidate)
        patches = []
        if validation.get("valid"):
            for name, metric in by_name.items():
                fields = {
                    key: metric[key]
                    for key in ("expression", "custom_extensions")
                    if metric.get(key) != original[name].get(key)
                }
                if fields:
                    patches.append({"name": name, "fields": fields})
        return {"valid": bool(validation.get("valid")), "patches": patches, "validation": validation}
    except (
        ValueError,
        SyntaxError,
        KeyError,
        TypeError,
        AttributeError,
        RecursionError,
        yaml.YAMLError,
        sqlglot.errors.SqlglotError,
    ) as exc:
        return {
            "valid": False,
            "patches": [],
            "validation": {
                "valid": False,
                "issues": [{"code": "metric_edit_rejected", "severity": "error", "message": str(exc)}],
            },
        }
