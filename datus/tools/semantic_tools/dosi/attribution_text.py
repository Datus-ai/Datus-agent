# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Dosi attribution results as the text a model reads.

``attribute_metrics`` and ``attribute_dimensions`` return a ``schema_version: 3``
result: a status, the issues that explain it, and one tree. A model needs the
tree's labels, the two values of each node and its effect on the top line's
change; the JSON around them (schema version, algorithm, execution counters,
bindings, each group's full key) costs tokens and says nothing it acts on. This
module writes the tree one node per line, indented by level:

    status complete
    issue warning <code>: <message> → <suggested action>
    note <a note of the tool's own>

    <metric> = <formula> | <baseline> → <current> | <effect> (<share>)
      <input> [input] | <baseline> → <current> | <effect> (<share>)
    <input> in all, <n> places | <effect> (<share>)

and the dimension tree as ``- <group> | <baseline> → <current> | <effect>
(<share>) | <n> below``. The tools' docstrings say how to read it; the two must
change together.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# A formula longer than this is cut: get_metric shows it in full.
EXPRESSION_MAX_CHARS = 80
# Significant digits of every value and effect.
SIGNIFICANT_DIGITS = 6

_TIME_SUFFIX = " 00:00:00"


def _fixed(value: float, digits: int) -> str:
    """``value`` with ``digits`` significant digits, never dropping an integer
    digit and never in exponent form; trailing zeros removed."""
    if value == 0:
        return "0"
    if not math.isfinite(value):
        return str(value)
    integer_digits = int(math.floor(math.log10(abs(value)))) + 1
    decimals = max(0, digits - integer_digits)
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def number(value: Optional[float]) -> str:
    """A value as the text shows it: six significant digits."""
    if value is None:
        return "null"
    return _fixed(float(value), SIGNIFICANT_DIGITS)


def signed(value: float, digits: int = SIGNIFICANT_DIGITS) -> str:
    """An effect or change, with its sign."""
    text = _fixed(float(value), digits)
    return text if text.startswith("-") or text == "0" else f"+{text}"


def percent(share: Optional[float]) -> str:
    """A share as a percentage with one decimal."""
    if share is None:
        return "n/a"
    return f"{share * 100:.1f}%"


def _one_line(text: Any) -> str:
    return " ".join(str(text).split())


def _clean(value: Any) -> str:
    """A coordinate value: a midnight timestamp is the day it starts."""
    text = str(value)
    return text[: -len(_TIME_SUFFIX)] if text.endswith(_TIME_SUFFIX) else text


def _values(node: Mapping[str, Any]) -> str:
    """``<baseline> → <current>``; a group without a value of its own (a rank
    among groups) says so, and a missing value says why."""
    if node.get("baseline_status") == "not_defined" and node.get("current_status") == "not_defined":
        return "no own value"

    def one(value: Any, status: Any) -> str:
        return number(value) if value is not None else str(status or "null")

    return (
        f"{one(node.get('baseline'), node.get('baseline_status'))} → "
        f"{one(node.get('current'), node.get('current_status'))}"
    )


def _uncertainty(effect: float, uncertainty: Any) -> str:
    """``±<standard error>`` when sampling leaves the shown digits uncertain."""
    if not isinstance(uncertainty, Mapping):
        return ""
    error = uncertainty.get("standard_error")
    if not isinstance(error, (int, float)) or error <= 0 or not math.isfinite(error):
        return ""
    # The last digit shown is 10^(magnitude - digits + 1); below that the
    # sampling error does not change what is written.
    if effect != 0:
        last_digit = 10 ** (math.floor(math.log10(abs(effect))) - SIGNIFICANT_DIGITS + 1)
        if error < last_digit:
            return ""
    # Two significant digits, integer digits included: 1234.5 is ±1200.
    return f" ±{_fixed(float(f'{error:.2g}'), SIGNIFICANT_DIGITS)}"


def _effect(node: Mapping[str, Any]) -> str:
    effect = node.get("effect")
    if effect is None:
        return "effect n/a"
    return f"{signed(effect)} ({percent(node.get('share'))}){_uncertainty(float(effect), node.get('uncertainty'))}"


Coordinates = Dict[str, Tuple[Any, Any]]


def _coordinates(raw: Any) -> Coordinates:
    """``{field: (baseline value, current value)}`` in the order the result
    gives them."""
    if not isinstance(raw, Mapping):
        return {}
    baseline = raw.get("baseline") if isinstance(raw.get("baseline"), Mapping) else {}
    current = raw.get("current") if isinstance(raw.get("current"), Mapping) else {}
    keys = list(dict.fromkeys([*baseline.keys(), *current.keys()]))
    return {key: (baseline.get(key), current.get(key)) for key in keys}


def _write_coordinates(coordinates: Coordinates) -> str:
    """``field=value`` when both periods read the same, else ``field=b / c``."""
    parts = []
    for key, (baseline, current) in coordinates.items():
        b = "-" if baseline is None else _clean(baseline)
        c = "-" if current is None else _clean(current)
        parts.append(f"{key}={b}" if b == c else f"{key}={b} / {c}")
    return ", ".join(parts)


def _header(result: Mapping[str, Any], notes: Iterable[str]) -> List[str]:
    status = str(result.get("status") or "unknown")
    accuracy = result.get("accuracy")
    lines = [f"status {status}" + (f", accuracy {accuracy}" if accuracy not in (None, "exact") else "")]
    for issue in result.get("issues") or []:
        if not isinstance(issue, Mapping):
            continue
        text = f"issue {issue.get('severity') or 'error'} {issue.get('code')}: {_one_line(issue.get('message') or '')}"
        if issue.get("suggested_action"):
            text += f" → {_one_line(issue['suggested_action'])}"
        lines.append(text)
    lines.extend(f"note {_one_line(note)}" for note in notes if note)
    return lines


def _expression(node: Mapping[str, Any]) -> str:
    text = _one_line(node.get("expression") or "")
    if len(text) > EXPRESSION_MAX_CHARS:
        text = text[: EXPRESSION_MAX_CHARS - 1] + "…"
    return text


# A label that is a name, not a formula.
_NAME = re.compile(r"[^\s()]+")


def _input_name(node: Mapping[str, Any]) -> str:
    """An input by its model metric: the one it can be queried as, else the
    metric it is written as (a member with a fill of its own reads the bare
    aggregate here), else the aggregate itself."""
    if node.get("metric"):
        return str(node["metric"])
    label = str(node.get("label") or "")
    if _NAME.fullmatch(label):
        return label
    return str(node["atom"])


def _metric_head(node: Mapping[str, Any], *, root: bool = False) -> str:
    """``<metric> = <formula>``, ``(<formula>)`` without a metric, ``<input>
    [input]`` for an aggregate read from data. The top line always names its
    metric and formula, even when it is itself one input."""
    if node.get("atom"):
        binding = node.get("binding") if isinstance(node.get("binding"), Mapping) else {}
        filters = [str(f) for f in binding.get("filters") or []]
        tag = "input" + (f", filter {' AND '.join(filters)}" if filters else "")
        if root and node.get("metric"):
            return f"{node['metric']} = {_expression(node)} [{tag}]"
        return f"{_input_name(node)} [{tag}]"
    if node.get("metric"):
        return f"{node['metric']} = {_expression(node)}"
    return f"({_expression(node)})"


def _group_label(group: Mapping[str, Any]) -> str:
    """A group's value. Dosi labels a NULL value ``(null)``, as it does the
    text ``'(null)'``; the key tells them apart, and a NULL is written ``NULL``
    so a follow-up filters it with ``IS NULL``."""
    label = str(group.get("label"))
    key = group.get("key")
    if label == "(null)" and isinstance(key, Mapping):
        values = list(key.values())
        if None in values and "(null)" not in values:
            return "NULL"
    return label


def render_attribute_metrics(result: Mapping[str, Any], notes: Sequence[str] = ()) -> str:
    """An ``attribute_metrics`` result as text: the formula tree, one node per line."""
    lines = _header(result, notes)
    tree = result.get("tree")
    if not isinstance(tree, Mapping):
        return "\n".join(lines)
    root = _coordinates(tree.get("coordinates"))
    lines.append("")
    lines.append(f"{_metric_head(tree, root=True)} | {_values(tree)} | {_effect(tree)}")
    if root:
        lines.append(f"row {_write_coordinates(root)}")

    def walk(node: Mapping[str, Any], depth: int) -> None:
        for child in node.get("children") or []:
            if not isinstance(child, Mapping):
                continue
            own = _coordinates(child.get("coordinates"))
            differs = {key: pair for key, pair in own.items() if root.get(key) != pair}
            at = f" @ {_write_coordinates(differs)}" if differs else ""
            lines.append(f"{'  ' * depth}{_metric_head(child)}{at} | {_values(child)} | {_effect(child)}")
            walk(child, depth + 1)

    walk(tree, 1)
    lines.extend(_input_totals(tree))
    return "\n".join(lines)


def _input_totals(tree: Mapping[str, Any]) -> List[str]:
    """One line per input read in more than one place: its whole effect is the
    sum of its lines, which the model is not to add up itself."""
    reads: Dict[str, List[Mapping[str, Any]]] = {}

    def collect(node: Mapping[str, Any]) -> None:
        for child in node.get("children") or []:
            if not isinstance(child, Mapping):
                continue
            if child.get("atom"):
                reads.setdefault(str(child["atom"]), []).append(child)
            collect(child)

    collect(tree)
    root_delta = tree.get("effect")
    out: List[str] = []
    for nodes in reads.values():
        effects = [node.get("effect") for node in nodes]
        if len(nodes) < 2 or any(not isinstance(e, (int, float)) for e in effects):
            continue
        total = float(sum(effects))
        share = total / root_delta if isinstance(root_delta, (int, float)) and root_delta else None
        out.append(f"{_input_name(nodes[0])} in all, {len(nodes)} places | {signed(total)} ({percent(share)})")
    return out


def render_attribute_dimensions(result: Mapping[str, Any], notes: Sequence[str] = ()) -> str:
    """An ``attribute_dimensions`` result as text: the metric, then its groups."""
    lines = _header(result, notes)
    tree = result.get("tree")
    if not isinstance(tree, Mapping):
        return "\n".join(lines)
    dimensions = [str(d) for d in result.get("dimensions") or []]
    delta = tree.get("delta")
    change = signed(delta) if delta is not None else "n/a"
    lines.append("")
    lines.append(f"{tree.get('label')} by {' > '.join(dimensions)} | {_values(tree)} | change {change}")
    root = _coordinates(tree.get("coordinates"))
    if root:
        lines.append(f"row {_write_coordinates(root)}")

    def level(node: Mapping[str, Any], depth: int) -> None:
        children = [c for c in node.get("children") or [] if isinstance(c, Mapping)]
        indent = "  " * depth
        for child in children:
            flag = " ~" if child.get("rank_unstable") else ""
            tail = ""
            if child.get("groups_total") is not None:
                listed = any(isinstance(c, Mapping) for c in child.get("children") or [])
                tail = f" | {child['groups_total']} below" + ("" if listed else ", none listed")
            lines.append(f"{indent}- {_group_label(child)}{flag} | {_values(child)} | {_effect(child)}{tail}")
            level(child, depth + 1)
        others = node.get("others")
        if children and isinstance(others, Mapping) and others.get("group_count"):
            count = int(others["group_count"])
            lines.append(
                f"{indent}- {count} other{'s' if count != 1 else ''} | "
                f"{signed(others.get('effect') or 0)} ({percent(others.get('share'))})"
            )

    level(tree, 0)
    return "\n".join(lines)


def render_attribution(result: Mapping[str, Any], notes: Sequence[str] = ()) -> str:
    """Either tool's result as text; a dimension result names its dimensions."""
    if "dimensions" in result:
        return render_attribute_dimensions(result, notes)
    return render_attribute_metrics(result, notes)


# ── One-line summary (CLI and UI) ───────────────────────────────────────

_STATUS = re.compile(r"^status (\w+)")
_ISSUE = re.compile(r"^issue (\w+) ([\w.-]+):")
_EFFECT = re.compile(r"([+-]?[\d.]+) \((-?[\d.]+)%\)")
_CHANGE = re.compile(r"\| change ([+-]?[\d.]+)\s*$")


def _effect_of(line: str) -> Optional[Tuple[float, float]]:
    """``(effect, share%)`` from the last field of a node line."""
    last = line.rsplit(" | ", 1)[-1]
    match = _EFFECT.search(last)
    if not match:
        return None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None


def summarize_attribution_text(text: Any) -> str:
    """``complete · Δ +1.08 · store_sales_amount +0.355 (32.9%)``: the status,
    the top line's change and the part with the largest effect — the input
    with the largest total over its lines, or the largest first-level group.
    A result without effects is its status and first error code."""
    if not isinstance(text, str):
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    status_match = _STATUS.match(lines[0])
    if not status_match:
        return ""
    status = status_match.group(1)
    body_start = next((i for i, line in enumerate(lines) if not line.strip()), len(lines))
    head, body = lines[:body_start], [line for line in lines[body_start + 1 :] if line.strip()]
    if status not in ("complete", "partial") or not body:
        codes = [m.group(2) for m in (_ISSUE.match(line) for line in head) if m and m.group(1) == "error"]
        return " · ".join([status, *codes[:1]])

    root = body[0]
    parts = [status]
    change = _CHANGE.search(root)
    if change:
        parts.append(f"Δ {change.group(1)}")
    else:
        root_effect = _effect_of(root)
        if root_effect:
            parts.append(f"Δ {signed(root_effect[0])}")

    best: Optional[Tuple[str, float, float]] = None
    if change:  # dimension tree: the first-level groups
        for line in body[1:]:
            if not line.startswith("- ") or line.count(" | ") < 2:
                continue  # deeper levels, and `others` lines
            parsed = _effect_of(_drop_tail(line))
            if parsed and (best is None or abs(parsed[0]) > abs(best[1])):
                best = (line[2:].split(" | ", 1)[0], parsed[0], parsed[1])
    else:  # metric tree: inputs, summed over their lines
        totals: Dict[str, List[float]] = {}
        for line in body[1:]:
            if " [input" not in line:
                continue
            parsed = _effect_of(line)
            if parsed is None:
                continue
            name = line.strip().split(" [input", 1)[0]
            total = totals.setdefault(name, [0.0, 0.0])
            total[0] += parsed[0]
            total[1] += parsed[1]
        for name, (effect, share) in totals.items():
            if best is None or abs(effect) > abs(best[1]):
                best = (name, effect, share)
    if best is not None:
        parts.append(f"{best[0]} {signed(best[1], 3)} ({best[2]:.1f}%)")
    return " · ".join(parts)


def _drop_tail(line: str) -> str:
    """A group line without its ``| <n> below`` field."""
    fields = line.split(" | ")
    if fields and re.match(r"^\d+ below", fields[-1]):
        fields = fields[:-1]
    return " | ".join(fields)
