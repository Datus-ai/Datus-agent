# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Static validation of the card primitives inside a ``render/`` tree.

Both visual-artifact kinds mount the same two runtime components from
``@datus/web-artifact``:

* ``<ChartCard>`` — chrome + chart-actions menu around a query-backed
  visualization.
* ``<BlockHandle>`` — a zero-chrome anchor around the blocks that aren't
  charts (KPI tiles, insight callouts, the filter strip).

Both carry an id that the viewer sends back to the agent when a user
points at that block — for the creator ("change *this* tile") while the
artifact is being authored, and for the reader ("why is *this* number
low") once it is published. Those two flows share one id namespace and
break the same way when it drifts: a malformed id produces an envelope
the host drops, and a duplicated id makes two different blocks pin the
identical chip, so the second one silently de-dupes away and the button
looks broken.

That failure is invisible until runtime, which is why it's checked here
instead. The checks started out dashboard-only; this module exists so
reports get the identical treatment rather than a second, drifting copy
— the only per-kind differences are where a query lives (a saved
template vs a saved result) and how to say so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set

from datus.schemas.gen_visual_report_models import extract_query_slug

# ``<ChartCard ... >`` opening tag, including the self-closing form
# ``<ChartCard ... />``. Group 1 captures the prop block. The attribute
# block is matched as a sequence of "atom" tokens (plain non-special
# chars, quoted strings, balanced brace expressions) so attributes whose
# values contain ``>`` — ``title={<Icon />}`` or
# ``titleRight={<span>a > b</span>}`` — are not truncated at the first
# stray angle bracket. Brace expressions are recognised up to depth 3,
# enough for ``style={{ color: '#fff' }}`` / nested JSX expressions.
_JSX_ATTR_BLOCK = r"""
    (
      (?:
        [^'"{}<>]                                          # plain char
        | '[^'\\]*(?:\\.[^'\\]*)*'                         # 'single-quoted'
        | "[^"\\]*(?:\\.[^"\\]*)*"                         # "double-quoted"
        | \{ (?: [^{}] | \{ (?: [^{}] | \{[^{}]*\} )* \} )* \}   # {balanced braces, depth ≤ 3}
      )*
    )
    /?\s*>
"""

CHART_CARD_OPEN_RE = re.compile(r"<ChartCard\b" + _JSX_ATTR_BLOCK, re.VERBOSE | re.DOTALL)


# ``<BlockHandle ... >`` opening tag — the entry point authors wrap around
# blocks that aren't ChartCards. Same atom-based attribute matching so
# ``name={<span>a > b</span>}`` doesn't truncate the prop block.
BLOCK_HANDLE_OPEN_RE = re.compile(r"<BlockHandle\b" + _JSX_ATTR_BLOCK, re.VERBOSE | re.DOTALL)


# Top-level JSX spread attribute: ``{...rest}`` sitting between other
# attributes. The match is anchored on a whitespace boundary so nested
# spreads inside JSX expressions (e.g. ``style={{...defaults}}``) — where
# the inner ``{`` is preceded by another ``{``, not whitespace — do not
# false-positive into "this card uses spread props".
SPREAD_ATTR_RE = re.compile(r"(?<=\s)\{\s*\.{3}")


# Presence of a prop regardless of whether its value is a literal. Used to
# tell "the author forgot ``handleId``" (an issue) apart from "``handleId``
# is forwarded from a wrapper component" (legitimate — see below).
BLOCK_HANDLE_ANY_ATTR_RE = re.compile(r"\b(handleId|name)\s*=")


# ``chartId`` shape — same slug grammar used elsewhere in the artifact path
# (artifact slug, query slug). Caps at 64 to keep the validate_render
# cards-registry payload compact.
CHART_ID_RE = re.compile(r"^[a-z0-9_]{1,64}$")


# ChartCard's chartType enum.
VALID_CHART_TYPES: Set[str] = {
    # recharts native chart components
    "bar",
    "line",
    "area",
    "pie",
    "scatter",
    "radar",
    "composed",
    "radial-bar",
    "treemap",
    "funnel",
    # Not recharts-native but common in BI dashboards; rendered via
    # custom SVG / RadialBarChart subsets. Declaring the type lets the
    # edit-time LLM see the intent without forcing every BI chart into
    # the catch-all ``custom`` bucket.
    "gauge",
    "heatmap",
    "waterfall",
    # Tabular + single-value cards.
    "table",
    "kpi",
    # Escape hatch for hand-rolled visuals that don't match any of the above.
    "custom",
}


# ``BlockHandle``'s kind enum. ``'chart'`` is deliberately absent: that kind
# belongs to ChartCard, and a chart wrapped in a bare BlockHandle would lose
# the chart-actions menu.
VALID_BLOCK_HANDLE_KINDS: Set[str] = {"kpi", "note", "filter"}

JS_NON_CODE_RE = re.compile(r"""//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`""")
_ATTR_NAME_RE = re.compile(r"[\w:$-]+")
_ARRAY_STRING = r"""(?:'[^'\\\n]*'|"[^"\\\n]*")"""
_QUERY_ARRAY_RE = re.compile(rf"\[\s*(?:{_ARRAY_STRING}(?:\s*,\s*{_ARRAY_STRING})*\s*,?)?\s*\]")


@dataclass
class CardAttributes:
    literals: dict[str, str]
    query_ids: list[str] | None = None
    dynamic_sources: bool = False
    source_error: str | None = None


def parse_card_attributes(attrs: str) -> CardAttributes:
    """Read top-level literal props and queryIds arrays, never executing JSX."""
    values: dict[str, str] = {}
    duplicates: set[str] = set()
    i = 0
    while i < len(attrs):
        name_match = _ATTR_NAME_RE.match(attrs, i)
        if not name_match:
            token = JS_NON_CODE_RE.match(attrs, i)
            i = token.end() if token else i + 1
            continue
        name = name_match.group()
        i = name_match.end()
        while i < len(attrs) and attrs[i].isspace():
            i += 1
        if i >= len(attrs) or attrs[i] != "=":
            continue
        i += 1
        while i < len(attrs) and attrs[i].isspace():
            i += 1
        start = i
        token = JS_NON_CODE_RE.match(attrs, i)
        if token:
            i = token.end()
        elif i < len(attrs) and attrs[i] == "{":
            depth = 1
            i += 1
            while i < len(attrs) and depth:
                token = JS_NON_CODE_RE.match(attrs, i)
                if token:
                    i = token.end()
                    continue
                if attrs[i] == "{":
                    depth += 1
                elif attrs[i] == "}":
                    depth -= 1
                i += 1
        else:
            while i < len(attrs) and not attrs[i].isspace():
                i += 1
        if name in values:
            duplicates.add(name)
        values[name] = attrs[start:i]

    result = CardAttributes(
        literals={name: value[1:-1] for name, value in values.items() if value.startswith(("'", '"'))}
    )
    raw = values.get("queryIds")
    if raw is None:
        return result
    if "queryIds" in duplicates:
        result.source_error = "queryIds must be declared once"
        return result
    if not (raw.startswith("{") and raw.endswith("}")):
        result.source_error = "queryIds must be a string array expression"
        return result
    expression = raw[1:-1].strip()
    # Comments are whitespace; retain strings verbatim, including comment-like text.
    expression = JS_NON_CODE_RE.sub(
        lambda match: " " if match.group().startswith(("//", "/*")) else match.group(), expression
    )
    if _QUERY_ARRAY_RE.fullmatch(expression):
        result.query_ids = [match.group()[1:-1] for match in re.finditer(_ARRAY_STRING, expression)]
    else:
        # Forwarding and computed arrays remain legal but statically partial.
        result.dynamic_sources = True
    return result


def _collect_query_ids(
    result: CardScanResult,
    props: CardAttributes,
    rel: str,
    query_exists: Callable[[str], bool],
    missing_query_hint: str,
) -> None:
    if props.source_error:
        result.issues.append(f"render/{rel}: {props.source_error}.")
    if props.dynamic_sources:
        result.warnings.append(f"render/{rel}: queryIds is dynamic — source validation is deferred to runtime.")
    for ref in props.query_ids or []:
        slug = extract_query_slug(ref)
        if slug is None:
            result.issues.append(f"render/{rel}: queryIds contains an invalid query reference: {ref!r}.")
            continue
        result.query_refs.add(f"queries/{slug}")
        if not query_exists(slug):
            result.issues.append(f"render/{rel}: queryIds reference 'queries/{slug}' points to {missing_query_hint}.")


@dataclass
class CardScanResult:
    """Everything a ``validate_render`` needs from the card scan."""

    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # ``queries/<slug>`` references discovered on card props. Merged into
    # the caller's set alongside the ``useQuerySql`` literals it collects
    # itself — a card can name a query no hook in that file reads.
    query_refs: Set[str] = field(default_factory=set)
    # chartId / handleId → render-file rel-path of its first declaration.
    # One namespace for both because they land in the same field of the
    # card-reference payload.
    ids_seen: Dict[str, str] = field(default_factory=dict)
    # Same keys → what the block is ("chart" for ChartCard, the BlockHandle
    # kind otherwise), so the registry can tell a KPI tile from a chart.
    kinds: Dict[str, str] = field(default_factory=dict)

    def registry(self) -> List[Dict[str, str]]:
        """Cards registry for the ``validate_render`` wire result.

        Sorted by id for stable output. BlockHandles whose ``handleId`` is
        forwarded from a wrapper component's props are absent — their ids
        only exist at runtime.
        """
        return [
            {"chart_id": cid, "jsx_path": f"render/{rel}", "kind": self.kinds.get(cid, "chart")}
            for cid, rel in sorted(self.ids_seen.items())
        ]


def scan_render_cards(
    modules: Dict[str, Dict[str, Any]],
    *,
    query_exists: Callable[[str], bool],
    missing_query_hint: str,
) -> CardScanResult:
    """Validate every ``<ChartCard>`` / ``<BlockHandle>`` in a render tree.

    Args:
        modules: ``{module_key: {"rel": str, "source": str}}`` — the same
            mapping both ``validate_render`` implementations already build
            for their import-graph walk.
        query_exists: Predicate answering "is ``queries/<slug>`` a real,
            saved query for this artifact kind?". Dashboards look it up in
            the saved-template map; reports check for the result JSON on
            disk.
        missing_query_hint: Tail of the message shown when ``query_exists``
            says no, e.g. ``"a template not produced via save_query_template"``.
            Kind-specific because the fix differs.

    Returns:
        A :class:`CardScanResult`. Issues are hard failures; warnings are
        advisory (spread props defeat static inspection but are legal).
    """
    result = CardScanResult()

    for mod in modules.values():
        rel = mod["rel"]
        source = mod["source"]
        non_code = [(m.start(), m.end()) for m in JS_NON_CODE_RE.finditer(source)]

        # ---- <ChartCard ... > opening tags — required props + enums.
        for cc_match in CHART_CARD_OPEN_RE.finditer(source):
            if any(start <= cc_match.start() < end for start, end in non_code):
                continue
            attrs = cc_match.group(1) or ""
            props = parse_card_attributes(attrs)
            attr_values: Dict[str, str] = props.literals

            # Spread props (``<ChartCard {...rest}>``) hide attributes from
            # static inspection. Surface as a warning, then bail on the
            # rest of the checks for this match to avoid false positives.
            if SPREAD_ATTR_RE.search(attrs):
                result.warnings.append(
                    f"render/{rel}: <ChartCard> uses spread props — static "
                    "validation of chartId / sqlId / chartType is deferred to runtime."
                )
                continue

            _collect_query_ids(result, props, rel, query_exists, missing_query_hint)
            missing = [k for k in ("chartId", "sqlId", "chartType") if k not in attr_values]
            if missing:
                result.issues.append(
                    f"render/{rel}: <ChartCard> is missing required string-literal "
                    f"props: {missing}. Each ChartCard must declare chartId, sqlId, and "
                    "chartType up front."
                )
                continue

            chart_id = attr_values["chartId"]
            sql_id = attr_values["sqlId"]
            chart_type = attr_values["chartType"]

            if not CHART_ID_RE.fullmatch(chart_id):
                result.issues.append(
                    f"render/{rel}: <ChartCard chartId={chart_id!r}> must match {CHART_ID_RE.pattern}."
                )
            elif chart_id in result.ids_seen:
                result.issues.append(
                    f"render/{rel}: <ChartCard chartId={chart_id!r}> duplicates the "
                    f"chartId already declared in render/{result.ids_seen[chart_id]}. chartId "
                    "must be globally unique across the artifact."
                )
            else:
                result.ids_seen[chart_id] = rel
                result.kinds[chart_id] = "chart"

            slug = extract_query_slug(sql_id)
            if slug is None:
                result.issues.append(
                    f"render/{rel}: <ChartCard sqlId={sql_id!r}> is not a valid queries/<slug> reference."
                )
            else:
                result.query_refs.add(f"queries/{slug}")
                if not query_exists(slug):
                    result.issues.append(
                        f"render/{rel}: <ChartCard sqlId='queries/{slug}'> points to {missing_query_hint}."
                    )

            if chart_type not in VALID_CHART_TYPES:
                result.issues.append(
                    f"render/{rel}: <ChartCard chartType={chart_type!r}> is not one of {sorted(VALID_CHART_TYPES)}."
                )

        # ---- <BlockHandle ... > opening tags. Checks are deliberately
        # looser than ChartCard's: a KPI tile is normally rendered through
        # a shared wrapper (``shared/kpi-card.jsx``) that forwards
        # ``handleId`` / ``name`` from its own props, so those values are
        # expressions rather than string literals at this call site.
        # Literals get the full shape + uniqueness treatment; forwarded
        # props are deferred to runtime.
        for eh_match in BLOCK_HANDLE_OPEN_RE.finditer(source):
            if any(start <= eh_match.start() < end for start, end in non_code):
                continue
            attrs = eh_match.group(1) or ""
            props = parse_card_attributes(attrs)
            attr_values = props.literals
            present = set(BLOCK_HANDLE_ANY_ATTR_RE.findall(attrs))

            if SPREAD_ATTR_RE.search(attrs):
                result.warnings.append(
                    f"render/{rel}: <BlockHandle> uses spread props — static "
                    "validation of handleId / name / kind is deferred to runtime."
                )
                continue

            _collect_query_ids(result, props, rel, query_exists, missing_query_hint)
            missing = [k for k in ("handleId", "name") if k not in present]
            if missing:
                result.issues.append(
                    f"render/{rel}: <BlockHandle> is missing required props: {missing}. "
                    "Every BlockHandle needs a globally-unique handleId and a human-readable "
                    "name (the label the user sees on the chat chip)."
                )
                continue

            kind = attr_values.get("kind")
            if kind is not None and kind not in VALID_BLOCK_HANDLE_KINDS:
                result.issues.append(
                    f"render/{rel}: <BlockHandle kind={kind!r}> is not one of "
                    f"{sorted(VALID_BLOCK_HANDLE_KINDS)}. Charts belong in <ChartCard>, "
                    "which carries its own edit entry point."
                )

            handle_sql_id = attr_values.get("sqlId")
            if handle_sql_id is not None:
                slug = extract_query_slug(handle_sql_id)
                if slug is None:
                    result.issues.append(
                        f"render/{rel}: <BlockHandle sqlId={handle_sql_id!r}> is not a valid "
                        "queries/<slug> reference. Omit sqlId entirely for a block with no query "
                        "behind it."
                    )
                else:
                    result.query_refs.add(f"queries/{slug}")
                    if not query_exists(slug):
                        result.issues.append(
                            f"render/{rel}: <BlockHandle sqlId='queries/{slug}'> points to {missing_query_hint}."
                        )

            handle_id = attr_values.get("handleId")
            if handle_id is None:
                # Forwarded from a wrapper component's props — the id only
                # exists at runtime, so shape and uniqueness can't be
                # checked here.
                continue

            if not CHART_ID_RE.fullmatch(handle_id):
                result.issues.append(
                    f"render/{rel}: <BlockHandle handleId={handle_id!r}> must match {CHART_ID_RE.pattern}."
                )
            elif handle_id in result.ids_seen:
                result.issues.append(
                    f"render/{rel}: <BlockHandle handleId={handle_id!r}> duplicates the id "
                    f"already declared in render/{result.ids_seen[handle_id]}. handleId shares one "
                    "namespace with ChartCard's chartId and must be globally unique across the "
                    "artifact."
                )
            else:
                result.ids_seen[handle_id] = rel
                result.kinds[handle_id] = kind or "note"

    return result
