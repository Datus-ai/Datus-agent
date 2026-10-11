# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Validate declared query sources without evaluating generated JSX."""

import pytest

from datus.tools.func_tool._visual_artifact_cards import scan_render_cards


def scan(source):
    return scan_render_cards(
        {"app": {"rel": "app.jsx", "source": source}},
        query_exists=lambda name: name in {"sales", "cost"},
        missing_query_hint="a missing saved query",
    )


@pytest.mark.parametrize(
    "tag,props",
    [
        ("ChartCard", 'chartId="combined" chartType="line" sqlId="queries/sales"'),
        ("BlockHandle", 'handleId="combined" name="Combined"'),
    ],
)
def test_multi_source_cards_validate_every_declared_query(tag, props):
    result = scan(f"<{tag} {props} queryIds={{['queries/sales', 'cost.json', 'cost.json',]}} />")
    assert result.issues == []
    assert result.query_refs == {"queries/sales", "queries/cost"}


@pytest.mark.parametrize("value", ["queries/missing", "queries/../../secret", "invalid-name"])
def test_missing_and_invalid_array_references_fail_validation(value):
    result = scan(f'<BlockHandle handleId="tile" name="Tile" queryIds={{["{value}"]}} />')
    assert result.issues
    assert any("queryIds" in issue for issue in result.issues)


def test_dynamic_forwarding_is_compatible_but_warns_about_partial_inspection():
    result = scan('<BlockHandle handleId={id} name={label} sqlId="queries/sales" queryIds={sources} />')
    assert result.issues == []
    assert result.query_refs == {"queries/sales"}
    assert any("queryIds" in warning for warning in result.warnings)


def test_nested_and_commented_arrays_are_not_sources_of_the_parent_card():
    result = scan("""// <BlockHandle handleId="fake" name="Fake" queryIds={['queries/missing']} />
        <ChartCard chartId="real" chartType="line" sqlId="queries/sales"
            title={<span queryIds={['queries/missing']} />} />""")
    assert result.issues == []
    assert result.query_refs == {"queries/sales"}
    assert list(result.ids_seen) == ["real"]


@pytest.mark.parametrize("array", ["[]", "[/* first */ 'sales', // second\n 'cost',]"])
def test_empty_arrays_and_comments_are_supported(array):
    result = scan(f'<BlockHandle handleId="tile" name="Tile" queryIds={{{array}}} />')
    assert result.issues == []
    assert result.warnings == []
    assert result.query_refs == (set() if array == "[]" else {"queries/sales", "queries/cost"})


@pytest.mark.parametrize("prop", ['queryIds="sales"', 'queryIds={["sales"]} queryIds={["cost"]}'])
def test_malformed_declarations_are_rejected(prop):
    result = scan(f'<BlockHandle handleId="tile" name="Tile" {prop} />')
    assert any("queryIds" in issue for issue in result.issues)


def test_computed_arrays_remain_partial_and_do_not_invent_static_sources():
    result = scan('<BlockHandle handleId="tile" name="Tile" sqlId="sales" queryIds={["cost"].concat(extra)} />')
    assert result.issues == []
    assert result.query_refs == {"queries/sales"}
    assert any("queryIds" in warning for warning in result.warnings)


def test_quoted_display_text_cannot_override_source_properties():
    result = scan("""<ChartCard chartId="real" chartType="line" sqlId="queries/sales"
        title="sqlId='queries/missing'" queryIds={['cost']} />""")
    assert result.issues == []
    assert result.query_refs == {"queries/sales", "queries/cost"}


@pytest.mark.parametrize(
    "source,expected",
    [
        ('<BlockHandle handleId={id} name={label} kind="kpi" sqlId={sqlId} />', "handleId"),
        ('<BlockHandle handleId="tile" name="Tile" sqlId={query} />', "sqlId"),
        ('<BlockHandle handleId="tile" name="Tile" queryIds={sources} />', "queryIds"),
        ('<BlockHandle handleId="tile" name="Tile" queryIds={["sales"].concat(extra)} />', "queryIds"),
        ("<BlockHandle {...props} />", "spread props"),
        ("<ChartCard {...props} />", "spread props"),
        ('<ChartCard chartId="chart" chartType="line" sqlId="sales" queryIds={sources} />', "queryIds"),
    ],
)
def test_authoring_rejects_sources_that_cannot_be_statically_attributed(source, expected):
    result = scan_render_cards(
        {"app": {"rel": "app.jsx", "source": source}},
        query_exists=lambda name: name in {"sales", "cost"},
        missing_query_hint="a missing saved query",
        require_static_sources=True,
    )
    assert any(expected in issue and "call site" in issue for issue in result.issues)


@pytest.mark.parametrize("tag", ["ChartCard", "BlockHandle"])
def test_legacy_spread_inspection_remains_compatible(tag):
    result = scan(f"<{tag} {{...props}} />")
    assert result.issues == []
    assert any("spread props" in warning for warning in result.warnings)
    assert result.registry() == []


def test_literal_kpi_wrapper_registers_the_content_and_all_contributing_queries():
    from datus.utils.artifact_insight import _blocks

    source = """<BlockHandle handleId="repeat_rate" name={label} kind="kpi"
        sqlId="sales" queryIds={['sales', 'cost']} params={params}>
        <KpiCard value={ratio} />
    </BlockHandle>"""
    result = scan_render_cards(
        {"app": {"rel": "app.jsx", "source": source}},
        query_exists=lambda name: name in {"sales", "cost"},
        missing_query_hint="a missing saved query",
        require_static_sources=True,
    )
    assert result.issues == []
    assert result.registry() == [{"chart_id": "repeat_rate", "jsx_path": "render/app.jsx", "kind": "kpi"}]
    blocks = _blocks({"render/app.jsx": source}, {"sales", "cost"})
    assert [(b.id, b.query_ids) for b in blocks] == [("repeat_rate", ["sales", "cost"])]


def test_nested_display_props_and_comments_do_not_trigger_strict_source_errors():
    source = """// <BlockHandle handleId={ignored} name={label} sqlId={query} />
    <BlockHandle handleId="tile" name="sqlId={query}" kind="note"
        style={{...defaults}}>
        <div sqlId={displayProp} />
    </BlockHandle>"""
    result = scan_render_cards(
        {"app": {"rel": "app.jsx", "source": source}},
        query_exists=lambda _: False,
        missing_query_hint="a missing saved query",
        require_static_sources=True,
    )
    assert result.issues == []
    assert result.query_refs == set()
    assert result.registry() == [{"chart_id": "tile", "jsx_path": "render/app.jsx", "kind": "note"}]


@pytest.mark.parametrize("kind", ["dashboard", "report"])
def test_system_prompt_kpi_example_finalizes_with_complete_content_sources(kind):
    import re
    from pathlib import Path

    from jinja2 import Environment, FileSystemLoader

    from datus.utils.artifact_insight import _blocks

    template_dir = Path(__file__).resolve().parents[4] / "datus/prompts/prompt_templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    prompt = env.get_template(f"gen_visual_{kind}_system_1.0.j2").render()
    examples = re.findall(r"```jsx\n(.*?)```", prompt, re.DOTALL)
    files = {
        "render/shared/kpi-card.jsx": next(s for s in examples if s.startswith("// render/shared/kpi-card.jsx")),
        "render/kpi-strip.jsx": next(s for s in examples if s.startswith("// render/kpi-strip.jsx")),
    }
    result = scan_render_cards(
        {path: {"rel": path.removeprefix("render/"), "source": source} for path, source in files.items()},
        query_exists=lambda name: name == "kpi_summary",
        missing_query_hint="a missing saved query",
        require_static_sources=True,
    )
    assert result.issues == []
    blocks = _blocks(files, {"kpi_summary"})
    assert {b.id for b in blocks} == {"kpi_total_revenue", "kpi_total_orders", "kpi_customers", "kpi_avg_order_value"}
    assert all(b.query_ids == ["kpi_summary"] for b in blocks)
    assert {card["chart_id"] for card in result.registry()} == {b.id for b in blocks}
