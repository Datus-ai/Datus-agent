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
