# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Read-model contracts: traversal over cycles, statement grouping, roles and components."""

import pytest

from datus.storage.lineage.graph import LineageGraph
from datus.storage.lineage.models import Evidence, LineageDocument, SourceContribution, SourceRecord, TempHop
from datus.storage.lineage.store import replace_source


def build(edges):
    """``edges``: (upstream, downstream, source, line[, via_temp])."""
    document = LineageDocument()
    by_source = {}
    for up, down, source, line, *rest in edges:
        evidence = Evidence(source=source, line=line, operation="insert", via_temp=rest[0] if rest else None)
        by_source.setdefault(source, []).append((up, down, evidence))
    for source, items in by_source.items():
        record = SourceRecord(kind="file", sha256=source, analyzer_version=1, analyzed_at="")
        replace_source(document, source, SourceContribution(record=record, edges=items))
    return LineageGraph(document)


@pytest.mark.parametrize("depth", [3, -1])
def test_traversal_through_a_cycle_terminates_and_reaches_every_member(depth):
    graph = build([("a", "b", "f", 1), ("b", "c", "f", 2), ("c", "a", "f", 3), ("c", "d", "f", 4)])
    nodes, edges = graph.traverse(["a"], "downstream", depth)
    assert nodes == {"a", "b", "c", "d"}
    assert edges == {("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")}


def test_depth_counts_table_hops_in_each_direction():
    graph = build([("a", "b", "f", 1), ("b", "c", "f", 2), ("c", "d", "f", 3)])
    assert graph.traverse(["c"], "upstream", 1)[0] == {"b", "c"}
    assert graph.traverse(["c"], "upstream", 2)[0] == {"a", "b", "c"}
    assert graph.traverse(["b"], "both", 1)[0] == {"a", "b", "c"}
    assert graph.traverse(["b"], "both", 0) == ({"b"}, set())


def test_one_statement_groups_all_its_sources_and_temp_hops():
    hops = [TempHop(table="tmp", lines=[1])]
    graph = build([("s1", "t", "f", 3, hops), ("s2", "t", "f", 3, hops), ("s1", "t", "g", 7)])
    statements = graph.statements_for({("s1", "t")})
    assert [(st.source, st.line, sorted(st.sources)) for st in statements] == [
        ("f", 3, ["s1", "s2"]),
        ("g", 7, ["s1"]),
    ]
    assert [hop.table for hop in statements[0].via_temp] == ["tmp"]


def test_roles_ignore_self_loops_and_components_number_largest_first():
    graph = build([("a", "b", "f", 1), ("b", "c", "f", 2), ("x", "y", "f", 3), ("inc", "inc", "f", 4)])
    assert [graph.role(t) for t in ("a", "b", "c", "inc")] == ["root", "intermediate", "leaf", "isolated"]
    components = graph.components()
    assert {components[t] for t in ("a", "b", "c")} == {1}
    assert components["x"] == components["y"] == 2
    assert components["inc"] == 3


def test_globs_resolve_against_full_and_default_database_names():
    graph = build([("ods.a", "dw.a_d", "f", 1), ("dw.a_d", "dw.b_d", "f", 2)])
    result = graph.resolve(["*_d", "dw.*", "ods.?"], "dw")
    assert result.resolved == {"*_d": ["dw.a_d", "dw.b_d"], "dw.*": ["dw.a_d", "dw.b_d"], "ods.?": ["ods.a"]}
    assert result.tables == ["dw.a_d", "dw.b_d", "ods.a"]


def test_exact_spelling_wins_over_case_variants():
    graph = build([("dw.Orders", "dw.x", "f", 1), ("dw.orders", "dw.x", "f", 2)])
    result = graph.resolve(["dw.orders", "DW.ORDERS"])
    assert result.resolved == {"dw.orders": ["dw.orders"]}
    assert result.ambiguous == {"DW.ORDERS": ["dw.Orders", "dw.orders"]}
