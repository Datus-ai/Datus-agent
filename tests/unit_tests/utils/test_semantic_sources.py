"""Exact calculation evidence across SQL scopes, statements and physical writes."""

from unittest.mock import patch

from datus.utils.semantic_sources import SemanticSourceAnalyzer


def analyze(sql, schemas=None):
    return SemanticSourceAnalyzer("mysql").analyze([{"name": "etl.sql", "sql": sql}], schemas)


def test_preserves_policy_literals_boolean_structure_group_aliases_and_passthrough():
    result = analyze("""
        WITH counts AS (
          SELECT DATE_FORMAT(dt, '%Y-%m') AS month, store,
            COUNT(DISTINCT CASE WHEN sla = '00022' THEN order_id ELSE NULL END) AS n
          FROM orders
          WHERE (brand = 'KFC' OR brand = 'PH') AND state = 1
          GROUP BY month, 2
        )
        SELECT month, store, n AS stored_count FROM counts
    """)
    inner, outer = result["queries"]
    assert inner["group_by"] == ["DATE_FORMAT(dt, '%Y-%m')", "store"]
    assert inner["filters"]["where"] == "WHERE (brand = 'KFC' OR brand = 'PH') AND state = 1"
    calculation = result["outputs"][2]
    assert "'00022'" in calculation["expression"]
    assert "COUNT(DISTINCT" in calculation["expression"]
    assert result["outputs"][-1]["dependencies"] == [calculation["id"]]
    assert result["outputs"][-1]["calculation_queries"] == [inner["id"]]
    assert outer["group_by"] == []
    assert result["complete"]


def test_multiple_statements_semicolons_in_comments_and_strings_and_upstream_write():
    result = analyze("""
        -- separator ; inside comment
        CREATE TABLE summary (market VARCHAR(40), num BIGINT);
        INSERT INTO summary SELECT market, COUNT(*) FROM raw WHERE label='a;b' GROUP BY market;
        SELECT market, num AS inherited FROM summary
    """)
    assert len(result["statements"]) == 3
    assert result["outputs"][-1]["dependencies"] == [result["statements"][1]["outputs"][1]]
    assert len(result["outputs"][-1]["calculation_queries"]) == 1
    assert "label = 'a;b'" in result["queries"][0]["filters"]["where"]
    assert result["complete"]


def test_union_all_branches_are_bound_by_position_even_when_names_differ():
    result = analyze("""
        INSERT INTO target(market,n)
        SELECT market, COUNT(*) AS n FROM a GROUP BY market
        UNION ALL
        SELECT region, SUM(value) AS other_name FROM b GROUP BY region
    """)
    union = result["queries"][-1]
    assert union["operation"] == "UNION" and not union["distinct"]
    assert len(union["branches"]) == 2
    assert len(result["outputs"][-1]["dependencies"]) == 2
    assert len(result["outputs"][-1]["calculation_queries"]) == 2
    assert result["outputs"][-1]["target_column"] == "n"
    assert result["complete"]


def test_windows_keep_partition_order_frame_and_post_window_selection():
    result = analyze("""
        SELECT market, month,
          SUM(n) OVER (PARTITION BY market ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS rolling
        FROM (SELECT market, month, COUNT(*) n FROM raw GROUP BY market,month) c
        QUALIFY rolling > 0
    """)
    output = result["outputs"][-1]
    assert output["kind"] == "window"
    assert "PARTITION BY market ORDER BY month" in output["windows"][0]
    assert "2 PRECEDING" in output["windows"][0]
    assert len(output["calculation_queries"]) == 2
    assert result["queries"][-1]["filters"]["qualify"] == "QUALIFY rolling > 0"


def test_live_schemas_disambiguate_qualified_sources_without_guessing():
    sql = "SELECT region, SUM(amount) n FROM sales.a a LEFT JOIN crm.a b ON a.customer=b.id GROUP BY region"
    result = analyze(sql, {"sales.a": ["amount", "customer"], "crm.a": ["id", "region"]})
    assert result["outputs"][0]["source_columns"] == [{"table": "crm.a", "column": "region"}]
    assert result["queries"][0]["joins"][0]["join_type"] == "LEFT"
    assert result["queries"][0]["joins"][0]["cardinality"] == "unverified"
    ambiguous = analyze(sql)
    assert any(i["reason"] == "column_origin_unresolved" for i in ambiguous["issues"])


def test_star_and_union_expansion_uses_live_columns():
    result = analyze("SELECT * FROM a UNION SELECT * FROM b", {"a": ["k", "v"], "b": ["x", "y"]})
    assert len(result["statements"][0]["outputs"]) == 2
    assert [o["name"] for o in result["outputs"][-2:]] == ["k", "v"]
    assert result["outputs"][0]["source_columns"] == [{"table": "a", "column": "k"}]
    unknown = analyze("SELECT * FROM missing")
    assert any(i["reason"] == "star_expansion_unresolved" for i in unknown["issues"])


def test_expanded_star_preserves_upstream_aggregate_lineage():
    result = analyze("SELECT * FROM (SELECT market, COUNT(*) n FROM raw GROUP BY market) c")
    inner = result["queries"][0]
    assert result["outputs"][-1]["dependencies"] == [inner["outputs"][-1]]
    assert result["outputs"][-1]["calculation_queries"] == [inner["id"]]
    assert result["outputs"][-2]["dependencies"] == [inner["outputs"][0]]


def test_unsupported_star_modifiers_do_not_invent_passthrough_outputs():
    for projection in ("* EXCLUDE (v)", "t.* REPLACE (v+1 AS v)"):
        result = SemanticSourceAnalyzer("duckdb").analyze(
            [{"name": "source", "sql": f"SELECT {projection} FROM t"}], {"t": ["k", "v"]}
        )
        assert result["outputs"] == []
        assert any(issue["reason"] == "star_expansion_unresolved" for issue in result["issues"])


def test_invalid_statement_does_not_hide_other_statements_or_unresolved_templates():
    result = analyze("SELECT ) FROM broken; SELECT COUNT(*) n FROM orders WHERE dt='${day}'")
    assert not result["complete"]
    assert result["statements"][0]["status"] == "parse_error"
    assert result["statements"][1]["parameters"] == ["day"]
    assert result["queries"][0]["filters"]["where"] == "WHERE dt = '${day}'"


def test_insert_column_mismatch_never_invents_physical_measure_mapping():
    result = analyze("INSERT INTO target SELECT k, COUNT(*) n FROM raw GROUP BY k", {"target": ["k"]})
    assert any(i["reason"] == "target_column_mapping_unresolved" for i in result["issues"])
    assert all("target_column" not in o for o in result["outputs"])


def test_ctas_intermediate_columns_and_sequential_reads_do_not_require_live_table():
    result = analyze(
        "SELECT n FROM summary; "
        "CREATE TABLE summary AS SELECT market, COUNT(*) n FROM orders GROUP BY market; "
        "SELECT market,n FROM summary"
    )
    assert result["outputs"][0]["dependencies"] == []
    assert result["outputs"][-1]["calculation_queries"] == [result["queries"][1]["id"]]
    assert result["outputs"][2]["target_column"] == "n"


def test_repeated_analysis_reuses_parse_but_rechecks_column_resolution():
    analyzer = SemanticSourceAnalyzer("mysql")
    entries = [{"name": "source", "sql": "SELECT a FROM t"}]
    with patch("datus.utils.semantic_sources.sqlglot.parse_one", wraps=__import__("sqlglot").parse_one) as parser:
        analyzer.analyze(entries)
        result = analyzer.analyze(entries, {"t": ["other"]})
    assert parser.call_count == 1
    assert any(i["reason"] == "source_column_missing" for i in result["issues"])


def test_output_ids_survive_unrelated_additions_but_change_with_dependencies():
    analyzer = SemanticSourceAnalyzer("mysql")
    producer = {"name": "upstream", "sql": "CREATE TABLE summary AS SELECT market, COUNT(*) n FROM raw GROUP BY market"}
    consumer = {"name": "downstream", "sql": "SELECT market,n FROM summary"}
    original = analyzer.analyze([producer, consumer])
    ids = [s["outputs"] for s in original["statements"]]
    extra = {"name": "extra", "sql": "SELECT SUM(price) total FROM unrelated"}
    appended = analyzer.analyze([extra, producer, consumer], {"unrelated": ["price"]})
    assert [s["outputs"] for s in appended["statements"][1:]] == ids
    changed = analyzer.analyze(
        [{**producer, "sql": producer["sql"].replace("COUNT(*)", "COUNT(DISTINCT market)")}, consumer]
    )
    assert changed["statements"][0]["outputs"] != ids[0]
    assert changed["statements"][1]["outputs"] != ids[1]
    schema_changed = analyzer.analyze([producer, consumer], {"raw": ["market", "other"]})
    assert schema_changed["statements"][0]["outputs"] != ids[0]
