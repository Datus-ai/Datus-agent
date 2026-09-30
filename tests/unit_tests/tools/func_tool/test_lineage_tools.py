# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Behavioral contracts for the compact, complete lineage tool response."""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from datus.tools.func_tool.lineage_tools import LineageTools


def make_tool(root, datasources=None, current="", **kwargs):
    config = SimpleNamespace(current_datasource=current, services=SimpleNamespace(datasources=datasources or {}))
    return LineageTools(config, root_path=str(root), **kwargs)


def extract(root, **kwargs):
    response = make_tool(root).extract_sql_lineage(dialect="mysql", default_database="dw", **kwargs)
    assert response.success == 1, response.error
    return response.result


def evidence(result, row):
    return [(result["files"][ref.split(":")[0]], int(ref.split(":")[1])) for ref in row["evidence"]]


def test_compact_contract_and_copy_provenance(tmp_path):
    sql = "INSERT INTO t SELECT a.id FROM a LEFT JOIN b ON a.id=b.id WHERE a.dt >= '${day}';"
    (tmp_path / "a.sql").write_text(sql)
    (tmp_path / "b.sql").write_text("-- copied script\n" + sql)
    result = extract(tmp_path, paths=["*.sql"])
    assert set(result) == {"schema_version", "complete", "files", "lineage", "tables", "joins", "stats"}
    assert result["schema_version"] == 5
    assert result["complete"] is True
    [edge] = result["lineage"]
    assert {k: v for k, v in edge.items() if k != "evidence"} == {
        "target": "t",
        "sources": ["a", "b"],
        "operation": "insert",
        "parameterized_predicates": ["dt >= '${day}'"],
    }
    assert evidence(result, edge) == [("a.sql", 1), ("b.sql", 2)]
    [join] = result["joins"]
    assert join["expression"] == "#1 LEFT JOIN #2 ON a.id = b.id"
    assert set(join) == {"expression", "evidence"}
    assert evidence(result, join) == [("a.sql", 1), ("b.sql", 2)]
    assert result["tables"] == {"#1": "a", "#2": "b"}
    assert result["stats"] == {"files": 2, "statements": 2, "parsed": 2, "dialect": "mysql", "default_database": "dw"}


def test_byte_identical_copies_are_analyzed_once_and_mapped_to_the_original(tmp_path):
    sql = "INSERT INTO t SELECT a.id FROM a LEFT JOIN b ON a.id=b.id"
    (tmp_path / "a.sql").write_text(sql)
    (tmp_path / "copy.sql").write_text(sql)
    (tmp_path / "c.sql").write_text("SELECT * FROM c")
    result = extract(tmp_path, paths=["*.sql"])
    assert result["files"] == {"f1": "a.sql", "f2": "c.sql", "f3": "=f1"}
    assert [row["evidence"] for row in result["lineage"] + result["joins"]] == [["f1:1"], ["f2:1"], ["f1:1"]]
    assert result["stats"]["files"] == 3
    assert result["complete"] is True
    # Identity is content, not name: a diverging copy is analyzed and cited on its own.
    (tmp_path / "copy.sql").write_text(sql + " WHERE a.flag = 1")
    result = extract(tmp_path, paths=["*.sql"])
    assert result["files"] == {"f1": "a.sql", "f2": "c.sql", "f3": "copy.sql"}
    assert sorted(ref for row in result["lineage"] for ref in row["evidence"]) == ["f1:1", "f2:1", "f3:1"]


def test_all_files_and_connected_edges_returned_together(tmp_path):
    """A graph exceeding the former file/page limits must arrive in one response."""
    for i in range(510):
        (tmp_path / f"{i:03}.sql").write_text(f"INSERT INTO t{i + 1} SELECT * FROM t{i}")
    result = extract(tmp_path, paths=["*.sql"])
    assert result["complete"] is True
    assert result["stats"]["files"] == result["stats"]["parsed"] == 510
    assert {(e["target"], tuple(e["sources"])) for e in result["lineage"]} == {
        (f"t{i + 1}", (f"t{i}",)) for i in range(510)
    }
    assert len(result["files"]) == 510


def test_distinct_operations_and_sql_for_one_target_are_not_mixed(tmp_path):
    (tmp_path / "q.sql").write_text(
        "INSERT INTO t SELECT * FROM a WHERE flag=1;\n"
        "INSERT INTO t SELECT * FROM a WHERE flag=2;\n"
        "INSERT INTO t SELECT * FROM b;\n"
        "TRUNCATE TABLE t; INSERT INTO t SELECT * FROM a;"
    )
    result = extract(tmp_path, paths=["*.sql"])
    assert [(e["target"], e["sources"], e["operation"]) for e in result["lineage"]] == [
        ("t", ["a"], "insert"),
        ("t", ["a"], "insert"),
        ("t", ["b"], "insert"),
        ("t", ["a"], "truncate_reload"),
    ]
    assert [evidence(result, e) for e in result["lineage"]] == [
        [("q.sql", 1)],
        [("q.sql", 2)],
        [("q.sql", 3)],
        [("q.sql", 4)],
    ]


def test_nested_temporary_folding_preserves_build_evidence(tmp_path):
    (tmp_path / "q.sql").write_text(
        "CREATE TABLE tmp_a AS SELECT * FROM src;\n"
        "CREATE TABLE tmp_b AS SELECT * FROM tmp_a;\n"
        "CREATE TABLE tmp_c AS SELECT * FROM tmp_b;\n"
        "INSERT INTO final SELECT * FROM tmp_c;\n"
        "DROP TABLE tmp_c; DROP TABLE tmp_b; DROP TABLE tmp_a;"
    )
    result = extract(tmp_path, paths=["*.sql"])
    [edge] = result["lineage"]
    assert edge["sources"] == ["src"]
    assert edge["target"] == "final"
    assert set(edge["via_temp"]) == {"tmp_a", "tmp_b", "tmp_c"}
    assert evidence(result, edge) == [("q.sql", 1), ("q.sql", 2), ("q.sql", 3), ("q.sql", 4)]


def test_cte_union_query_inventory_and_self_read_write(tmp_path):
    (tmp_path / "q.sql").write_text(
        "WITH x AS (SELECT id FROM a), y AS (SELECT id FROM x), z AS (SELECT id FROM y) "
        "SELECT * FROM z UNION ALL SELECT id FROM b;\n"
        "INSERT INTO t SELECT * FROM t;"
    )
    result = extract(tmp_path, paths=["*.sql"])
    assert {(e["target"], tuple(e["sources"]), e["operation"]) for e in result["lineage"]} == {
        (None, ("a", "b"), "select"),
        ("t", ("t",), "insert"),
    }


@pytest.mark.parametrize(
    "condition,kind",
    [
        ("a.id=b.id", "LEFT"),
        ("LEFT(a.id,6)=b.id", "LEFT"),
        ("a.id=b.id AND a.dt=b.dt", "LEFT"),
    ],
)
def test_join_expression_and_composite_keys(tmp_path, condition, kind):
    (tmp_path / "q.sql").write_text(f"SELECT * FROM a LEFT JOIN b ON {condition}")
    result = extract(tmp_path, paths=["*.sql"])
    [join] = result["joins"]
    assert (
        join["expression"]
        == {
            "a.id=b.id": "#1 LEFT JOIN #2 ON a.id = b.id",
            "LEFT(a.id,6)=b.id": "#1 LEFT JOIN #2 ON LEFT(a.id, 6) = b.id",
            "a.id=b.id AND a.dt=b.dt": "#1 LEFT JOIN #2 ON a.id = b.id AND a.dt = b.dt",
        }[condition]
    )


def test_join_transform_outer_direction_and_self_aliases(tmp_path):
    (tmp_path / "q.sql").write_text(
        "SELECT * FROM z LEFT JOIN a ON DATE(z.dt)=a.day;\n"
        "SELECT * FROM employee e LEFT JOIN employee m ON CAST(e.manager AS BIGINT)=m.id AND e.x=e.y"
    )
    result = extract(tmp_path, paths=["*.sql"])
    first, second = result["joins"]
    assert result["tables"] == {"#1": "a", "#2": "employee", "#3": "z"}
    assert first["expression"] == "#3 LEFT JOIN #1 ON DATE(z.dt) = a.day"
    assert second["expression"] == "#2 AS e LEFT JOIN #2 AS m ON CAST(e.manager AS SIGNED) = m.id AND e.x = e.y"


def test_table_ids_are_shared_across_files_and_keep_foreign_database_qualifiers(tmp_path):
    (tmp_path / "one.sql").write_text("SELECT * FROM dw.orders o JOIN ods.store s ON o.store_id = s.id")
    (tmp_path / "two.sql").write_text(
        "SELECT * FROM (SELECT store_id AS sid, SUM(amt) AS amt FROM orders GROUP BY 1) agg\n"
        "LEFT JOIN ods.store s ON agg.sid = s.id"
    )
    result = extract(tmp_path, paths=["*.sql"])
    assert result["tables"] == {"#1": "orders", "#2": "ods.store"}
    assert [row["expression"] for row in result["joins"]] == [
        "#1 AS o INNER JOIN #2 AS s ON o.store_id = s.id",
        "agg{#1} LEFT JOIN #2 AS s ON agg.sid = s.id",
    ]
    assert not any("/*" in row["expression"] for row in result["joins"])


def test_unresolved_join_expression_marks_coverage_incomplete(tmp_path):
    (tmp_path / "q.sql").write_text("SELECT * FROM a JOIN b ON a.id=b.id OR a.alt=b.alt")
    result = extract(tmp_path, paths=["*.sql"])
    assert result["complete"] is False
    assert result["lineage"][0]["sources"] == ["a", "b"]
    assert result["joins"] == []


def test_parse_failure_and_dynamic_python_keep_surviving_graph(tmp_path):
    (tmp_path / "q.sql").write_text("SELECT * FROM a;\nSELECT FROM WHERE (((;\nSELECT * FROM b;")
    (tmp_path / "q.py").write_text("query = prefix + table")
    result = extract(tmp_path, paths=["*.sql", "*.py"])
    assert result["complete"] is False
    assert [e["sources"] for e in result["lineage"]] == [["a"], ["b"]]
    assert result["stats"]["parsed"] == 2
    assert result["stats"]["statements"] == 3


def test_python_literals_and_templates(tmp_path):
    (tmp_path / "q.py").write_text('def build():\n    run("""\nINSERT INTO t SELECT * FROM {{ ref(\'s\') }}\n""")\n')
    result = extract(tmp_path, paths=["*.py"])
    [edge] = result["lineage"]
    assert (edge["target"], edge["sources"], edge["operation"]) == ("t", ["s"], "insert")
    assert evidence(result, edge) == [("q.py", 3)]


@pytest.mark.parametrize("mode", ["external", "symlink", "hidden"])
def test_unreadable_zones_cannot_be_read(tmp_path, tmp_path_factory, mode):
    outside = tmp_path_factory.mktemp("outside") / "q.sql"
    outside.write_text("SELECT * FROM private_table")
    paths = [str(outside)]
    if mode == "symlink":
        (tmp_path / "q.sql").symlink_to(outside)
        paths = ["*.sql"]
    elif mode == "hidden":
        (tmp_path / ".datus").mkdir()
        (tmp_path / ".datus" / "q.sql").write_text("SELECT * FROM private_table")
        paths = [".datus/q.sql"]
    result = extract(tmp_path, paths=paths)
    assert result["complete"] is False
    assert result["lineage"] == []


def test_missing_oversized_and_duplicate_inputs(tmp_path):
    (tmp_path / "q.sql").write_text("SELECT * FROM s")
    (tmp_path / "large.sql").write_text("--" + "x" * (2 * 1024 * 1024))
    result = extract(tmp_path, paths=["q.sql", "q.sql", "missing.sql", "missing/*.sql", "large.sql"])
    assert len(result["lineage"]) == 1
    assert result["complete"] is False
    assert result["lineage"][0]["sources"] == ["s"]
    assert result["stats"]["files"] == 1


def test_datasource_defaults_and_explicit_override(tmp_path):
    (tmp_path / "q.sql").write_text("SELECT * FROM dw.s")
    tool = make_tool(tmp_path, {"wh": SimpleNamespace(type="sqlite", database="dw")}, "wh")
    response = tool.extract_sql_lineage(paths=["q.sql"], dialect="starrocks")
    assert response.success == 1
    assert response.result["stats"]["dialect"] == "starrocks"
    assert response.result["stats"]["default_database"] == "dw"
    assert response.result["lineage"][0]["sources"] == ["s"]
    assert tool.extract_sql_lineage(paths=["q.sql"], datasource="missing").success == 0
    assert tool.extract_sql_lineage(paths=[]).success == 0


def test_content_and_permissions_rechecked_on_cached_input(tmp_path):
    path = tmp_path / "q.sql"
    path.write_text("SELECT * FROM a")
    stat = path.stat()
    tool = make_tool(tmp_path)
    assert tool.extract_sql_lineage(paths=["q.sql"]).result["lineage"][0]["sources"] == ["a"]
    path.write_text("SELECT * FROM b")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert tool.extract_sql_lineage(paths=["q.sql"]).result["lineage"][0]["sources"] == ["b"]
    path.unlink()
    response = tool.extract_sql_lineage(paths=["q.sql"])
    assert response.result["complete"] is False
    assert response.result["lineage"] == []


@pytest.mark.asyncio
async def test_sdk_invocation_returns_independent_complete_graphs(tmp_path):
    (tmp_path / "a.sql").write_text("SELECT * FROM a")
    (tmp_path / "b.sql").write_text("SELECT * FROM b")
    [tool] = make_tool(tmp_path).available_tools()
    names = ["a", "b"] * 4
    results = await asyncio.gather(
        *[tool.on_invoke_tool(None, json.dumps({"paths": [f"{name}.sql"], "dialect": "mysql"})) for name in names]
    )
    assert [r["success"] for r in results] == [1] * len(names)
    assert [r["result"]["lineage"][0]["sources"] for r in results] == [[name] for name in names]
    assert [r["result"]["complete"] for r in results] == [True] * len(names)
    # The SDK schema is the public invocation contract, including the removal of sections/pagination.
    assert set(tool.params_json_schema["properties"]) == {"paths", "datasource", "dialect", "default_database"}


@pytest.mark.parametrize("dialect", ["mysql", "postgres", "snowflake", "spark", "tsql"])
def test_join_evidence_locates_complete_conditions_in_source(tmp_path, dialect):
    sql = (
        "-- header\n"
        "SELECT * FROM a LEFT JOIN b\n"
        "ON a.id = b.id\n"
        "AND COALESCE(a.code,\n"
        "  'x') = COALESCE(b.code,\n"
        "  'x'\n"
        ")\n"
        "WHERE a.flag = 1;\n"
        "SELECT * FROM a LEFT JOIN b\n"
        "ON a.id = b.id\n"
        "AND COALESCE(a.code,\n"
        "  'x') = COALESCE(b.code,\n"
        "  'x'\n"
        ")\n"
    )
    (tmp_path / "q.sql").write_text(sql)
    response = make_tool(tmp_path).extract_sql_lineage(paths=["q.sql"], dialect=dialect)
    assert response.success == 1
    [join] = response.result["joins"]
    assert join["evidence"] == ["f1:3-7", "f1:10-14"]
    assert [row["evidence"] for row in response.result["lineage"]] == [["f1:2"], ["f1:9"]]


def test_join_evidence_python_templates_nested_scopes_and_using(tmp_path):
    sql = '''# python header
query = """
SELECT * FROM (
 SELECT a.id FROM {{
 ref('a')
 }} a JOIN b
 USING (
 id
 )
) x JOIN c
ON x.id = c.id
WHERE x.id >
 c.id;
SELECT * FROM a, b
WHERE a.id = b.id
"""
'''
    (tmp_path / "q.py").write_text(sql)
    result = extract(tmp_path, paths=["q.py"])
    assert {row["expression"]: row["evidence"] for row in result["joins"]} == {
        "#1 AS a INNER JOIN #2 USING (id)": ["f1:7-9"],
        "#1 CROSS JOIN #2 WHERE a.id = b.id": ["f1:15"],
        "x{#1,#2} INNER JOIN #3 ON x.id = c.id": ["f1:11"],
    }
    assert result["complete"] is False


def test_missing_parser_positions_explicitly_mark_statement_fallback(tmp_path, monkeypatch):
    import sqlglot

    monkeypatch.setattr(
        "datus.utils.sql_lineage._parse_with_positions", lambda text, dialect: sqlglot.parse_one(text, read=dialect)
    )
    (tmp_path / "q.sql").write_text("-- header\nSELECT * FROM a JOIN b\nON a.id = b.id\nWHERE a.id > b.id")
    result = extract(tmp_path, paths=["q.sql"])
    assert result["joins"][0]["evidence"] == ["f1:2@statement"]


def test_cte_and_union_relations_keep_projected_names_and_all_physical_sources(tmp_path):
    (tmp_path / "q.sql").write_text(
        "WITH unused AS (SELECT * FROM secret), x AS (SELECT LEFT(code,6) AS id FROM raw_a "
        "UNION ALL SELECT id FROM raw_b)\n"
        "SELECT * FROM x AS p LEFT JOIN dimension d\nON p.id = d.id;"
    )
    result = extract(tmp_path, paths=["q.sql"])
    assert result["complete"] is True
    assert result["joins"] == [
        {
            "expression": "x{#2,#3} AS p LEFT JOIN #1 AS d ON p.id = d.id",
            "evidence": ["f1:3"],
        }
    ]
    # Tables read only by the unused CTE are not part of any join, so they get no ID.
    assert result["tables"] == {"#1": "dimension", "#2": "raw_a", "#3": "raw_b"}


def test_multi_alias_predicate_identifies_additional_scope(tmp_path):
    (tmp_path / "q.sql").write_text("SELECT * FROM a JOIN b ON a.id=b.id LEFT JOIN c\nON a.id=c.id AND b.flag=c.flag")
    result = extract(tmp_path, paths=["q.sql"])
    assert result["joins"][1] == {
        "expression": "#1, #2 LEFT JOIN #3 ON a.id = c.id AND b.flag = c.flag",
        "evidence": ["f1:2"],
    }


def test_composite_using_has_one_complete_expression(tmp_path):
    (tmp_path / "q.sql").write_text("SELECT * FROM a LEFT JOIN b USING (\n id,\n dt\n)")
    result = extract(tmp_path, paths=["q.sql"])
    assert result["joins"] == [{"expression": "#1 LEFT JOIN #2 USING (id, dt)", "evidence": ["f1:1-4"]}]
