# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Behavioral contracts for the persisted project lineage tools."""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from datus.storage.lineage import analysis
from datus.tools.func_tool import lineage_tools as lineage_module
from datus.tools.func_tool.lineage_tools import LineageTools


def make_tool(root, datasources=None, current="", **kwargs):
    config = SimpleNamespace(current_datasource=current, services=SimpleNamespace(datasources=datasources or {}))
    return LineageTools(config, root_path=str(root), **kwargs)


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def upsert(tool, **kwargs):
    kwargs.setdefault("dialect", "mysql")
    kwargs.setdefault("default_database", "dw")
    response = tool.upsert_lineage(**kwargs)
    assert response.success == 1, response.error
    return response.result


def query(tool, **kwargs):
    response = tool.query_lineage(**kwargs)
    assert response.success == 1, response.error
    return response.result


def saved(root):
    return json.loads((root / "lineage" / "lineage.json").read_text())


def records(result):
    """(target, sources, operation, [(path, line), ...]) per statement record."""
    return [
        (
            row["target"],
            row["sources"],
            row["operation"],
            [(result["files"][ref.split(":")[0]], int(ref.split(":")[1])) for ref in row["evidence"]],
        )
        for row in result["lineage"]
    ]


# -- upsert ---------------------------------------------------------------------------------


def test_upsert_persists_one_project_graph_across_directories(tmp_path):
    write(tmp_path, "ods/load.sql", "INSERT INTO dwd_order SELECT * FROM ods_order;")
    write(tmp_path, "dws/build.sql", "INSERT OVERWRITE TABLE dws_order SELECT * FROM dwd_order;")
    tool = make_tool(tmp_path)

    result = upsert(tool, paths=["**/*.sql"])

    assert result["added"] == ["dws/build.sql", "ods/load.sql"]
    assert result["delta"] == {"nodes_added": 3, "nodes_removed": 0, "edges_added": 2, "edges_removed": 0}
    document = saved(tmp_path)
    assert document["schema_version"] == 1
    assert document["revision"] == result["revision"] == 1
    assert set(document["sources"]) == {"dws/build.sql", "ods/load.sql"}
    assert document["nodes"] == {
        "dw.dwd_order": {"kind": "table"},
        "dw.dws_order": {"kind": "table"},
        "dw.ods_order": {"kind": "table"},
    }
    # Files in different directories meet at the shared table: one graph, one component.
    whole = query(tool)
    assert whole["nodes"]["dwd_order"] == {"role": "intermediate", "component": 1}
    assert whole["components"] == [{"id": 1, "tables": 3, "roots": ["ods_order"], "leaves": ["dws_order"]}]
    assert records(whole) == [
        ("dwd_order", ["ods_order"], "insert", [("ods/load.sql", 1)]),
        ("dws_order", ["dwd_order"], "overwrite", [("dws/build.sql", 1)]),
    ]


def test_unchanged_sources_are_skipped_without_rewriting_the_file(tmp_path):
    write(tmp_path, "a.sql", "INSERT INTO t SELECT * FROM s;")
    tool = make_tool(tmp_path)
    upsert(tool, paths=["*.sql"])
    before = (tmp_path / "lineage" / "lineage.json").read_bytes()

    result = upsert(tool, paths=["*.sql"])

    assert (result["added"], result["updated"], result["unchanged"]) == ([], [], 1)
    assert result["revision"] == 1
    assert (tmp_path / "lineage" / "lineage.json").read_bytes() == before


@pytest.mark.parametrize("change", [{"dialect": "hive"}, {"default_database": "ods"}])
def test_changed_analysis_settings_reanalyze_identical_content(tmp_path, change):
    write(tmp_path, "a.sql", "INSERT INTO t SELECT * FROM s;")
    tool = make_tool(tmp_path)
    upsert(tool, paths=["*.sql"])

    result = upsert(tool, paths=["*.sql"], **change)

    assert result["updated"] == ["a.sql"]
    record = saved(tmp_path)["sources"]["a.sql"]
    assert {key: record[key] for key in change} == change


def test_reanalysis_replaces_everything_the_source_contributed(tmp_path):
    path = write(tmp_path, "a.sql", "INSERT INTO t SELECT * FROM s1;\nINSERT INTO u SELECT * FROM s2;")
    tool = make_tool(tmp_path)
    upsert(tool, paths=["a.sql"])

    path.write_text("INSERT INTO t SELECT * FROM s1;")
    result = upsert(tool, paths=["a.sql"])

    assert result["updated"] == ["a.sql"]
    assert result["delta"] == {"nodes_added": 0, "nodes_removed": 2, "edges_added": 0, "edges_removed": 1}
    document = saved(tmp_path)
    assert set(document["nodes"]) == {"dw.s1", "dw.t"}
    assert [(e["from"], e["to"]) for e in document["edges"]] == [("dw.s1", "dw.t")]


def test_shared_edge_survives_until_its_last_source_is_removed(tmp_path):
    sql = "INSERT INTO t SELECT * FROM s;"
    write(tmp_path, "a.sql", sql)
    write(tmp_path, "b.sql", "-- copy\n" + sql)
    tool = make_tool(tmp_path)
    upsert(tool, paths=["*.sql"])
    [edge] = saved(tmp_path)["edges"]
    assert [(ev["source"], ev["line"]) for ev in edge["evidence"]] == [("a.sql", 1), ("b.sql", 2)]

    first = tool.delete_lineage(sources=["a.sql"]).result
    assert first["removed"] == ["a.sql"]
    assert first["delta"]["edges_removed"] == 0
    assert [ev["source"] for ev in saved(tmp_path)["edges"][0]["evidence"]] == ["b.sql"]

    second = tool.delete_lineage(sources=["b.sql"]).result
    assert second["delta"] == {"nodes_added": 0, "nodes_removed": 2, "edges_added": 0, "edges_removed": 1}
    assert saved(tmp_path)["nodes"] == {}
    assert saved(tmp_path)["edges"] == []


def test_prune_missing_removes_only_deleted_files_inside_the_patterns(tmp_path):
    write(tmp_path, "etl/a.sql", "INSERT INTO a SELECT * FROM s;")
    gone = write(tmp_path, "etl/b.sql", "INSERT INTO b SELECT * FROM s;")
    nested = write(tmp_path, "etl/sub/c.sql", "INSERT INTO c SELECT * FROM s;")
    other = write(tmp_path, "other/d.sql", "INSERT INTO d SELECT * FROM s;")
    tool = make_tool(tmp_path)
    upsert(tool, paths=["etl/**/*.sql", "other/*.sql"])
    for path in (gone, nested, other):
        path.unlink()

    result = upsert(tool, paths=["etl/*.sql"], prune_missing=True)

    # "*" does not cross directories, and other/ lies outside the patterns entirely.
    assert result["removed"] == ["etl/b.sql"]
    assert set(saved(tmp_path)["sources"]) == {"etl/a.sql", "etl/sub/c.sql", "other/d.sql"}


def test_inline_sql_is_an_idempotent_source(tmp_path):
    tool = make_tool(tmp_path)
    sql = "-- daily GMV by shop\nSELECT shop_id, SUM(gmv) FROM dws_order GROUP BY shop_id"

    first = upsert(tool, sql=sql)
    again = upsert(tool, sql=sql)
    named = upsert(tool, sql=sql, source_id="gmv")

    [default_id] = first["added"]
    assert default_id.startswith("inline:adhoc-")
    assert again["unchanged"] == 1
    assert named["added"] == ["inline:gmv"]
    sources = saved(tmp_path)["sources"]
    assert {sid: sources[sid]["kind"] for sid in sources} == {default_id: "inline", "inline:gmv": "inline"}
    # Both sources carry the same query: one node, cited twice.
    [node_id] = [n for n in saved(tmp_path)["nodes"] if n.startswith("query:")]
    assert saved(tmp_path)["nodes"][node_id] == {"kind": "query", "label": "daily GMV by shop"}

    result = tool.delete_lineage(sources=["inline:*"]).result
    assert sorted(result["removed"]) == sorted([default_id, "inline:gmv"])
    assert saved(tmp_path)["nodes"] == {}


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({}, "exactly one of paths or sql"),
        ({"paths": ["a.sql"], "sql": "SELECT 1"}, "exactly one of paths or sql"),
        ({"paths": ["a.sql"], "source_id": "x"}, "source_id only applies to sql"),
        ({"sql": "SELECT * FROM a", "prune_missing": True}, "prune_missing only applies to paths"),
    ],
)
def test_upsert_rejects_ambiguous_inputs(tmp_path, kwargs, error):
    response = make_tool(tmp_path).upsert_lineage(**kwargs)
    assert response.success == 0
    assert error in response.error
    assert not (tmp_path / "lineage" / "lineage.json").exists()


def test_parse_failures_are_saved_and_reported_with_lines(tmp_path):
    write(tmp_path, "q.sql", "INSERT INTO t SELECT * FROM a;\nSELECT FROM WHERE (((;\nINSERT INTO t SELECT * FROM b;")
    tool = make_tool(tmp_path)

    result = upsert(tool, paths=["q.sql"])

    [entry] = result["incomplete"]
    assert entry["source"] == "q.sql"
    assert [issue["line"] for issue in entry["issues"]] == [2]
    record = saved(tmp_path)["sources"]["q.sql"]
    assert (record["statements"], record["parsed"], record["complete"]) == (3, 2, False)
    assert [i["line"] for i in record["issues"]] == [2]
    # The surviving statements still contribute their lineage.
    lineage = query(tool, tables=["t"], direction="upstream")
    assert lineage["complete"] is False
    assert lineage["incomplete"] == ["f1"]
    assert {tuple(r[1]) for r in records(lineage)} == {("a",), ("b",)}


def test_reported_issues_are_capped_but_saved_in_full(tmp_path, monkeypatch):
    monkeypatch.setattr(lineage_module, "_MAX_REPORTED_ISSUES", 2)
    write(tmp_path, "a.sql", "SELECT FROM (;\nSELECT FROM (;")
    write(tmp_path, "b.sql", "SELECT FROM (;")

    result = upsert(make_tool(tmp_path), paths=["*.sql"])

    assert [(e["source"], len(e["issues"]), e.get("issues_omitted")) for e in result["incomplete"]] == [
        ("a.sql", 2, None),
        ("b.sql", 0, 1),
    ]
    assert len(saved(tmp_path)["sources"]["b.sql"]["issues"]) == 1


@pytest.mark.parametrize("mode", ["external", "symlink", "hidden"])
def test_unreadable_zones_are_skipped_and_never_saved(tmp_path, tmp_path_factory, mode):
    outside = tmp_path_factory.mktemp("outside") / "q.sql"
    outside.write_text("SELECT * FROM private_table")
    paths = [str(outside)]
    if mode == "symlink":
        (tmp_path / "q.sql").symlink_to(outside)
        paths = ["*.sql"]
    elif mode == "hidden":
        write(tmp_path, ".datus/q.sql", "SELECT * FROM private_table")
        paths = [".datus/q.sql"]

    result = upsert(make_tool(tmp_path), paths=paths)

    assert result["added"] == []
    assert result["skipped"]
    # Nothing readable changed the graph, so the lineage file is never even created.
    assert not (tmp_path / "lineage" / "lineage.json").exists()


def test_missing_and_oversized_inputs_are_skipped(tmp_path):
    write(tmp_path, "q.sql", "INSERT INTO t SELECT * FROM s")
    write(tmp_path, "large.sql", "--" + "x" * (2 * 1024 * 1024))

    result = upsert(make_tool(tmp_path), paths=["q.sql", "q.sql", "missing.sql", "missing/*.sql", "large.sql"])

    assert result["added"] == ["q.sql"]
    assert {(s["file"], s["reason"]) for s in result["skipped"]} == {
        ("missing.sql", "not a readable file"),
        ("missing/*.sql", "no files matched"),
        ("large.sql", "file too large"),
    }


def test_python_literals_and_templates(tmp_path):
    write(tmp_path, "q.py", 'def build():\n    run("""\nINSERT INTO t SELECT * FROM {{ ref(\'s\') }}\n""")\n')
    tool = make_tool(tmp_path)
    upsert(tool, paths=["*.py"])
    assert records(query(tool)) == [("t", ["s"], "insert", [("q.py", 3)])]


def test_datasource_supplies_dialect_and_database(tmp_path):
    write(tmp_path, "q.sql", "INSERT INTO t SELECT * FROM s")
    tool = make_tool(tmp_path, {"wh": SimpleNamespace(type="sqlite", database="dw")}, "wh")

    response = tool.upsert_lineage(paths=["q.sql"], dialect="starrocks")

    assert response.success == 1
    record = saved(tmp_path)["sources"]["q.sql"]
    assert (record["dialect"], record["default_database"]) == ("starrocks", "dw")
    assert set(saved(tmp_path)["nodes"]) == {"dw.s", "dw.t"}
    assert tool.upsert_lineage(paths=["q.sql"], datasource="missing").success == 0


def test_names_are_shortened_by_the_database_the_sources_were_analyzed_with(tmp_path):
    write(tmp_path, "q.sql", "INSERT INTO t SELECT * FROM s")
    tool = make_tool(tmp_path, {"wh": SimpleNamespace(type="starrocks", database="warehouse")}, "wh")
    tool.upsert_lineage(paths=["q.sql"], default_database="scripts_db")

    result = query(tool)

    assert result["stats"]["default_database"] == "scripts_db"
    assert records(result) == [("t", ["s"], "insert", [("q.sql", 1)])]


def test_corrupt_or_foreign_lineage_file_is_reported_not_overwritten(tmp_path):
    write(tmp_path, "q.sql", "INSERT INTO t SELECT * FROM s")
    target = write(tmp_path, "lineage/lineage.json", "{not json")
    tool = make_tool(tmp_path)
    assert tool.upsert_lineage(paths=["q.sql"]).success == 0
    assert tool.query_lineage().success == 0
    assert target.read_text() == "{not json"

    target.write_text(json.dumps({"schema_version": 99}))
    response = tool.upsert_lineage(paths=["q.sql"])
    assert response.success == 0
    assert "schema_version" in response.error


def test_lineage_file_defaults_to_the_project_lineage_dir(tmp_path):
    lineage_dir = tmp_path / "project" / "lineage"
    config = SimpleNamespace(
        current_datasource="",
        services=SimpleNamespace(datasources={}),
        path_manager=SimpleNamespace(lineage_dir=lineage_dir),
    )
    write(tmp_path, "ws/q.sql", "INSERT INTO t SELECT * FROM s")

    LineageTools(config, root_path=str(tmp_path / "ws")).upsert_lineage(paths=["q.sql"])

    assert (lineage_dir / "lineage.json").exists()


# -- delete ---------------------------------------------------------------------------------


def test_delete_matches_saved_ids_even_when_files_are_gone(tmp_path):
    write(tmp_path, "etl/a.sql", "INSERT INTO a SELECT * FROM s;")
    gone = write(tmp_path, "etl/b.sql", "INSERT INTO b SELECT * FROM s;")
    tool = make_tool(tmp_path)
    upsert(tool, paths=["etl/*.sql"])
    gone.unlink()

    result = tool.delete_lineage(sources=["etl/b.sql", str(tmp_path / "etl" / "a.sql"), "nope/*.sql"]).result

    assert sorted(result["removed"]) == ["etl/a.sql", "etl/b.sql"]
    assert result["not_found"] == ["nope/*.sql"]
    assert tool.delete_lineage(sources=[]).success == 0


# -- query ----------------------------------------------------------------------------------


def _chain(tmp_path):
    """ods.raw -> dwd -> dws -> ads, a separate pair, and a self-reading incremental table."""
    write(
        tmp_path,
        "etl.sql",
        "INSERT INTO dwd SELECT * FROM ods.raw;\n"
        "INSERT INTO dws SELECT * FROM dwd JOIN dim ON 1 = 1;\n"
        "INSERT INTO ads SELECT * FROM dws;\n"
        "INSERT INTO other_t SELECT * FROM other_s;\n"
        "INSERT INTO inc SELECT * FROM inc;\n",
    )
    tool = make_tool(tmp_path)
    upsert(tool, paths=["etl.sql"])
    return tool


def test_query_walks_the_requested_direction_and_depth(tmp_path):
    tool = _chain(tmp_path)

    up = query(tool, tables=["dws"], direction="upstream", depth=1)
    assert set(up["nodes"]) == {"dws", "dwd", "dim"}
    assert [r[0] for r in records(up)] == ["dws"]

    deep = query(tool, tables=["dws"], direction="upstream", depth=-1)
    assert set(deep["nodes"]) == {"dws", "dwd", "dim", "ods.raw"}

    down = query(tool, tables=["dwd"], direction="downstream", depth=1)
    assert set(down["nodes"]) == {"dwd", "dws"}
    # A statement lists every table it reads, even those beyond the walked depth.
    assert records(down) == [("dws", ["dim", "dwd"], "insert", [("etl.sql", 2)])]

    both = query(tool, tables=["dws"], depth=1)
    assert set(both["nodes"]) == {"dwd", "dim", "dws", "ads"}


def test_byte_identical_copies_are_cited_once_until_they_diverge(tmp_path):
    sql = "INSERT INTO t SELECT * FROM s;"
    write(tmp_path, "a.sql", sql)
    write(tmp_path, "a_1.sql", sql)
    write(tmp_path, "b.sql", "INSERT INTO t SELECT * FROM s2;")
    tool = make_tool(tmp_path)
    upsert(tool, paths=["*.sql"])

    result = query(tool, tables=["t"], direction="upstream")
    assert result["files"] == {"f1": "a.sql", "f2": "b.sql"}
    assert result["copies"] == {"f1": ["a_1.sql"]}
    assert [r[3] for r in records(result)] == [[("a.sql", 1)], [("b.sql", 1)]]

    (tmp_path / "a_1.sql").write_text(sql + "\nINSERT INTO t SELECT * FROM s3;")
    upsert(tool, paths=["*.sql"])
    diverged = query(tool, tables=["t"], direction="upstream")
    assert "copies" not in diverged
    assert sorted(diverged["files"].values()) == ["a.sql", "a_1.sql", "b.sql"]


def test_scoped_queries_summarize_components_without_listing_their_members(tmp_path):
    tool = _chain(tmp_path)
    assert query(tool, tables=["dws"])["components"] == [{"id": 1, "tables": 5}]


def test_cycles_terminate_and_self_reads_do_not_make_a_table_intermediate(tmp_path):
    tool = _chain(tmp_path)
    result = query(tool, tables=["inc"], depth=-1)
    assert result["nodes"] == {"inc": {"role": "isolated", "component": 3}}
    assert records(result) == [("inc", ["inc"], "insert", [("etl.sql", 5)])]


def test_whole_graph_lists_disconnected_subgraphs(tmp_path):
    tool = _chain(tmp_path)
    result = query(tool)
    assert result["components"] == [
        {"id": 1, "tables": 5, "roots": ["dim", "ods.raw"], "leaves": ["ads"]},
        {"id": 2, "tables": 2, "roots": ["other_s"], "leaves": ["other_t"]},
    ]
    assert result["stats"] == {"default_database": "dw", "tables": 8, "statements": 5, "revision": 1}


def test_table_names_resolve_without_guessing(tmp_path):
    write(
        tmp_path,
        "a.sql",
        "INSERT INTO dw.Orders SELECT * FROM ods.orders;\n"
        "INSERT INTO dw.shop SELECT * FROM ods.shop;\n"
        "INSERT INTO stg.item SELECT * FROM ods.item;",
    )
    tool = make_tool(tmp_path)
    upsert(tool, paths=["a.sql"])

    result = query(tool, tables=["orders", "ODS.ORDERS", "dw.s*", "item", "missing"])

    # A name without database means the default database, matched case-insensitively.
    assert result["resolved"] == {"orders": ["Orders"], "ODS.ORDERS": ["ods.orders"], "dw.s*": ["shop"]}
    # A bare name matching tables in two other databases is reported, never picked.
    assert result["ambiguous"] == {"item": ["ods.item", "stg.item"]}
    assert result["not_found"] == ["missing"]


def test_queries_are_counted_by_default_and_expanded_on_request(tmp_path):
    write(tmp_path, "etl.sql", "INSERT INTO dws SELECT * FROM dwd;")
    write(
        tmp_path,
        "q/report.sql",
        "-- Q1: monthly GMV by shop\nSELECT * FROM dws;\n-- top shops\nSELECT * FROM dws JOIN dim ON 1 = 1;\nSELECT 1 FROM dws;",
    )
    tool = make_tool(tmp_path)
    upsert(tool, paths=["**/*.sql"])

    counted = query(tool, tables=["dws"])
    assert counted["nodes"]["dws"] == {"role": "leaf", "component": 1, "queried_by": 3}
    assert all(not r["target"].startswith("query:") for r in counted["lineage"])
    assert counted["files"] == {"f1": "etl.sql"}

    expanded = query(tool, tables=["dws"], include_queries=True)
    queries = {r["target"]: r for r in expanded["lineage"] if r["target"].startswith("query:")}
    labels = {expanded["nodes"][q].get("label"): sorted(r["sources"]) for q, r in queries.items()}
    assert labels == {"Q1: monthly GMV by shop": ["dws"], "top shops": ["dim", "dws"], None: ["dws"]}
    assert {r["operation"] for r in queries.values()} == {"select"}
    # Roles describe the table graph; queries never turn a leaf into an intermediate table.
    assert expanded["nodes"]["dws"]["role"] == "leaf"


def test_stale_sources_make_the_result_incomplete_until_reanalyzed(tmp_path, monkeypatch):
    edited = write(tmp_path, "a.sql", "INSERT INTO t SELECT * FROM s;")
    removed = write(tmp_path, "b.sql", "INSERT INTO u SELECT * FROM s;")
    tool = make_tool(tmp_path)
    upsert(tool, paths=["*.sql"])
    assert query(tool)["complete"] is True

    edited.write_text("INSERT INTO t SELECT * FROM s2;")
    removed.unlink()
    result = query(tool)
    assert result["complete"] is False
    stale = {result["files"].get(s["source"], s["source"]): s["reason"] for s in result["stale"]}
    assert stale == {"a.sql": "modified", "b.sql": "deleted"}

    upsert(tool, paths=["*.sql"], prune_missing=True)
    assert query(tool)["complete"] is True

    monkeypatch.setattr(lineage_module, "ANALYZER_VERSION", lineage_module.ANALYZER_VERSION + 1)
    upgraded = query(tool)
    assert [s["reason"] for s in upgraded["stale"]] == ["analyzer_upgraded"]


def test_same_content_with_a_new_mtime_is_not_stale(tmp_path):
    path = write(tmp_path, "a.sql", "INSERT INTO t SELECT * FROM s;")
    tool = make_tool(tmp_path)
    upsert(tool, paths=["a.sql"])
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert query(tool)["complete"] is True


def test_empty_project_hints_at_upsert(tmp_path):
    result = query(make_tool(tmp_path))
    assert result["lineage"] == []
    assert "upsert_lineage" in result["hint"]


@pytest.mark.parametrize("kwargs", [{"direction": "sideways"}, {"depth": -2}])
def test_query_rejects_invalid_walks(tmp_path, kwargs):
    assert make_tool(tmp_path).query_lineage(tables=["t"], **kwargs).success == 0


def test_temporary_folding_keeps_builder_evidence(tmp_path):
    write(
        tmp_path,
        "q.sql",
        "CREATE TABLE tmp_a AS SELECT * FROM src;\n"
        "CREATE TABLE tmp_b AS SELECT * FROM tmp_a;\n"
        "INSERT INTO final SELECT * FROM tmp_b;\n"
        "DROP TABLE tmp_b; DROP TABLE tmp_a;",
    )
    tool = make_tool(tmp_path)
    upsert(tool, paths=["q.sql"])

    [row] = query(tool, tables=["final"], direction="upstream")["lineage"]

    assert (row["sources"], sorted(row["via_temp"])) == (["src"], ["tmp_a", "tmp_b"])
    assert row["evidence"][0] == "f1:3"
    assert sorted(row["evidence"][1:]) == ["f1:1", "f1:2"]
    assert not any(node.endswith("tmp_a") for node in saved(tmp_path)["nodes"])


def test_distinct_statements_for_one_target_stay_separate(tmp_path):
    write(
        tmp_path,
        "q.sql",
        "INSERT INTO t SELECT * FROM a WHERE flag=1;\n"
        "INSERT INTO t SELECT * FROM a WHERE flag=2;\n"
        "TRUNCATE TABLE t; INSERT INTO t SELECT * FROM a;",
    )
    tool = make_tool(tmp_path)
    upsert(tool, paths=["q.sql"])
    assert [(r[2], r[3]) for r in records(query(tool, tables=["t"]))] == [
        ("insert", [("q.sql", 1)]),
        ("insert", [("q.sql", 2)]),
        ("truncate_reload", [("q.sql", 3)]),
    ]


# -- SDK ------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_sdk_upserts_all_persist(tmp_path):
    names = [f"t{i}" for i in range(8)]
    for name in names:
        write(tmp_path, f"{name}.sql", f"INSERT INTO {name} SELECT * FROM s")
    tools = {tool.name: tool for tool in make_tool(tmp_path).available_tools()}

    results = await asyncio.gather(
        *[
            tools["upsert_lineage"].on_invoke_tool(None, json.dumps({"paths": [f"{name}.sql"], "dialect": "mysql"}))
            for name in names
        ]
    )

    assert [r["success"] for r in results] == [1] * len(names)
    assert set(saved(tmp_path)["sources"]) == {f"{name}.sql" for name in names}
    assert saved(tmp_path)["revision"] == len(names)
    # The SDK schema is the public invocation contract.
    assert set(tools) == {"upsert_lineage", "delete_lineage", "query_lineage"}
    assert set(tools["upsert_lineage"].params_json_schema["properties"]) == {
        "paths",
        "sql",
        "source_id",
        "datasource",
        "dialect",
        "default_database",
        "prune_missing",
    }
    assert set(tools["delete_lineage"].params_json_schema["properties"]) == {"sources"}
    assert set(tools["query_lineage"].params_json_schema["properties"]) == {
        "tables",
        "direction",
        "depth",
        "include_queries",
    }


def test_analysis_runs_once_per_changed_source(tmp_path, monkeypatch):
    write(tmp_path, "a.sql", "INSERT INTO t SELECT * FROM s;")
    write(tmp_path, "b.sql", "INSERT INTO u SELECT * FROM s;")
    tool = make_tool(tmp_path)
    upsert(tool, paths=["*.sql"])
    calls = []
    original = analysis.analyze_source

    def counting(source_id, *args, **kwargs):
        calls.append(source_id)
        return original(source_id, *args, **kwargs)

    monkeypatch.setattr(lineage_module, "analyze_source", counting)
    (tmp_path / "b.sql").write_text("INSERT INTO u SELECT * FROM s2;")
    upsert(tool, paths=["*.sql"])
    assert calls == ["b.sql"]
