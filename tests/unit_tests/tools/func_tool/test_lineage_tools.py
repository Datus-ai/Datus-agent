# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the ``extract_sql_lineage`` function tool."""

from types import SimpleNamespace

import pytest

from datus.tools.func_tool.lineage_tools import LineageTools

ETL_ORDERS = """-- build the order detail
insert into dw.dwd_orders
select o.order_id, o.store_code, -- store the order belongs to
       s.brand_name,
       case when o.order_type = 1 then 'MA' when o.order_type = 2 then 'IMAC' end as order_type_name,
       count(distinct o.item_id) as item_num -- each item counted once
from dw.ods_orders o
left join dw.dim_store s on o.store_code = s.store_code
where s.billing_status = 1 and s.brand_name not in ('X') and o.pt_date >= '${pt_date}'
"""

ETL_ORDERS_V2 = ETL_ORDERS.replace("-- build the order detail", "-- build the order detail (v2)")

ETL_SUMMARY = """insert into dw.dws_orders_daily
select d.store_code, count(*) as order_num
from (
    select store_code, row_number() over (partition by order_id order by update_time desc) as rn
    from dw.dwd_orders
) d
join dw.dim_store s on d.store_code = s.store_code
where d.rn = 1 and s.billing_status = 1 and s.brand_name not in ('X')
group by d.store_code
"""

DAG = '''
def build():
    run("""
    insert into dw.ads_orders select * from dw.dws_orders_daily
    """)
'''


@pytest.fixture
def workspace(tmp_path):
    etl = tmp_path / "etl"
    etl.mkdir()
    (etl / "dwd_orders.sql").write_text(ETL_ORDERS, encoding="utf-8")
    (etl / "dwd_orders_v2.sql").write_text(ETL_ORDERS_V2, encoding="utf-8")
    (etl / "dws_orders_daily.sql").write_text(ETL_SUMMARY, encoding="utf-8")
    (tmp_path / "dags").mkdir()
    (tmp_path / "dags" / "ads.py").write_text(DAG, encoding="utf-8")
    return tmp_path


def make_tool(root, datasources=None, current=""):
    config = SimpleNamespace(
        current_datasource=current,
        services=SimpleNamespace(datasources=datasources or {}),
    )
    return LineageTools(config, root_path=str(root))


def extract(root, **kwargs):
    kwargs.setdefault("dialect", "mysql")
    kwargs.setdefault("default_database", "dw")
    result = make_tool(root).extract_sql_lineage(**kwargs)
    assert result.success == 1, result.error
    return result.result


def test_full_result_contract(workspace):
    result = extract(workspace, paths=["etl/**/*.sql", "dags/*.py"])
    assert set(result) == {
        "schema_version",
        "lineage",
        "roots",
        "tables",
        "joins",
        "rules",
        "conditions",
        "comments",
        "unresolved",
        "stats",
        "pagination",
    }

    lineage = {e["target"]: e for e in result["lineage"]}
    assert set(lineage) == {"dwd_orders", "dws_orders_daily", "ads_orders"}
    # Versioned copies collapse into one target listing both scripts.
    assert lineage["dwd_orders"]["scripts"] == ["etl/dwd_orders.sql", "etl/dwd_orders_v2.sql"]
    assert lineage["dwd_orders"]["load_modes"] == ["insert"]
    assert lineage["dwd_orders"]["parameterized_predicates"] == ["pt_date >= '${pt_date}'"]
    assert lineage["ads_orders"]["scripts"] == ["dags/ads.py"]
    assert result["roots"] == ["dim_store", "ods_orders"]

    joins = {tuple(j["tables"]): j for j in result["joins"]}
    assert joins[("dim_store", "ods_orders")]["on"] == ["store_code"]
    assert joins[("dim_store", "ods_orders")]["occurrences"] == 2

    stats = result["stats"]
    assert stats["files_scanned"] == 4
    assert stats["statements"] == stats["parsed"] == 4
    # dwd_orders x2 (target + 2 sources), dws (target + 2 sources), ads (target + 1 source)
    assert stats["databases_referenced"] == {"dw": 11}


def test_rules_are_ranked_by_how_many_files_use_them(workspace):
    rules = extract(workspace, paths=["etl/*.sql"], sections=["rules"])["rules"]
    top = rules["filters"][:2]
    assert {(f["column"], f["predicate"], f["files"]) for f in top} == {
        ("dim_store.billing_status", "= 1", 3),
        ("dim_store.brand_name", "NOT IN ('X')", 3),
    }
    [mapping] = rules["value_mappings"]
    assert mapping["column"] == "ods_orders.order_type"
    assert mapping["values"] == {"1": "MA", "2": "IMAC"}
    assert mapping["occurrences"] == 4
    assert mapping["partial"] is True
    assert mapping["distinct_statements"] == 1
    assert rules["dedup"][0]["table"] == "dwd_orders"
    assert rules["dedup"][0]["partition_by"] == ["order_id"]
    assert rules["dedup"][0]["order_by"] == ["update_time DESC"]


def test_comments_split_into_labels_metrics_and_notes_and_dedupe_copies(workspace):
    comments = extract(workspace, paths=["etl/*.sql"], sections=["comments"])["comments"]
    assert comments["column_labels"] == {"store_code": "store the order belongs to"}
    [metric] = comments["metric_notes"]
    assert metric["column"] == "item_num"
    assert metric["label"] == "each item counted once"
    assert metric["copies"] == 2
    # The opening comment of each script is its header, kept whole and deduplicated across copies.
    assert [h["text"] for h in comments["file_headers"]] == ["build the order detail", "build the order detail (v2)"]
    assert comments["notes"] == []


def test_sections_filter_output_and_reject_unknown_names(workspace):
    result = extract(workspace, paths=["etl/*.sql"], sections=["joins"])
    assert "rules" not in result and "comments" not in result and "joins" in result
    bad = make_tool(workspace).extract_sql_lineage(paths=["etl/*.sql"], sections=["lineage", "bogus"])
    assert bad.success == 0
    assert "bogus" in bad.error


def test_max_items_truncation_is_reported(workspace):
    result = extract(workspace, paths=["etl/*.sql"], sections=["rules"], max_items=1)
    assert len(result["rules"]["filters"]) == 1
    assert result["stats"]["truncated_lists"]["rules.filters"] > 1


def test_paths_outside_workspace_are_skipped_not_read(workspace, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.sql"
    outside.write_text("insert into dw.t select * from dw.s", encoding="utf-8")
    result = extract(workspace, paths=[str(outside), "missing/*.sql"])
    assert result["lineage"] == []
    reasons = sorted(u["reason"] for u in result["unresolved"])
    assert reasons == ["no files matched", "outside the readable workspace"]


def test_empty_paths_is_an_error(workspace):
    result = make_tool(workspace).extract_sql_lineage(paths=[])
    assert result.success == 0


def test_datasource_supplies_dialect_and_default_database(workspace):
    datasources = {"wh": SimpleNamespace(type="mysql", database="dw")}
    tool = make_tool(workspace, datasources=datasources, current="wh")
    result = tool.extract_sql_lineage(paths=["etl/dws_orders_daily.sql"], sections=[])
    assert result.success == 1
    assert result.result["stats"]["default_database"] == "dw"
    assert result.result["stats"]["dialect"] == "mysql"
    assert result.result["lineage"][0]["target"] == "dws_orders_daily"


def test_unknown_datasource_is_an_error(workspace):
    result = make_tool(workspace).extract_sql_lineage(paths=["etl/*.sql"], datasource="nope")
    assert result.success == 0
    assert "nope" in result.error


def test_explicit_dialect_overrides_datasource(workspace):
    datasources = {"wh": SimpleNamespace(type="sqlite", database="")}
    tool = make_tool(workspace, datasources=datasources, current="wh")
    result = tool.extract_sql_lineage(paths=["etl/*.sql"], dialect="starrocks", sections=[])
    assert result.result["stats"]["dialect"] == "starrocks"


def test_available_tools_exposes_one_function(workspace):
    tools = make_tool(workspace).available_tools()
    assert [t.name for t in tools] == ["extract_sql_lineage"]
    assert LineageTools.all_tools_name() == ["extract_sql_lineage"]


def test_query_only_corpus_reports_read_tables(tmp_path):
    """Read inventory and database statistics must include SELECT-only corpora."""
    (tmp_path / "q.sql").write_text("SELECT * FROM other.orders")
    result = extract(tmp_path, paths=["*.sql"])
    assert result["roots"] == ["other.orders"]
    assert result["stats"]["tables_read_only"] == 1
    assert result["stats"]["databases_referenced"] == {"other": 1}
    assert result["tables"][0]["table"] == "other.orders"


def test_rule_frequency_deduplicates_copies_and_exposes_denominator(tmp_path):
    """Copied SQL must not inflate independent support for a business rule."""
    for name, sql in {
        "a": "SELECT * FROM s WHERE flag = 1",
        "b": "-- copy\nSELECT * FROM s WHERE flag = 1",
        "c": "SELECT * FROM s",
    }.items():
        (tmp_path / f"{name}.sql").write_text(sql)
    result = extract(tmp_path, paths=["*.sql"], sections=["rules"])
    [rule] = result["rules"]["filters"]
    assert (rule["files"], rule["distinct_statements"], rule["table_read_statements"]) == (2, 1, 2)
    assert rule["evidence_total"] == 2
    assert {e["file"] for e in rule["evidence"]} == {"a.sql", "b.sql"}


@pytest.mark.parametrize("max_files", [0, -1])
def test_invalid_scan_limit_is_rejected(tmp_path, max_files):
    """Invalid limits must not silently select an arbitrary file subset."""
    result = make_tool(tmp_path).extract_sql_lineage(paths=["*.sql"], max_files=max_files)
    assert result.success == 0
    assert "max_files" in result.error


def test_pagination_recovers_all_query_tables(tmp_path):
    """Every omitted table must remain accessible by the returned offset."""
    (tmp_path / "q.sql").write_text("SELECT * FROM a; SELECT * FROM b; SELECT * FROM c")
    first = extract(tmp_path, paths=["*.sql"], sections=[], max_items=2)
    assert first["pagination"]["tables"]["total"] == 3
    second = extract(
        tmp_path,
        paths=["*.sql"],
        sections=[],
        max_items=2,
        offset=first["pagination"]["tables"]["next_offset"],
        result_path="tables",
    )
    assert [t["table"] for t in first["tables"] + second["tables"]] == ["a", "b", "c"]


def test_output_budget_provides_resumable_results(tmp_path):
    """A large header corpus must respect the budget without losing continuation."""
    import json

    for i in range(8):
        (tmp_path / f"{i}.sql").write_text("-- " + str(i) + "x" * 1500 + "\nSELECT * FROM s")
    result = extract(tmp_path, paths=["*.sql"], sections=["comments"], max_output_chars=6000)
    assert len(json.dumps(result, ensure_ascii=False)) <= 6000
    assert result["pagination"]["comments.file_headers"]["next_offset"] == len(result["comments"]["file_headers"])


def test_transform_variants_do_not_collapse(tmp_path):
    """Identical physical columns with different transformations remain distinct relationships."""
    (tmp_path / "q.sql").write_text("SELECT * FROM a JOIN b ON a.id=b.id; SELECT * FROM a JOIN b ON LEFT(a.id,6)=b.id")
    result = extract(tmp_path, paths=["*.sql"], sections=["joins"])
    assert len(result["joins"]) == 2
    assert [r.get("transforms", {}) for r in result["joins"]] == [{}, {"a.id": "LEFT(a.id, 6)"}]


def test_full_evidence_can_be_retrieved_after_summary_sampling(tmp_path):
    """Evidence sampling has an explicit detail path that recovers every occurrence."""
    for i in range(5):
        (tmp_path / f"{i}.sql").write_text("SELECT * FROM s WHERE flag=1")
    summary = extract(tmp_path, paths=["*.sql"], sections=["rules"])
    [rule] = summary["rules"]["filters"]
    assert rule["evidence_total"] == 5
    assert len(rule["evidence"]) == 3
    detail = extract(tmp_path, paths=["*.sql"], sections=["rules"], result_path="filter_occurrences")
    assert {r["file"] for r in detail["filter_occurrences"]} == {f"{i}.sql" for i in range(5)}
    assert {r["predicate"] for r in detail["filter_occurrences"]} == {"= 1"}


def test_budgeted_header_pages_recover_every_header(tmp_path):
    """Pagination must progress and recover all headers within the same character budget."""
    import json

    for i in range(9):
        (tmp_path / f"{i}.sql").write_text("-- " + str(i) + "x" * 1800 + "\nSELECT * FROM s")
    collected = []
    offset = 0
    for _ in range(9):
        page = extract(
            tmp_path,
            paths=["*.sql"],
            sections=["comments"],
            result_path="comments.file_headers",
            offset=offset,
            max_output_chars=6000,
        )
        assert len(json.dumps(page, ensure_ascii=False)) <= 6000
        collected.extend(h["file"] for h in page["comments"]["file_headers"])
        next_offset = page["pagination"]["comments.file_headers"]["next_offset"]
        if next_offset is None:
            break
        assert next_offset > offset
        offset = next_offset
    assert collected == [f"{i}.sql" for i in range(9)]


def test_workspace_symlink_does_not_bypass_path_policy(tmp_path, tmp_path_factory):
    """Resolving a symlink must not turn an external file into workspace data."""
    outside = tmp_path_factory.mktemp("external") / "q.sql"
    outside.write_text("SELECT * FROM secret")
    (tmp_path / "q.sql").symlink_to(outside)
    result = extract(tmp_path, paths=["*.sql"])
    assert result["tables"] == []
    assert result["unresolved"][0]["reason"] == "outside the readable workspace"


def test_hidden_config_sql_is_not_read(tmp_path):
    """Explicit paths must obey the same hidden-directory rule as globs."""
    (tmp_path / ".datus").mkdir()
    (tmp_path / ".datus" / "q.sql").write_text("SELECT * FROM private_config")
    result = extract(tmp_path, paths=[".datus/q.sql"])
    assert result["tables"] == []
    assert result["unresolved"][0]["reason"] == "outside the readable workspace"


def test_empty_python_extraction_is_visible(tmp_path):
    """Dynamic Python SQL cannot silently look like a successfully analyzed empty file."""
    (tmp_path / "q.py").write_text("query = prefix + table")
    result = extract(tmp_path, paths=["*.py"])
    assert result["unresolved"][0]["reason"] == "no supported literal SQL fragments"


def test_scan_limit_stops_with_explicit_truncation(tmp_path):
    """Scanning a subset must not be reported as complete corpus coverage."""
    for i in range(4):
        (tmp_path / f"{i}.sql").write_text("SELECT * FROM s")
    result = extract(tmp_path, paths=["*.sql"], max_files=2)
    assert result["stats"]["files_scanned"] == 2
    assert result["stats"]["files_truncated"] is True


def test_unknown_detail_collection_returns_actionable_error(tmp_path):
    """Invalid detail paths must report valid alternatives in the tool envelope."""
    result = make_tool(tmp_path).extract_sql_lineage(paths=["*.sql"], result_path="bad")
    assert result.success == 0
    assert "Unknown result_path: bad" in result.error


def test_lineage_detail_preserves_multiple_statements_on_same_line(tmp_path):
    """Statement IDs disambiguate source evidence even when start lines are equal."""
    (tmp_path / "q.sql").write_text("INSERT INTO x SELECT * FROM a; INSERT INTO y SELECT * FROM b")
    result = extract(tmp_path, paths=["*.sql"])
    assert [e["evidence"][0]["statement_id"] for e in result["lineage"]] == ["q.sql:0:1", "q.sql:0:2"]


def test_cached_corpus_is_invalidated_by_content_not_mtime(tmp_path):
    """Same-size edits with a restored mtime must not serve stale table facts."""
    import os

    path = tmp_path / "q.sql"
    path.write_text("SELECT * FROM a")
    stat = path.stat()
    tool = make_tool(tmp_path)
    first = tool.extract_sql_lineage(paths=["*.sql"], dialect="mysql")
    assert first.result["roots"] == ["a"]
    path.write_text("SELECT * FROM b")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = tool.extract_sql_lineage(paths=["*.sql"], dialect="mysql")
    assert second.result["roots"] == ["b"]
    detail = tool.extract_sql_lineage(paths=["*.sql"], dialect="mysql", result_path="statements")
    assert detail.result["statements"][0]["read_tables"] == ["b"]


def test_statement_ids_survive_narrowing_paths(tmp_path):
    """A source reference identifies the same statement in corpus and file views."""
    (tmp_path / "a.sql").write_text("SELECT * FROM a")
    (tmp_path / "b.sql").write_text("SELECT * FROM b")
    full = extract(tmp_path, paths=["*.sql"], result_path="statements")
    detail = extract(tmp_path, paths=["b.sql"], result_path="statements")
    assert full["statements"][1]["statement_id"] == detail["statements"][0]["statement_id"]


@pytest.mark.asyncio
async def test_concurrent_sdk_calls_keep_each_corpus_isolated(tmp_path):
    """The registered tool can analyze independent corpora concurrently without cache cross-talk."""
    import asyncio
    import json

    (tmp_path / "a.sql").write_text("SELECT * FROM a")
    (tmp_path / "b.sql").write_text("SELECT * FROM b")
    [tool] = make_tool(tmp_path).available_tools()
    names = ["a", "b"] * 8
    results = await asyncio.gather(
        *[
            tool.on_invoke_tool(None, json.dumps({"paths": [f"{name}.sql"], "dialect": "mysql", "sections": []}))
            for name in names
        ]
    )
    assert [r["success"] for r in results] == [1] * len(names)
    assert [r["result"]["roots"] for r in results] == [[name] for name in names]


def test_oversized_single_rule_returns_source_reference(tmp_path):
    """An indivisible record must make progress and point back to its complete source."""
    import json

    values = ",".join(str(i) for i in range(2000))
    (tmp_path / "q.sql").write_text(f"SELECT * FROM s WHERE id IN ({values})")
    result = extract(tmp_path, paths=["*.sql"], sections=["rules"], result_path="rules.filters", max_output_chars=6000)
    [item] = result["rules"]["filters"]
    assert item["detail_omitted"] is True
    assert item["evidence"] == [{"file": "q.sql", "line": 1}]
    assert result["pagination"]["rules.filters"]["next_offset"] is None
    assert len(json.dumps(result, ensure_ascii=False)) <= 6000
