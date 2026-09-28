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
    assert set(result) == {"lineage", "roots", "joins", "rules", "comments", "unresolved", "stats"}

    lineage = {e["target"]: e for e in result["lineage"]}
    assert set(lineage) == {"dwd_orders", "dws_orders_daily", "ads_orders"}
    # Versioned copies collapse into one target listing both scripts.
    assert lineage["dwd_orders"]["scripts"] == ["etl/dwd_orders.sql", "etl/dwd_orders_v2.sql"]
    assert lineage["dwd_orders"]["load_modes"] == ["incremental"]
    assert lineage["dwd_orders"]["window"] == ["pt_date >= '${pt_date}'"]
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
    assert rules["value_mappings"] == [
        {
            "column": "ods_orders.order_type",
            "values": {"1": "MA", "2": "IMAC"},
            "occurrences": 4,
            "seen_in": ["dwd_orders"],
        }
    ]
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
    assert result["stats"]["truncated_lists"]["filters"] > 1


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
