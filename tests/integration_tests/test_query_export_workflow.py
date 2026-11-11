"""Synthetic 114k-ID workflow: Excel -> SQL files -> exports -> reconciliation.

Uses an owned DuckDB file, without external credentials or business data.
StarRocks driver/server acceptance remains a separate integration boundary.
"""

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from datus.tools.db_tools.config import DuckDBConfig
from datus.tools.db_tools.duckdb_connector import DuckdbConnector
from datus.tools.func_tool.database import DBFuncTool

pytestmark = pytest.mark.integration


def test_excel_batches_export_and_reconcile_114444_ids(tmp_path):
    count = 114444
    ids = [f"{i:019d}" for i in range(count)]
    input_path = tmp_path / "orders.xlsx"
    pd.DataFrame({"orderid": ids + ids[:2]}).to_excel(input_path, index=False, engine="xlsxwriter")
    input_ids = pd.read_excel(input_path, dtype={"orderid": "string"})["orderid"]
    assert input_ids.tolist() == ids + ids[:2]
    keys = input_ids.drop_duplicates().tolist()
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    manifest = []
    for start in range(0, len(keys), 5000):
        index = len(manifest) + 1
        literals = ",".join("'" + value + "'" for value in keys[start : start + 5000])
        sql = f"SELECT resultorderid, sessionid, creator, createtime FROM order_log WHERE resultorderid IN ({literals})"
        sql_path = sql_dir / f"batch_{index:03d}.sql"
        sql_path.write_text(sql)
        manifest.append(
            {
                "sql": str(sql_path.relative_to(tmp_path)),
                "file": f"exports/order_match/parts/{index:03d}.parquet",
                "sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
            }
        )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert len(manifest) == 23

    connector = DuckdbConnector(DuckDBConfig(db_path=str(tmp_path / "warehouse.duckdb")))
    try:
        created = connector.execute_ddl(
            "CREATE TABLE order_log AS "
            "SELECT lpad(i::VARCHAR,19,'0') AS resultorderid, 's' || i AS sessionid, "
            "'creator' AS creator, TIMESTAMP '2026-01-01' AS createtime "
            "FROM range(114444) t(i) WHERE i % 2 = 0 "
            "UNION ALL SELECT lpad(i::VARCHAR,19,'0'), 'second-' || i, 'creator', TIMESTAMP '2026-01-02' "
            "FROM range(114444) t(i) WHERE i % 100 = 0 "
            "UNION ALL SELECT lpad(i::VARCHAR,19,'0'), 'outside', 'creator', TIMESTAMP '2026-01-01' "
            "FROM range(114444,114454) t(i)"
        )
        assert created.success, created.error
        tool = DBFuncTool(connector, filesystem_root=str(tmp_path))
        tool.agent_config = SimpleNamespace(bulk_query={"batch_rows": 512}, policy_context={}, sql_read_only=True)
        wire_bytes = []
        for batch in manifest:
            result = tool.export_query(batch["sql"], output_file=batch["file"])
            assert result.success == 1, result.error
            wire_bytes.append(len(json.dumps(result.model_dump()).encode()))
            assert wire_bytes[-1] <= 16384

        frames = []
        exported_rows = 0
        for batch in json.loads((tmp_path / "manifest.json").read_text()):
            path = tmp_path / batch["file"]
            metadata = json.loads(path.with_name(path.name + ".meta.json").read_text())
            assert metadata["complete"] is True
            assert metadata["sql_sha256"] == batch["sql_sha256"]
            frame = pd.read_parquet(path)
            assert len(frame) == metadata["row_count"]
            frames.append(frame)
            exported_rows += metadata["row_count"]
        merged = pd.concat(frames, ignore_index=True)
        matched = set(merged["resultorderid"])
        unmatched = set(keys) - matched
        assert matched == {f"{i:019d}" for i in range(0, count, 2)}
        assert matched <= set(keys)
        assert len(matched) + len(unmatched) == len(keys)
        assert len(merged) == exported_rows == 58367
        assert merged["resultorderid"].duplicated().sum() == 1145
        final_path = tmp_path / "exports/order_match/matched_orders.xlsx"
        merged.to_excel(final_path, index=False, engine="xlsxwriter")
        roundtrip = pd.read_excel(final_path, dtype={"resultorderid": "string"})
        assert roundtrip["resultorderid"].tolist() == merged["resultorderid"].tolist()
        assert final_path.stat().st_size < 50 * 1024 * 1024
        summary = {
            "input_rows": len(input_ids),
            "unique_ids": len(keys),
            "batch_count": len(manifest),
            "matched_ids": len(matched),
            "unmatched_ids": len(unmatched),
            "result_rows": len(merged),
            "max_tool_output_bytes": max(wire_bytes),
            "final_file_bytes": final_path.stat().st_size,
            "complete": True,
            "dataset": "synthetic",
        }
        (tmp_path / "summary.json").write_text(json.dumps(summary))
    finally:
        connector.close()
