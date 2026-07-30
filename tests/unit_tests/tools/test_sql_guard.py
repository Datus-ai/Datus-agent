"""Tests for the EXPLAIN-based row-count guard (datus/tools/sql_guard.py)."""

from datus.tools.db_tools import connector_registry
from datus.tools.sql_guard import (
    MAX_ESTIMATED_ROWS,
    build_oversize_message,
    estimate_rows_from_explain,
)


def test_ceiling_is_a_sane_positive_constant():
    # Guards the calibration: above legitimate multi-million-row scans, well
    # below the ~9e9 cartesian-blowup regime it must catch.
    assert 1_000_000 < MAX_ESTIMATED_ROWS < 1_000_000_000


class TestEstimateRowsFromExplain:
    def test_starrocks_takes_max_cardinality(self):
        # The exploding join node's cardinality bounds the result magnitude.
        rows = [
            {"Explain String": "  |  0:OlapScanNode  cardinality: 96478"},
            {"Explain String": "  |  4:CROSS JOIN  cardinality: 9006993124"},
            {"Explain String": "  |  1:OlapScanNode  cardinality: 93358"},
        ]
        assert estimate_rows_from_explain("starrocks", rows) == 9_006_993_124

    def test_starrocks_cardinality_equals_form(self):
        rows = [{"plan": "node cardinality=12345"}]
        assert estimate_rows_from_explain("starrocks", rows) == 12345

    def test_duckdb_estimated_cardinality(self):
        rows = [{"explain_value": "HASH_JOIN\nEC: 8000000000"}, {"explain_value": "SEQ_SCAN\nEC: 100"}]
        assert estimate_rows_from_explain("duckdb", rows) == 8_000_000_000

    def test_postgres_takes_max_rows_across_nodes(self):
        rows = [
            {"QUERY PLAN": "Nested Loop  (cost=0.00..1.00 rows=9000000000 width=8)"},
            {"QUERY PLAN": "  ->  Seq Scan on a  (cost=0.00..1.00 rows=96478 width=4)"},
        ]
        assert estimate_rows_from_explain("postgres", rows) == 9_000_000_000

    def test_postgres_aggregated_join_does_not_bypass_via_small_root(self):
        # GROUP BY/LIMIT shrinks the ROOT node's rows, but the inner join is the
        # real blowup — taking the max (not the root) keeps it enforceable.
        rows = [
            {"QUERY PLAN": "Limit  (cost=0.00..1.00 rows=100 width=8)"},
            {"QUERY PLAN": "  ->  GroupAggregate  (cost=0.00..1.00 rows=500 width=8)"},
            {"QUERY PLAN": "        ->  Nested Loop  (cost=0.00..1.00 rows=9000000000 width=8)"},
        ]
        assert estimate_rows_from_explain("postgres", rows) == 9_000_000_000

    def test_adapter_parser_dialect_keeps_postgres_guard(self, monkeypatch):
        monkeypatch.setattr(
            connector_registry,
            "get_parser_dialect",
            lambda dialect: "postgres" if dialect == "hologres" else None,
            raising=False,
        )
        rows = [{"QUERY PLAN": "Nested Loop  (cost=0.00..1.00 rows=9000000000 width=8)"}]
        assert estimate_rows_from_explain("hologres", rows) == 9_000_000_000

    def test_mysql_multiplies_per_table_rows(self):
        rows = [{"id": 1, "table": "a", "rows": 96478}, {"id": 1, "table": "b", "rows": 93358}]
        assert estimate_rows_from_explain("mysql", rows) == 96478 * 93358

    def test_mysql_rows_key_case_insensitive(self):
        rows = [{"ROWS": 1000}, {"ROWS": 2000}]
        assert estimate_rows_from_explain("mysql", rows) == 2_000_000

    def test_mysql_no_rows_column_returns_none(self):
        assert estimate_rows_from_explain("mysql", [{"id": 1, "table": "a"}]) is None

    def test_unknown_dialect_returns_none(self):
        assert estimate_rows_from_explain("sqlite", [{"detail": "SCAN t"}]) is None
        assert estimate_rows_from_explain("snowflake", [{"plan": "cardinality: 5"}]) is None

    def test_empty_explain_returns_none(self):
        assert estimate_rows_from_explain("starrocks", []) is None

    def test_no_parseable_number_returns_none(self):
        assert estimate_rows_from_explain("starrocks", [{"plan": "no numbers here"}]) is None


class TestBuildOversizeMessage:
    def test_message_carries_estimate_threshold_and_guidance(self):
        msg = build_oversize_message(9_006_993_124, 100_000_000)
        assert "9,006,993,124" in msg  # thousands-separated estimate
        assert "100,000,000" in msg  # thousands-separated ceiling
        assert "JOIN" in msg  # actionable rewrite guidance
        assert "LIMIT" in msg
