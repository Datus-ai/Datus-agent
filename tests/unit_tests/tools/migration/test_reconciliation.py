"""Tests for reconciliation check generation (datus/tools/migration/reconciliation.py)."""

import pytest

from datus.tools.migration.reconciliation import build_reconciliation_checks


class TestBuildReconciliationChecks:
    """Test that all 7 check types are generated correctly."""

    @pytest.fixture
    def sample_columns(self):
        return [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "name", "type": "VARCHAR", "nullable": True},
            {"name": "amount", "type": "DECIMAL(10,2)", "nullable": True},
            {"name": "created_at", "type": "DATE", "nullable": True},
            {"name": "is_active", "type": "BOOLEAN", "nullable": True},
        ]

    def test_returns_list_of_checks(self, sample_columns):
        checks = build_reconciliation_checks(
            source_table="src.users",
            target_table="tgt.users",
            columns=sample_columns,
            key_columns=["id"],
        )
        assert isinstance(checks, list)
        assert len(checks) == 7

    def test_all_checks_have_required_fields(self, sample_columns):
        checks = build_reconciliation_checks(
            source_table="src.users",
            target_table="tgt.users",
            columns=sample_columns,
            key_columns=["id"],
        )
        assert len(checks) > 0
        for check in checks:
            assert "name" in check
            assert "source_query" in check
            assert "target_query" in check
            assert isinstance(check["source_query"], str)
            assert len(check["source_query"]) > 0
            assert isinstance(check["target_query"], str)
            assert len(check["target_query"]) > 0

    def test_check_names_present(self, sample_columns):
        checks = build_reconciliation_checks(
            source_table="src.users",
            target_table="tgt.users",
            columns=sample_columns,
            key_columns=["id"],
        )
        check_names = [c["name"] for c in checks]
        assert "row_count" in check_names
        assert "null_ratio" in check_names
        assert "min_max" in check_names
        assert "distinct_count" in check_names
        assert "duplicate_key" in check_names
        assert "sample_diff" in check_names
        assert "numeric_aggregate" in check_names


class TestIndividualCheckSQL:
    """Test SQL correctness for each individual check type."""

    def test_row_count_sql(self):
        checks = build_reconciliation_checks(
            source_table="src.t",
            target_table="tgt.t",
            columns=[{"name": "id", "type": "INTEGER", "nullable": False}],
            key_columns=["id"],
        )
        row_count = next(c for c in checks if c["name"] == "row_count")
        assert "COUNT(*)" in row_count["source_query"].upper()
        assert "src.t" in row_count["source_query"]
        assert "COUNT(*)" in row_count["target_query"].upper()
        assert "tgt.t" in row_count["target_query"]
        assert "SELECT" in row_count["source_query"].upper()
        assert "FROM" in row_count["source_query"].upper()

    def test_null_ratio_for_nullable_columns(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "name", "type": "VARCHAR", "nullable": True},
        ]
        checks = build_reconciliation_checks(
            source_table="src.t",
            target_table="tgt.t",
            columns=columns,
            key_columns=["id"],
        )
        null_checks = [c for c in checks if c["name"] == "null_ratio"]
        assert len(null_checks) > 0
        sql = null_checks[0]["source_query"].upper()
        assert "COUNT" in sql
        assert "NULL" in sql
        assert "name" in null_checks[0]["source_query"]
        # non-nullable column "id" should NOT appear in null ratio query
        assert '"id"' not in null_checks[0]["source_query"]

    def test_min_max_for_numeric_columns(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "amount", "type": "DECIMAL(10,2)", "nullable": True},
            {"name": "name", "type": "VARCHAR", "nullable": True},
        ]
        checks = build_reconciliation_checks(
            source_table="src.t",
            target_table="tgt.t",
            columns=columns,
            key_columns=["id"],
        )
        min_max_checks = [c for c in checks if c["name"] == "min_max"]
        assert len(min_max_checks) > 0
        sql = min_max_checks[0]["source_query"].upper()
        assert "MIN" in sql
        assert "MAX" in sql

    def test_min_max_includes_date_columns(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "created_at", "type": "DATE", "nullable": True},
        ]
        checks = build_reconciliation_checks(
            source_table="src.t",
            target_table="tgt.t",
            columns=columns,
            key_columns=["id"],
        )
        min_max_checks = [c for c in checks if c["name"] == "min_max"]
        assert len(min_max_checks) > 0
        sql = min_max_checks[0]["source_query"].upper()
        assert "MIN" in sql
        assert "MAX" in sql
        assert "created_at" in min_max_checks[0]["source_query"]

    def test_aggregate_for_numeric_columns(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "amount", "type": "DECIMAL(10,2)", "nullable": True},
            {"name": "name", "type": "VARCHAR", "nullable": True},
        ]
        checks = build_reconciliation_checks(
            source_table="src.t",
            target_table="tgt.t",
            columns=columns,
            key_columns=["id"],
        )
        agg_checks = [c for c in checks if c["name"] == "numeric_aggregate"]
        assert len(agg_checks) > 0
        sql = agg_checks[0]["source_query"].upper()
        assert "SUM" in sql
        assert "AVG" in sql

    def test_duplicate_key_sql(self):
        columns = [{"name": "id", "type": "INTEGER", "nullable": False}]
        checks = build_reconciliation_checks(
            source_table="src.t",
            target_table="tgt.t",
            columns=columns,
            key_columns=["id"],
        )
        dup_check = next(c for c in checks if c["name"] == "duplicate_key")
        sql = dup_check["target_query"].upper()
        assert "GROUP BY" in sql
        assert "HAVING" in sql
        assert "COUNT" in sql

    def test_sample_diff_sql(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "name", "type": "VARCHAR", "nullable": True},
        ]
        checks = build_reconciliation_checks(
            source_table="src.t",
            target_table="tgt.t",
            columns=columns,
            key_columns=["id"],
        )
        sample_check = next(c for c in checks if c["name"] == "sample_diff")
        assert "ORDER BY" in sample_check["source_query"].upper()
        assert "LIMIT" in sample_check["source_query"].upper()


class TestCheckSkipping:
    """Test that checks are skipped when required columns are absent."""

    def test_skips_key_dependent_checks(self):
        columns = [
            {"name": "name", "type": "VARCHAR", "nullable": True},
            {"name": "amount", "type": "DECIMAL", "nullable": True},
        ]
        checks = build_reconciliation_checks(
            source_table="src.t",
            target_table="tgt.t",
            columns=columns,
            key_columns=None,
        )
        check_names = [c["name"] for c in checks]
        # row_count and numeric_aggregate should always be present
        assert "row_count" in check_names
        assert "numeric_aggregate" in check_names
        # key-dependent checks should be skipped
        assert "duplicate_key" not in check_names
        assert "sample_diff" not in check_names

    def test_empty_key_columns_same_as_none(self):
        columns = [
            {"name": "name", "type": "VARCHAR", "nullable": True},
            {"name": "amount", "type": "DECIMAL", "nullable": True},
        ]
        checks_none = build_reconciliation_checks("s", "t", columns, key_columns=None)
        checks_empty = build_reconciliation_checks("s", "t", columns, key_columns=[])
        assert [c["name"] for c in checks_none] == [c["name"] for c in checks_empty]

    def test_all_non_nullable_skips_null_ratio(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "code", "type": "VARCHAR", "nullable": False},
            {"name": "amount", "type": "DECIMAL(10,2)", "nullable": False},
        ]
        checks = build_reconciliation_checks(
            source_table="src.t",
            target_table="tgt.t",
            columns=columns,
            key_columns=["id"],
        )
        check_names = [c["name"] for c in checks]
        assert "null_ratio" not in check_names

    def test_no_numeric_columns_skips_aggregates(self):
        columns = [
            {"name": "id", "type": "VARCHAR", "nullable": False},
            {"name": "label", "type": "VARCHAR", "nullable": True},
            {"name": "active", "type": "BOOLEAN", "nullable": True},
        ]
        checks = build_reconciliation_checks(
            source_table="src.t",
            target_table="tgt.t",
            columns=columns,
            key_columns=["id"],
        )
        check_names = [c["name"] for c in checks]
        assert "numeric_aggregate" not in check_names
        assert "min_max" not in check_names


class TestIdentifierQuoting:
    """Test that column identifiers are quoted in generated SQL."""

    def test_column_names_are_quoted(self):
        columns = [
            {"name": "order", "type": "INTEGER", "nullable": False},
            {"name": "group", "type": "VARCHAR", "nullable": True},
        ]
        checks = build_reconciliation_checks(
            source_table="src.t",
            target_table="tgt.t",
            columns=columns,
            key_columns=["order"],
        )
        null_check = next(c for c in checks if c["name"] == "null_ratio")
        assert '"group"' in null_check["source_query"]

        distinct_check = next(c for c in checks if c["name"] == "distinct_count")
        assert '"order"' in distinct_check["source_query"]

    def test_composite_key_distinct_count_uses_subquery(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "region", "type": "VARCHAR", "nullable": False},
            {"name": "amount", "type": "DECIMAL", "nullable": True},
        ]
        checks = build_reconciliation_checks(
            source_table="src.t",
            target_table="tgt.t",
            columns=columns,
            key_columns=["id", "region"],
        )
        distinct_check = next(c for c in checks if c["name"] == "distinct_count")
        sql = distinct_check["source_query"].upper()
        # Composite key should use SELECT DISTINCT subquery, not per-column COUNT(DISTINCT)
        assert "SELECT DISTINCT" in sql
        assert "COUNT(*)" in sql
        assert '"id"' in distinct_check["source_query"]
        assert '"region"' in distinct_check["source_query"]

    def test_composite_key_duplicate_check(self):
        columns = [
            {"name": "id", "type": "INTEGER", "nullable": False},
            {"name": "region", "type": "VARCHAR", "nullable": False},
            {"name": "amount", "type": "DECIMAL", "nullable": True},
        ]
        checks = build_reconciliation_checks(
            source_table="src.t",
            target_table="tgt.t",
            columns=columns,
            key_columns=["id", "region"],
        )
        dup_check = next(c for c in checks if c["name"] == "duplicate_key")
        sql = dup_check["target_query"]
        assert '"id"' in sql
        assert '"region"' in sql
        assert "GROUP BY" in sql.upper()
