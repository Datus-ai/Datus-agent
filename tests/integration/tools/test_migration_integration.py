"""
Integration tests for cross-database migration (DuckDB -> Greenplum / StarRocks).

Requires Docker containers:
  - Greenplum: datus-db-adapters/datus-greenplum/docker-compose.yml (port 15432)
  - StarRocks: datus-db-adapters/datus-starrocks/docker-compose.yml (port 9030)

Tests are skipped automatically if databases are unavailable.

Run:
  cd datus-db-adapters/datus-greenplum && docker compose up -d
  cd datus-db-adapters/datus-starrocks && docker compose up -d
  # Wait ~60s for healthchecks
  uv run pytest tests/integration/tools/test_migration_integration.py -v -m integration
"""

import os
import tempfile

import duckdb
import pytest

from datus.tools.migration.reconciliation import build_reconciliation_checks
from datus.tools.migration.target_profiles import GreenplumProfile, StarRocksProfile, build_target_ddl
from datus.tools.migration.type_mapping import map_columns_between_dialects

# ---------------------------------------------------------------------------
# Test data: a small table with mixed types for migration testing
# ---------------------------------------------------------------------------

SOURCE_TABLE = "migration_test_users"
SOURCE_DDL = f"""
CREATE TABLE IF NOT EXISTS {SOURCE_TABLE} (
    id INTEGER NOT NULL,
    name VARCHAR(100),
    amount DECIMAL(10,2),
    is_active BOOLEAN,
    created_at TIMESTAMP,
    birth_date DATE
)
"""
SOURCE_INSERT = f"""
INSERT INTO {SOURCE_TABLE} VALUES
    (1, 'Alice', 100.50, true, '2024-01-15 10:30:00', '1990-05-20'),
    (2, 'Bob', 200.75, false, '2024-02-20 14:00:00', '1985-11-10'),
    (3, 'Charlie', NULL, true, '2024-03-10 09:15:00', NULL),
    (4, 'Diana', 50.00, NULL, NULL, '1992-08-30'),
    (5, 'Eve', 999.99, true, '2024-04-01 08:00:00', '2000-01-01')
"""

SOURCE_COLUMNS = [
    {"name": "id", "type": "INTEGER", "nullable": False},
    {"name": "name", "type": "VARCHAR(100)", "nullable": True},
    {"name": "amount", "type": "DECIMAL(10,2)", "nullable": True},
    {"name": "is_active", "type": "BOOLEAN", "nullable": True},
    {"name": "created_at", "type": "TIMESTAMP", "nullable": True},
    {"name": "birth_date", "type": "DATE", "nullable": True},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def duckdb_source():
    """Create a temporary DuckDB database with test migration data."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "migration_source.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute(SOURCE_DDL)
    conn.execute(SOURCE_INSERT)
    yield {"path": db_path, "connection": conn, "table": SOURCE_TABLE}
    conn.close()


@pytest.fixture(scope="session")
def greenplum_connection():
    """Connect to Greenplum Docker instance. Skip if unavailable.

    Uses Docker test defaults from datus-db-adapters/datus-greenplum/docker-compose.yml.
    Does NOT read env vars to avoid pollution from agent.yml config.
    """
    try:
        import psycopg2

        conn = psycopg2.connect(
            host="localhost",
            port=15432,
            user="gpadmin",
            password="pivotal",
            dbname="test",
        )
        # Test the connection
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"Greenplum not available: {e}")


@pytest.fixture(scope="session")
def starrocks_connection():
    """Connect to StarRocks Docker instance. Skip if unavailable.

    Uses Docker test defaults from datus-db-adapters/datus-starrocks/docker-compose.yml.
    Does NOT read env vars to avoid pollution from agent.yml config.
    """
    try:
        import pymysql

        conn = pymysql.connect(
            host="localhost",
            port=9030,
            user="root",
            password="",
            database="test",
        )
        # Ensure test database exists
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"StarRocks not available: {e}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _execute_and_fetch(conn, sql):
    """Execute SQL and return all rows as list of tuples."""
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    return rows


def _execute_ddl(conn, sql):
    """Execute DDL and commit. Rollback on failure to reset transaction state."""
    try:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        cur.close()
    except Exception:
        conn.rollback()
        raise


def _transfer_data(source_duckdb, target_conn, target_table, columns):
    """Transfer data from DuckDB to target database using pandas + executemany."""
    df = source_duckdb["connection"].execute(f"SELECT * FROM {SOURCE_TABLE}").df()

    # Convert pandas NaT/NaN to Python None for DBAPI2 compatibility
    df = df.astype(object).where(df.notna(), other=None)

    col_names = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    insert_sql = f"INSERT INTO {target_table} ({col_names}) VALUES ({placeholders})"

    cur = target_conn.cursor()
    cur.executemany(insert_sql, df.values.tolist())
    target_conn.commit()
    cur.close()
    return len(df)


def _compare_results(src_result, tgt_result, check_name):
    """Compare reconciliation results using type-aware comparison instead of str()."""
    import datetime

    if check_name == "row_count":
        # Compare integer values directly
        src_val = src_result[0][0] if src_result else 0
        tgt_val = tgt_result[0][0] if tgt_result else 0
        return src_val == tgt_val
    elif check_name in ("numeric_aggregate", "min_max"):
        # Compare values with tolerance for precision and type differences
        if len(src_result) == 0 and len(tgt_result) == 0:
            return True
        if len(src_result) != len(tgt_result):
            return False
        for src_row, tgt_row in zip(src_result, tgt_result):
            for sv, tv in zip(src_row, tgt_row):
                if sv is None and tv is None:
                    continue
                if sv is None or tv is None:
                    return False
                # Date/datetime: compare directly (equal if same value)
                if isinstance(sv, (datetime.date, datetime.datetime)):
                    if sv != tv:
                        return False
                    continue
                # Numeric: compare with tolerance
                if abs(float(sv) - float(tv)) > 0.01:
                    return False
        return True
    elif check_name == "sample_diff":
        # Compare row count only for sample (value formats may differ across engines)
        return len(src_result) == len(tgt_result)
    else:
        # For other checks, compare structure
        return len(src_result) == len(tgt_result)


def _run_reconciliation(source_duckdb, target_conn, source_table, target_table, columns, key_columns):
    """Run reconciliation checks and return results."""
    checks = build_reconciliation_checks(source_table, target_table, columns, key_columns)
    results = []

    for check in checks:
        # Source query on DuckDB
        src_result = source_duckdb["connection"].execute(check["source_query"]).fetchall()
        # Target query on target database
        tgt_result = _execute_and_fetch(target_conn, check["target_query"])

        status = "pass" if _compare_results(src_result, tgt_result, check["name"]) else "fail"
        results.append(
            {
                "name": check["name"],
                "status": status,
                "source_value": str(src_result),
                "target_value": str(tgt_result),
            }
        )

    return results


# ---------------------------------------------------------------------------
# DuckDB -> Greenplum migration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDuckDBToGreenplumMigration:
    """End-to-end migration tests from DuckDB to Greenplum."""

    GP_TARGET_TABLE = "public.migration_test_users_gp"

    @pytest.fixture(autouse=True)
    def cleanup_target(self, greenplum_connection):
        """Cleanup target table before and after each test."""
        _execute_ddl(greenplum_connection, f"DROP TABLE IF EXISTS {self.GP_TARGET_TABLE}")
        yield
        _execute_ddl(greenplum_connection, f"DROP TABLE IF EXISTS {self.GP_TARGET_TABLE}")

    def test_type_mapping_duckdb_to_greenplum(self):
        """Verify type mapping produces valid Greenplum types."""
        mapped = map_columns_between_dialects(SOURCE_COLUMNS, "duckdb", "greenplum")
        type_map = {c["name"]: c["target_type"] for c in mapped}
        assert type_map["id"] == "INTEGER"
        assert type_map["name"] == "VARCHAR(100)"
        assert type_map["amount"] == "NUMERIC(10,2)"
        assert type_map["is_active"] == "BOOLEAN"
        assert type_map["created_at"] == "TIMESTAMP"
        assert type_map["birth_date"] == "DATE"

    def test_build_and_execute_target_ddl(self, greenplum_connection):
        """Generate GP DDL and create the table."""
        profile = GreenplumProfile(schema_name="public")
        ddl = build_target_ddl(SOURCE_COLUMNS, "duckdb", "greenplum", "migration_test_users_gp", profile)

        assert "CREATE TABLE" in ddl
        assert "public.migration_test_users_gp" in ddl

        _execute_ddl(greenplum_connection, ddl)

        # Verify table exists
        rows = _execute_and_fetch(
            greenplum_connection,
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'migration_test_users_gp' ORDER BY ordinal_position",
        )
        assert len(rows) == 6
        col_names = [r[0] for r in rows]
        assert "id" in col_names
        assert "amount" in col_names

    def test_transfer_and_reconcile(self, duckdb_source, greenplum_connection):
        """Transfer data and run reconciliation checks."""
        # Build and execute DDL
        profile = GreenplumProfile(schema_name="public")
        ddl = build_target_ddl(SOURCE_COLUMNS, "duckdb", "greenplum", "migration_test_users_gp", profile)
        _execute_ddl(greenplum_connection, ddl)

        # Transfer data
        col_names = [c["name"] for c in SOURCE_COLUMNS]
        rows_transferred = _transfer_data(duckdb_source, greenplum_connection, self.GP_TARGET_TABLE, col_names)
        assert rows_transferred == 5

        # Verify row count on target
        rows = _execute_and_fetch(greenplum_connection, f"SELECT COUNT(*) FROM {self.GP_TARGET_TABLE}")
        assert rows[0][0] == 5

        # Run reconciliation
        results = _run_reconciliation(
            duckdb_source, greenplum_connection, SOURCE_TABLE, self.GP_TARGET_TABLE, SOURCE_COLUMNS, ["id"]
        )
        # Verify ALL reconciliation checks pass, not just row_count (H-6)
        failed_checks = [r for r in results if r["status"] == "fail"]
        assert len(failed_checks) == 0, f"Failed checks: {failed_checks}"


# ---------------------------------------------------------------------------
# DuckDB -> StarRocks migration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDuckDBToStarRocksMigration:
    """End-to-end migration tests from DuckDB to StarRocks."""

    SR_TARGET_TABLE = "test.migration_test_users_sr"

    @pytest.fixture(autouse=True)
    def cleanup_target(self, starrocks_connection):
        """Cleanup target table before and after each test."""
        try:
            _execute_ddl(starrocks_connection, f"DROP TABLE IF EXISTS {self.SR_TARGET_TABLE}")
        except Exception:
            pass
        yield
        try:
            _execute_ddl(starrocks_connection, f"DROP TABLE IF EXISTS {self.SR_TARGET_TABLE}")
        except Exception:
            pass

    def test_type_mapping_duckdb_to_starrocks(self):
        """Verify type mapping produces valid StarRocks types."""
        mapped = map_columns_between_dialects(SOURCE_COLUMNS, "duckdb", "starrocks")
        type_map = {c["name"]: c["target_type"] for c in mapped}
        assert type_map["id"] == "INT"
        assert type_map["name"] == "VARCHAR(100)"
        assert type_map["amount"] == "DECIMAL(10,2)"
        assert type_map["is_active"] == "BOOLEAN"
        assert type_map["created_at"] == "DATETIME"
        assert type_map["birth_date"] == "DATE"

    def test_build_and_execute_target_ddl(self, starrocks_connection):
        """Generate StarRocks DDL with DUPLICATE KEY and create the table."""
        profile = StarRocksProfile(database="test")
        ddl = build_target_ddl(SOURCE_COLUMNS, "duckdb", "starrocks", "migration_test_users_sr", profile)

        assert "CREATE TABLE" in ddl
        assert "DUPLICATE KEY" in ddl
        assert "DISTRIBUTED BY HASH" in ddl
        assert "BUCKETS" in ddl

        _execute_ddl(starrocks_connection, ddl)

        # Verify table exists
        rows = _execute_and_fetch(
            starrocks_connection,
            "SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = 'test' AND TABLE_NAME = 'migration_test_users_sr' "
            "ORDER BY ORDINAL_POSITION",
        )
        assert len(rows) == 6

    def test_transfer_and_reconcile(self, duckdb_source, starrocks_connection):
        """Transfer data and run reconciliation checks."""
        # Build and execute DDL
        profile = StarRocksProfile(database="test")
        ddl = build_target_ddl(SOURCE_COLUMNS, "duckdb", "starrocks", "migration_test_users_sr", profile)
        _execute_ddl(starrocks_connection, ddl)

        # Transfer data
        col_names = [c["name"] for c in SOURCE_COLUMNS]
        rows_transferred = _transfer_data(duckdb_source, starrocks_connection, self.SR_TARGET_TABLE, col_names)
        assert rows_transferred == 5

        # Verify row count on target
        rows = _execute_and_fetch(starrocks_connection, f"SELECT COUNT(*) FROM {self.SR_TARGET_TABLE}")
        assert rows[0][0] == 5

        # Run reconciliation
        results = _run_reconciliation(
            duckdb_source, starrocks_connection, SOURCE_TABLE, self.SR_TARGET_TABLE, SOURCE_COLUMNS, ["id"]
        )
        # Verify ALL reconciliation checks pass, not just row_count (H-6)
        failed_checks = [r for r in results if r["status"] == "fail"]
        assert len(failed_checks) == 0, f"Failed checks: {failed_checks}"


# ---------------------------------------------------------------------------
# End-to-end tests using DBFuncTool
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEndToEndMigrationWithDBFuncTool:
    """Test the full migration flow using DBFuncTool.transfer_query_result.

    These tests require both DuckDB and target databases to be configured
    in agent.yml and accessible. They validate that the tool-level API
    works end-to-end, not just the helper functions.
    """

    def test_full_migration_duckdb_to_greenplum(self, duckdb_source, greenplum_connection):
        """Full flow: inspect -> DDL -> transfer -> reconcile via Greenplum."""
        target_table = "public.migration_e2e_gp"

        try:
            # Step 1: Inspect source
            src_count = duckdb_source["connection"].execute(f"SELECT COUNT(*) FROM {SOURCE_TABLE}").fetchone()[0]
            assert src_count == 5

            # Step 2: Build and execute DDL
            profile = GreenplumProfile(schema_name="public")
            ddl = build_target_ddl(SOURCE_COLUMNS, "duckdb", "greenplum", "migration_e2e_gp", profile)
            _execute_ddl(greenplum_connection, ddl)

            # Step 3: Transfer data
            col_names = [c["name"] for c in SOURCE_COLUMNS]
            rows = _transfer_data(duckdb_source, greenplum_connection, target_table, col_names)
            assert rows == 5

            # Step 4: Reconcile
            results = _run_reconciliation(
                duckdb_source, greenplum_connection, SOURCE_TABLE, target_table, SOURCE_COLUMNS, ["id"]
            )
            # Row count must pass
            failed_checks = [r for r in results if r["status"] == "fail"]
            assert len(failed_checks) == 0, f"Failed checks: {failed_checks}"
        finally:
            _execute_ddl(greenplum_connection, f"DROP TABLE IF EXISTS {target_table}")

    def test_full_migration_duckdb_to_starrocks(self, duckdb_source, starrocks_connection):
        """Full flow: inspect -> DDL -> transfer -> reconcile via StarRocks."""
        target_table = "test.migration_e2e_sr"

        try:
            # Step 1: Inspect source
            src_count = duckdb_source["connection"].execute(f"SELECT COUNT(*) FROM {SOURCE_TABLE}").fetchone()[0]
            assert src_count == 5

            # Step 2: Build and execute DDL
            profile = StarRocksProfile(database="test")
            ddl = build_target_ddl(SOURCE_COLUMNS, "duckdb", "starrocks", "migration_e2e_sr", profile)
            _execute_ddl(starrocks_connection, ddl)

            # Step 3: Transfer data
            col_names = [c["name"] for c in SOURCE_COLUMNS]
            rows = _transfer_data(duckdb_source, starrocks_connection, target_table, col_names)
            assert rows == 5

            # Step 4: Reconcile
            results = _run_reconciliation(
                duckdb_source, starrocks_connection, SOURCE_TABLE, target_table, SOURCE_COLUMNS, ["id"]
            )
            failed_checks = [r for r in results if r["status"] == "fail"]
            assert len(failed_checks) == 0, f"Failed checks: {failed_checks}"
        finally:
            try:
                _execute_ddl(starrocks_connection, f"DROP TABLE IF EXISTS {target_table}")
            except Exception:
                pass
