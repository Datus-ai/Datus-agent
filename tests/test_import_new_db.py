import glob
import os
from pathlib import Path

import duckdb
import lancedb

# new db path
HOME_DIR = Path.home()
NEW_DB_PATH = str(HOME_DIR) + "/.metricflow/duck_new.db"
EXPORT_DIR = str(HOME_DIR) + "/duckdb_export"


def test_import_from_file():
    conn = duckdb.connect(NEW_DB_PATH)
    conn.execute("CREATE SCHEMA IF NOT EXISTS mf_demo;")
    # import all sql files
    csv_files = glob.glob(os.path.join(EXPORT_DIR, "*.csv"))
    for csv_file in csv_files:
        full_file_name = os.path.basename(csv_file)
        file_name = full_file_name.split(".")[0]
        table_name = f"mf_demo.{file_name}"
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_file}', header=TRUE)")
        print(f"finish the import of {csv_file}")
    conn.close()


def test_duckdb_query():
    tmp_db_path = Path(__file__).parent / "data" / "datus_metricflow_db" / "duck.db"
    conn = duckdb.connect(tmp_db_path)
    result = conn.execute(
        "select database_name, schema_name, table_name, 'sql' from duckdb_tables() where database_name != 'system'"
    )
    print(result.fetchall())
    assert result is not None
    conn.close()


def test_lancedb_query():
    db_path = Path(__file__).parent.parent.resolve() / "data/datus_db_local_duckdb/"
    conn = lancedb.connect(db_path)
    table_names = conn.table_names()
    for table_name in table_names:
        result = conn.open_table(table_name).to_pandas().iterrows()
        for row in result:
            print(f"query the table:{table_name}, row:\n{row}")
    assert table_names is not None
