import duckdb
import os
import glob
import lancedb
from pathlib import Path
from tabulate import tabulate
# 新数据库路径
HOME_DIR = Path.home()
NEW_DB_PATH = str(HOME_DIR) + "/.metricflow/duck_new.db"
EXPORT_DIR = str(HOME_DIR) + "/duckdb_export"

def test_import_from_file():
    # 连接新数据库（自动创建新版本文件）
    conn = duckdb.connect(NEW_DB_PATH)
    conn.execute("CREATE SCHEMA IF NOT EXISTS mf_demo;")
    # 导入所有SQL文件
    csv_files = glob.glob(os.path.join(EXPORT_DIR, "*.csv"))
    for csv_file in csv_files:
        full_file_name = os.path.basename(csv_file)
        file_name = full_file_name.split(".")[0]
        table_name = f"mf_demo.{file_name}" 
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_file}', header=TRUE)")
        print(f"导入 {csv_file} 完成")
    conn.close()

def test_duckdb_query():
    conn = duckdb.connect(NEW_DB_PATH)
    # result = conn.execute("SELECT * FROM mf_demo.mf_demo_countries")
    result = conn.execute("SELECT * FROM duckdb_tables()")
    print(result.fetchall())
    assert result is not None
    conn.close()

def test_lancedb_query():
    db_path = str(HOME_DIR) + "/AIdeaProjects/Datus-agent/data/datus_db_local_duckdb"
    conn = lancedb.connect(db_path)
    table_names = conn.table_names()
    for table_name in table_names:
        result = conn.open_table(table_name).to_pandas().iterrows()
        # ascii_table = tabulate(result, headers="keys", tablefmt="pretty", showindex=False)
        # 查询所有数据（返回 PyArrow Table）
        # print(f'query the table:{table_name}, result:\n{ascii_table}')
        for row in result:
            print(f'query the table:{table_name}, row:\n{row}')
    assert table_names is not None

