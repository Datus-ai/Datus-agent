import sqlite3
from pathlib import Path

from datus.tools.func_tool.database import DBFuncTool
from tests.unit_tests.conftest import CALIFORNIA_SCHOOLS_DB


def _sqlite_uri_to_path(uri: str) -> Path:
    if uri.startswith("sqlite:///"):
        return Path(uri.removeprefix("sqlite:///")).resolve()
    return Path(uri).resolve()


def test_real_agent_config_uses_shared_readonly_sample_db(real_agent_config, tmp_path):
    datasource = real_agent_config.services.datasources["california_schools"]
    db_path = _sqlite_uri_to_path(datasource.uri)

    assert db_path == Path(CALIFORNIA_SCHOOLS_DB).resolve()
    assert datasource.extra == {"read_only": True}
    assert not (tmp_path / "california_schools.sqlite").exists()

    tool = DBFuncTool(agent_config=real_agent_config)
    read_result = tool.read_query("SELECT COUNT(*) AS count FROM schools")
    assert read_result.success == 1

    write_result = tool.execute_ddl("CREATE TABLE datus_readonly_probe (id INT)")
    assert write_result.success == 0
    assert "readonly" in write_result.error.lower().replace("-", "")


def test_mutable_real_agent_config_gets_isolated_writable_copy(mutable_real_agent_config, tmp_path):
    datasource = mutable_real_agent_config.services.datasources["california_schools"]
    db_path = _sqlite_uri_to_path(datasource.uri)

    assert db_path == tmp_path / "california_schools.sqlite"
    assert db_path.exists()
    assert datasource.extra is None

    tool = DBFuncTool(agent_config=mutable_real_agent_config)
    write_result = tool.execute_ddl("CREATE TABLE datus_mutable_probe (id INT)")
    assert write_result.success == 1

    read_result = tool.read_query(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'datus_mutable_probe'"
    )
    assert read_result.success == 1
    assert "datus_mutable_probe" in str(read_result.result)

    with sqlite3.connect(CALIFORNIA_SCHOOLS_DB) as conn:
        source_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'datus_mutable_probe'"
        ).fetchone()
    assert source_table is None
