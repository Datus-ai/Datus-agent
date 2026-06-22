import sqlite3
from pathlib import Path
from types import SimpleNamespace

from datus.configuration.agent_config import DbConfig
from tests.conftest import isolate_bird_sqlite_databases


def _table_names(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


def test_isolate_bird_sqlite_databases_copies_and_sanitizes_generated_tables(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    source_root = fake_home / "benchmark" / "bird" / "dev_20240627" / "dev_databases"
    source_db = source_root / "california_schools" / "california_schools.sqlite"
    source_db.parent.mkdir(parents=True)

    conn = sqlite3.connect(str(source_db))
    try:
        conn.execute("CREATE TABLE schools (id INT)")
        conn.execute("CREATE TABLE mf_time_spine (ds DATETIME)")
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(Path, "home", lambda: fake_home)

    agent_config = SimpleNamespace(
        services=SimpleNamespace(
            datasources={
                "bird_sqlite": DbConfig(
                    type="sqlite",
                    path_pattern=str(source_root / "**/*.sqlite"),
                ),
                "bird_school": DbConfig(
                    type="sqlite",
                    uri=f"sqlite:///{source_db}",
                    database="california_schools",
                ),
            }
        )
    )

    isolated_root = isolate_bird_sqlite_databases(
        agent_config,
        tmp_path / "isolated",
        ("california_schools",),
    )
    isolated_db = isolated_root / "california_schools" / "california_schools.sqlite"

    assert isolated_db.exists()
    assert agent_config.services.datasources["bird_sqlite"].path_pattern == str(isolated_root / "**/*.sqlite")
    assert agent_config.services.datasources["bird_school"].uri == f"sqlite:///{isolated_db.resolve().as_posix()}"
    assert _table_names(isolated_db) == {"schools"}
    assert _table_names(source_db) == {"schools", "mf_time_spine"}
