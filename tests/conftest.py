import argparse
import os
import shutil
import sqlite3
import sys
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROJECT_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_CONF_DIR = Path(__file__).parent / "conf"
RERUN_LOG_MAX_LINES = int(os.getenv("DATUS_RERUN_LOG_MAX_LINES", "80"))
RERUN_CAPTURE_LOG_MAX_LINES = int(os.getenv("DATUS_RERUN_CAPTURE_LOG_MAX_LINES", "40"))
_DATUS_RERUN_REPORTS: list[dict[str, object]] = []
_BIRD_DEV_DATABASES = Path("benchmark/bird/dev_20240627/dev_databases")
_GENERATED_SQLITE_TABLES = ("mf_time_spine",)


@pytest.fixture
def mock_args():
    """Create a mock arguments object for testing."""
    args = argparse.Namespace(
        model="deepseek-v3",
        temperature=0.5,
        top_p=0.9,
        max_tokens=2500,
        task="Select all employees who earn more than $50,000",
        task_type="local",
        db_path="test_db.sqlite",
        schema_path="test_schema.sql",
        plan=True,
        max_steps=20,
        human_in_loop=False,
        output_dir="test_output",
    )
    return args


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.generate.return_value = "Generated text response"
    model.generate_with_json_output.return_value = {"result": "success"}
    model.gen_sql.return_value = "SELECT * FROM employees WHERE salary > 50000;"
    return model


# @pytest.fixture
# def sample_workflow():
#     """Create a sample workflow for testing."""
#     from datus.agent.workflow import Node, Workflow

#     workflow = Workflow("Test Workflow", "A workflow for testing")

#     # Add some tasks to the workflow
#     task1 = Node(
#         "task1",
#         "Parse the query",
#         "query_processing",
#         "Select all employees who earn more than $50,000",
#     )
#     task2 = Node("task2", "Generate SQL", "sql_generation", "Parsed query data")
#     task3 = Node(
#         "task3",
#         "Execute SQL",
#         "sql_execution",
#         "SELECT * FROM employees WHERE salary > 50000;",
#     )

#     workflow.add_task(task1)
#     workflow.add_task(task2)
#     workflow.add_task(task3)

#     return workflow


@pytest.fixture
def sample_database_schema():
    """Create a sample database schema for testing."""
    return """
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        salary REAL NOT NULL,
        hire_date TEXT NOT NULL
    );

    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        budget REAL NOT NULL
    );
    """


@pytest.fixture
def sample_database_data():
    """Create sample database data for testing."""
    return [
        {
            "id": 1,
            "name": "John Doe",
            "department": "Engineering",
            "salary": 75000,
            "hire_date": "2020-01-15",
        },
        {
            "id": 2,
            "name": "Jane Smith",
            "department": "Marketing",
            "salary": 65000,
            "hire_date": "2019-05-20",
        },
        {
            "id": 3,
            "name": "Bob Johnson",
            "department": "Engineering",
            "salary": 85000,
            "hire_date": "2018-11-10",
        },
        {
            "id": 4,
            "name": "Alice Brown",
            "department": "HR",
            "salary": 45000,
            "hire_date": "2021-03-01",
        },
        {
            "id": 5,
            "name": "Charlie Wilson",
            "department": "Marketing",
            "salary": 55000,
            "hire_date": "2020-07-30",
        },
    ]


def load_acceptance_config(datasource: str = "snowflake", home: str = "") -> AgentConfig:
    return load_agent_config(
        config=str(TEST_CONF_DIR / "agent.yml"), datasource=datasource, home=home, reload=True, force=True, yes=True
    )


def _sqlite_uri_to_path(uri: str) -> Path:
    if uri.startswith("sqlite:///"):
        return Path(uri.removeprefix("sqlite:///")).resolve()
    return Path(uri).expanduser().resolve()


def _sqlite_path_to_uri(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def _drop_generated_sqlite_tables(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        for table in _GENERATED_SQLITE_TABLES:
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.commit()
    finally:
        conn.close()


def isolate_bird_sqlite_databases(
    agent_config: AgentConfig,
    tmp_root: Path,
    database_names: Iterable[str],
    *,
    reuse_existing: bool = False,
) -> Path:
    """Repoint BIRD SQLite datasources at isolated writable database copies.

    Nightly/acceptance configs intentionally reference the developer or CI
    benchmark checkout under ``~/benchmark``. Tests that initialize adapters
    capable of DDL, such as MetricFlow, must not write support tables back into
    that shared fixture. This helper copies the requested databases into a
    tmp-root benchmark layout, removes known generated support tables from the
    copies, and rewrites both the ``bird_sqlite`` glob datasource and any
    single-file datasource that points at a copied database.
    """
    names = tuple(dict.fromkeys(database_names))
    if not names:
        raise ValueError("database_names must contain at least one database")

    source_root = Path.home() / _BIRD_DEV_DATABASES
    if not source_root.exists():
        pytest.skip(f"BIRD benchmark database root not found: {source_root}")

    dest_root = Path(tmp_root) / _BIRD_DEV_DATABASES
    copied_paths: dict[str, Path] = {}
    for name in names:
        src = source_root / name / f"{name}.sqlite"
        if not src.exists():
            pytest.skip(f"BIRD benchmark SQLite database not found: {src}")
        dst = dest_root / name / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not reuse_existing or not dst.exists():
            shutil.copy2(src, dst)
        _drop_generated_sqlite_tables(dst)
        copied_paths[name] = dst.resolve()

    for cfg in agent_config.services.datasources.values():
        if cfg.path_pattern:
            pattern_path = Path(os.path.expanduser(cfg.path_pattern)).resolve()
            if source_root in pattern_path.parents or pattern_path == source_root:
                cfg.path_pattern = str(dest_root / "**/*.sqlite")

        if not cfg.uri:
            continue
        try:
            db_path = _sqlite_uri_to_path(cfg.uri)
        except Exception:
            continue
        copied = copied_paths.get(db_path.stem)
        if copied and source_root in db_path.parents:
            cfg.uri = _sqlite_path_to_uri(copied)

    from datus.tools.db_tools import db_manager as _db_manager

    _db_manager._cli_cache.clear()
    return dest_root


def _tail_lines(text: str, max_lines: int) -> list[str]:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return lines
    omitted = len(lines) - max_lines
    return [f"... omitted {omitted} earlier line(s) ...", *lines[-max_lines:]]


def _format_report_sections(sections: Iterable[tuple[str, str]]) -> list[str]:
    formatted: list[str] = []
    for name, content in sections:
        if not content:
            continue
        formatted.append(f"-- {name} --")
        formatted.extend(_tail_lines(content, RERUN_CAPTURE_LOG_MAX_LINES))
    return formatted


def pytest_configure(config) -> None:
    _DATUS_RERUN_REPORTS.clear()


def pytest_runtest_logreport(report) -> None:
    if report.outcome != "rerun":
        return

    longrepr_text = getattr(report, "longreprtext", "") or str(report.longrepr or "")
    _DATUS_RERUN_REPORTS.append(
        {
            "nodeid": report.nodeid,
            "when": report.when,
            "duration": report.duration,
            "rerun": getattr(report, "rerun", "?"),
            "worker": getattr(report, "worker_id", os.getenv("PYTEST_XDIST_WORKER", "main")),
            "longrepr": _tail_lines(longrepr_text, RERUN_LOG_MAX_LINES),
            "sections": _format_report_sections(getattr(report, "sections", [])),
        }
    )


def pytest_terminal_summary(terminalreporter) -> None:
    if not _DATUS_RERUN_REPORTS:
        return

    terminalreporter.section("Datus rerun diagnostics", sep="=")
    for report in _DATUS_RERUN_REPORTS:
        terminalreporter.write_line(
            "RERUN "
            f"{report['nodeid']} "
            f"when={report['when']} "
            f"attempt={report['rerun']} "
            f"worker={report['worker']} "
            f"duration={report['duration']:.2f}s"
        )
        if report["longrepr"]:
            terminalreporter.write_line("First failure traceback summary:")
            for line in report["longrepr"]:
                terminalreporter.write_line(f"  {line}")
        if report["sections"]:
            terminalreporter.write_line("Captured output summary:")
            for line in report["sections"]:
                terminalreporter.write_line(f"  {line}")
