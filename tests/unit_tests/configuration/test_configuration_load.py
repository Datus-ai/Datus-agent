import os

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException
from tests.conftest import TEST_CONF_DIR


@pytest.fixture
def agent_config(tmp_path) -> AgentConfig:
    # ``home=tmp_path`` pins every derived path inside the pytest-managed tmp
    # dir. The unit-tests autouse ``_isolate_project_cwd`` fixture already
    # chdir-s into tmp_path, so the yml's relative paths never resolve under
    # the repo root.
    return load_agent_config(config=str(TEST_CONF_DIR / "agent.yml"), home=str(tmp_path), reload=True)


def test_config_exception(tmp_path):
    with pytest.raises(DatusException, match="Agent configuration file not found: not_found.yml"):
        load_agent_config(config="not_found.yml", reload=True)

    with pytest.raises(
        DatusException,
        match="Unexcepted value of Node Type, excepted value:",
    ):
        load_agent_config(config=str(TEST_CONF_DIR / "wrong_nodes_agent.yml"), home=str(tmp_path), reload=True)

    agent_config = load_agent_config(config=str(TEST_CONF_DIR / "agent.yml"), home=str(tmp_path), reload=True)

    with pytest.raises(DatusException, match="Unsupported value `abc` for field `benchmark`"):
        agent_config.override_by_args(database="snowflake", benchmark="abc")

    with pytest.raises(DatusException, match="Unsupported value `abc` for field `database`"):
        agent_config.override_by_args(database="abc")


def test_service_config_structure(agent_config: AgentConfig):
    """Verify service config sections load into AgentConfig."""
    assert agent_config.services is not None
    assert len(agent_config.services.databases) > 0
    assert "bird_school" in agent_config.services.databases
    assert "snowflake" in agent_config.services.databases
    assert "local_duckdb" in agent_config.services.databases
    assert "metricflow" in agent_config.services.semantic_layer
    assert "superset" in agent_config.services.bi_tools
    assert "airflow_local" in agent_config.services.schedulers


@pytest.mark.parametrize("database", ["bird_school", "snowflake", "local_duckdb"])
def test_configuration_load(database: str, agent_config: AgentConfig):
    assert agent_config.target
    assert agent_config.models
    assert agent_config.active_model()
    assert agent_config.rag_base_path

    assert agent_config.nodes

    agent_config.override_by_args(
        **{
            "schema_linking_rate": "slow",
            "database": database,
        }
    )

    assert agent_config.schema_linking_rate == "slow"
    # rag_storage_path() lives under the project-sharded data_dir now:
    # ``{home}/data/{project_name}/datus_db``. The name ``datus_db`` is fixed;
    # project isolation happens via the parent directory.
    storage_path = agent_config.rag_storage_path()
    assert storage_path.endswith("datus_db")
    assert f"/data/{agent_config.project_name}/datus_db" in storage_path.replace(os.sep, "/")

    with pytest.raises(DatusException, match="Missing required field: database"):
        agent_config.current_namespace = ""

    error_db = "abc"
    with pytest.raises(DatusException, match=f"Unsupported value `{error_db}` for field `database`"):
        agent_config.current_namespace = error_db

    error_benchmark = "abc"
    with pytest.raises(DatusException, match=f"Unsupported value `{error_benchmark}` for field `benchmark`"):
        agent_config.benchmark_path(error_benchmark)


def test_benchmark_db_check(agent_config: AgentConfig):
    db_name = "snowflake"
    agent_config.services.databases[db_name].type = DBType.SQLITE

    with pytest.raises(DatusException, match="spider2 only support snowflake"):
        agent_config.override_by_args(
            **{
                "benchmark": "spider2",
                "database": db_name,
            }
        )


@pytest.mark.parametrize(
    argnames=["database", "benchmark"],
    argvalues=[("bird_school", "bird_dev"), ("snowflake", "spider2")],
)
def test_benchmark_config(database: str, benchmark: str, agent_config: AgentConfig):
    agent_config.override_by_args(
        **{
            "database": database,
            "benchmark": benchmark,
        }
    )
    benchmark_path = agent_config.benchmark_path(benchmark)
    assert benchmark_path is not None
    assert "benchmark" in benchmark_path


def test_storage_config(agent_config: AgentConfig):
    assert agent_config.storage_configs is not None


def test_get_db_name_type(agent_config: AgentConfig):
    db_name, db_type = agent_config.current_db_name_type(db_name="bird_school")
    assert db_name == "bird_school"
    assert db_type == DBType.SQLITE

    db_name, db_type = agent_config.current_db_name_type(db_name="local_duckdb")
    assert db_name == "local_duckdb"
    assert db_type == DBType.DUCKDB

    db_name, db_type = agent_config.current_db_name_type(db_name="starrocks")
    assert db_name == "starrocks"
    assert db_type == "starrocks"
