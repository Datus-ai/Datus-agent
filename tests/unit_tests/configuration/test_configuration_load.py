import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException
from tests.conftest import TEST_CONF_DIR


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_agent_config(config=str(TEST_CONF_DIR / "agent.yml"), reload=True)


def test_config_exception():
    with pytest.raises(DatusException, match="Agent configuration file not found: not_found.yml"):
        load_agent_config(config="not_found.yml", reload=True)

    with pytest.raises(
        DatusException,
        match="Unexcepted value of Node Type, excepted value:",
    ):
        load_agent_config(config=str(TEST_CONF_DIR / "wrong_nodes_agent.yml"), reload=True)

    agent_config = load_agent_config(config=str(TEST_CONF_DIR / "agent.yml"), reload=True)

    with pytest.raises(DatusException, match="Unsupported value `abc` for field `benchmark`"):
        agent_config.override_by_args(database="snowflake", benchmark="abc")

    with pytest.raises(DatusException, match="Unsupported value `abc` for field `database`"):
        agent_config.override_by_args(database="abc")


def test_service_config_structure(agent_config: AgentConfig):
    """Verify legacy namespace config is migrated to service.databases."""
    assert agent_config.service is not None
    assert len(agent_config.service.databases) > 0
    # bird_school should be migrated as a database entry
    assert "bird_school" in agent_config.service.databases
    assert "snowflake" in agent_config.service.databases
    assert "local_duckdb" in agent_config.service.databases


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
    # rag_storage_path() now uses project_name (from cwd), not database name
    assert "data/datus_db_" in agent_config.rag_storage_path()

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
    agent_config.service.databases[db_name].type = DBType.SQLITE

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
