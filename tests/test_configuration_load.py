import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.storage_cfg import check_storage_config
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_agent_config()


@pytest.mark.acceptance
def test_config_exception():
    with pytest.raises(DatusException, match="Agent configuration file not found: not_found.yml"):
        load_agent_config(config="not_found.yml", reload=True)

    with pytest.raises(
        DatusException,
        match="Unexcepted value of Node Type, excepted value:",
    ):
        load_agent_config(config="tests/conf/wrong_nodes_agent.yml", reload=True)

    agent_config = load_agent_config(config="tests/conf/agent.yml", reload=True)

    with pytest.raises(DatusException, match="Unsupported value `abc` for field `benchmark`"):
        agent_config.override_by_args(namespace="snowflake", benchmark="abc")

    with pytest.raises(DatusException, match="Unsupported value `abc` for field `namespace`"):
        agent_config.override_by_args(namespace="abc")


@pytest.mark.parametrize("namespace", ["bird_sqlite", "snowflake", "local_duckdb"])
def test_configuration_load(namespace: str, agent_config: AgentConfig):
    assert agent_config.target
    assert agent_config.models
    assert agent_config.active_model()
    assert agent_config.rag_base_path

    assert agent_config.nodes
    assert agent_config.current_namespace == ""

    agent_config.override_by_args(
        **{
            "schema_linking_rate": "slow",
            "namespace": namespace,
        }
    )

    assert agent_config.schema_linking_rate == "slow"
    assert agent_config.rag_storage_path() == f"data/datus_db_{namespace}"

    with pytest.raises(DatusException, match="Missing required field: namespace"):
        agent_config.current_namespace = ""

    error_namespace = "abc"
    with pytest.raises(DatusException, match=f"Unsupported value {error_namespace} for namespace"):
        agent_config.current_namespace = error_namespace

    error_benchmark = "abc"
    with pytest.raises(DatusException, match=f"Unsupported value {error_namespace} for benchmark"):
        agent_config.benchmark_path(error_benchmark)


def test_benchmark_db_check(agent_config: AgentConfig, namespace: str = "snowflake"):
    agent_config.namespaces[namespace][namespace].type = DBType.SQLITE

    with pytest.raises(DatusException, match="spider2 only support snowflake"):
        agent_config.override_by_args(
            **{
                "benchmark": "spider2",
                "namespace": namespace,
            }
        )


@pytest.mark.parametrize(
    argnames=["namespace", "benchmark"],
    argvalues=[("bird_sqlite", "bird_dev"), ("snowflake", "spider2")],
)
def test_benchmark_config(namespace: str, benchmark: str, agent_config: AgentConfig):
    # Benchmark paths are now fixed at {agent.home}/benchmark/{name}
    agent_config.override_by_args(
        **{
            "namespace": namespace,
            "benchmark": benchmark,
        }
    )
    # Verify the benchmark path is constructed correctly
    benchmark_path = agent_config.benchmark_path(benchmark)
    assert benchmark_path is not None
    assert "benchmark" in benchmark_path


def test_storage_config(agent_config: AgentConfig):
    assert agent_config.storage_configs is not None
    # assert agent_config.storage_configs["database"]


def test_check_storage_config(agent_config: AgentConfig):
    agent_config.current_namespace = "bird_sqlite"
    rag_path = agent_config.rag_storage_path()
    if "database" in agent_config.storage_configs:
        agent_config.check_init_storage_config("database", save_config=False)

    error_config = {f: v.to_dict() for f, v in agent_config.storage_configs.items()}
    if "database" not in error_config:
        error_config["database"] = {}
    error_config["database"]["model_name"] = "123123"
    with pytest.raises(DatusException, match=ErrorCode.COMMON_CONFIG_ERROR.code):
        check_storage_config("database", error_config["database"], rag_path, save_config=False)


def test_get_db_name_type(agent_config: AgentConfig):
    agent_config.current_namespace = "bird_sqlite"
    db_name, db_type = agent_config.current_db_name_type(db_name="california_schools")
    assert db_name == "california_schools"
    assert db_type == DBType.SQLITE

    agent_config.current_namespace = "local_duckdb"
    db_name, db_type = agent_config.current_db_name_type(db_name="ssb")
    assert db_name == "ssb"
    assert db_type == DBType.DUCKDB

    agent_config.current_namespace = "starrocks"
    db_name, db_type = agent_config.current_db_name_type(db_name="ssb")
    assert db_name == "ssb"
    assert db_type == DBType.STARROCKS
