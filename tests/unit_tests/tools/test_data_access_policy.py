from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from datus.tools.data_access_policy import (
    DataAccessConfig,
    DataAccessProviderError,
    EnforcementResult,
    NoopDataAccessEnforcer,
    load_data_access_enforcer,
)
from datus.tools.func_tool.database import DBFuncTool
from datus.utils.exceptions import DatusException


class FakeDataAccessEnforcer:
    last_config: DataAccessConfig | None = None

    def __init__(self, config: DataAccessConfig) -> None:
        self.config = config
        FakeDataAccessEnforcer.last_config = config

    def enforce_read(
        self,
        sql: str,
        *,
        datasource: str,
        dialect: str,
        principal: dict[str, Any] | None,
    ) -> EnforcementResult:
        assert datasource == "default"
        assert dialect == "sqlite"
        assert principal == {"store_ids": ["S001"]}
        return EnforcementResult(
            allowed=True,
            sql="SELECT * FROM orders WHERE store_id = 'S001'",
            applied_policies=["store_scope"],
        )


class NonCallableDataAccessEnforcer:
    def __init__(self, config: DataAccessConfig) -> None:
        self.config = config

    enforce_read = None


class DatasourceCaptureEnforcer:
    last_datasource: str | None = None

    def __init__(self, config: DataAccessConfig) -> None:
        self.config = config

    def enforce_read(
        self,
        sql: str,
        *,
        datasource: str,
        dialect: str,
        principal: dict[str, Any] | None,
    ) -> EnforcementResult:
        DatasourceCaptureEnforcer.last_datasource = datasource
        return EnforcementResult(allowed=True, sql=sql)


class DenyDataAccessEnforcer:
    def __init__(self, config: DataAccessConfig) -> None:
        self.config = config

    def enforce_read(
        self,
        sql: str,
        *,
        datasource: str,
        dialect: str,
        principal: dict[str, Any] | None,
    ) -> EnforcementResult:
        return EnforcementResult(allowed=False, reason="store scope missing")


class UnsafeRewriteEnforcer:
    def __init__(self, config: DataAccessConfig) -> None:
        self.config = config

    def enforce_read(
        self,
        sql: str,
        *,
        datasource: str,
        dialect: str,
        principal: dict[str, Any] | None,
    ) -> EnforcementResult:
        return EnforcementResult(allowed=True, sql="DELETE FROM orders")


def _provider_config() -> DataAccessConfig:
    return DataAccessConfig.from_dict(
        {
            "enabled": True,
            "provider": "tests.unit_tests.tools.test_data_access_policy:FakeDataAccessEnforcer",
            "policies": [
                {
                    "name": "store_scope",
                    "type": "row_filter",
                    "applies_to": {
                        "datasources": ["default"],
                        "tables": ["orders", "store_sales"],
                    },
                    "condition": {
                        "column": "store_id",
                        "operator": "in",
                        "value_from": "principal.store_ids",
                    },
                    "enforcement": {
                        "on_read": "filter",
                        "on_unhandled": "deny",
                    },
                }
            ],
        }
    )


def test_data_access_config_preserves_provider_and_raw_policy_config():
    config = _provider_config()

    assert config.enabled is True
    assert config.provider == "tests.unit_tests.tools.test_data_access_policy:FakeDataAccessEnforcer"
    assert config.raw["policies"][0]["name"] == "store_scope"


def test_data_access_config_rejects_non_mapping():
    with pytest.raises(DatusException, match="agent.data_access must be a mapping"):
        DataAccessConfig.from_dict(False)  # type: ignore[arg-type]


def test_disabled_data_access_uses_noop_enforcer():
    result = load_data_access_enforcer(DataAccessConfig()).enforce_read(
        "SELECT * FROM orders",
        datasource="default",
        dialect="sqlite",
        principal=None,
    )

    assert isinstance(load_data_access_enforcer(DataAccessConfig()), NoopDataAccessEnforcer)
    assert result.allowed is True
    assert result.sql == "SELECT * FROM orders"


def test_enabled_data_access_requires_provider():
    config = DataAccessConfig.from_dict({"enabled": True})

    with pytest.raises(DataAccessProviderError, match="provider is not configured"):
        load_data_access_enforcer(config)


def test_data_access_enabled_must_be_boolean():
    with pytest.raises(DatusException, match="enabled must be a boolean"):
        DataAccessConfig.from_dict({"enabled": "false"})


def test_data_access_provider_requires_colon_path():
    config = DataAccessConfig.from_dict(
        {
            "enabled": True,
            "provider": "tests.unit_tests.tools.test_data_access_policy.FakeDataAccessEnforcer",
        }
    )

    with pytest.raises(DataAccessProviderError, match="package.module:ProviderClass"):
        load_data_access_enforcer(config)


def test_data_access_provider_can_be_loaded_from_config():
    enforcer = load_data_access_enforcer(_provider_config())

    assert isinstance(enforcer, FakeDataAccessEnforcer)
    assert FakeDataAccessEnforcer.last_config.raw["policies"][0]["type"] == "row_filter"


def test_data_access_provider_must_implement_callable_enforce_read():
    config = DataAccessConfig.from_dict(
        {
            "enabled": True,
            "provider": "tests.unit_tests.tools.test_data_access_policy:NonCallableDataAccessEnforcer",
        }
    )

    with pytest.raises(DataAccessProviderError, match="must implement enforce_read"):
        load_data_access_enforcer(config)


def test_db_func_tool_applies_data_access_provider_before_query_execution():
    connector = Mock()
    connector.dialect = "sqlite"
    connector.get_databases.return_value = []
    query_result = Mock()
    query_result.success = True
    query_result.sql_return = [{"order_id": 1}]
    connector.execute_query.return_value = query_result

    agent_config = SimpleNamespace(
        active_model=lambda: SimpleNamespace(model="test-model"),
        data_access_config=_provider_config(),
        principal={"store_ids": ["S001"]},
    )

    with (
        patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
        patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
    ):
        mock_rag.return_value.schema_store.table_size.return_value = 0
        mock_sem.return_value.get_size.return_value = 0
        tool = DBFuncTool(connector, agent_config=agent_config)

    result = tool.read_query("SELECT * FROM orders", datasource="default")

    assert result.success == 1
    executed_sql = connector.execute_query.call_args.args[0]
    assert executed_sql == "SELECT * FROM orders WHERE store_id = 'S001'"


def test_db_func_tool_ignores_mock_data_access_config():
    connector = Mock()
    connector.dialect = "sqlite"
    connector.get_databases.return_value = []
    query_result = Mock()
    query_result.success = True
    query_result.sql_return = [{"order_id": 1}]
    connector.execute_query.return_value = query_result

    agent_config = Mock()
    agent_config.active_model.return_value = SimpleNamespace(model="test-model")

    with (
        patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
        patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
    ):
        mock_rag.return_value.schema_store.table_size.return_value = 0
        mock_sem.return_value.get_size.return_value = 0
        tool = DBFuncTool(connector, agent_config=agent_config)

    result = tool.read_query("SELECT * FROM orders", datasource="default")

    assert result.success == 1
    connector.execute_query.assert_called_once()


def test_db_func_tool_uses_configured_default_datasource_for_policy_enforcement():
    connector = Mock()
    connector.dialect = "sqlite"
    connector.get_databases.return_value = []
    query_result = Mock()
    query_result.success = True
    query_result.sql_return = [{"order_id": 1}]
    connector.execute_query.return_value = query_result

    agent_config = SimpleNamespace(
        active_model=lambda: SimpleNamespace(model="test-model"),
        data_access_config=DataAccessConfig.from_dict(
            {
                "enabled": True,
                "provider": "tests.unit_tests.tools.test_data_access_policy:DatasourceCaptureEnforcer",
            }
        ),
        principal={},
        services=SimpleNamespace(default_datasource="analytics"),
    )

    with (
        patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
        patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
    ):
        mock_rag.return_value.schema_store.table_size.return_value = 0
        mock_sem.return_value.get_size.return_value = 0
        tool = DBFuncTool(connector, agent_config=agent_config)

    result = tool.read_query("SELECT * FROM orders")

    assert result.success == 1
    assert DatasourceCaptureEnforcer.last_datasource == "analytics"


def test_db_func_tool_returns_datus_error_when_policy_denies_query():
    connector = Mock()
    connector.dialect = "sqlite"
    connector.get_databases.return_value = []

    agent_config = SimpleNamespace(
        active_model=lambda: SimpleNamespace(model="test-model"),
        data_access_config=DataAccessConfig.from_dict(
            {
                "enabled": True,
                "provider": "tests.unit_tests.tools.test_data_access_policy:DenyDataAccessEnforcer",
            }
        ),
        principal={},
    )

    with (
        patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
        patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
    ):
        mock_rag.return_value.schema_store.table_size.return_value = 0
        mock_sem.return_value.get_size.return_value = 0
        tool = DBFuncTool(connector, agent_config=agent_config)

    result = tool.read_query("SELECT * FROM orders", datasource="default")

    assert result.success == 0
    assert "error_code=400002" in result.error
    assert "store scope missing" in result.error
    connector.execute_query.assert_not_called()


def test_db_func_tool_revalidates_sql_after_policy_rewrite():
    connector = Mock()
    connector.dialect = "sqlite"
    connector.get_databases.return_value = []

    agent_config = SimpleNamespace(
        active_model=lambda: SimpleNamespace(model="test-model"),
        data_access_config=DataAccessConfig.from_dict(
            {
                "enabled": True,
                "provider": "tests.unit_tests.tools.test_data_access_policy:UnsafeRewriteEnforcer",
            }
        ),
        principal={},
    )

    with (
        patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
        patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
    ):
        mock_rag.return_value.schema_store.table_size.return_value = 0
        mock_sem.return_value.get_size.return_value = 0
        tool = DBFuncTool(connector, agent_config=agent_config)

    result = tool.read_query("SELECT * FROM orders", datasource="default")

    assert result.success == 0
    assert "Only read-only queries" in result.error
    connector.execute_query.assert_not_called()
