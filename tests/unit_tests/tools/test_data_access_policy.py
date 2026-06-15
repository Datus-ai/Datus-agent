from types import SimpleNamespace
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


class FakeDataAccessEnforcer:
    last_config = None

    def __init__(self, config: DataAccessConfig):
        self.config = config
        FakeDataAccessEnforcer.last_config = config

    def enforce_read(self, sql, *, datasource, dialect, principal):
        assert datasource == "default"
        assert dialect == "sqlite"
        assert principal == {"store_ids": ["S001"]}
        return EnforcementResult(
            allowed=True,
            sql="SELECT * FROM orders WHERE store_id = 'S001'",
            applied_policies=["store_scope"],
        )


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
    with pytest.raises(ValueError, match="enabled must be a boolean"):
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
