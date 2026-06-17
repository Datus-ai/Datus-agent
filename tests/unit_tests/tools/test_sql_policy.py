from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.database import DBFuncTool
from datus.tools.sql_policy import (
    EnforcementResult,
    NoopSqlPolicyEnforcer,
    SqlPolicyConfig,
    SqlPolicyProviderError,
    load_sql_policy_enforcer,
)
from datus.utils.exceptions import DatusException


class FakeSqlPolicyEnforcer:
    last_config: SqlPolicyConfig | None = None

    def __init__(self, config: SqlPolicyConfig) -> None:
        self.config = config
        FakeSqlPolicyEnforcer.last_config = config

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


class NonCallableSqlPolicyEnforcer:
    def __init__(self, config: SqlPolicyConfig) -> None:
        self.config = config

    enforce_read = None


class DatasourceCaptureEnforcer:
    last_datasource: str | None = None

    def __init__(self, config: SqlPolicyConfig) -> None:
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


class DenySqlPolicyEnforcer:
    def __init__(self, config: SqlPolicyConfig) -> None:
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
    def __init__(self, config: SqlPolicyConfig) -> None:
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


class EmptyRewriteEnforcer:
    def __init__(self, config: SqlPolicyConfig) -> None:
        self.config = config

    def enforce_read(
        self,
        sql: str,
        *,
        datasource: str,
        dialect: str,
        principal: dict[str, Any] | None,
    ) -> EnforcementResult:
        return EnforcementResult(allowed=True, sql="")


def _provider_config() -> SqlPolicyConfig:
    return SqlPolicyConfig.from_dict(
        {
            "enabled": True,
            "provider": "tests.unit_tests.tools.test_sql_policy:FakeSqlPolicyEnforcer",
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


def test_sql_policy_config_preserves_provider_and_raw_policy_config():
    config = _provider_config()

    assert config.enabled is True
    assert config.provider == "tests.unit_tests.tools.test_sql_policy:FakeSqlPolicyEnforcer"
    assert config.raw["policies"][0]["name"] == "store_scope"


def test_sql_policy_config_rejects_non_mapping():
    with pytest.raises(DatusException, match="agent.sql_policy must be a mapping"):
        SqlPolicyConfig.from_dict(False)  # type: ignore[arg-type]


def test_disabled_sql_policy_uses_noop_enforcer():
    result = load_sql_policy_enforcer(SqlPolicyConfig()).enforce_read(
        "SELECT * FROM orders",
        datasource="default",
        dialect="sqlite",
        principal=None,
    )

    assert isinstance(load_sql_policy_enforcer(SqlPolicyConfig()), NoopSqlPolicyEnforcer)
    assert result.allowed is True
    assert result.sql == "SELECT * FROM orders"


def test_enabled_sql_policy_requires_provider():
    config = SqlPolicyConfig.from_dict({"enabled": True})

    with pytest.raises(SqlPolicyProviderError, match="provider is not configured"):
        load_sql_policy_enforcer(config)


def test_sql_policy_enabled_must_be_boolean():
    with pytest.raises(DatusException, match="enabled must be a boolean"):
        SqlPolicyConfig.from_dict({"enabled": "false"})


def test_sql_policy_provider_requires_colon_path():
    config = SqlPolicyConfig.from_dict(
        {
            "enabled": True,
            "provider": "tests.unit_tests.tools.test_sql_policy.FakeSqlPolicyEnforcer",
        }
    )

    with pytest.raises(SqlPolicyProviderError, match="package.module:ProviderClass"):
        load_sql_policy_enforcer(config)


def test_sql_policy_provider_can_be_loaded_from_config():
    enforcer = load_sql_policy_enforcer(_provider_config())

    assert isinstance(enforcer, FakeSqlPolicyEnforcer)
    assert FakeSqlPolicyEnforcer.last_config.raw["policies"][0]["type"] == "row_filter"


def test_sql_policy_provider_must_implement_callable_enforce_read():
    config = SqlPolicyConfig.from_dict(
        {
            "enabled": True,
            "provider": "tests.unit_tests.tools.test_sql_policy:NonCallableSqlPolicyEnforcer",
        }
    )

    with pytest.raises(SqlPolicyProviderError, match="must implement enforce_read"):
        load_sql_policy_enforcer(config)


def test_db_func_tool_applies_sql_policy_provider_before_query_execution():
    connector = Mock()
    connector.dialect = "sqlite"
    connector.get_databases.return_value = []
    query_result = Mock()
    query_result.success = True
    query_result.sql_return = [{"order_id": 1}]
    connector.execute_query.return_value = query_result

    agent_config = SimpleNamespace(
        active_model=lambda: SimpleNamespace(model="test-model"),
        sql_policy_config=_provider_config(),
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


def test_db_func_tool_ignores_mock_sql_policy_config():
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
        sql_policy_config=SqlPolicyConfig.from_dict(
            {
                "enabled": True,
                "provider": "tests.unit_tests.tools.test_sql_policy:DatasourceCaptureEnforcer",
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
        sql_policy_config=SqlPolicyConfig.from_dict(
            {
                "enabled": True,
                "provider": "tests.unit_tests.tools.test_sql_policy:DenySqlPolicyEnforcer",
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
        sql_policy_config=SqlPolicyConfig.from_dict(
            {
                "enabled": True,
                "provider": "tests.unit_tests.tools.test_sql_policy:UnsafeRewriteEnforcer",
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


def test_db_func_tool_does_not_fall_back_for_empty_policy_rewrite():
    connector = Mock()
    connector.dialect = "sqlite"
    connector.get_databases.return_value = []

    agent_config = SimpleNamespace(
        active_model=lambda: SimpleNamespace(model="test-model"),
        sql_policy_config=SqlPolicyConfig.from_dict(
            {
                "enabled": True,
                "provider": "tests.unit_tests.tools.test_sql_policy:EmptyRewriteEnforcer",
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
