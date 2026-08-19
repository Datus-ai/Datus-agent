from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.database import DBFuncTool
from datus.tools.policy_runtime import PolicyRuntime
from datus.utils.exceptions import DatusException


class FakeRuntime:
    def __init__(
        self,
        *,
        sql: str | None = None,
        result: Any = None,
        allowed: bool = True,
        result_allowed: bool | None = None,
        reason: str | None = None,
    ) -> None:
        self.sql = sql
        self.result = result
        self.allowed = allowed
        self.result_allowed = allowed if result_allowed is None else result_allowed
        self.reason = reason
        self.contexts: list[dict[str, Any]] = []

    def validate_context(self, policy_context: dict[str, Any]) -> SimpleNamespace:
        self.contexts.append(policy_context)
        return SimpleNamespace(allowed=self.allowed, reason=self.reason)

    def before_sql_read(
        self,
        sql: str,
        *,
        datasource: str,
        dialect: str,
        policy_context: dict[str, Any],
    ) -> SimpleNamespace:
        self.contexts.append(policy_context)
        return SimpleNamespace(
            allowed=self.allowed,
            sql=self.sql or sql,
            reason=self.reason,
            applied_policies=["row_scope"] if self.sql else [],
        )

    def after_read_result(
        self,
        result: Any,
        *,
        sql: str,
        datasource: str,
        dialect: str,
        policy_context: dict[str, Any],
    ) -> SimpleNamespace:
        self.contexts.append(policy_context)
        return SimpleNamespace(
            allowed=self.result_allowed,
            result=result if self.result is None else self.result,
            reason=self.reason,
            applied_policies=["email_mask"] if self.result is not None else [],
        )


def config(policy_context: dict[str, Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        active_plugin_names=lambda: {"sql-policy"},
        get_plugin_profile=lambda name: {"policies": []},
        policy_context=policy_context or {},
        active_model=lambda: SimpleNamespace(model="test-model"),
    )


def runtime_with(fake: Any, monkeypatch: pytest.MonkeyPatch) -> PolicyRuntime:
    monkeypatch.setattr(
        "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
        lambda active: {"sql-policy": lambda profile: fake},
    )
    return PolicyRuntime(config())


def test_no_policy_runtime_is_pass_through(monkeypatch):
    monkeypatch.setattr("datus.tools.policy_runtime.collect_plugin_policy_runtime_factories", lambda active: {})
    runtime = PolicyRuntime(config())
    assert runtime.validate_context({}).allowed
    assert (
        runtime.before_sql_read("SELECT 1", datasource="warehouse", dialect="postgres", policy_context={}).sql
        == "SELECT 1"
    )


def test_runtime_chains_sql_and_result_hooks(monkeypatch):
    first = FakeRuntime(sql="SELECT 1 WHERE tenant_id = 1")
    second = FakeRuntime(result=[{"email": "***"}])
    monkeypatch.setattr(
        "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
        lambda active: {"row": lambda profile: first, "mask": lambda profile: second},
    )
    cfg = SimpleNamespace(
        active_plugin_names=lambda: {"row", "mask"},
        get_plugin_profile=lambda name: {},
    )
    runtime = PolicyRuntime(cfg)
    sql = runtime.before_sql_read("SELECT 1", datasource="warehouse", dialect="postgres", policy_context={})
    assert sql.sql == "SELECT 1 WHERE tenant_id = 1"
    assert sql.applied_policies == ["row_scope"]
    result = runtime.after_read_result(
        [{"email": "a@example.com"}],
        sql=sql.sql,
        datasource="warehouse",
        dialect="postgres",
        policy_context={},
    )
    assert result.result == [{"email": "***"}]
    assert result.applied_policies == ["email_mask"]


def test_denial_stops_runtime_chain(monkeypatch):
    runtime = runtime_with(FakeRuntime(allowed=False, reason="access denied"), monkeypatch)
    result = runtime.before_sql_read("SELECT 1", datasource="warehouse", dialect="postgres", policy_context={})
    assert not result.allowed
    assert result.reason == "access denied"


def test_invalid_runtime_decision_fails_closed(monkeypatch):
    class InvalidRuntime:
        def validate_context(self, policy_context):
            return SimpleNamespace(allowed="yes")

    runtime = runtime_with(InvalidRuntime(), monkeypatch)
    with pytest.raises(DatusException, match="boolean allowed"):
        runtime.validate_context({})


def make_db_tool(connector, agent_config):
    with (
        patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag,
        patch("datus.tools.func_tool.database.SemanticModelRAG") as mock_sem,
    ):
        mock_rag.return_value.schema_store.table_size.return_value = 0
        mock_sem.return_value.get_size.return_value = 0
        return DBFuncTool(connector, agent_config=agent_config)


def connector():
    value = Mock()
    value.dialect = "sqlite"
    value.get_databases.return_value = []
    value.execute_query.return_value = SimpleNamespace(
        success=True,
        sql_return=[{"email": "a@example.com"}],
        error=None,
    )
    return value


def test_db_read_applies_before_and_after_hooks(monkeypatch):
    fake = FakeRuntime(sql="SELECT * FROM orders WHERE store_id = 1", result=[{"email": "***"}])
    monkeypatch.setattr(
        "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
        lambda active: {"sql-policy": lambda profile: fake},
    )
    cfg = config({"row_filter": {"access_mode": "scoped", "store_ids": [1]}})
    db = connector()
    result = make_db_tool(db, cfg).execute_read_enforced("SELECT * FROM orders", db, datasource="warehouse")
    assert result.success is True
    assert db.execute_query.call_args.args[0] == "SELECT * FROM orders WHERE store_id = 1"
    assert result.sql_return == [{"email": "***"}]
    assert fake.contexts[-1] == cfg.policy_context


def test_db_policy_denial_does_not_execute(monkeypatch):
    fake = FakeRuntime(allowed=False, reason="access denied")
    monkeypatch.setattr(
        "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
        lambda active: {"sql-policy": lambda profile: fake},
    )
    db = connector()
    result = make_db_tool(db, config()).execute_read_enforced("SELECT * FROM orders", db)
    assert not result.success
    assert "access denied" in result.error
    db.execute_query.assert_not_called()


def test_db_result_policy_denial_fails_after_one_execution(monkeypatch):
    fake = FakeRuntime(result_allowed=False, reason="result denied")
    monkeypatch.setattr(
        "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
        lambda active: {"sql-policy": lambda profile: fake},
    )
    db = connector()
    result = make_db_tool(db, config()).execute_read_enforced("SELECT * FROM orders", db)
    assert result.success is False
    assert "result denied" in result.error
    assert result.sql_return is None
    db.execute_query.assert_called_once()


def test_db_revalidates_policy_rewrite(monkeypatch):
    fake = FakeRuntime(sql="DROP TABLE orders")
    monkeypatch.setattr(
        "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
        lambda active: {"sql-policy": lambda profile: fake},
    )
    db = connector()
    result = make_db_tool(db, config()).execute_read_enforced("SELECT * FROM orders", db)
    assert not result.success
    assert "Only read-only queries" in result.error
    db.execute_query.assert_not_called()


def test_db_rejects_empty_policy_rewrite(monkeypatch):
    class EmptyRewriteRuntime(FakeRuntime):
        def before_sql_read(self, sql, *, datasource, dialect, policy_context):
            return SimpleNamespace(allowed=True, sql="", applied_policies=["broken_rewrite"])

    monkeypatch.setattr(
        "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
        lambda active: {"sql-policy": lambda profile: EmptyRewriteRuntime()},
    )
    db = connector()
    result = make_db_tool(db, config()).execute_read_enforced("SELECT * FROM orders", db)
    assert not result.success
    assert "Only read-only queries" in result.error
    db.execute_query.assert_not_called()
