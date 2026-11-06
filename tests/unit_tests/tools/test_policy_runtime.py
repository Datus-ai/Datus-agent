import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.database import DBFuncTool
from datus.tools.policy_runtime import PolicyRuntime
from datus.utils.exceptions import DatusException, ErrorCode


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


def test_effective_snapshot_without_provider_fails_closed(monkeypatch):
    monkeypatch.setattr("datus.tools.policy_runtime.collect_plugin_policy_runtime_factories", lambda active: {})
    runtime = PolicyRuntime(config())
    with pytest.raises(DatusException, match="no implementation is loaded"):
        runtime.before_sql_read(
            "SELECT 1",
            datasource="warehouse",
            dialect="duckdb",
            policy_context={"sql_policy": {"version": "v1", "policies": {}}},
        )


@pytest.mark.parametrize("active_names", [None, {"sql-policy"}])
def test_configured_but_missing_plugin_fails_closed(monkeypatch, active_names):
    monkeypatch.setattr("datus.tools.policy_runtime.collect_plugin_policy_runtime_factories", lambda active: {})
    cfg = config()
    cfg.active_plugin_names = lambda: active_names
    cfg.get_plugin_profile = lambda _: {"column_masking": [{"column": "db.main.orders.phone", "mask": "last4"}]}
    with pytest.raises(DatusException, match="Required SQL policy plugin"):
        PolicyRuntime(cfg)


def test_empty_legacy_metric_provider_needs_no_connection(monkeypatch):
    legacy = SimpleNamespace(policies=[], before_metric_read=lambda **kwargs: None)
    runtime = runtime_with(legacy, monkeypatch)
    assert not runtime.requires_bound_read({})
    assert runtime.bind_read(connector=None, datasource="warehouse", policy_context={}) is None


@pytest.mark.parametrize("sql", ["SHOW TABLES", "DESCRIBE main.orders"])
def test_only_plain_metadata_inspection_can_bypass_source_rewrite(sql):
    from datus.tools.policy_runtime import _plain_metadata_read

    assert _plain_metadata_read(sql, "duckdb")


@pytest.mark.parametrize(
    "sql",
    [
        "DESCRIBE SELECT * FROM orders",
        "PRAGMA storage_info('orders')",
        "SHOW TABLES; SELECT * FROM orders",
        "EXPLAIN ANALYZE SELECT * FROM orders",
    ],
)
def test_query_bodies_cannot_bypass_rewrite_as_metadata(sql):
    from datus.tools.policy_runtime import _plain_metadata_read

    assert not _plain_metadata_read(sql, "duckdb")


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
        patch("datus.tools.func_tool.database.SemanticDatasetRAG") as mock_sem,
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


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO])
def test_sql_debug_logs_match_the_statement_sent_to_the_connector(monkeypatch, caplog, level):
    original = "SELECT * FROM orders"
    rewritten = "SELECT * FROM orders WHERE store_id = 1"
    fake = FakeRuntime(sql=rewritten)
    monkeypatch.setattr(
        "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
        lambda active: {"sql-policy": lambda profile: fake},
    )
    cfg = config({"row_filter": {"access_mode": "scoped", "store_ids": [1]}, "private_marker": "not-for-logs"})
    db = connector()
    with caplog.at_level(level):
        result = make_db_tool(db, cfg).execute_read_enforced(original, db, datasource="warehouse")
    assert result.success
    db.execute_query.assert_called_once_with(rewritten, result_format="list")
    messages = [record.getMessage() for record in caplog.records]
    sql_logs = [message for message in messages if "sql.policy.rewrite" in message or "sql.execute" in message]
    if level == logging.INFO:
        assert sql_logs == []
    else:
        assert len(sql_logs) == 2
        assert original in sql_logs[0] and rewritten in sql_logs[0]
        assert rewritten in sql_logs[1]
        assert "not-for-logs" not in "\n".join(sql_logs)


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


def test_db_policy_denial_is_coded_and_unprefixed(monkeypatch):
    """A refusal has to be distinguishable from a broken query.

    Both used to arrive as ``TOOL_INVALID_INPUT`` inside a string, so the only
    way to tell "you may not see this" from "your SQL is wrong" was to match
    the sentence — and an API surface has to tell them apart to know whether a
    retry button means anything.
    """
    fake = FakeRuntime(allowed=False, reason="you have no value for store_id")
    monkeypatch.setattr(
        "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
        lambda active: {"sql-policy": lambda profile: fake},
    )
    db = connector()
    result = make_db_tool(db, config()).execute_read_enforced("SELECT * FROM orders", db)

    assert result.error_code == ErrorCode.POLICY_DENIED.code
    # Reaches the reader as written, without the wire prefix in front of it.
    assert result.error == "you have no value for store_id"


def test_db_result_policy_denial_is_coded_too(monkeypatch):
    fake = FakeRuntime(result_allowed=False, reason="you may not see these rows")
    monkeypatch.setattr(
        "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
        lambda active: {"sql-policy": lambda profile: fake},
    )
    db = connector()
    result = make_db_tool(db, config()).execute_read_enforced("SELECT * FROM orders", db)

    assert result.error_code == ErrorCode.POLICY_DENIED.code
    assert result.error == "you may not see these rows"


def test_db_non_policy_failure_keeps_the_prefixed_form(monkeypatch):
    """Only a refusal is unwrapped — everything else keeps what the logs expect."""
    db = connector()
    result = make_db_tool(db, config()).execute_read_enforced("SELECT 1; DROP TABLE t", db)

    assert result.success is False
    assert result.error_code is None
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


class TestDenialExplanation:
    """SaaS knows why a caller was denied; the plugin does not.

    A plugin can only report that context validation failed. "Policy context
    denies all data reads" is true and useless: the reader cannot tell a gap in
    their own profile from a misconfigured policy, and has nothing to ask for.
    Datus-backend attaches the reason to the context under a key the plugins
    ignore, and it is appended here.
    """

    DENIED = {
        "row_filter": {"access_mode": "denied"},
        "x_saas_denial": "you have no value for store_ids, which this project's row policies filter by.",
    }

    def test_validate_context_appends_it(self, monkeypatch):
        rt = runtime_with(FakeRuntime(allowed=False, reason="Policy context denies all data reads"), monkeypatch)

        decision = rt.validate_context(self.DENIED)

        assert decision.allowed is False
        assert "denies all data reads" in decision.reason
        assert "store_ids" in decision.reason

    def test_before_sql_read_appends_it(self, monkeypatch):
        rt = runtime_with(FakeRuntime(allowed=False, reason="Policy context denies all data reads"), monkeypatch)

        decision = rt.before_sql_read(
            "SELECT * FROM orders", datasource="w", dialect="postgres", policy_context=self.DENIED
        )

        assert decision.allowed is False
        assert "store_ids" in decision.reason

    def test_without_the_key_the_reason_is_untouched(self, monkeypatch):
        rt = runtime_with(FakeRuntime(allowed=False, reason="Policy context denies all data reads"), monkeypatch)

        decision = rt.validate_context({"row_filter": {"access_mode": "denied"}})

        assert decision.reason == "Policy context denies all data reads"

    @pytest.mark.parametrize("junk", [None, "", "   ", 42, {"a": 1}])
    def test_a_malformed_key_is_ignored(self, monkeypatch, junk):
        """It arrives over the wire; it must not be able to break a refusal."""
        rt = runtime_with(FakeRuntime(allowed=False, reason="denied"), monkeypatch)

        decision = rt.validate_context({"row_filter": {"access_mode": "denied"}, "x_saas_denial": junk})

        assert decision.allowed is False
        assert decision.reason == "denied"

    def test_an_allowed_context_is_unaffected(self, monkeypatch):
        rt = runtime_with(FakeRuntime(), monkeypatch)

        assert rt.validate_context(self.DENIED).allowed is True


@pytest.mark.parametrize("initial_where", [None, "amount > 0"])
@pytest.mark.parametrize("returned_fields", [{}, {"where": None}])
def test_metric_hook_without_rewrite_preserves_input(monkeypatch, initial_where, returned_fields):
    hook = Mock(return_value=SimpleNamespace(allowed=True, **returned_fields))
    runtime = runtime_with(SimpleNamespace(before_metric_read=hook), monkeypatch)

    result = runtime.before_metric_read(datasets={"orders"}, where=initial_where, policy_context={})

    assert result.allowed
    assert result.where == initial_where


def test_metric_hook_none_preserves_earlier_policy_filters(monkeypatch):
    original = "amount > 0"
    scoped = "(amount > 0) AND store_id = 7"
    first = Mock(return_value=SimpleNamespace(allowed=True, where=scoped, applied_policies=["store_scope"]))
    second = Mock(return_value=SimpleNamespace(allowed=True, where=None))
    third = Mock(return_value=SimpleNamespace(allowed=True))
    monkeypatch.setattr(
        "datus.tools.policy_runtime.collect_plugin_policy_runtime_factories",
        lambda _: {
            "scope": lambda _: SimpleNamespace(before_metric_read=first),
            "check": lambda _: SimpleNamespace(before_metric_read=second),
            "observe": lambda _: SimpleNamespace(before_metric_read=third),
        },
    )
    runtime = PolicyRuntime(config())

    result = runtime.before_metric_read(datasets={"orders"}, where=original, policy_context={})

    assert first.call_args.kwargs["where"] == original
    assert second.call_args.kwargs["where"] == scoped
    assert third.call_args.kwargs["where"] == scoped
    assert result.where == scoped
    assert result.applied_policies == ["store_scope"]


@pytest.mark.parametrize("invalid_where", [False, 0, [], {}])
def test_metric_hook_rejects_non_string_filters(monkeypatch, invalid_where):
    hook = Mock(return_value=SimpleNamespace(allowed=True, where=invalid_where))
    runtime = runtime_with(SimpleNamespace(before_metric_read=hook), monkeypatch)

    with pytest.raises(DatusException, match="non-string filter") as exc:
        runtime.before_metric_read(datasets={"orders"}, where="amount > 0", policy_context={})

    assert exc.value.code == ErrorCode.COMMON_CONFIG_ERROR
