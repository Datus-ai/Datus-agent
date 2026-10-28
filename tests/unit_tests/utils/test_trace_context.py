# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datus.utils.trace_context import (
    TRACE_IDENTITY_ATTRIBUTE_PREFIX,
    build_benchmark_trace_context,
    build_bootstrap_trace_context,
    build_chat_trace_context,
    build_trace_span_attributes,
    build_workflow_trace_context_from_runner,
    resolve_trace_identity,
    set_trace_identity_provider,
    trace_context,
)


def test_benchmark_context_uses_run_id_as_session_group():
    ctx = build_benchmark_trace_context(
        benchmark="baisheng",
        run_id="semantic_model_20260520_054027",
        task_id="1",
        workflow="baisheng_semantic_model",
        datasource="starrocks",
    )

    assert ctx.name == "benchmark/baisheng/semantic_model/task-1"
    assert ctx.session_id == "benchmark:semantic_model_20260520_054027"
    assert "task:1" in ctx.tags
    assert ctx.metadata["benchmark_run_id"] == "semantic_model_20260520_054027"
    assert ctx.metadata["context_type"] == "semantic_model"


def test_bootstrap_context_names_datasource_and_components():
    ctx = build_bootstrap_trace_context(
        datasource="starrocks",
        components=["metadata", "semantic_model"],
        strategy="incremental",
        stream_id="stream-1",
    )

    assert ctx.name == "bootstrap-kb/starrocks/metadata+semantic_model"
    assert ctx.session_id == "bootstrap:stream-1"
    assert "component:metadata" in ctx.tags
    assert ctx.metadata["components"] == ["metadata", "semantic_model"]


def test_generated_session_ids_keep_operation_prefixes():
    benchmark_ctx = build_benchmark_trace_context(
        benchmark="baisheng",
        run_id="",
        task_id="1",
    )
    bootstrap_ctx = build_bootstrap_trace_context(
        datasource="starrocks",
        components=["metadata"],
    )

    assert benchmark_ctx.session_id.startswith("benchmark:20")
    assert not benchmark_ctx.session_id.startswith("benchmark:cli:")
    assert bootstrap_ctx.session_id.startswith("bootstrap:20")
    assert not bootstrap_ctx.session_id.startswith("bootstrap:cli:")


def test_workflow_context_uses_workflow_operation_prefix():
    class Args:
        workflow = None
        datasource = "starrocks"

    class Config:
        current_datasource = "mysql"
        home = "/tmp/datus"

    class Runner:
        args = Args()
        global_config = Config()
        run_id = "run-1"

    ctx = build_workflow_trace_context_from_runner(Runner())

    assert ctx.name == "workflow/default"
    assert ctx.session_id == "workflow:run:run-1"
    assert "workflow" in ctx.tags
    assert "workflow:default" in ctx.tags
    assert "cli" not in ctx.tags
    assert ctx.metadata["datus_component"] == "workflow_run"
    assert ctx.metadata["workflow"] == "default"


def test_chat_context_uses_chat_session_as_group_not_name():
    ctx = build_chat_trace_context(
        session_id="gen_sql_summary_session_ab12cd34",
        llm_session_id="gen_sql_summary_session_ab12cd34",
        node_name="gen_sql_summary",
        datasource="starrocks",
    )

    assert ctx.name == "agent/gen_sql_summary"
    assert ctx.session_id == "gen_sql_summary_session_ab12cd34"
    assert "gen_sql_summary_session_ab12cd34" not in ctx.name
    assert ctx.metadata["service_session_id"] == "gen_sql_summary_session_ab12cd34"


def test_agents_run_config_does_not_repeat_trace_leaf_agent_name():
    cases = [
        ("chat", "agent/chat"),
        ("chatbot", "agent/chatbot"),
    ]
    for agent_name, expected_name in cases:
        ctx = build_chat_trace_context(
            session_id=f"{agent_name}_session_ab12cd34",
            llm_session_id=f"{agent_name}_session_ab12cd34",
            node_name=agent_name,
        )

        kwargs = ctx.agents_run_config_kwargs(agent_name=agent_name)

        assert ctx.name == expected_name
        assert kwargs["workflow_name"] == expected_name
        assert kwargs["group_id"] == f"{agent_name}_session_ab12cd34"
        assert kwargs["trace_metadata"]["agent_name"] == agent_name


def test_agents_run_config_appends_distinct_agent_name():
    ctx = build_benchmark_trace_context(
        benchmark="baisheng",
        run_id="semantic_model_20260520_054027",
        task_id="1",
        workflow="baisheng_semantic_model",
    )

    kwargs = ctx.agents_run_config_kwargs(agent_name="gen_sql")

    assert kwargs["workflow_name"] == "benchmark/baisheng/semantic_model/task-1/gen_sql"
    assert kwargs["group_id"] == "benchmark:semantic_model_20260520_054027"
    assert kwargs["trace_metadata"]["agent_name"] == "gen_sql"


def test_trace_span_attributes_include_provider_neutral_identity():
    ctx = build_chat_trace_context(
        session_id="gen_sql_summary_session_ab12cd34",
        llm_session_id="gen_sql_summary_session_ab12cd34",
        node_name="gen_sql_summary",
        user_id="user-1",
        datasource="starrocks",
        extra={"run_id": "run-1"},
    )

    with trace_context(ctx, replace=True):
        attrs = build_trace_span_attributes(operation="gen_sql_summary", run_type="llm")

    assert attrs["datus.operation"] == "gen_sql_summary"
    assert attrs["datus.run_type"] == "llm"
    assert attrs["datus.trace.name"] == "agent/gen_sql_summary"
    assert attrs["datus.session_id"] == "gen_sql_summary_session_ab12cd34"
    assert attrs["datus.user_id"] == "user-1"
    assert attrs["datus.run_id"] == "run-1"
    assert attrs["datus.metadata.service_session_id"] == "gen_sql_summary_session_ab12cd34"


def test_trace_span_attributes_normalize_complex_metadata_values():
    ctx = build_chat_trace_context(
        session_id="chat_session_ab12cd34",
        node_name="chat",
        extra={"payload": {"tables": ["schools"]}, "attempt": 2},
    )

    attrs = build_trace_span_attributes(operation="chat", run_type="llm", ctx=ctx)

    assert attrs["datus.metadata.payload"] == '{"tables": ["schools"]}'
    assert attrs["datus.metadata.attempt"] == 2


def test_chat_context_carries_agent_owned_join_keys():
    """max_turns and release are Datus' own facts and stay first-class."""
    ctx = build_chat_trace_context(
        session_id="chat_session_1",
        node_name="olist_order_analyst",
        user_id="u1",
        max_turns=30,
        release="0.4.0",
    )

    assert ctx.metadata["max_turns"] == 30
    assert ctx.metadata["release"] == "0.4.0"
    assert ctx.metadata["node_name"] == "olist_order_analyst"


def test_chat_context_forwards_host_identity_verbatim():
    """Datus must not interpret the host's identity fields, only carry them."""
    ctx = build_chat_trace_context(
        session_id="chat_session_1",
        node_name="chat",
        identity={"workspace_id": "ws_abc", "tenant": "acme"},
    )

    assert ctx.identity == {"workspace_id": "ws_abc", "tenant": "acme"}
    # Tags are the only filterable trace field, so each identity field is tagged
    # under the host's own key -- no Datus-side vocabulary.
    assert "workspace_id:ws_abc" in ctx.tags
    assert "tenant:acme" in ctx.tags


def test_chat_context_without_a_host_carries_no_identity():
    """A plain OSS install registers no provider and must produce no empty tags."""
    ctx = build_chat_trace_context(session_id="chat_session_1", node_name="chat")

    assert ctx.identity == {}
    assert ctx.tags == ("chat", "agent:chat")


def test_identity_reaches_span_attributes_under_its_own_namespace():
    """Exporters forward this prefix wholesale; a field that stops here is lost."""
    ctx = build_chat_trace_context(
        session_id="chat_session_1",
        node_name="chat",
        max_turns=30,
        identity={"workspace_id": "ws_abc", "request_trace_id": "abc123"},
    )
    attrs = build_trace_span_attributes(operation="chat", ctx=ctx)

    assert attrs[f"{TRACE_IDENTITY_ATTRIBUTE_PREFIX}workspace_id"] == "ws_abc"
    assert attrs[f"{TRACE_IDENTITY_ATTRIBUTE_PREFIX}request_trace_id"] == "abc123"
    assert attrs["datus.metadata.max_turns"] == 30


class TestTraceIdentityProvider:
    """The host names its runs; Datus only asks."""

    def teardown_method(self):
        set_trace_identity_provider(None)

    def test_no_provider_means_no_identity(self):
        set_trace_identity_provider(None)
        assert resolve_trace_identity() == {}

    def test_provider_reads_the_hosts_own_request_state(self):
        """The hook takes no arguments; a contextvar the host set is still visible."""
        import contextvars

        host_var = contextvars.ContextVar("host_request", default=None)
        set_trace_identity_provider(lambda: {"workspace_id": host_var.get()})

        host_var.set("ws_abc")
        assert resolve_trace_identity() == {"workspace_id": "ws_abc"}

    def test_values_are_coerced_and_blanks_dropped(self):
        set_trace_identity_provider(lambda: {"a": 7, "b": None, "c": "  ", "d": " x "})
        assert resolve_trace_identity() == {"a": "7", "d": "x"}

    def test_a_broken_provider_costs_identity_not_the_run(self):
        def boom():
            raise RuntimeError("host hook is broken")

        set_trace_identity_provider(boom)
        assert resolve_trace_identity() == {}

    def test_a_provider_returning_junk_is_ignored(self):
        set_trace_identity_provider(lambda: "not a mapping")
        assert resolve_trace_identity() == {}
