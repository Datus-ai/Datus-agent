# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datus.utils.trace_context import (
    build_benchmark_trace_context,
    build_bootstrap_trace_context,
    build_chat_trace_context,
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


def test_chat_context_uses_chat_session_as_group_not_name():
    ctx = build_chat_trace_context(
        session_id="gen_sql_summary_session_ab12cd34",
        llm_session_id="gen_sql_summary_session_ab12cd34",
        node_name="gen_sql_summary",
        datasource="starrocks",
    )

    assert ctx.name == "chat/gen_sql_summary"
    assert ctx.session_id == "gen_sql_summary_session_ab12cd34"
    assert "gen_sql_summary_session_ab12cd34" not in ctx.name
    assert ctx.metadata["service_session_id"] == "gen_sql_summary_session_ab12cd34"
