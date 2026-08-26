# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker

from datus.schemas.node_models import SqlTask
from datus.utils.benchmark_artifacts import allocate_benchmark_attempt, finalize_benchmark_attempt
from datus.utils.benchmark_trajectory import (
    build_trajectory_payload,
    trajectory_file_path,
    write_benchmark_trajectory,
)

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "benchmark_artifacts" / "v1" / "trajectory.schema.json"

ORDERS_DDL = "CREATE TABLE analytics.orders (\n    order_id BIGINT,\n    amount DECIMAL(18, 2)\n)"


def _validator() -> Draft202012Validator:
    """Build a format-checking validator for the v1 trajectory schema."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _model_config() -> SimpleNamespace:
    """Model config stub carrying private fields that must never reach trajectories."""
    return SimpleNamespace(
        type="openai",
        model="gpt-5.5",
        api_key="must-not-leak",
        base_url="https://private.invalid",
        reasoning_effort=None,
        temperature=None,
        top_p=None,
        enable_thinking=None,
    )


def _agent_config() -> MagicMock:
    """Agent config mock exposing the active model."""
    config = MagicMock()
    config.active_model.return_value = _model_config()
    return config


def _table_schema() -> dict:
    """One table schema entry as workflow nodes carry it today."""
    return {
        "identifier": "analytics.orders",
        "table_name": "orders",
        "database_name": "analytics",
        "definition": ORDERS_DDL,
        "table_type": "table",
    }


def _success_workflow() -> SimpleNamespace:
    """Workflow stub: two nodes sharing one schema body, timed, with actions."""
    now = time.time()
    schema_node = SimpleNamespace(
        id="schema_linking",
        type="schema_linking",
        status="completed",
        start_time=now - 10.0,
        end_time=now - 8.0,
        input={"input_text": "List order amounts", "table_schemas": [_table_schema()]},
        result=SimpleNamespace(
            table_schemas=[_table_schema()],
            action_history=[
                {
                    "role": "tool",
                    "tool_name": "search_table",
                    "status": "completed",
                    "output": {"raw_output": {"result": [{"table_name": "analytics.orders"}]}},
                }
            ],
            execution_stats={"tool_calls_count": 1},
        ),
        model=None,
    )
    gen_node = SimpleNamespace(
        id="gen_sql",
        type="gen_sql",
        status="completed",
        start_time=now - 8.0,
        end_time=now - 1.0,
        input={"table_schemas": [_table_schema()]},
        result=SimpleNamespace(
            action_history=[
                {
                    "role": "assistant",
                    "action_type": "message",
                    "status": "completed",
                    "content": "Generated SQL",
                    "output": {"usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}},
                }
            ],
            execution_stats={},
        ),
        model=SimpleNamespace(model_config=_model_config()),
    )
    pending_node = SimpleNamespace(
        id="output",
        type="output",
        status="pending",
        start_time=None,
        end_time=None,
        input={},
        result=None,
        model=None,
    )
    return SimpleNamespace(
        task=SqlTask(id="42", task="List order amounts", artifact_profile="benchmark_v1"),
        status="completed",
        nodes={"schema_linking": schema_node, "gen_sql": gen_node, "output": pending_node},
        node_order=["schema_linking", "gen_sql", "output"],
        context=SimpleNamespace(table_schemas=[_table_schema()], sql_contexts=[]),
        metadata={"trace_id": "trace-42", "trace_provider": "langfuse"},
    )


def _failed_workflow() -> SimpleNamespace:
    """Workflow stub whose only node failed."""
    now = time.time()
    node = SimpleNamespace(
        id="gen_sql",
        type="gen_sql",
        status="failed",
        start_time=now - 5.0,
        end_time=now - 1.0,
        input={},
        result=SimpleNamespace(error="No schema found", action_history=None, execution_stats=None),
        model=SimpleNamespace(model_config=_model_config()),
    )
    return SimpleNamespace(
        task=SqlTask(id="42", task="List order amounts", artifact_profile="benchmark_v1"),
        status="running",
        nodes={"gen_sql": node},
        node_order=["gen_sql"],
        context=SimpleNamespace(table_schemas=[], sql_contexts=[SimpleNamespace(sql_error=None, row_count=0)]),
        metadata={},
    )


def _allocate(tmp_path: Path):
    """Allocate an attempt under nested save/trajectory run roots."""
    save_root = tmp_path / "save" / "analytics" / "run-1"
    trajectory_root = tmp_path / "trajectory" / "analytics" / "run-1"
    trajectory_root.mkdir(parents=True, exist_ok=True)
    return allocate_benchmark_attempt(save_root, trajectory_root, run_id="run-1", task_id="42")


def test_success_payload_validates_and_stores_each_schema_once(tmp_path: Path) -> None:
    """A completed workflow yields a schema-valid payload with a deduplicated registry."""
    attempt = _allocate(tmp_path)
    payload = build_trajectory_payload(attempt, workflow=_success_workflow(), agent_config=_agent_config())

    _validator().validate(payload)
    assert payload["status"] == "completed"
    assert payload["run_id"] == "run-1"
    assert payload["attempt_id"] == "attempt-1"
    assert payload["partial"] is False
    assert payload["model"]["name"] == "gpt-5.5"
    assert payload["usage"] == {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    assert payload["trace"] == {"trace_id": "trace-42", "provider": "langfuse"}

    # The DDL body appears exactly once; nodes carry references only.
    assert list(payload["schemas"]) == ["table:analytics.orders"]
    assert payload["schemas"]["table:analytics.orders"]["ddl"] == ORDERS_DDL
    assert json.dumps(payload["nodes"]).count("CREATE TABLE") == 0
    assert [node["schema_refs"] for node in payload["nodes"]] == [
        ["table:analytics.orders"],
        ["table:analytics.orders"],
    ]
    assert "must-not-leak" not in json.dumps(payload)


def test_nodes_have_normalized_timing_actions_and_stats(tmp_path: Path) -> None:
    """Executed nodes expose RFC3339 timing and native-shape actions; pending nodes are excluded."""
    attempt = _allocate(tmp_path)
    payload = build_trajectory_payload(attempt, workflow=_success_workflow(), agent_config=_agent_config())

    assert [node["id"] for node in payload["nodes"]] == ["schema_linking", "gen_sql"]
    schema_node = payload["nodes"][0]
    assert schema_node["started_at"].endswith("Z") and schema_node["completed_at"].endswith("Z")
    assert schema_node["duration_seconds"] == pytest.approx(2.0)
    # Tool result is lifted out of output.raw_output.result into content.
    tool_action = schema_node["actions"][0]
    assert tool_action["name"] == "search_table"
    assert tool_action["content"] == [{"table_name": "analytics.orders"}]
    assert schema_node["execution_stats"] == {"tool_calls_count": 1}
    assert payload["execution_stats"] == {"total_nodes": 2, "completed_nodes": 2}


def test_failure_payload_validates_with_error_and_failure_types(tmp_path: Path) -> None:
    """A failed workflow yields a schema-valid payload with structured error data."""
    attempt = _allocate(tmp_path)
    payload = build_trajectory_payload(attempt, workflow=_failed_workflow(), agent_config=_agent_config())

    _validator().validate(payload)
    assert payload["status"] == "failed"
    assert payload["failure_types"] == [payload["error"]["type"]]
    assert payload["nodes"][0]["error"]["message"] == "No schema found"


def test_exception_payload_validates_without_workflow(tmp_path: Path) -> None:
    """A crash before workflow init still yields a schema-valid failed payload."""
    attempt = _allocate(tmp_path)
    payload = build_trajectory_payload(
        attempt,
        workflow=None,
        agent_config=_agent_config(),
        exception=RuntimeError("plan generation failed"),
    )

    _validator().validate(payload)
    assert payload["status"] == "failed"
    assert payload["error"]["message"] == "plan generation failed"
    assert payload["nodes"] == []


def test_written_yaml_uses_block_scalars_and_deterministic_path(tmp_path: Path) -> None:
    """The YAML file keeps DDL readable and uses the attempt-based filename."""
    attempt = _allocate(tmp_path)
    path = write_benchmark_trajectory(attempt, workflow=_success_workflow(), agent_config=_agent_config())

    assert path == trajectory_file_path(attempt)
    assert path.name == "task_42.attempt-1.yaml"
    text = path.read_text(encoding="utf-8")
    assert "ddl: |" in text
    assert "\\n" not in text.split("ddl: |")[1].split("metadata:")[0]
    assert yaml.safe_load(text)["artifact_type"] == "trajectory"
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_manifest_references_native_trajectory(tmp_path: Path) -> None:
    """finalize_benchmark_attempt records a native_v1 trajectory reference."""
    attempt = _allocate(tmp_path)
    workflow = _success_workflow()
    trajectory_path = write_benchmark_trajectory(attempt, workflow=workflow, agent_config=_agent_config())
    (attempt.output_dir / "42.sql").write_text("SELECT 1", encoding="utf-8")
    (attempt.output_dir / "42.csv").write_text("a\n1\n", encoding="utf-8")

    # The success stub has no completed output node, so add one for finalize.
    workflow.nodes["output"] = SimpleNamespace(
        id="output",
        type="output",
        status="completed",
        start_time=None,
        end_time=None,
        input={},
        result=SimpleNamespace(success=True, action_history=None, execution_stats=None),
        model=None,
    )
    manifest_path = finalize_benchmark_attempt(
        attempt,
        task=workflow.task,
        workflow=workflow,
        trajectory_path=trajectory_path,
        agent_config=_agent_config(),
        trajectory_profile="native_v1",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["trajectory"] == {
        "root": "trajectory",
        "path": "task_42.attempt-1.yaml",
        "format": "yaml",
        "schema_version": 1,
        "contract_profile": "native_v1",
        "attempt_id": "attempt-1",
    }
