# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from datus.agent.agent import Agent
from datus.schemas.node_models import SqlTask
from datus.utils.benchmark_artifacts import (
    BenchmarkAttempt,
    allocate_benchmark_attempt,
    finalize_benchmark_attempt,
    load_task_output_manifest,
    resolve_task_output_path,
    resolve_task_trajectory_path,
)
from datus.utils.benchmark_utils import AgentResultSqlProvider, CsvPerTaskResultProvider
from datus.utils.exceptions import DatusException

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "benchmark_artifacts" / "v1" / "task-output.schema.json"
)


def _validator() -> Draft202012Validator:
    """Build a format-checking validator for the v1 task-output schema."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _model_config() -> SimpleNamespace:
    """Model config stub carrying private fields that must never reach manifests."""
    return SimpleNamespace(
        type="openai",
        model="gpt-5.5",
        api_key="must-not-leak",
        base_url="https://private.invalid",
        reasoning_effort="medium",
        temperature=None,
        top_p=None,
        enable_thinking=False,
    )


def _agent_config() -> MagicMock:
    """Agent config mock exposing the active model."""
    config = MagicMock()
    config.active_model.return_value = _model_config()
    return config


def _task(task_id: str = "42") -> SqlTask:
    """Benchmark SqlTask fixture."""
    return SqlTask(
        id=task_id,
        datasource="analytics",
        database_name="warehouse",
        task="List order amounts",
        artifact_profile="benchmark_v1",
    )


def _success_workflow(task: SqlTask) -> SimpleNamespace:
    """Workflow stub with token usage and a completed output node."""
    model = SimpleNamespace(model_config=_model_config())
    gen_result = SimpleNamespace(
        action_history=[
            {
                "role": "assistant",
                "action_type": "message",
                "output": {"usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}},
            },
            {
                "role": "workflow",
                "action_type": "token_usage",
                "output": {
                    "delta": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
                    "cumulative": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
                },
            },
        ],
        execution_stats={"total_tokens": 14},
    )
    nodes = {
        "gen_sql": SimpleNamespace(
            id="gen_sql",
            type="gen_sql",
            status="completed",
            result=gen_result,
            model=model,
        ),
        "output": SimpleNamespace(
            id="output",
            type="output",
            status="completed",
            result=SimpleNamespace(success=True),
            model=None,
        ),
    }
    return SimpleNamespace(
        task=task,
        status="completed",
        nodes=nodes,
        node_order=["gen_sql", "output"],
        context=SimpleNamespace(sql_contexts=[SimpleNamespace(row_count=2, sql_error=None)]),
    )


def _failed_workflow(task: SqlTask, node_type: str, message: str) -> SimpleNamespace:
    """Workflow stub whose node failed with the given type and message."""
    node = SimpleNamespace(
        id="failed_node",
        type=node_type,
        status="failed",
        result=SimpleNamespace(error=message, action_history=None, execution_stats=None),
        model=SimpleNamespace(model_config=_model_config()),
    )
    return SimpleNamespace(
        task=task,
        status="running",
        nodes={"failed_node": node},
        node_order=["failed_node"],
        context=SimpleNamespace(
            sql_contexts=[SimpleNamespace(row_count=0, sql_error=message if node_type == "execute_sql" else None)]
        ),
    )


def _allocate(tmp_path: Path, task_id: str = "42") -> BenchmarkAttempt:
    """Allocate an attempt under a nested datasource/run root."""
    save_root = tmp_path / "save" / "analytics" / "run-1"
    trajectory_root = tmp_path / "trajectory" / "analytics" / "run-1"
    trajectory_root.mkdir(parents=True, exist_ok=True)
    return allocate_benchmark_attempt(
        save_root,
        trajectory_root,
        run_id="run-1",
        task_id=task_id,
    )


def _write_success_files(attempt: BenchmarkAttempt) -> Path:
    """Write canonical SQL/CSV outputs and a trajectory file for the attempt."""
    (attempt.output_dir / f"{attempt.task_id}.sql").write_text(
        "SELECT order_id, amount FROM analytics.orders",
        encoding="utf-8",
    )
    (attempt.output_dir / f"{attempt.task_id}.csv").write_text(
        "order_id,amount\n1,10.00\n2,15.00\n",
        encoding="utf-8",
    )
    trajectory = attempt.trajectory_run_root / f"{attempt.task_id}_123.yaml"
    trajectory.write_text("schema_version: 1\nworkflow: {}\n", encoding="utf-8")
    return trajectory


def test_success_manifest_validates_and_does_not_inline_query_result(tmp_path: Path) -> None:
    """A successful attempt yields a schema-valid manifest without inlined query results."""
    attempt = _allocate(tmp_path)
    task = _task()
    trajectory = _write_success_files(attempt)

    manifest_path = finalize_benchmark_attempt(
        attempt,
        task=task,
        workflow=_success_workflow(task),
        trajectory_path=trajectory,
        agent_config=_agent_config(),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    _validator().validate(payload)
    assert payload["status"] == "completed"
    assert payload["attempt_id"] == "attempt-1"
    assert payload["usage"] == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    assert payload["model"] == {
        "provider": "openai",
        "name": "gpt-5.5",
        "configuration": {"reasoning_effort": "medium", "enable_thinking": False},
    }
    assert payload["trajectory"]["contract_profile"] == "compatibility_v1"
    assert payload["trajectory"]["attempt_id"] == "attempt-1"
    assert "sql_result" not in payload
    assert "1,10.00" not in json.dumps(payload)
    assert "must-not-leak" not in json.dumps(payload)
    assert all(not Path(output["path"]).is_absolute() for output in payload["outputs"])
    assert not list(manifest_path.parent.glob(f".{manifest_path.name}.*.tmp"))

    legacy_json = json.loads((attempt.save_run_root / "42.json").read_text(encoding="utf-8"))
    assert legacy_json["finished"] is True
    assert "sql_result" not in legacy_json
    assert "gen_sql" not in legacy_json
    assert legacy_json["result"] == "42.csv"
    assert (attempt.save_run_root / "42.csv").read_text(encoding="utf-8").startswith("order_id,amount")


@pytest.mark.parametrize(
    ("node_type", "expected_error_type"),
    [("execute_sql", "sql_execution"), ("schema_linking", "node_failure")],
)
def test_failure_manifest_validates_before_output_node(
    tmp_path: Path, node_type: str, expected_error_type: str
) -> None:
    """Failures before the output node still produce schema-valid failure manifests."""
    attempt = _allocate(tmp_path)
    task = _task()
    trajectory = attempt.trajectory_run_root / "42_456.yaml"
    trajectory.write_text("schema_version: 1\nworkflow: {}\n", encoding="utf-8")

    manifest_path = finalize_benchmark_attempt(
        attempt,
        task=task,
        workflow=_failed_workflow(task, node_type, "Unknown column customer_tier"),
        trajectory_path=trajectory,
        agent_config=_agent_config(),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    _validator().validate(payload)
    assert payload["status"] == "failed"
    assert payload["outputs"] == []
    assert payload["error"]["type"] == expected_error_type
    assert payload["error"]["details"]["node_type"] == node_type
    assert not (attempt.save_run_root / "42.csv").exists()
    assert json.loads((attempt.save_run_root / "42.json").read_text())["finished"] is False


def test_uninitialized_workflow_still_writes_valid_failure_manifest(tmp_path: Path) -> None:
    """A crash before workflow init still yields a valid failure manifest."""
    attempt = _allocate(tmp_path)

    manifest_path = finalize_benchmark_attempt(
        attempt,
        task=_task(),
        workflow=None,
        trajectory_path=None,
        agent_config=_agent_config(),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    _validator().validate(payload)
    assert payload["status"] == "failed"
    assert payload["error"] == {
        "type": "benchmark_execution",
        "message": "Workflow did not initialize",
        "code": None,
        "retryable": None,
        "details": {},
    }


def test_manifest_uses_token_event_deltas_when_actions_have_no_usage(tmp_path: Path) -> None:
    """Token event deltas are used when assistant actions carry no usage."""
    attempt = _allocate(tmp_path)
    task = _task()
    trajectory = _write_success_files(attempt)
    workflow = _success_workflow(task)
    workflow.nodes["gen_sql"].result.action_history = [
        {
            "action_type": "token_usage",
            "output": {
                "delta": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cached_tokens": 3,
                    "total_tokens": 14,
                }
            },
        }
    ]

    manifest_path = finalize_benchmark_attempt(
        attempt,
        task=task,
        workflow=workflow,
        trajectory_path=trajectory,
        agent_config=_agent_config(),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["usage"] == {
        "input_tokens": 10,
        "output_tokens": 4,
        "cached_input_tokens": 3,
        "total_tokens": 14,
    }


def test_attempt_allocation_is_retry_and_concurrency_safe(tmp_path: Path) -> None:
    """Concurrent allocations receive distinct attempt directories."""
    save_root = tmp_path / "save"
    trajectory_root = tmp_path / "trajectory"

    def allocate(_: int) -> BenchmarkAttempt:
        """Allocate one attempt for the shared task."""
        return allocate_benchmark_attempt(
            save_root,
            trajectory_root,
            run_id="run-1",
            task_id="42",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        attempts = list(executor.map(allocate, range(8)))

    assert {attempt.attempt_id for attempt in attempts} == {f"attempt-{index}" for index in range(1, 9)}
    assert len({attempt.output_dir for attempt in attempts}) == 8


def test_attempt_requires_run_id_and_missing_manifest_is_explicit(tmp_path: Path) -> None:
    """Empty run ids are rejected and missing manifests load as None."""
    with pytest.raises(DatusException, match="run_id must be non-empty"):
        allocate_benchmark_attempt(tmp_path / "save", tmp_path / "trajectory", run_id="", task_id="42")

    assert load_task_output_manifest(tmp_path / "save", "42") is None


def test_retry_updates_manifest_without_overwriting_prior_attempt(tmp_path: Path) -> None:
    """A retry refreshes the manifest while keeping the prior attempt's files."""
    task = _task()
    first = _allocate(tmp_path)
    first_trajectory = _write_success_files(first)
    finalize_benchmark_attempt(
        first,
        task=task,
        workflow=_success_workflow(task),
        trajectory_path=first_trajectory,
        agent_config=_agent_config(),
    )

    second = _allocate(tmp_path)
    second_trajectory = second.trajectory_run_root / "42_456.yaml"
    second_trajectory.write_text("schema_version: 1\nworkflow: {}\n", encoding="utf-8")
    finalize_benchmark_attempt(
        second,
        task=task,
        workflow=_failed_workflow(task, "schema_linking", "No schema found"),
        trajectory_path=second_trajectory,
        agent_config=_agent_config(),
    )

    manifest = load_task_output_manifest(first.save_run_root, "42")
    assert manifest["attempt_id"] == "attempt-2"
    assert manifest["status"] == "failed"
    assert (first.output_dir / "42.csv").exists()
    assert second.output_dir.exists()
    assert not (first.save_run_root / "42.csv").exists()


@pytest.mark.parametrize("task_id", ["../42", "task/42", "task\\42", "C:drive", "."])
def test_attempt_rejects_nonportable_task_ids(tmp_path: Path, task_id: str) -> None:
    """Task ids that are not portable path segments are rejected."""
    with pytest.raises(DatusException, match="portable path segment"):
        allocate_benchmark_attempt(tmp_path / "save", tmp_path / "trajectory", run_id="run-1", task_id=task_id)


def test_manifest_resolvers_use_authoritative_relative_paths(tmp_path: Path) -> None:
    """Resolvers return the manifest-declared files inside the run roots."""
    attempt = _allocate(tmp_path)
    task = _task()
    trajectory = _write_success_files(attempt)
    finalize_benchmark_attempt(
        attempt,
        task=task,
        workflow=_success_workflow(task),
        trajectory_path=trajectory,
        agent_config=_agent_config(),
    )

    assert (
        resolve_task_output_path(attempt.save_run_root, "42", "sql_result") == (attempt.output_dir / "42.csv").resolve()
    )
    assert resolve_task_trajectory_path(attempt.save_run_root, attempt.trajectory_run_root, "42") == (
        trajectory.resolve()
    )


@pytest.mark.parametrize("bad_path", ["../outside.sql", "/etc/passwd", "tasks\\42\\evil.sql"])
def test_manifest_resolvers_reject_escaping_paths(tmp_path: Path, bad_path: str) -> None:
    """Lexically escaping manifest paths raise DatusException."""
    attempt = _allocate(tmp_path)
    task = _task()
    trajectory = _write_success_files(attempt)
    finalize_benchmark_attempt(
        attempt,
        task=task,
        workflow=_success_workflow(task),
        trajectory_path=trajectory,
        agent_config=_agent_config(),
    )
    payload = json.loads(attempt.manifest_path.read_text(encoding="utf-8"))
    payload["outputs"][0]["path"] = bad_path
    payload["trajectory"]["path"] = bad_path
    attempt.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatusException, match="invalid benchmark output path"):
        resolve_task_output_path(attempt.save_run_root, "42", "generated_sql")
    with pytest.raises(DatusException, match="invalid benchmark trajectory path"):
        resolve_task_trajectory_path(attempt.save_run_root, attempt.trajectory_run_root, "42")


def test_manifest_resolvers_reject_symlink_escapes(tmp_path: Path) -> None:
    """Symlinked manifest paths resolving outside the run root raise DatusException."""
    attempt = _allocate(tmp_path)
    task = _task()
    trajectory = _write_success_files(attempt)
    finalize_benchmark_attempt(
        attempt,
        task=task,
        workflow=_success_workflow(task),
        trajectory_path=trajectory,
        agent_config=_agent_config(),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil.sql").write_text("SELECT 999", encoding="utf-8")
    (outside / "evil.yaml").write_text("workflow: {}\n", encoding="utf-8")
    try:
        # Setup-only guard: the OSError comes from symlink creation on platforms
        # without the privilege (e.g. Windows sans Developer Mode), never from
        # the resolvers under test.
        (attempt.save_run_root / "link").symlink_to(outside, target_is_directory=True)
        (attempt.trajectory_run_root / "link").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"platform cannot create symlinks: {exc}")  # audit-noqa: try_except_skip

    payload = json.loads(attempt.manifest_path.read_text(encoding="utf-8"))
    payload["outputs"][0]["path"] = "link/evil.sql"
    payload["trajectory"]["path"] = "link/evil.yaml"
    attempt.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatusException, match="escapes its run root"):
        resolve_task_output_path(attempt.save_run_root, "42", "generated_sql")
    with pytest.raises(DatusException, match="escapes its run root"):
        resolve_task_trajectory_path(attempt.save_run_root, attempt.trajectory_run_root, "42")


def test_internal_evaluators_prefer_manifest_over_stale_flat_files(tmp_path: Path) -> None:
    """Providers read attempt files via the manifest, ignoring stale flat aliases."""
    attempt = _allocate(tmp_path)
    task = _task()
    trajectory = _write_success_files(attempt)
    finalize_benchmark_attempt(
        attempt,
        task=task,
        workflow=_success_workflow(task),
        trajectory_path=trajectory,
        agent_config=_agent_config(),
    )
    (attempt.save_run_root / "42.csv").unlink()
    (attempt.save_run_root / "42.csv").write_text("wrong\n999\n", encoding="utf-8")
    (attempt.save_run_root / "42.sql").unlink()
    (attempt.save_run_root / "42.sql").write_text("SELECT 999", encoding="utf-8")

    result = CsvPerTaskResultProvider(str(attempt.save_run_root)).fetch("42")
    sql = AgentResultSqlProvider(str(tmp_path), datasource="save", run_id="analytics/run-1").fetch("42")

    assert result.dataframe["order_id"].tolist() == [1, 2]
    assert sql.sql == "SELECT order_id, amount FROM analytics.orders"


def test_sql_provider_reports_missing_canonical_sql_instead_of_legacy_json(tmp_path: Path) -> None:
    """A manifest with missing canonical SQL errors out instead of using legacy JSON."""
    attempt = _allocate(tmp_path)
    task = _task()
    trajectory = _write_success_files(attempt)
    finalize_benchmark_attempt(
        attempt,
        task=task,
        workflow=_success_workflow(task),
        trajectory_path=trajectory,
        agent_config=_agent_config(),
    )
    (attempt.output_dir / "42.sql").unlink()
    (attempt.save_run_root / "42.json").write_text(
        json.dumps({"finished": True, "instance_id": "42", "gen_sql": "SELECT 999"}),
        encoding="utf-8",
    )

    sql = AgentResultSqlProvider(str(tmp_path), datasource="save", run_id="analytics/run-1").fetch("42")

    assert sql.sql is None
    assert "generated_sql file is missing" in sql.error


def test_ensure_benchmark_run_id_generates_and_preserves_ids() -> None:
    """Missing run ids are generated in the shared timestamp format."""
    assert Agent._ensure_benchmark_run_id("run-7") == "run-7"
    for missing in (None, ""):
        generated = Agent._ensure_benchmark_run_id(missing)
        # The generated id must satisfy attempt allocation, which rejects empty ids.
        assert re.fullmatch(r"\d{8}_\d{6}", generated)


def test_agent_wrapper_writes_manifest_when_runner_raises_before_output(tmp_path: Path) -> None:
    """Runner exceptions still produce a schema-valid failure manifest."""
    save_root = tmp_path / "save" / "analytics" / "run-1"
    trajectory_root = tmp_path / "trajectory" / "analytics" / "run-1"
    config = MagicMock()
    config.save_run_dir.return_value = save_root
    config.trajectory_run_dir.return_value = trajectory_root
    config.current_datasource = "analytics"
    config.active_model.return_value = _model_config()
    runner = SimpleNamespace(
        workflow=None,
        last_run_metadata={},
        run=MagicMock(side_effect=RuntimeError("plan generation failed")),
    )
    agent = SimpleNamespace(global_config=config, create_workflow_runner=MagicMock(return_value=runner))

    with pytest.raises(RuntimeError, match="plan generation failed"):
        Agent._run_benchmark_task(agent, _task(), run_id="run-1")

    payload = json.loads((save_root / "tasks" / "42" / "task-output.json").read_text(encoding="utf-8"))
    _validator().validate(payload)
    assert payload["status"] == "failed"
    assert payload["error"]["type"] == "benchmark_execution"
    assert payload["trajectory"] is None
    config.save_run_dir.assert_called_once_with("analytics", "run-1")
    config.trajectory_run_dir.assert_called_once_with("analytics", "run-1")
