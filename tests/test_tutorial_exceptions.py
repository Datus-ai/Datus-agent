from argparse import Namespace
from io import StringIO
from types import SimpleNamespace

import pandas as pd
import pytest
from rich.console import Console

import datus.cli.tutorial as tutorial_module
import datus.storage.metric.metric_init as metric_init
import datus.storage.reference_sql.reference_sql_init as reference_sql_init
from datus.schemas.action_history import ActionStatus


class DummyAgentConfig:
    """Lightweight AgentConfig stand-in used to satisfy process_line dependencies."""

    def __init__(self):
        self.db_type = "sqlite"
        self._db_config = SimpleNamespace(catalog="catalog", database="database", schema="schema")

    def current_db_config(self):
        return self._db_config


class DummyReferenceStorage:
    """Minimal storage stub that exposes the methods init_reference_sql expects."""

    def __init__(self, size: int = 0):
        self.size = size

    def get_reference_sql_size(self):
        return self.size

    def after_init(self):
        return None


class AsyncIteratorStub:
    """Async iterator that either yields preset actions or raises an exception immediately."""

    def __init__(self, actions=None, exc: Exception | None = None):
        self._actions = actions or []
        self._exc = exc
        self._iter = iter(self._actions)

    def __aiter__(self):
        self._iter = iter(self._actions)
        return self

    async def __anext__(self):
        if self._exc:
            raise self._exc
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def patch_node_class(monkeypatch, module, class_name: str, behavior_map: dict[str, object]):
    """Patch an Agentic node class so each node_name follows the provided behavior."""

    class _NodeStub:
        def __init__(self, *args, **kwargs):
            # For new node classes, determine node_name from the class itself
            self.node_name = getattr(self, "NODE_NAME", kwargs.get("node_name"))
            self.input = None

        def execute_stream(self, action_history_manager):
            behavior = behavior_map.get(self.node_name, [])
            if isinstance(behavior, Exception):
                return AsyncIteratorStub(exc=behavior)
            return AsyncIteratorStub(actions=behavior)

    monkeypatch.setattr(module, class_name, _NodeStub)


class FakeAgent:
    """Agent stub used to drive BenchmarkTutorial._init_metrics behavior."""

    def __init__(self, bootstrap_result, metrics_size: int | None = None, bootstrap_exception: Exception | None = None):
        self._bootstrap_result = bootstrap_result
        self._bootstrap_exception = bootstrap_exception
        self.metrics_store = (
            SimpleNamespace(get_metrics_size=lambda: metrics_size) if metrics_size is not None else None
        )

    def bootstrap_kb(self):
        if self._bootstrap_exception:
            raise self._bootstrap_exception
        return self._bootstrap_result


def _make_console(monkeypatch):
    buffer = StringIO()
    test_console = Console(file=buffer, force_terminal=False, color_system=None)
    monkeypatch.setattr(tutorial_module, "console", test_console)
    return test_console, buffer


def _make_tutorial(tmp_path):
    tutorial = tutorial_module.BenchmarkTutorial.__new__(tutorial_module.BenchmarkTutorial)
    tutorial.namespace_name = "california_schools"
    tutorial.benchmark_path = tmp_path
    tutorial.config_path = "conf.yml"
    return tutorial


@pytest.mark.asyncio
async def test_process_line_returns_error_without_table(monkeypatch):
    row = {"sql": "SELECT 1", "question": "q"}
    agent_config = DummyAgentConfig()
    monkeypatch.setattr(metric_init, "extract_table_names", lambda sql, db_type: [])

    result = await metric_init.process_line_metrics_only(row, agent_config)

    assert result["successful"] is False
    assert result["error"] == "No table name found in SQL query"


@pytest.mark.asyncio
async def test_process_line_reports_metrics_generation_exception_old(monkeypatch):
    # This test is now deprecated since we split semantic model and metrics generation
    # Keeping for reference but it doesn't test the new flow
    row = {"sql": "SELECT * FROM schools", "question": "Describe schools"}
    agent_config = DummyAgentConfig()

    behavior_map = {
        "gen_metrics": RuntimeError("metrics failure"),
    }
    patch_node_class(monkeypatch, metric_init, "GenMetricsAgenticNode", behavior_map)
    monkeypatch.setattr(metric_init, "extract_table_names", lambda sql, db_type: ["schools"])

    result = await metric_init.process_line_metrics_only(row, agent_config)

    assert result["successful"] is False
    assert "Error generating metrics" in result["error"]
    assert "metrics failure" in result["error"]


@pytest.mark.asyncio
async def test_process_line_reports_metrics_generation_exception(monkeypatch):
    row = {"sql": "SELECT * FROM metrics", "question": "Describe metrics"}
    agent_config = DummyAgentConfig()

    behavior_map = {
        "gen_metrics": RuntimeError("metrics failure"),
    }
    patch_node_class(monkeypatch, metric_init, "GenMetricsAgenticNode", behavior_map)
    monkeypatch.setattr(metric_init, "extract_table_names", lambda sql, db_type: ["metrics"])

    result = await metric_init.process_line_metrics_only(row, agent_config)

    assert result["successful"] is False
    assert "Error generating metrics" in result["error"]
    assert "metrics failure" in result["error"]


def test_init_success_story_metrics_collects_all_errors(monkeypatch):
    df = pd.DataFrame(
        [
            {"sql": "SELECT * FROM schools", "question": "Q1"},
            {"sql": "SELECT * FROM students", "question": "Q2"},
        ]
    )
    monkeypatch.setattr(metric_init.pd, "read_csv", lambda path: df)

    responses = [
        {"successful": False, "error": "LLM refused to run"},
        RuntimeError("agent blew up"),
    ]

    async def fake_process_line(*args, **kwargs):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(metric_init, "process_line_metrics_only", fake_process_line)

    args = Namespace(success_story="anything.csv")
    success, error_message = metric_init.init_success_story_metrics(args, DummyAgentConfig())

    assert success is False
    assert "Error processing row 1: LLM refused to run" in error_message
    assert "Error processing row 2: agent blew up" in error_message


@pytest.mark.asyncio
async def test_process_sql_item_returns_none_when_node_raises(monkeypatch):
    behavior_map = {"gen_sql_summary": RuntimeError("summary failure")}
    patch_node_class(monkeypatch, reference_sql_init, "SqlSummaryAgenticNode", behavior_map)

    item = {"sql": "SELECT * FROM schools", "comment": "desc", "filepath": "file.sql"}

    result = await reference_sql_init.process_sql_item(item, SimpleNamespace(), build_mode="overwrite")

    assert result is None


@pytest.mark.asyncio
async def test_process_sql_item_returns_none_without_output(monkeypatch):
    action = SimpleNamespace(status=ActionStatus.SUCCESS, output={}, messages="")
    behavior_map = {"gen_sql_summary": [action]}
    patch_node_class(monkeypatch, reference_sql_init, "SqlSummaryAgenticNode", behavior_map)

    item = {"sql": "SELECT * FROM schools", "comment": "desc", "filepath": "file.sql"}

    result = await reference_sql_init.process_sql_item(item, SimpleNamespace(), build_mode="overwrite")

    assert result is None


def test_init_reference_sql_reports_process_errors(monkeypatch):
    valid_items = [{"sql": "SELECT * FROM schools", "comment": "desc"}]
    monkeypatch.setattr(reference_sql_init, "process_sql_files", lambda sql_dir: (valid_items, []))

    async def failing_process_sql_item(*args, **kwargs):
        raise RuntimeError("worker failure")

    monkeypatch.setattr(reference_sql_init, "process_sql_item", failing_process_sql_item)

    storage = DummyReferenceStorage()

    result = reference_sql_init.init_reference_sql(
        storage,
        global_config=SimpleNamespace(),
        build_mode="overwrite",
        sql_dir="dummy",
        validate_only=False,
        pool_size=1,
        subject_tree=None,
    )

    assert result["status"] == "success"
    assert result["processed_entries"] == 0
    assert result["process_error"] is not None
    assert "SQL processing failed with exception `worker failure`" in result["process_error"]


def test_tutorial_init_metrics_logs_partial_success(monkeypatch, tmp_path):
    tutorial = _make_tutorial(tmp_path)
    _, buffer = _make_console(monkeypatch)
    fake_agent = FakeAgent({"status": "success", "error": "Row 2 failed"}, metrics_size=3)
    monkeypatch.setattr(tutorial_module, "create_agent", lambda **kwargs: fake_agent)

    assert tutorial._init_metrics(tmp_path / "success.csv")
    output = buffer.getvalue()
    assert "Processed 3 metrics" in output
    assert "The metrics has not been fully initialised successfully" in output
    assert "Row 2 failed" in output


def test_tutorial_init_metrics_logs_error_when_no_metrics(monkeypatch, tmp_path):
    tutorial = _make_tutorial(tmp_path)
    _, buffer = _make_console(monkeypatch)
    fake_agent = FakeAgent({"status": "success", "error": "fatal error"}, metrics_size=0)
    monkeypatch.setattr(tutorial_module, "create_agent", lambda **kwargs: fake_agent)

    assert tutorial._init_metrics(tmp_path / "success.csv")
    output = buffer.getvalue()
    assert "There are some errors in the processing" in output
    assert "fatal error" in output


def test_tutorial_init_metrics_logs_failure_from_result(monkeypatch, tmp_path):
    tutorial = _make_tutorial(tmp_path)
    _, buffer = _make_console(monkeypatch)
    fake_agent = FakeAgent({"status": "failed", "message": "boot failure"}, metrics_size=5)
    monkeypatch.setattr(tutorial_module, "create_agent", lambda **kwargs: fake_agent)

    assert tutorial._init_metrics(tmp_path / "success.csv")
    output = buffer.getvalue()
    assert "Metrics initialization failed" in output
    assert "boot failure" in output


def test_tutorial_init_metrics_handles_exception(monkeypatch, tmp_path):
    tutorial = _make_tutorial(tmp_path)
    called = {}

    def mock_print(console, exc, description, _logger):
        called["exc"] = exc
        called["description"] = description

    def raising_agent(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(tutorial_module, "print_rich_exception", mock_print)
    monkeypatch.setattr(tutorial_module, "create_agent", raising_agent)

    assert tutorial._init_metrics(tmp_path / "success.csv") is False
    assert called["description"] == "Metrics initialization failed"
    assert str(called["exc"]) == "boom"
