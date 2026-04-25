# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CI-level unit tests for SchedulerTools and Spark DAG template.

All external calls (adapter, filesystem) are mocked so these tests run
with zero network access and zero pre-built data.
"""

import json
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# Mock datus_scheduler_core if not installed. This MUST run at module scope —
# the `from datus.tools.func_tool.scheduler_tools import ...` below transitively
# imports datus_scheduler_core, so a fixture-scoped patch would happen too late.
# The mock is idempotent (guarded by `not in sys.modules`) and the modules are
# namespaced under `datus_scheduler_core.*`, so there's no bleed into other tests.
if "datus_scheduler_core" not in sys.modules:

    class _MockPayload:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    _mock_core = MagicMock()
    _mock_core.models.SchedulerJobPayload = _MockPayload
    sys.modules["datus_scheduler_core"] = _mock_core  # audit-noqa: module_level_sys_modules
    sys.modules["datus_scheduler_core.models"] = _mock_core.models  # audit-noqa: module_level_sys_modules
    sys.modules["datus_scheduler_core.registry"] = _mock_core.registry  # audit-noqa: module_level_sys_modules
    sys.modules["datus_scheduler_core.config"] = _mock_core.config  # audit-noqa: module_level_sys_modules

from datus.tools.func_tool.scheduler_tools import SchedulerTools
from datus.utils.exceptions import DatusException, ErrorCode

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_agent_config(scheduler_config=None):
    cfg = MagicMock()
    if scheduler_config is None:
        cfg.scheduler_config = {
            "name": "airflow_local",
            "type": "airflow",
            "api_base_url": "http://localhost:8080/api/v1",
            "username": "admin",
            "password": "admin123",
            "dags_folder": "/tmp/dags",
        }
    else:
        cfg.scheduler_config = scheduler_config
    cfg.scheduler_services = {"airflow_local": cfg.scheduler_config} if cfg.scheduler_config else {}

    def _get_scheduler_config(service_name=None):
        if service_name:
            if service_name not in cfg.scheduler_services:
                raise DatusException(
                    ErrorCode.COMMON_CONFIG_ERROR,
                    message=f"No scheduler service named `{service_name}` found.",
                )
            return cfg.scheduler_services[service_name]
        if cfg.scheduler_services:
            return next(iter(cfg.scheduler_services.values()))
        raise DatusException(
            ErrorCode.COMMON_CONFIG_ERROR,
            message="No scheduler configured in `agent.services.schedulers`.",
        )

    cfg.get_scheduler_config.side_effect = _get_scheduler_config
    return cfg


class _SchedulerPage:
    """Minimal stand-in for ``PaginatedScheduledResult`` / ``ListJobsResult`` /
    ``ListRunsResult``. The SchedulerTools envelope builder only looks at
    ``.items`` and ``.total``, so mirroring those two attributes is enough.
    """

    def __init__(self, items, total=None):
        self.items = list(items)
        self.total = total


def _make_scheduled_job(job_id="spark_pi_test"):
    job = MagicMock()
    job.job_id = job_id
    job.job_name = job_id
    job.status.value = "active"
    job.schedule = "0 8 * * *"
    job.description = "test"
    job.platform = "airflow"
    return job


def _make_job_run(run_id="manual__2025-01-01"):
    run = MagicMock()
    run.run_id = run_id
    run.job_id = "spark_pi_test"
    run.status.value = "running"
    return run


# ── SchedulerTools._get_adapter ─────────────────────────────────────────────


class TestGetAdapter:
    def test_no_scheduler_config_raises(self):
        from datus.utils.exceptions import DatusException

        tools = SchedulerTools(_make_agent_config(scheduler_config={}))
        with pytest.raises(DatusException):
            tools._get_adapter()

    def test_success_with_mocked_registry(self):
        mock_adapter = MagicMock()
        tools = SchedulerTools(_make_agent_config())
        mock_registry = MagicMock()
        mock_registry.create_adapter.return_value = mock_adapter
        with patch("datus.tools.func_tool.scheduler_tools.SchedulerAdapterRegistry", mock_registry):
            adapter = tools._get_adapter()
        assert adapter is mock_adapter

    def test_airflow_injects_project_name_as_file_scope_only(self):
        """Datus auto-injects ``agent.project_name`` into the Airflow
        adapter config — but *only* for the filesystem-scoping role
        (DAG subdirectory under ``dags_folder_root``). In the adapter
        0.2.0+ schema ``project_name`` no longer drives ``dag_id_prefix``
        defaulting, so list/get operations aren't silently filtered by
        the Datus workspace. Users who want list-level multi-tenant
        isolation set ``dag_id_prefix`` explicitly in agent.yml.
        """
        agent_cfg = _make_agent_config(
            scheduler_config={
                "name": "airflow_local",
                "type": "airflow",
                "api_base_url": "http://localhost:8080/api/v1",
                "username": "admin",
                "password": "admin",
                "dags_folder_root": "/opt/airflow/dags",
                # Deliberately no explicit project_name — adapter expects
                # Datus to fill it from agent.project_name.
            }
        )
        agent_cfg.project_name = "reports-team"
        tools = SchedulerTools(agent_cfg)

        mock_registry = MagicMock()
        mock_registry.create_adapter.return_value = MagicMock()
        with patch("datus.tools.func_tool.scheduler_tools.SchedulerAdapterRegistry", mock_registry):
            tools._get_adapter()

        call_kwargs = mock_registry.create_adapter.call_args.kwargs
        assert call_kwargs["platform"] == "airflow"
        # File-scoping is auto-filled so DAG files land in a per-workspace subdir.
        assert call_kwargs["config"]["project_name"] == "reports-team"
        # dag_id_prefix is NOT auto-set — that's an explicit opt-in.
        assert "dag_id_prefix" not in call_kwargs["config"]

    def test_airflow_explicit_project_name_takes_precedence(self):
        """setdefault semantics: if user writes project_name in agent.yml, Datus
        must NOT overwrite it with agent.project_name."""
        agent_cfg = _make_agent_config(
            scheduler_config={
                "name": "airflow_local",
                "type": "airflow",
                "api_base_url": "http://localhost:8080/api/v1",
                "username": "admin",
                "password": "admin",
                "dags_folder_root": "/opt/airflow/dags",
                "project_name": "explicit-override",
            }
        )
        agent_cfg.project_name = "reports-team"
        tools = SchedulerTools(agent_cfg)

        mock_registry = MagicMock()
        mock_registry.create_adapter.return_value = MagicMock()
        with patch("datus.tools.func_tool.scheduler_tools.SchedulerAdapterRegistry", mock_registry):
            tools._get_adapter()

        call_kwargs = mock_registry.create_adapter.call_args.kwargs
        assert call_kwargs["config"]["project_name"] == "explicit-override"

    def test_non_airflow_platform_not_injected(self):
        """Only Airflow config schema has a project_name field; don't inject for
        DS/Azkaban (their 'project' semantics are platform-side, not Datus)."""
        agent_cfg = _make_agent_config(
            scheduler_config={
                "name": "ds_prod",
                "type": "dolphinscheduler",
                "api_base_url": "http://localhost:12345/dolphinscheduler",
                "token": "fake-token",
            }
        )
        agent_cfg.project_name = "reports-team"
        tools = SchedulerTools(agent_cfg)

        mock_registry = MagicMock()
        mock_registry.create_adapter.return_value = MagicMock()
        with patch("datus.tools.func_tool.scheduler_tools.SchedulerAdapterRegistry", mock_registry):
            tools._get_adapter()

        call_kwargs = mock_registry.create_adapter.call_args.kwargs
        assert "project_name" not in call_kwargs["config"]


# ── SchedulerTools.available_tools ─────────────────────────────────────────


class TestAvailableTools:
    def test_returns_tool_list(self):
        tools = SchedulerTools(_make_agent_config())
        result = tools.available_tools()
        assert isinstance(result, list)
        assert len(result) > 0
        tool_names = {t.name for t in result}
        for expected in ["submit_sql_job", "submit_sparksql_job", "trigger_scheduler_job", "get_scheduler_job"]:
            assert expected in tool_names


# ── adapter.close() error handling ─────────────────────────────────────────


class TestAdapterCloseError:
    """adapter.close() failure should not affect the method result."""

    @pytest.mark.parametrize(
        "method_name, call_args, call_kwargs, adapter_setup",
        [
            (
                "trigger_scheduler_job",
                ("dag_1",),
                {},
                lambda a: setattr(a, "trigger_job", MagicMock(return_value=_make_job_run())),
            ),
            (
                "get_scheduler_job",
                ("dag_1",),
                {},
                lambda a: setattr(a, "get_job", MagicMock(return_value=_make_scheduled_job())),
            ),
            (
                "list_scheduler_jobs",
                (),
                {},
                lambda a: setattr(a, "list_jobs", MagicMock(return_value=_SchedulerPage(items=[], total=0))),
            ),
            ("pause_job", ("dag_1",), {}, None),
            ("resume_job", ("dag_1",), {}, None),
            ("delete_job", ("dag_1",), {}, None),
            (
                "list_job_runs",
                ("dag_1",),
                {},
                lambda a: setattr(a, "list_job_runs", MagicMock(return_value=_SchedulerPage(items=[], total=0))),
            ),
            (
                "get_run_log",
                ("dag_1", "run_1"),
                {},
                lambda a: setattr(a, "get_run_log", MagicMock(return_value="log text")),
            ),
        ],
    )
    def test_close_exception_still_returns(self, method_name, call_args, call_kwargs, adapter_setup):
        mock_adapter = MagicMock()
        mock_adapter.close.side_effect = Exception("close failed")
        if adapter_setup is not None:
            adapter_setup(mock_adapter)

        tools = SchedulerTools(_make_agent_config())
        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = getattr(tools, method_name)(*call_args, **call_kwargs)

        assert result.success == 1

    def test_submit_sql_close_exception_still_returns(self, tmp_path):
        sql_file = tmp_path / "q.sql"
        sql_file.write_text("SELECT 1")

        mock_job = _make_scheduled_job("j1")
        mock_adapter = MagicMock()
        mock_adapter.submit_job.return_value = mock_job
        mock_adapter.close.side_effect = Exception("close failed")

        tools = SchedulerTools(_make_agent_config())
        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.submit_sql_job(job_name="j1", sql_file_path=str(sql_file), conn_id="my_conn")

        assert result.success == 1

    def test_submit_sparksql_close_exception_still_returns(self, tmp_path):
        sql_file = tmp_path / "q.sql"
        sql_file.write_text("SELECT 1")

        mock_job = _make_scheduled_job("j1")
        mock_adapter = MagicMock()
        mock_adapter.submit_job.return_value = mock_job
        mock_adapter.close.side_effect = Exception("close failed")

        tools = SchedulerTools(_make_agent_config())
        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.submit_sparksql_job(job_name="j1", sql_file_path=str(sql_file))

        assert result.success == 1

    def test_update_close_exception_still_returns(self, tmp_path):
        sql_file = tmp_path / "q.sql"
        sql_file.write_text("SELECT 1")

        mock_job = _make_scheduled_job("j1")
        mock_adapter = MagicMock()
        mock_adapter.update_job.return_value = mock_job
        mock_adapter.close.side_effect = Exception("close failed")

        tools = SchedulerTools(_make_agent_config())
        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.update_job(job_id="j1", sql_file_path=str(sql_file), job_name="J1", conn_id="my_conn")

        assert result.success == 1


# ── _get_adapter error paths in tool methods ───────────────────────────────


class TestAdapterCreationErrors:
    @pytest.mark.parametrize(
        "method_name, call_args, call_kwargs",
        [
            ("trigger_scheduler_job", ("dag_1",), {}),
            ("get_scheduler_job", ("dag_1",), {}),
            ("list_scheduler_jobs", (), {}),
            ("pause_job", ("dag_1",), {}),
            ("resume_job", ("dag_1",), {}),
            ("delete_job", ("dag_1",), {}),
            ("list_job_runs", ("dag_1",), {}),
            ("get_run_log", ("dag_1", "run_1"), {}),
        ],
    )
    def test_no_scheduler_config_returns_failure(self, method_name, call_args, call_kwargs):
        tools = SchedulerTools(_make_agent_config(scheduler_config={}))
        result = getattr(tools, method_name)(*call_args, **call_kwargs)
        assert result.success == 0

    def test_submit_sql_no_scheduler_config(self, tmp_path):
        sql_file = tmp_path / "q.sql"
        sql_file.write_text("SELECT 1")
        tools = SchedulerTools(_make_agent_config(scheduler_config={}))
        result = tools.submit_sql_job(job_name="j1", sql_file_path=str(sql_file), conn_id="my_conn")
        assert result.success == 0
        assert "scheduler" in (result.error or "").lower()

    def test_submit_sparksql_no_scheduler_config(self, tmp_path):
        sql_file = tmp_path / "q.sql"
        sql_file.write_text("SELECT 1")
        tools = SchedulerTools(_make_agent_config(scheduler_config={}))
        result = tools.submit_sparksql_job(job_name="j1", sql_file_path=str(sql_file))
        assert result.success == 0
        assert "scheduler" in (result.error or "").lower()

    def test_update_no_scheduler_config(self, tmp_path):
        sql_file = tmp_path / "q.sql"
        sql_file.write_text("SELECT 1")
        tools = SchedulerTools(_make_agent_config(scheduler_config={}))
        result = tools.update_job(job_id="j1", sql_file_path=str(sql_file), job_name="J1", conn_id="my_conn")
        assert result.success == 0
        assert "scheduler" in (result.error or "").lower()


# ── DAG template tests ─────────────────────────────────────────────────────


try:
    from datus_scheduler_airflow.adapter import AirflowSchedulerAdapter
    from datus_scheduler_airflow.dag_template import render_spark_dag_source
    from datus_scheduler_core.config import AirflowConfig
    from datus_scheduler_core.models import SchedulerJobPayload

    _HAS_SCHEDULER_AIRFLOW = True
except ImportError:
    _HAS_SCHEDULER_AIRFLOW = False


@pytest.mark.skipif(not _HAS_SCHEDULER_AIRFLOW, reason="datus-scheduler-airflow not installed")
class TestRenderSparkDagSource:
    def test_renders_valid_python(self):
        """Generated DAG source must be valid Python and carry the given dag_id."""
        source = render_spark_dag_source(
            dag_id="test_spark_pi",
            job_name="test_spark_pi",
            spark_script='print("hello")',
        )
        # compile() raises SyntaxError on invalid Python — that's the primary contract.
        compile(source, "<test_dag>", "exec")
        # Verify the rendered source actually incorporates the caller's arguments.
        assert "test_spark_pi" in source, "dag_id should appear in rendered source"
        assert isinstance(source, str) and len(source) > 0

    def test_embeds_spark_script(self):
        """The spark_script content must appear in the rendered source."""
        script = "print('[Datus] Pi test')"
        source = render_spark_dag_source(
            dag_id="test_embed",
            job_name="test_embed",
            spark_script=script,
        )
        assert json.dumps(script) in source

    def test_embeds_spark_master(self):
        """Custom spark_master must appear in the rendered source."""
        source = render_spark_dag_source(
            dag_id="test_master",
            job_name="test_master",
            spark_script="pass",
            spark_master="spark://localhost:7077",
        )
        assert "spark://localhost:7077" in source

    def test_default_spark_master(self):
        """Default spark master should be local[*]."""
        source = render_spark_dag_source(
            dag_id="test_default",
            job_name="test_default",
            spark_script="pass",
        )
        assert "local[*]" in source

    def test_schedule_embedded(self):
        """Cron schedule must appear in the rendered source."""
        source = render_spark_dag_source(
            dag_id="test_schedule",
            job_name="test_schedule",
            spark_script="pass",
            schedule="0 8 * * *",
        )
        assert "0 8 * * *" in source


# ── SchedulerTools.trigger_scheduler_job ─────────────────────────────────


class TestTriggerSchedulerJob:
    def test_trigger_success(self):
        """trigger_scheduler_job returns run_id on success."""
        mock_run = _make_job_run()
        mock_adapter = MagicMock()
        mock_adapter.trigger_job.return_value = mock_run

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.trigger_scheduler_job("spark_pi_test")

        assert result.success == 1
        assert result.result["run_id"] == "manual__2025-01-01"

    def test_trigger_adapter_exception(self):
        """trigger_scheduler_job returns error when adapter raises."""
        mock_adapter = MagicMock()
        mock_adapter.trigger_job.side_effect = Exception("dag not found")

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.trigger_scheduler_job("missing_dag")

        assert result.success == 0
        assert "dag not found" in (result.error or "")


# ── SchedulerTools.get_scheduler_job ─────────────────────────────────────


class TestGetSchedulerJob:
    def test_get_existing_job(self):
        """get_scheduler_job returns found=True for an existing job."""
        mock_adapter = MagicMock()
        mock_adapter.get_job.return_value = _make_scheduled_job()

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.get_scheduler_job("spark_pi_test")

        assert result.success == 1
        assert result.result["found"] is True
        assert result.result["job_id"] == "spark_pi_test"

    def test_get_missing_job(self):
        """get_scheduler_job returns found=False when job does not exist."""
        mock_adapter = MagicMock()
        mock_adapter.get_job.return_value = None

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.get_scheduler_job("ghost_dag")

        assert result.success == 1
        assert result.result["found"] is False


# ── SchedulerTools.list_scheduler_jobs ───────────────────────────────────


class TestListSchedulerJobs:
    def test_list_jobs(self):
        """list_scheduler_jobs returns the canonical FuncToolListResult envelope."""
        mock_adapter = MagicMock()
        mock_adapter.list_jobs.return_value = _SchedulerPage(
            items=[_make_scheduled_job("dag_a"), _make_scheduled_job("dag_b")],
            total=2,
        )

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.list_scheduler_jobs(limit=10)

        assert result.success == 1
        envelope = result.result
        assert envelope["total"] == 2
        assert len(envelope["items"]) == 2
        assert envelope["items"][0]["job_id"] == "dag_a"
        # 2 items with total=2 and offset=0 → last page, no next_offset.
        assert envelope["has_more"] is False
        assert envelope["extra"] is None


# ── adapter.py: submit_job with job_type=spark ───────────────────────────


@pytest.mark.skipif(not _HAS_SCHEDULER_AIRFLOW, reason="datus-scheduler-airflow not installed")
class TestAdapterSparkBranch:
    def test_submit_job_spark_calls_render_spark(self):
        """adapter.submit_job with job_type='spark' uses render_spark_dag_source."""
        config = AirflowConfig(
            name="test",
            type="airflow",
            api_base_url="http://localhost:8080/api/v1",
            username="admin",
            password="admin123",
            dags_folder="/tmp/dags",
        )
        adapter = AirflowSchedulerAdapter.__new__(AirflowSchedulerAdapter)
        adapter._config = config
        adapter._session = MagicMock()
        adapter._session.get.return_value = MagicMock(status_code=404)

        written_source = {}

        def fake_write(dag_id, source):
            written_source["source"] = source

        def fake_wait(dag_id):
            pass

        def fake_get(dag_id):
            from datus_scheduler_core.models import JobStatus, ScheduledJob

            return ScheduledJob(
                scheduler_name="test",
                platform="airflow",
                job_id=dag_id,
                job_name=dag_id,
                status=JobStatus.ACTIVE,
            )

        adapter._write_dag_file = fake_write
        adapter._wait_for_dag_discovery = fake_wait
        adapter.get_job = MagicMock(side_effect=[None, fake_get("test_spark")])

        payload = SchedulerJobPayload(
            job_name="test_spark",
            extra={
                "job_type": "spark",
                "spark_script": 'print("pi")',
                "spark_master": "local[*]",
            },
        )
        job = adapter.submit_job(payload)

        assert job.job_id == "test_spark"
        assert "DatusSparkJob" in written_source["source"]
        assert "_run_spark_script" in written_source["source"]


# ── SchedulerTools.submit_sql_job ────────────────────────────────────────


class TestSubmitSqlJob:
    def test_submit_success_with_conn_id(self, tmp_path):
        sql_file = tmp_path / "query.sql"
        sql_file.write_text("SELECT 1")

        mock_job = _make_scheduled_job("sql_job_1")
        mock_adapter = MagicMock()
        mock_adapter.submit_job.return_value = mock_job

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.submit_sql_job(
                job_name="sql_job_1",
                sql_file_path=str(sql_file),
                conn_id="starrocks_default",
            )

        assert result.success == 1
        assert result.result["job_id"] == "sql_job_1"
        payload = mock_adapter.submit_job.call_args[0][0]
        assert payload.db_connection == {"conn_id": "starrocks_default"}

    def test_missing_sql_file(self, tmp_path):
        tools = SchedulerTools(_make_agent_config())
        result = tools.submit_sql_job(
            job_name="test",
            sql_file_path=str(tmp_path / "nonexistent.sql"),
            conn_id="my_conn",
        )
        assert result.success == 0
        assert "not found" in (result.error or "").lower()

    def test_empty_sql_file(self, tmp_path):
        sql_file = tmp_path / "empty.sql"
        sql_file.write_text("   ")
        tools = SchedulerTools(_make_agent_config())
        result = tools.submit_sql_job(
            job_name="test",
            sql_file_path=str(sql_file),
            conn_id="my_conn",
        )
        assert result.success == 0
        assert "empty" in (result.error or "").lower()

    def test_adapter_exception(self, tmp_path):
        sql_file = tmp_path / "query.sql"
        sql_file.write_text("SELECT 1")

        mock_adapter = MagicMock()
        mock_adapter.submit_job.side_effect = Exception("Connection failed")
        mock_adapter.close.return_value = None

        tools = SchedulerTools(_make_agent_config())
        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.submit_sql_job(job_name="j1", sql_file_path=str(sql_file), conn_id="my_conn")

        assert result.success == 0
        assert "Connection failed" in (result.error or "")


# ── SchedulerTools.submit_sparksql_job ───────────────────────────────────


class TestSubmitSparksqlJob:
    def test_submit_success(self, tmp_path):
        sql_file = tmp_path / "sparksql.sql"
        sql_file.write_text("SELECT * FROM t")

        mock_job = _make_scheduled_job("sparksql_1")
        mock_adapter = MagicMock()
        mock_adapter.submit_job.return_value = mock_job

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.submit_sparksql_job(
                job_name="sparksql_1",
                sql_file_path=str(sql_file),
            )

        assert result.success == 1
        assert result.result["job_id"] == "sparksql_1"

    def test_missing_sql_file(self, tmp_path):
        tools = SchedulerTools(_make_agent_config())
        result = tools.submit_sparksql_job(
            job_name="test",
            sql_file_path=str(tmp_path / "missing.sql"),
        )
        assert result.success == 0
        assert "not found" in (result.error or "").lower()

    def test_adapter_exception(self, tmp_path):
        sql_file = tmp_path / "sparksql.sql"
        sql_file.write_text("SELECT 1")

        mock_adapter = MagicMock()
        mock_adapter.submit_job.side_effect = Exception("timeout")

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.submit_sparksql_job(
                job_name="test",
                sql_file_path=str(sql_file),
            )

        assert result.success == 0
        assert "timeout" in (result.error or "")


# ── SchedulerTools.pause_job ─────────────────────────────────────────────


class TestPauseJob:
    def test_pause_success(self):
        mock_adapter = MagicMock()
        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.pause_job("my_dag")

        assert result.success == 1
        assert result.result["status"] == "paused"
        mock_adapter.pause_job.assert_called_once_with("my_dag")

    def test_pause_adapter_exception(self):
        mock_adapter = MagicMock()
        mock_adapter.pause_job.side_effect = Exception("not found")
        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.pause_job("missing")

        assert result.success == 0
        assert "not found" in (result.error or "")


# ── SchedulerTools.resume_job ────────────────────────────────────────────


class TestResumeJob:
    def test_resume_success(self):
        mock_adapter = MagicMock()
        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.resume_job("my_dag")

        assert result.success == 1
        assert result.result["status"] == "active"
        mock_adapter.resume_job.assert_called_once_with("my_dag")

    def test_resume_adapter_exception(self):
        mock_adapter = MagicMock()
        mock_adapter.resume_job.side_effect = Exception("forbidden")
        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.resume_job("my_dag")

        assert result.success == 0
        assert "forbidden" in (result.error or "")


# ── SchedulerTools.delete_job ────────────────────────────────────────────


class TestDeleteJob:
    def test_delete_success(self):
        mock_adapter = MagicMock()
        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.delete_job("old_dag")

        assert result.success == 1
        assert result.result["status"] == "deleted"
        mock_adapter.delete_job.assert_called_once_with("old_dag")

    def test_delete_adapter_exception(self):
        mock_adapter = MagicMock()
        mock_adapter.delete_job.side_effect = Exception("permission denied")
        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.delete_job("old_dag")

        assert result.success == 0
        assert "permission denied" in (result.error or "")


# ── SchedulerTools.update_job ────────────────────────────────────────────


class TestUpdateJob:
    def test_update_success_with_conn_id(self, tmp_path):
        sql_file = tmp_path / "updated.sql"
        sql_file.write_text("SELECT 2")

        mock_job = _make_scheduled_job("dag_to_update")
        mock_adapter = MagicMock()
        mock_adapter.update_job.return_value = mock_job

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.update_job(
                job_id="dag_to_update",
                sql_file_path=str(sql_file),
                job_name="DAG To Update",
                conn_id="starrocks_default",
            )

        assert result.success == 1
        assert result.result["job_id"] == "dag_to_update"
        payload = mock_adapter.update_job.call_args[0][1]
        assert payload.db_connection == {"conn_id": "starrocks_default"}

    def test_update_missing_sql_file(self, tmp_path):
        tools = SchedulerTools(_make_agent_config())
        result = tools.update_job(
            job_id="dag_x",
            sql_file_path=str(tmp_path / "gone.sql"),
            job_name="DAG X",
            conn_id="my_conn",
        )
        assert result.success == 0
        assert "not found" in (result.error or "").lower()

    def test_update_no_conn_id_returns_error(self, tmp_path):
        sql_file = tmp_path / "updated.sql"
        sql_file.write_text("SELECT 2")
        tools = SchedulerTools(_make_agent_config())
        result = tools.update_job(
            job_id="dag_x",
            sql_file_path=str(sql_file),
            job_name="DAG X",
            job_type="sql",
        )
        assert result.success == 0
        assert "conn_id" in (result.error or "").lower()

    def test_update_invalid_job_type(self, tmp_path):
        sql_file = tmp_path / "updated.sql"
        sql_file.write_text("SELECT 2")
        tools = SchedulerTools(_make_agent_config())
        result = tools.update_job(
            job_id="dag_x",
            sql_file_path=str(sql_file),
            job_name="DAG X",
            job_type="pyspark",
        )
        assert result.success == 0
        assert "Unsupported job_type" in (result.error or "")

    def test_update_sparksql_success(self, tmp_path):
        sql_file = tmp_path / "spark_updated.sql"
        sql_file.write_text("SELECT * FROM t")

        mock_job = _make_scheduled_job("dag_sparksql_update")
        mock_adapter = MagicMock()
        mock_adapter.update_job.return_value = mock_job

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.update_job(
                job_id="dag_sparksql_update",
                sql_file_path=str(sql_file),
                job_name="SparkSQL Update Job",
                job_type="sparksql",
                spark_master="spark://localhost:7077",
            )

        assert result.success == 1
        assert result.result["job_id"] == "dag_sparksql_update"
        # Verify adapter was called with sparksql payload
        call_args = mock_adapter.update_job.call_args
        payload = call_args[0][1]
        assert payload.extra["job_type"] == "sparksql"
        assert payload.extra["sparksql"] == "SELECT * FROM t"
        assert payload.extra["spark_master"] == "spark://localhost:7077"

    def test_update_sparksql_default_master(self, tmp_path):
        sql_file = tmp_path / "spark_updated.sql"
        sql_file.write_text("SELECT 1")

        mock_job = _make_scheduled_job("dag_sparksql_default")
        mock_adapter = MagicMock()
        mock_adapter.update_job.return_value = mock_job

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.update_job(
                job_id="dag_sparksql_default",
                sql_file_path=str(sql_file),
                job_name="SparkSQL Default Job",
                job_type="sparksql",
            )

        assert result.success == 1
        call_args = mock_adapter.update_job.call_args
        payload = call_args[0][1]
        assert payload.extra["spark_master"] == "local[*]"

    def test_update_sparksql_no_conn_id_needed(self, tmp_path):
        """SparkSQL update should succeed without conn_id."""
        sql_file = tmp_path / "spark.sql"
        sql_file.write_text("SELECT 1")

        mock_job = _make_scheduled_job("dag_spark_no_db")
        mock_adapter = MagicMock()
        mock_adapter.update_job.return_value = mock_job

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.update_job(
                job_id="dag_spark_no_db",
                sql_file_path=str(sql_file),
                job_name="Spark No DB Job",
                job_type="sparksql",
            )

        assert result.success == 1

    def test_update_adapter_exception(self, tmp_path):
        sql_file = tmp_path / "updated.sql"
        sql_file.write_text("SELECT 2")

        mock_adapter = MagicMock()
        mock_adapter.update_job.side_effect = Exception("update failed")
        mock_adapter.close.return_value = None

        tools = SchedulerTools(_make_agent_config())
        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.update_job(job_id="j1", sql_file_path=str(sql_file), job_name="J1", conn_id="my_conn")

        assert result.success == 0
        assert "update failed" in (result.error or "")


# ── SchedulerTools.list_scheduler_connections ─────────────────────────────


class TestListSchedulerConnections:
    def test_returns_configured_connections(self):
        cfg = _make_agent_config()
        cfg.scheduler_config["connections"] = {
            "starrocks_default": "StarRocks ac_manage",
            "pg_conn": "PostgreSQL test DB",
        }
        tools = SchedulerTools(cfg)
        result = tools.list_scheduler_connections()

        assert result.success == 1
        assert result.result["total"] == 2
        conn_ids = [c["conn_id"] for c in result.result["connections"]]
        assert "starrocks_default" in conn_ids
        assert "pg_conn" in conn_ids

    def test_empty_connections(self):
        cfg = _make_agent_config()
        # No connections key
        tools = SchedulerTools(cfg)
        result = tools.list_scheduler_connections()

        assert result.success == 1
        assert result.result["total"] == 0
        assert "hint" in result.result

    def test_no_scheduler_config(self):
        cfg = _make_agent_config(scheduler_config={})
        tools = SchedulerTools(cfg)
        result = tools.list_scheduler_connections()

        assert result.success == 0
        assert "scheduler" in (result.error or "").lower()


# ── available_tools: conn_id injection into description ──────────────────


class TestConnIdDescriptionInjection:
    def test_connections_injected_into_submit_and_update(self):
        cfg = _make_agent_config()
        cfg.scheduler_config["connections"] = {"sr_default": "StarRocks DB"}
        tools = SchedulerTools(cfg)
        tool_list = tools.available_tools()
        tool_map = {t.name: t for t in tool_list}

        assert "sr_default" in tool_map["submit_sql_job"].description
        assert "sr_default" in tool_map["update_job"].description
        # Other tools should NOT have the suffix
        assert "sr_default" not in tool_map["pause_job"].description

    def test_no_connections_no_injection(self):
        cfg = _make_agent_config()
        tools = SchedulerTools(cfg)
        tool_list = tools.available_tools()
        tool_map = {t.name: t for t in tool_list}

        assert "Available conn_id" not in tool_map["submit_sql_job"].description


# ── SchedulerTools.list_job_runs ─────────────────────────────────────────


class TestListJobRuns:
    def test_list_runs_success(self):
        mock_run = MagicMock()
        mock_run.run_id = "run_001"
        mock_run.status.value = "success"
        mock_run.started_at = datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        mock_run.ended_at = datetime(2025, 1, 1, 8, 5, 0, tzinfo=timezone.utc)

        mock_adapter = MagicMock()
        mock_adapter.list_job_runs.return_value = _SchedulerPage(items=[mock_run], total=1)

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.list_job_runs("my_dag", limit=5)

        assert result.success == 1
        envelope = result.result
        assert envelope["total"] == 1
        assert len(envelope["items"]) == 1
        run = envelope["items"][0]
        assert run["run_id"] == "run_001"
        assert run["started_at"] == "2025-01-01T08:00:00+00:00"
        assert run["ended_at"] == "2025-01-01T08:05:00+00:00"
        assert envelope["has_more"] is False

    def test_list_runs_string_timestamps(self):
        """Runs with string timestamps should pass through as-is."""
        mock_run = MagicMock()
        mock_run.run_id = "run_002"
        mock_run.status.value = "running"
        mock_run.started_at = "2025-01-01T08:00:00Z"
        mock_run.ended_at = None

        mock_adapter = MagicMock()
        mock_adapter.list_job_runs.return_value = _SchedulerPage(items=[mock_run], total=None)

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.list_job_runs("my_dag")

        assert result.success == 1
        run = result.result["items"][0]
        assert run["started_at"] == "2025-01-01T08:00:00Z"
        assert run["ended_at"] is None

    def test_list_runs_adapter_exception(self):
        mock_adapter = MagicMock()
        mock_adapter.list_job_runs.side_effect = Exception("api error")
        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.list_job_runs("my_dag")

        assert result.success == 0
        assert "api error" in (result.error or "")


# ── SchedulerTools.get_run_log ───────────────────────────────────────────


class TestGetRunLog:
    def test_get_log_success(self):
        mock_adapter = MagicMock()
        mock_adapter.get_run_log.return_value = "[Datus] Running SQL: SELECT 1\n[Datus] SQL completed. rows=1"

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.get_run_log("my_dag", "run_001")

        assert result.success == 1
        assert "SELECT 1" in result.result["log"]
        assert result.result["run_id"] == "run_001"

    def test_get_log_adapter_exception(self):
        mock_adapter = MagicMock()
        mock_adapter.get_run_log.side_effect = Exception("run not found")
        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.get_run_log("my_dag", "bad_run")

        assert result.success == 0
        assert "run not found" in (result.error or "")


# ── SchedulerTools deliverable_target self-reporting ──────────────────────


class TestSchedulerDeliverableTarget:
    """The 3 mutating scheduler tools (submit_sql_job / submit_sparksql_job /
    update_job) must attach a ``SchedulerJobTarget`` to ``result.result`` so
    ValidationHook can see the delivered job."""

    def _mock_adapter(self, job_id="job_x"):
        mock_job = _make_scheduled_job(job_id)
        mock_adapter = MagicMock()
        mock_adapter.submit_job.return_value = mock_job
        mock_adapter.update_job.return_value = mock_job
        return mock_adapter

    def test_submit_sql_job_emits_deliverable_target(self, tmp_path):
        sql_file = tmp_path / "q.sql"
        sql_file.write_text("SELECT 1")
        tools = SchedulerTools(_make_agent_config())
        with patch.object(tools, "_get_adapter", return_value=self._mock_adapter("job_x")):
            result = tools.submit_sql_job(job_name="job_x", sql_file_path=str(sql_file), conn_id="c1")
        assert result.success == 1
        target = result.result.get("deliverable_target")
        assert target is not None
        assert target["type"] == "scheduler_job"
        assert target["platform"] == "airflow"
        assert target["job_id"] == "job_x"
        assert target["job_name"] == "job_x"

    def test_submit_sparksql_job_emits_deliverable_target(self, tmp_path):
        sql_file = tmp_path / "q.sql"
        sql_file.write_text("SELECT 1")
        tools = SchedulerTools(_make_agent_config())
        with patch.object(tools, "_get_adapter", return_value=self._mock_adapter("spark_x")):
            result = tools.submit_sparksql_job(job_name="spark_x", sql_file_path=str(sql_file))
        assert result.success == 1
        target = result.result.get("deliverable_target")
        assert target is not None
        assert target["type"] == "scheduler_job"
        assert target["job_id"] == "spark_x"

    def test_update_job_emits_deliverable_target(self, tmp_path):
        sql_file = tmp_path / "q.sql"
        sql_file.write_text("SELECT 1")
        mock_adapter = self._mock_adapter("job_u")
        tools = SchedulerTools(_make_agent_config())
        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.update_job(
                job_id="job_u",
                sql_file_path=str(sql_file),
                job_name="job_u",
                job_type="sql",
                conn_id="c1",
            )
        assert result.success == 1
        target = result.result.get("deliverable_target")
        assert target is not None
        assert target["type"] == "scheduler_job"
        assert target["job_id"] == "job_u"

    def test_failure_does_not_attach_target(self, tmp_path):
        """When the adapter fails, no deliverable_target is emitted (since nothing was delivered)."""
        sql_file = tmp_path / "q.sql"
        sql_file.write_text("SELECT 1")
        mock_adapter = MagicMock()
        mock_adapter.submit_job.side_effect = Exception("boom")
        tools = SchedulerTools(_make_agent_config())
        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.submit_sql_job(job_name="job_x", sql_file_path=str(sql_file), conn_id="c1")
        assert result.success == 0
        # On failure the tool sets result=None, so there's no dict to contain a target.
        assert result.result is None or "deliverable_target" not in (result.result or {})
