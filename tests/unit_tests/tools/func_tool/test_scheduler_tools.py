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

# Mock datus_scheduler_core if not installed (CI has no external scheduler deps)
if "datus_scheduler_core" not in sys.modules:

    class _MockPayload:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    _mock_core = MagicMock()
    _mock_core.models.SchedulerJobPayload = _MockPayload
    sys.modules["datus_scheduler_core"] = _mock_core
    sys.modules["datus_scheduler_core.models"] = _mock_core.models
    sys.modules["datus_scheduler_core.registry"] = _mock_core.registry
    sys.modules["datus_scheduler_core.config"] = _mock_core.config

from datus.tools.func_tool.scheduler_tools import (
    SchedulerTools,
    _build_connection_url,
    _redact_url,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_agent_config(scheduler_config=None, namespaces=None):
    cfg = MagicMock()
    cfg.scheduler_config = scheduler_config or {
        "name": "airflow_local",
        "type": "airflow",
        "api_base_url": "http://localhost:8080/api/v1",
        "username": "admin",
        "password": "admin123",
        "dags_folder": "/tmp/dags",
    }
    cfg.namespaces = namespaces
    return cfg


def _make_db_config(
    db_type="starrocks", host="127.0.0.1", port="9030", username="admin", password="pass@123", database="mydb"
):
    db_cfg = MagicMock()
    db_cfg.type = db_type
    db_cfg.host = host
    db_cfg.port = port
    db_cfg.username = username
    db_cfg.password = password
    db_cfg.database = database
    return db_cfg


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


# ── DAG template tests ─────────────────────────────────────────────────────


try:
    from datus_scheduler_airflow.dag_template import render_spark_dag_source

    _HAS_SCHEDULER_AIRFLOW = True
except ImportError:
    _HAS_SCHEDULER_AIRFLOW = False


@pytest.mark.skipif(not _HAS_SCHEDULER_AIRFLOW, reason="datus-scheduler-airflow not installed")
class TestRenderSparkDagSource:
    def test_renders_valid_python(self):
        """Generated DAG source must compile without errors."""
        source = render_spark_dag_source(
            dag_id="test_spark_pi",
            job_name="test_spark_pi",
            spark_script='print("hello")',
        )
        compile(source, "<test_dag>", "exec")

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
        """list_scheduler_jobs returns a list of job summaries."""
        mock_adapter = MagicMock()
        mock_adapter.list_jobs.return_value = [_make_scheduled_job("dag_a"), _make_scheduled_job("dag_b")]

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.list_scheduler_jobs(limit=10)

        assert result.success == 1
        assert result.result["total"] == 2
        assert result.result["jobs"][0]["job_id"] == "dag_a"


# ── adapter.py: submit_job with job_type=spark ───────────────────────────


class TestAdapterSparkBranch:
    def test_submit_job_spark_calls_render_spark(self):
        """adapter.submit_job with job_type='spark' uses render_spark_dag_source."""
        try:
            from datus_scheduler_airflow.adapter import AirflowSchedulerAdapter
            from datus_scheduler_core.config import AirflowConfig
            from datus_scheduler_core.models import SchedulerJobPayload
        except ImportError:
            pytest.skip("datus-airflow not installed")

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


# ── _build_connection_url ────────────────────────────────────────────────


class TestBuildConnectionUrl:
    def test_starrocks_url(self):
        db_cfg = _make_db_config(db_type="starrocks")
        url = _build_connection_url(db_cfg)
        assert url.startswith("mysql+pymysql://")
        assert "127.0.0.1:9030/mydb" in url

    def test_postgresql_url(self):
        db_cfg = _make_db_config(db_type="postgresql", port="5432")
        url = _build_connection_url(db_cfg)
        assert url.startswith("postgresql+psycopg2://")

    def test_unknown_dialect_fallback(self):
        db_cfg = _make_db_config(db_type="oracle")
        url = _build_connection_url(db_cfg)
        assert url.startswith("oracle://")

    def test_password_url_encoded(self):
        db_cfg = _make_db_config(password="p@ss:word/123")
        url = _build_connection_url(db_cfg)
        assert "p%40ss%3Aword%2F123" in url

    def test_username_url_encoded(self):
        db_cfg = _make_db_config(username="user@domain")
        url = _build_connection_url(db_cfg)
        assert "user%40domain" in url

    def test_empty_password(self):
        db_cfg = _make_db_config(password="")
        url = _build_connection_url(db_cfg)
        assert ":@" in url

    def test_empty_host_raises(self):
        db_cfg = _make_db_config(host="")
        with pytest.raises(ValueError, match="Incomplete DB config"):
            _build_connection_url(db_cfg)

    def test_empty_port_raises(self):
        db_cfg = _make_db_config(port="")
        with pytest.raises(ValueError, match="Incomplete DB config"):
            _build_connection_url(db_cfg)

    def test_empty_database_raises(self):
        db_cfg = _make_db_config(database="")
        with pytest.raises(ValueError, match="Incomplete DB config"):
            _build_connection_url(db_cfg)


# ── _redact_url ──────────────────────────────────────────────────────────


class TestRedactUrl:
    def test_redacts_password(self):
        url = "mysql+pymysql://admin:secret123@host:3306/db"
        assert _redact_url(url) == "mysql+pymysql://admin:***@host:3306/db"

    def test_redacts_encoded_password(self):
        url = "mysql+pymysql://admin:p%40ss@host:3306/db"
        assert _redact_url(url) == "mysql+pymysql://admin:***@host:3306/db"

    def test_malformed_url_returns_redacted(self):
        assert _redact_url("not-a-url") == "<redacted URL>"


# ── SchedulerTools.submit_sql_job ────────────────────────────────────────


class TestSubmitSqlJob:
    def test_submit_success_with_connection_url(self, tmp_path):
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
                connection_url="mysql+pymysql://user:pass@host:3306/db",
            )

        assert result.success == 1
        assert result.result["job_id"] == "sql_job_1"

    def test_submit_success_with_namespace(self, tmp_path):
        sql_file = tmp_path / "query.sql"
        sql_file.write_text("SELECT 1")

        mock_job = _make_scheduled_job("sql_job_ns")
        mock_adapter = MagicMock()
        mock_adapter.submit_job.return_value = mock_job

        db_cfg = _make_db_config()
        namespaces = {"starrocks": {"default": db_cfg}}
        tools = SchedulerTools(_make_agent_config(namespaces=namespaces))

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.submit_sql_job(
                job_name="sql_job_ns",
                sql_file_path=str(sql_file),
                namespace="starrocks",
            )

        assert result.success == 1

    def test_missing_sql_file(self, tmp_path):
        tools = SchedulerTools(_make_agent_config())
        result = tools.submit_sql_job(
            job_name="test",
            sql_file_path=str(tmp_path / "nonexistent.sql"),
            connection_url="mysql://x:y@h:3306/db",
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
            connection_url="mysql://x:y@h:3306/db",
        )
        assert result.success == 0
        assert "empty" in (result.error or "").lower()

    def test_no_namespace_no_url_returns_error(self, tmp_path):
        sql_file = tmp_path / "query.sql"
        sql_file.write_text("SELECT 1")
        tools = SchedulerTools(_make_agent_config())
        result = tools.submit_sql_job(
            job_name="test",
            sql_file_path=str(sql_file),
        )
        assert result.success == 0
        assert "namespace" in (result.error or "").lower()

    def test_namespace_not_found(self, tmp_path):
        sql_file = tmp_path / "query.sql"
        sql_file.write_text("SELECT 1")
        tools = SchedulerTools(_make_agent_config(namespaces={"other_ns": {}}))
        result = tools.submit_sql_job(
            job_name="test",
            sql_file_path=str(sql_file),
            namespace="missing_ns",
        )
        assert result.success == 0
        assert "missing_ns" in (result.error or "")


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
    def test_update_success_with_url(self, tmp_path):
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
                connection_url="mysql+pymysql://u:p@h:3306/db",
            )

        assert result.success == 1
        assert result.result["job_id"] == "dag_to_update"

    def test_update_missing_sql_file(self, tmp_path):
        tools = SchedulerTools(_make_agent_config())
        result = tools.update_job(
            job_id="dag_x",
            sql_file_path=str(tmp_path / "gone.sql"),
            job_name="DAG X",
            connection_url="mysql://u:p@h:3306/db",
        )
        assert result.success == 0
        assert "not found" in (result.error or "").lower()

    def test_update_no_namespace_no_url(self, tmp_path):
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
        assert "namespace" in (result.error or "").lower()

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

    def test_update_sparksql_no_db_connection_needed(self, tmp_path):
        """SparkSQL update should succeed without namespace or connection_url."""
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

    def test_update_with_namespace(self, tmp_path):
        sql_file = tmp_path / "updated.sql"
        sql_file.write_text("SELECT 2")

        mock_job = _make_scheduled_job("dag_ns_update")
        mock_adapter = MagicMock()
        mock_adapter.update_job.return_value = mock_job

        db_cfg = _make_db_config()
        namespaces = {"starrocks": {"default": db_cfg}}
        tools = SchedulerTools(_make_agent_config(namespaces=namespaces))

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.update_job(
                job_id="dag_ns_update",
                sql_file_path=str(sql_file),
                job_name="DAG NS Update",
                namespace="starrocks",
            )

        assert result.success == 1


# ── SchedulerTools.list_job_runs ─────────────────────────────────────────


class TestListJobRuns:
    def test_list_runs_success(self):
        mock_run = MagicMock()
        mock_run.run_id = "run_001"
        mock_run.status.value = "success"
        mock_run.started_at = datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        mock_run.ended_at = datetime(2025, 1, 1, 8, 5, 0, tzinfo=timezone.utc)

        mock_adapter = MagicMock()
        mock_adapter.list_job_runs.return_value = [mock_run]

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.list_job_runs("my_dag", limit=5)

        assert result.success == 1
        assert result.result["total"] == 1
        run = result.result["runs"][0]
        assert run["run_id"] == "run_001"
        assert run["started_at"] == "2025-01-01T08:00:00+00:00"
        assert run["ended_at"] == "2025-01-01T08:05:00+00:00"

    def test_list_runs_string_timestamps(self):
        """Runs with string timestamps should pass through as-is."""
        mock_run = MagicMock()
        mock_run.run_id = "run_002"
        mock_run.status.value = "running"
        mock_run.started_at = "2025-01-01T08:00:00Z"
        mock_run.ended_at = None

        mock_adapter = MagicMock()
        mock_adapter.list_job_runs.return_value = [mock_run]

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.list_job_runs("my_dag")

        assert result.success == 1
        run = result.result["runs"][0]
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
