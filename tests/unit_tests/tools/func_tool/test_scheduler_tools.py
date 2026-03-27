# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CI-level unit tests for SchedulerTools and Spark DAG template.

All external calls (adapter, filesystem) are mocked so these tests run
with zero network access and zero pre-built data.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from datus.tools.func_tool.scheduler_tools import SchedulerTools

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_agent_config(schedulers_config=None):
    cfg = MagicMock()
    cfg.schedulers_config = schedulers_config or {
        "airflow_local": {
            "type": "airflow",
            "api_base_url": "http://localhost:8080/api/v1",
            "username": "admin",
            "password": "admin123",
            "dags_folder": "/tmp/dags",
        }
    }
    return cfg


def _make_scheduled_job(job_id="spark_pi_test"):
    from unittest.mock import MagicMock

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


class TestRenderSparkDagSource:
    def test_renders_valid_python(self):
        """Generated DAG source must compile without errors."""
        from datus_airflow.dag_template import render_spark_dag_source

        source = render_spark_dag_source(
            dag_id="test_spark_pi",
            job_name="test_spark_pi",
            spark_script='print("hello")',
        )
        compile(source, "<test_dag>", "exec")

    def test_embeds_spark_script(self):
        """The spark_script content must appear in the rendered source."""
        from datus_airflow.dag_template import render_spark_dag_source

        script = "print('[Datus] Pi test')"
        source = render_spark_dag_source(
            dag_id="test_embed",
            job_name="test_embed",
            spark_script=script,
        )
        # The script is JSON-serialised inside the source
        assert json.dumps(script) in source

    def test_embeds_spark_master(self):
        """Custom spark_master must appear in the rendered source."""
        from datus_airflow.dag_template import render_spark_dag_source

        source = render_spark_dag_source(
            dag_id="test_master",
            job_name="test_master",
            spark_script="pass",
            spark_master="spark://localhost:7077",
        )
        assert "spark://localhost:7077" in source

    def test_default_spark_master(self):
        """Default spark master should be local[*]."""
        from datus_airflow.dag_template import render_spark_dag_source

        source = render_spark_dag_source(
            dag_id="test_default",
            job_name="test_default",
            spark_script="pass",
        )
        assert "local[*]" in source

    def test_schedule_embedded(self):
        """Cron schedule must appear in the rendered source."""
        from datus_airflow.dag_template import render_spark_dag_source

        source = render_spark_dag_source(
            dag_id="test_schedule",
            job_name="test_schedule",
            spark_script="pass",
            schedule="0 8 * * *",
        )
        assert "0 8 * * *" in source


# ── SchedulerTools.submit_spark_job ───────────────────────────────────────


class TestSubmitSparkJob:
    def test_submit_success(self, tmp_path):
        """submit_spark_job returns job_id on success."""
        script_file = tmp_path / "pi.py"
        script_file.write_text('print("pi")')

        mock_job = _make_scheduled_job("spark_pi_test")
        mock_adapter = MagicMock()
        mock_adapter.submit_job.return_value = mock_job

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.submit_spark_job(
                scheduler_name="airflow_local",
                job_name="spark_pi_test",
                spark_script_path=str(script_file),
            )

        assert result.success == 1
        assert result.result["job_id"] == "spark_pi_test"
        assert result.result["status"] == "active"

    def test_submit_missing_script_file(self, tmp_path):
        """submit_spark_job returns error when script path does not exist."""
        tools = SchedulerTools(_make_agent_config())
        result = tools.submit_spark_job(
            scheduler_name="airflow_local",
            job_name="spark_pi_test",
            spark_script_path=str(tmp_path / "nonexistent.py"),
        )
        assert result.success == 0
        assert "not found" in (result.error or "").lower()

    def test_submit_unknown_scheduler(self, tmp_path):
        """submit_spark_job returns error when scheduler name not in config."""
        script_file = tmp_path / "pi.py"
        script_file.write_text("pass")

        tools = SchedulerTools(_make_agent_config())
        result = tools.submit_spark_job(
            scheduler_name="nonexistent_scheduler",
            job_name="spark_pi_test",
            spark_script_path=str(script_file),
        )
        assert result.success == 0
        assert "nonexistent_scheduler" in (result.error or "")

    def test_submit_adapter_exception(self, tmp_path):
        """submit_spark_job returns error when adapter raises."""
        script_file = tmp_path / "pi.py"
        script_file.write_text("pass")

        mock_adapter = MagicMock()
        mock_adapter.submit_job.side_effect = Exception("connection refused")

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.submit_spark_job(
                scheduler_name="airflow_local",
                job_name="spark_pi_test",
                spark_script_path=str(script_file),
            )

        assert result.success == 0
        assert "connection refused" in (result.error or "")


# ── SchedulerTools.trigger_scheduler_job ─────────────────────────────────


class TestTriggerSchedulerJob:
    def test_trigger_success(self):
        """trigger_scheduler_job returns run_id on success."""
        mock_run = _make_job_run()
        mock_adapter = MagicMock()
        mock_adapter.trigger_job.return_value = mock_run

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.trigger_scheduler_job("airflow_local", "spark_pi_test")

        assert result.success == 1
        assert result.result["run_id"] == "manual__2025-01-01"

    def test_trigger_adapter_exception(self):
        """trigger_scheduler_job returns error when adapter raises."""
        mock_adapter = MagicMock()
        mock_adapter.trigger_job.side_effect = Exception("dag not found")

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.trigger_scheduler_job("airflow_local", "missing_dag")

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
            result = tools.get_scheduler_job("airflow_local", "spark_pi_test")

        assert result.success == 1
        assert result.result["found"] is True
        assert result.result["job_id"] == "spark_pi_test"

    def test_get_missing_job(self):
        """get_scheduler_job returns found=False when job does not exist."""
        mock_adapter = MagicMock()
        mock_adapter.get_job.return_value = None

        tools = SchedulerTools(_make_agent_config())

        with patch.object(tools, "_get_adapter", return_value=mock_adapter):
            result = tools.get_scheduler_job("airflow_local", "ghost_dag")

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
            result = tools.list_scheduler_jobs("airflow_local", limit=10)

        assert result.success == 1
        assert result.result["total"] == 2
        assert result.result["jobs"][0]["job_id"] == "dag_a"


# ── adapter.py: submit_job with job_type=spark ───────────────────────────


class TestAdapterSparkBranch:
    def test_submit_job_spark_calls_render_spark(self):
        """adapter.submit_job with job_type='spark' uses render_spark_dag_source."""
        try:
            from datus_airflow.adapter import AirflowSchedulerAdapter
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
