# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Scheduler tools for submitting and managing jobs via Datus scheduler adapters."""

from pathlib import Path
from typing import List, Optional

from agents import Tool

from datus.tools import BaseTool
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SchedulerTools(BaseTool):
    """Function tools for interacting with configured scheduler platforms (e.g. Airflow).

    Requires ``schedulers:`` section in ``agent.yml``.  Each entry maps a logical
    name (e.g. ``airflow_local``) to an adapter config dict.
    """

    def __init__(self, agent_config, **kwargs):
        super().__init__(**kwargs)
        self.agent_config = agent_config

    # ── Adapter factory ────────────────────────────────────────────────────

    def _get_adapter(self, scheduler_name: str):
        """Lazily create a scheduler adapter for the named configuration."""
        try:
            from datus_scheduler_core.registry import SchedulerAdapterRegistry
        except ImportError as exc:
            raise ImportError(
                "datus-scheduler-core is required for scheduler tools. "
                "Install it with: pip install datus-scheduler-core"
            ) from exc

        schedulers_config = getattr(self.agent_config, "schedulers_config", {}) or {}
        raw_config = schedulers_config.get(scheduler_name)
        if not raw_config:
            raise ValueError(
                f"Scheduler '{scheduler_name}' not found in agent.yml schedulers configuration. "
                f"Available: {list(schedulers_config.keys())}"
            )

        config = dict(raw_config)
        config["name"] = scheduler_name
        platform = config.get("type", "airflow")
        return SchedulerAdapterRegistry.create_adapter(platform=platform, config=config)

    # ── Tool methods ───────────────────────────────────────────────────────

    def submit_spark_job(
        self,
        scheduler_name: str,
        job_name: str,
        spark_script_path: str,
        spark_master: Optional[str] = None,
        schedule: Optional[str] = None,
        description: Optional[str] = None,
    ) -> FuncToolResult:
        """Submit a PySpark script as a scheduled DAG to the specified scheduler.

        Args:
            scheduler_name:    Scheduler instance name from agent.yml (e.g. 'airflow_local').
            job_name:          Human-readable job name; used to derive the DAG/job ID.
            spark_script_path: Local path to the PySpark script file.  The file content is
                               embedded into the DAG at submission time — the path is not
                               referenced at runtime.
            spark_master:      Spark master URL (default: 'local[*]').
            schedule:          Cron expression, e.g. '0 8 * * *'.  None = manual trigger only.
            description:       Optional human-readable description for the DAG.

        Returns:
            FuncToolResult with result containing job_id and status.
        """
        try:
            from datus_scheduler_core.models import SchedulerJobPayload
        except ImportError as exc:
            return FuncToolResult(success=0, error=f"datus-scheduler-core not installed: {exc}")

        try:
            script_path = Path(spark_script_path).expanduser()
            if not script_path.exists():
                return FuncToolResult(success=0, error=f"Spark script not found: {spark_script_path}")
            spark_script = script_path.read_text(encoding="utf-8")
        except Exception as exc:
            return FuncToolResult(success=0, error=f"Failed to read spark script '{spark_script_path}': {exc}")

        try:
            adapter = self._get_adapter(scheduler_name)
        except (ImportError, ValueError) as exc:
            return FuncToolResult(success=0, error=str(exc))

        try:
            payload = SchedulerJobPayload(
                job_name=job_name,
                schedule=schedule,
                description=description,
                extra={
                    "job_type": "spark",
                    "spark_script": spark_script,
                    "spark_master": spark_master or "local[*]",
                },
            )
            job = adapter.submit_job(payload)
            return FuncToolResult(
                success=1,
                result={
                    "job_id": job.job_id,
                    "job_name": job.job_name,
                    "status": job.status.value,
                    "scheduler": scheduler_name,
                    "platform": job.platform,
                },
            )
        except Exception as exc:
            logger.error("submit_spark_job failed: %s", exc)
            return FuncToolResult(success=0, error=str(exc))
        finally:
            try:
                adapter.close()
            except Exception:
                pass

    def trigger_scheduler_job(
        self,
        scheduler_name: str,
        job_id: str,
    ) -> FuncToolResult:
        """Trigger an immediate run of an existing scheduled job.

        Args:
            scheduler_name: Scheduler instance name from agent.yml.
            job_id:         The job/DAG identifier to trigger.

        Returns:
            FuncToolResult with result containing run_id and status.
        """
        try:
            adapter = self._get_adapter(scheduler_name)
        except (ImportError, ValueError) as exc:
            return FuncToolResult(success=0, error=str(exc))

        try:
            run = adapter.trigger_job(job_id)
            return FuncToolResult(
                success=1,
                result={
                    "run_id": run.run_id,
                    "job_id": run.job_id,
                    "status": run.status.value,
                },
            )
        except Exception as exc:
            logger.error("trigger_scheduler_job failed: %s", exc)
            return FuncToolResult(success=0, error=str(exc))
        finally:
            try:
                adapter.close()
            except Exception:
                pass

    def get_scheduler_job(
        self,
        scheduler_name: str,
        job_id: str,
    ) -> FuncToolResult:
        """Get the current status and metadata of a scheduled job.

        Args:
            scheduler_name: Scheduler instance name from agent.yml.
            job_id:         The job/DAG identifier to query.

        Returns:
            FuncToolResult with result containing job details, or found=False if not found.
        """
        try:
            adapter = self._get_adapter(scheduler_name)
        except (ImportError, ValueError) as exc:
            return FuncToolResult(success=0, error=str(exc))

        try:
            job = adapter.get_job(job_id)
            if job is None:
                return FuncToolResult(success=1, result={"found": False, "job_id": job_id})
            return FuncToolResult(
                success=1,
                result={
                    "found": True,
                    "job_id": job.job_id,
                    "job_name": job.job_name,
                    "status": job.status.value,
                    "schedule": job.schedule,
                    "description": job.description,
                    "platform": job.platform,
                },
            )
        except Exception as exc:
            logger.error("get_scheduler_job failed: %s", exc)
            return FuncToolResult(success=0, error=str(exc))
        finally:
            try:
                adapter.close()
            except Exception:
                pass

    def list_scheduler_jobs(
        self,
        scheduler_name: str,
        limit: int = 20,
    ) -> FuncToolResult:
        """List all scheduled jobs on the specified scheduler.

        Args:
            scheduler_name: Scheduler instance name from agent.yml.
            limit:          Maximum number of jobs to return (default 20).

        Returns:
            FuncToolResult with result containing a list of job summaries.
        """
        try:
            adapter = self._get_adapter(scheduler_name)
        except (ImportError, ValueError) as exc:
            return FuncToolResult(success=0, error=str(exc))

        try:
            jobs = adapter.list_jobs(limit=limit)
            return FuncToolResult(
                success=1,
                result={
                    "total": len(jobs),
                    "jobs": [
                        {
                            "job_id": j.job_id,
                            "job_name": j.job_name,
                            "status": j.status.value,
                            "schedule": j.schedule,
                        }
                        for j in jobs
                    ],
                },
            )
        except Exception as exc:
            logger.error("list_scheduler_jobs failed: %s", exc)
            return FuncToolResult(success=0, error=str(exc))
        finally:
            try:
                adapter.close()
            except Exception:
                pass

    # ── Tool registration ──────────────────────────────────────────────────

    def available_tools(self) -> List[Tool]:
        """Return all scheduler tool functions as FunctionTool objects."""
        methods = [
            self.submit_spark_job,
            self.trigger_scheduler_job,
            self.get_scheduler_job,
            self.list_scheduler_jobs,
        ]
        return [trans_to_function_tool(m) for m in methods]
