# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Scheduler tools for submitting and managing jobs via Datus scheduler adapters."""

from pathlib import Path
from typing import Any, List, Optional

from agents import Tool
from datus_scheduler_core.models import SchedulerJobPayload
from datus_scheduler_core.registry import SchedulerAdapterRegistry

from datus.configuration.agent_config import AgentConfig
from datus.tools import BaseTool
from datus.tools.func_tool.base import FuncToolListResult, FuncToolResult, trans_to_function_tool
from datus.utils.exceptions import DatusException
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SchedulerTools(BaseTool):
    """Function tools for interacting with the configured scheduler platform (e.g. Airflow)."""

    tool_name = "scheduler_tools"
    tool_description = "Tools for submitting and managing scheduled jobs via Airflow"

    def __init__(self, agent_config: AgentConfig, scheduler_service: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.agent_config = agent_config
        self.scheduler_service = scheduler_service

    def _selected_scheduler_config(self) -> dict:
        return dict(self.agent_config.get_scheduler_config(self.scheduler_service))

    # ── Adapter factory ────────────────────────────────────────────────────

    def _get_adapter(self):
        """Create a scheduler adapter from the configured scheduler."""
        config = self._selected_scheduler_config()
        platform = config.get("type", "airflow")

        # For Airflow, the adapter config's ``project_name`` is a
        # filesystem-scoped knob ONLY — it controls the DAG subdirectory
        # (``{dags_folder_root}/{project_name}/``) so multiple Datus
        # instances writing DAG files to the same Airflow cluster never
        # collide on disk. It does NOT auto-derive a ``dag_id_prefix``
        # anymore (the two concerns were split in datus-scheduler-airflow
        # 0.2.0), so reads aren't silently filtered by the Datus workspace.
        # Auto-injecting the Datus workspace identifier is safe: it stays
        # on the file side, and users who want list-level multi-tenant
        # isolation opt in by setting ``dag_id_prefix`` explicitly in
        # ``services.schedulers.<name>``.
        if platform == "airflow":
            config.setdefault("project_name", self.agent_config.project_name)

        return SchedulerAdapterRegistry.create_adapter(platform=platform, config=config)

    # ── Tool methods ───────────────────────────────────────────────────────

    def submit_sql_job(
        self,
        job_name: str,
        sql_file_path: str,
        conn_id: str,
        schedule: Optional[str] = None,
        description: Optional[str] = None,
    ) -> FuncToolResult:
        """Submit a SQL file as a scheduled job.

        The scheduler worker executes the SQL using the database connection
        referenced by ``conn_id``.  The connection is managed by the scheduler
        platform (e.g. Airflow Connections) and resolved at runtime.

        Args:
            job_name:        Human-readable job name; used to derive the DAG/job ID.
            sql_file_path:   Local path to the .sql file.
            conn_id:         Scheduler connection ID (e.g. Airflow Connection ID).
            schedule:        Cron expression, e.g. '0 8 * * *'.  None = manual trigger only.
            description:     Optional human-readable description for the DAG.

        Returns:
            FuncToolResult with result containing job_id and status.
        """
        # Read SQL file
        try:
            sql_path = Path(sql_file_path).expanduser()
            if not sql_path.exists():
                return FuncToolResult(success=0, error=f"SQL file not found: {sql_file_path}")
            sql_content = sql_path.read_text(encoding="utf-8").strip()
            if not sql_content:
                return FuncToolResult(success=0, error=f"SQL file is empty: {sql_file_path}")
        except Exception as exc:
            return FuncToolResult(success=0, error=f"Failed to read SQL file '{sql_file_path}': {exc}")

        # Submit
        try:
            adapter = self._get_adapter()
        except (ValueError, DatusException) as exc:
            return FuncToolResult(success=0, error=str(exc))

        try:
            payload = SchedulerJobPayload(
                job_name=job_name,
                sql=sql_content,
                db_connection={"conn_id": conn_id},
                schedule=schedule,
                description=description,
            )
            job = adapter.submit_job(payload)
            return FuncToolResult(
                success=1,
                result={
                    "job_id": job.job_id,
                    "job_name": job.job_name,
                    "status": job.status.value,
                    "scheduler": job.platform,
                    "platform": job.platform,
                    "deliverable_target": self._build_scheduler_target(job),
                },
            )
        except Exception as exc:
            logger.error("submit_sql_job failed: %s", exc)
            return FuncToolResult(success=0, error=str(exc))
        finally:
            try:
                adapter.close()
            except Exception as close_exc:
                logger.debug("adapter.close() failed: %s", close_exc)

    def submit_sparksql_job(
        self,
        job_name: str,
        sql_file_path: str,
        spark_master: Optional[str] = None,
        schedule: Optional[str] = None,
        description: Optional[str] = None,
    ) -> FuncToolResult:
        """Submit a SparkSQL file as a scheduled job.

        Use this after generating SparkSQL and writing it to a .sql file.
        The SQL is executed via SparkSession.sql() by the scheduler worker.

        Args:
            job_name:        Human-readable job name; used to derive the DAG/job ID.
            sql_file_path:   Local path to the .sql file containing SparkSQL.
            spark_master:    Spark master URL (default: 'local[*]').
            schedule:        Cron expression, e.g. '0 8 * * *'.  None = manual trigger only.
            description:     Optional human-readable description for the DAG.

        Returns:
            FuncToolResult with result containing job_id and status.
        """
        try:
            sql_path = Path(sql_file_path).expanduser()
            if not sql_path.exists():
                return FuncToolResult(success=0, error=f"SQL file not found: {sql_file_path}")
            sql_content = sql_path.read_text(encoding="utf-8").strip()
            if not sql_content:
                return FuncToolResult(success=0, error=f"SQL file is empty: {sql_file_path}")
        except Exception as exc:
            return FuncToolResult(success=0, error=f"Failed to read SQL file '{sql_file_path}': {exc}")

        try:
            adapter = self._get_adapter()
        except (ValueError, DatusException) as exc:
            return FuncToolResult(success=0, error=str(exc))

        try:
            payload = SchedulerJobPayload(
                job_name=job_name,
                schedule=schedule,
                description=description,
                extra={
                    "job_type": "sparksql",
                    "sparksql": sql_content,
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
                    "scheduler": job.platform,
                    "platform": job.platform,
                    "deliverable_target": self._build_scheduler_target(job),
                },
            )
        except Exception as exc:
            logger.error("submit_sparksql_job failed: %s", exc)
            return FuncToolResult(success=0, error=str(exc))
        finally:
            try:
                adapter.close()
            except Exception as close_exc:
                logger.debug("adapter.close() failed: %s", close_exc)

    def trigger_scheduler_job(
        self,
        job_id: str,
    ) -> FuncToolResult:
        """Trigger an immediate run of an existing scheduled job.

        Args:
            job_id: The job/DAG identifier to trigger.

        Returns:
            FuncToolResult with result containing run_id and status.
        """
        try:
            adapter = self._get_adapter()
        except (ValueError, DatusException) as exc:
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
            except Exception as close_exc:
                logger.debug("adapter.close() failed: %s", close_exc)

    def get_scheduler_job(
        self,
        job_id: str,
    ) -> FuncToolResult:
        """Get the current status and metadata of a scheduled job.

        Args:
            job_id: The job/DAG identifier to query.

        Returns:
            FuncToolResult with result containing job details, or found=False if not found.
        """
        try:
            adapter = self._get_adapter()
        except (ValueError, DatusException) as exc:
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
            except Exception as close_exc:
                logger.debug("adapter.close() failed: %s", close_exc)

    def list_scheduler_jobs(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> FuncToolResult:
        """List all scheduled jobs on the configured scheduler.

        Args:
            limit: Maximum number of jobs to return (default 20).
            offset: Pagination offset (default 0).

        Returns:
            FuncToolResult with result as FuncToolListResult:
              - items (List[Dict]): job summary rows
              - total (int | None): upstream total when the platform reports
                it; None in multi-tenant modes where the server's count
                would be misleading
              - has_more (bool | None): next-page hint
              - extra (dict | None): {"next_offset": int} when has_more=True

            Pagination: call again with offset=extra.next_offset until
            has_more is False.
        """
        try:
            adapter = self._get_adapter()
        except (ValueError, DatusException) as exc:
            return FuncToolResult(success=0, error=str(exc))

        try:
            page = adapter.list_jobs(limit=limit, offset=offset)
            rows = [
                {
                    "job_id": j.job_id,
                    "job_name": j.job_name,
                    "status": j.status.value,
                    "schedule": j.schedule,
                }
                for j in page.items
            ]
            return self._build_scheduler_envelope(rows, total=page.total, offset=offset, limit=limit)
        except Exception as exc:
            logger.error("list_scheduler_jobs failed: %s", exc)
            return FuncToolResult(success=0, error=str(exc))
        finally:
            try:
                adapter.close()
            except Exception as close_exc:
                logger.debug("adapter.close() failed: %s", close_exc)

    @staticmethod
    def _build_scheduler_envelope(
        rows: list,
        *,
        total: "int | None",
        offset: int,
        limit: int,
    ) -> FuncToolResult:
        """Wrap adapter ListJobsResult / ListRunsResult rows into FuncToolListResult.

        When ``total`` is known, ``has_more`` is exact. When it's ``None``
        (multi-tenant Airflow or status-filtered runs), fall back to
        ``len(rows) == limit`` as the "looks like another page" heuristic.
        """
        if total is not None:
            has_more: "bool | None" = offset + len(rows) < total
        elif limit > 0:
            has_more = len(rows) == limit
        else:
            has_more = None
        extra = {"next_offset": offset + len(rows)} if has_more else None
        return FuncToolResult(
            success=1,
            result=FuncToolListResult(items=rows, total=total, has_more=has_more, extra=extra).model_dump(),
        )

    def pause_job(
        self,
        job_id: str,
    ) -> FuncToolResult:
        """Pause a scheduled job so it will not be triggered by the scheduler.

        Args:
            job_id: The job/DAG identifier to pause.

        Returns:
            FuncToolResult indicating success or failure.
        """
        try:
            adapter = self._get_adapter()
        except (ValueError, DatusException) as exc:
            return FuncToolResult(success=0, error=str(exc))

        try:
            adapter.pause_job(job_id)
            return FuncToolResult(success=1, result={"job_id": job_id, "status": "paused"})
        except Exception as exc:
            logger.error("pause_job failed: %s", exc)
            return FuncToolResult(success=0, error=str(exc))
        finally:
            try:
                adapter.close()
            except Exception as close_exc:
                logger.debug("adapter.close() failed: %s", close_exc)

    def resume_job(
        self,
        job_id: str,
    ) -> FuncToolResult:
        """Resume a paused job so the scheduler will trigger it again.

        Args:
            job_id: The job/DAG identifier to resume.

        Returns:
            FuncToolResult indicating success or failure.
        """
        try:
            adapter = self._get_adapter()
        except (ValueError, DatusException) as exc:
            return FuncToolResult(success=0, error=str(exc))

        try:
            adapter.resume_job(job_id)
            return FuncToolResult(success=1, result={"job_id": job_id, "status": "active"})
        except Exception as exc:
            logger.error("resume_job failed: %s", exc)
            return FuncToolResult(success=0, error=str(exc))
        finally:
            try:
                adapter.close()
            except Exception as close_exc:
                logger.debug("adapter.close() failed: %s", close_exc)

    def delete_job(
        self,
        job_id: str,
    ) -> FuncToolResult:
        """Delete a scheduled job permanently. Removes the DAG file and metadata.

        Args:
            job_id: The job/DAG identifier to delete.

        Returns:
            FuncToolResult indicating success or failure.
        """
        try:
            adapter = self._get_adapter()
        except (ValueError, DatusException) as exc:
            return FuncToolResult(success=0, error=str(exc))

        try:
            adapter.delete_job(job_id)
            return FuncToolResult(success=1, result={"job_id": job_id, "status": "deleted"})
        except Exception as exc:
            logger.error("delete_job failed: %s", exc)
            return FuncToolResult(success=0, error=str(exc))
        finally:
            try:
                adapter.close()
            except Exception as close_exc:
                logger.debug("adapter.close() failed: %s", close_exc)

    def update_job(
        self,
        job_id: str,
        sql_file_path: str,
        job_name: str,
        job_type: str = "sql",
        conn_id: Optional[str] = None,
        spark_master: Optional[str] = None,
        schedule: Optional[str] = None,
        description: Optional[str] = None,
    ) -> FuncToolResult:
        """Update an existing scheduled job with new SQL or configuration.

        Re-renders the job definition with updated content.  The scheduler reloads
        it automatically.  Supports both SQL and SparkSQL job types.

        For SQL jobs, ``conn_id`` is required — it references the scheduler-managed
        connection (e.g. Airflow Connection ID) resolved at runtime.

        Args:
            job_id:         The existing job/DAG identifier to update.
            sql_file_path:  Local path to the new .sql file.
            job_name:       Human-readable job name (used for rendering the job definition).
            job_type:       'sql' (default) or 'sparksql'.
            conn_id:        Scheduler connection ID (e.g. Airflow Connection ID). Required for sql jobs.
            spark_master:   Spark master URL, default 'local[*]' (sparksql only).
            schedule:       Cron expression, e.g. '0 8 * * *'.  None = manual trigger only.
            description:    Optional human-readable description.

        Returns:
            FuncToolResult with result containing updated job details.
        """
        if job_type not in ("sql", "sparksql"):
            return FuncToolResult(success=0, error=f"Unsupported job_type '{job_type}'. Use 'sql' or 'sparksql'.")

        # Read SQL file
        try:
            sql_path = Path(sql_file_path).expanduser()
            if not sql_path.exists():
                return FuncToolResult(success=0, error=f"SQL file not found: {sql_file_path}")
            sql_content = sql_path.read_text(encoding="utf-8").strip()
            if not sql_content:
                return FuncToolResult(success=0, error=f"SQL file is empty: {sql_file_path}")
        except Exception as exc:
            return FuncToolResult(success=0, error=f"Failed to read SQL file '{sql_file_path}': {exc}")

        # Validate conn_id for sql jobs
        if job_type == "sql" and not conn_id:
            return FuncToolResult(
                success=0,
                error="'conn_id' is required for sql job_type. "
                "Set it to the Airflow Connection ID for the target database.",
            )

        try:
            adapter = self._get_adapter()
        except (ValueError, DatusException) as exc:
            return FuncToolResult(success=0, error=str(exc))

        try:
            if job_type == "sparksql":
                payload = SchedulerJobPayload(
                    job_name=job_name,
                    schedule=schedule,
                    description=description,
                    extra={
                        "job_type": "sparksql",
                        "sparksql": sql_content,
                        "spark_master": spark_master or "local[*]",
                    },
                )
            else:
                payload = SchedulerJobPayload(
                    job_name=job_name,
                    sql=sql_content,
                    db_connection={"conn_id": conn_id},
                    schedule=schedule,
                    description=description,
                )
            job = adapter.update_job(job_id, payload)
            return FuncToolResult(
                success=1,
                result={
                    "job_id": job.job_id,
                    "job_name": job.job_name,
                    "status": job.status.value,
                    "scheduler": job.platform,
                    "platform": job.platform,
                    "deliverable_target": self._build_scheduler_target(job),
                },
            )
        except Exception as exc:
            logger.error("update_job failed: %s", exc)
            return FuncToolResult(success=0, error=str(exc))
        finally:
            try:
                adapter.close()
            except Exception as close_exc:
                logger.debug("adapter.close() failed: %s", close_exc)

    def list_job_runs(
        self,
        job_id: str,
        limit: int = 10,
        offset: int = 0,
    ) -> FuncToolResult:
        """List recent runs of a scheduled job, showing status and timing.

        Args:
            job_id: The job/DAG identifier to query.
            limit: Maximum number of runs to return (default 10).
            offset: Pagination offset (default 0).

        Returns:
            FuncToolResult with result as FuncToolListResult. See
            ``list_scheduler_jobs`` for field semantics.
        """
        try:
            adapter = self._get_adapter()
        except (ValueError, DatusException) as exc:
            return FuncToolResult(success=0, error=str(exc))

        try:
            page = adapter.list_job_runs(job_id, limit=limit, offset=offset)
            rows = [
                {
                    "run_id": r.run_id,
                    "status": r.status.value,
                    "started_at": r.started_at.isoformat() if hasattr(r.started_at, "isoformat") else r.started_at,
                    "ended_at": r.ended_at.isoformat() if hasattr(r.ended_at, "isoformat") else r.ended_at,
                }
                for r in page.items
            ]
            return self._build_scheduler_envelope(rows, total=page.total, offset=offset, limit=limit)
        except Exception as exc:
            logger.error("list_job_runs failed: %s", exc)
            return FuncToolResult(success=0, error=str(exc))
        finally:
            try:
                adapter.close()
            except Exception as close_exc:
                logger.debug("adapter.close() failed: %s", close_exc)

    def get_run_log(
        self,
        job_id: str,
        run_id: str,
    ) -> FuncToolResult:
        """Get the execution log of a specific job run.

        Args:
            job_id:  The job/DAG identifier.
            run_id:  The run identifier (from list_job_runs or trigger_scheduler_job).

        Returns:
            FuncToolResult with result containing the log text.
        """
        try:
            adapter = self._get_adapter()
        except (ValueError, DatusException) as exc:
            return FuncToolResult(success=0, error=str(exc))

        try:
            log_text = adapter.get_run_log(job_id, run_id)
            return FuncToolResult(
                success=1,
                result={
                    "job_id": job_id,
                    "run_id": run_id,
                    "log": log_text,
                },
            )
        except Exception as exc:
            logger.error("get_run_log failed: %s", exc)
            return FuncToolResult(success=0, error=str(exc))
        finally:
            try:
                adapter.close()
            except Exception as close_exc:
                logger.debug("adapter.close() failed: %s", close_exc)

    def list_scheduler_connections(self) -> FuncToolResult:
        """List available scheduler connection IDs (conn_id) and their descriptions.

        Returns the connections configured in the ``scheduler.connections`` section
        of agent.yml.  Use the returned conn_id values when calling submit_sql_job
        or update_job.

        Returns:
            FuncToolResult with result containing a list of {conn_id, description}.
        """
        try:
            scheduler_config = self._selected_scheduler_config()
        except DatusException as exc:
            return FuncToolResult(success=0, error=str(exc))
        connections = scheduler_config.get("connections", {})
        if not connections:
            return FuncToolResult(
                success=1,
                result={
                    "total": 0,
                    "connections": [],
                    "hint": "No connections configured in agent.yml. "
                    "Check Airflow (Admin > Connections) for available connections.",
                },
            )
        conn_list = [{"conn_id": k, "description": v} for k, v in connections.items()]
        return FuncToolResult(
            success=1,
            result={
                "total": len(conn_list),
                "connections": conn_list,
            },
        )

    # ── Tool registration ──────────────────────────────────────────────────

    def _connections_description(self) -> str:
        """Build a suffix describing available conn_ids from scheduler config."""
        try:
            scheduler_config = self._selected_scheduler_config()
        except DatusException:
            return ""
        connections = scheduler_config.get("connections", {})
        if not connections:
            return ""
        items = ", ".join(f"'{k}' ({v})" for k, v in connections.items())
        return f"\n\nAvailable conn_id values: {items}"

    @staticmethod
    def _build_scheduler_target(job: Any) -> dict:
        """Build a ``SchedulerJobTarget`` dict from a scheduler-core
        ``ScheduledJob`` return, for ValidationHook consumption. Attached by
        mutating methods (``submit_*`` / ``update_job``).
        """
        from datus.validation.report import SchedulerJobTarget

        return SchedulerJobTarget(
            platform=str(getattr(job, "platform", "") or "unknown"),
            job_id=str(getattr(job, "job_id", "") or ""),
            job_name=getattr(job, "job_name", None),
        ).model_dump(exclude_none=True)

    def available_tools(self) -> List[Tool]:
        """Return all scheduler tool functions as FunctionTool objects."""
        methods = [
            self.submit_sql_job,
            self.submit_sparksql_job,
            self.trigger_scheduler_job,
            self.pause_job,
            self.resume_job,
            self.delete_job,
            self.update_job,
            self.get_scheduler_job,
            self.list_scheduler_jobs,
            self.list_scheduler_connections,
            self.list_job_runs,
            self.get_run_log,
        ]
        tools = [trans_to_function_tool(m) for m in methods]

        # Inject available conn_ids into tool descriptions so the LLM knows
        # which connections are available without an extra tool call.
        conn_suffix = self._connections_description()
        if conn_suffix:
            for tool in tools:
                if tool.name in ("submit_sql_job", "update_job"):
                    tool.description += conn_suffix

        return tools
