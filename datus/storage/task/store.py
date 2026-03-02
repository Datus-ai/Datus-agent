# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Task storage implementation using pluggable RDB backend.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from datus.storage.rdb.base import ColumnDef, IndexDef, IntegrityError, TableDefinition, WhereOp
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_TASKS_TABLE = TableDefinition(
    table_name="tasks",
    columns=[
        ColumnDef(name="id", col_type="INTEGER", primary_key=True, autoincrement=True),
        ColumnDef(name="task_id", col_type="TEXT", nullable=False),
        ColumnDef(name="task_query", col_type="TEXT", nullable=False),
        ColumnDef(name="sql_query", col_type="TEXT", default=""),
        ColumnDef(name="sql_result", col_type="TEXT", default=""),
        ColumnDef(name="status", col_type="TEXT", default="running"),
        ColumnDef(name="user_feedback", col_type="TEXT", default=""),
        ColumnDef(name="created_at", col_type="TEXT", nullable=False),
        ColumnDef(name="updated_at", col_type="TEXT", nullable=False),
    ],
    indices=[
        IndexDef(name="idx_tasks_task_id", columns=["task_id"]),
    ],
    constraints=["UNIQUE(task_id)"],
)


@dataclass
class TaskRecord:
    """Typed record for the tasks table."""

    task_id: str = ""
    task_query: str = ""
    sql_query: str = ""
    sql_result: str = ""
    status: str = "running"
    user_feedback: str = ""
    created_at: str = ""
    updated_at: str = ""
    id: Optional[int] = None


def _task_to_dict(record: TaskRecord) -> Dict[str, Any]:
    """Convert a TaskRecord to the dict format expected by callers."""
    return {
        "task_id": record.task_id,
        "task_query": record.task_query,
        "sql_query": record.sql_query,
        "sql_result": record.sql_result,
        "status": record.status,
        "user_feedback": record.user_feedback,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


class TaskStore:
    """RDB-backed storage for task and feedback data."""

    def __init__(self):
        """Initialize the task store."""
        from datus.storage.backend_holder import create_rdb_for_store

        self._rdb = create_rdb_for_store("task")
        self._table = self._rdb.ensure_table(_TASKS_TABLE)

    def record_feedback(self, task_id: str, status: str) -> Dict[str, Any]:
        """Record user feedback for a task."""
        try:
            updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            self._table.update(
                {"user_feedback": status, "updated_at": updated_at},
                where={"task_id": task_id},
            )

            rows = self._table.query(TaskRecord, where={"task_id": task_id})

            if not rows:
                raise DatusException(ErrorCode.STORAGE_FAILED, message=f"Task {task_id} not found")

            row = rows[0]
            logger.info(f"Recorded feedback for task {task_id}: {status}")
            result = _task_to_dict(row)
            result["recorded_at"] = row.updated_at
            return result

        except DatusException:
            raise
        except Exception as e:
            raise DatusException(
                ErrorCode.STORAGE_FAILED, message=f"Failed to record feedback for task {task_id}: {str(e)}"
            ) from e

    def get_feedback(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback for a specific task."""
        try:
            rows = self._table.query(
                TaskRecord,
                where=[("task_id", WhereOp.EQ, task_id), ("user_feedback", WhereOp.NE, "")],
            )
            if rows:
                row = rows[0]
                result = _task_to_dict(row)
                result["recorded_at"] = row.updated_at
                return result
            return None
        except Exception as e:
            raise DatusException(
                ErrorCode.STORAGE_FAILED, message=f"Failed to get feedback for task {task_id}: {str(e)}"
            ) from e

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        """Get all recorded feedback."""
        try:
            rows = self._table.query(
                TaskRecord,
                where=[("user_feedback", WhereOp.NE, "")],
                order_by=["-updated_at"],
            )
            results = []
            for row in rows:
                d = _task_to_dict(row)
                d["recorded_at"] = row.updated_at
                results.append(d)
            return results
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_FAILED, message=f"Failed to get all feedback: {str(e)}") from e

    def delete_feedback(self, task_id: str) -> bool:
        """Clear feedback for a specific task."""
        try:
            updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            count = self._table.update(
                {"user_feedback": "", "updated_at": updated_at},
                where=[("task_id", WhereOp.EQ, task_id), ("user_feedback", WhereOp.NE, "")],
            )
            if count > 0:
                logger.info(f"Cleared feedback for task {task_id}")
                return True
            return False
        except Exception as e:
            raise DatusException(
                ErrorCode.STORAGE_FAILED, message=f"Failed to clear feedback for task {task_id}: {str(e)}"
            ) from e

    def create_task(self, task_id: str, task_query: str) -> Dict[str, Any]:
        """Create a new task record (idempotent — does not overwrite existing tasks)."""
        try:
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            record = TaskRecord(task_id=task_id, task_query=task_query, created_at=now, updated_at=now)
            try:
                self._table.insert(record)
            except IntegrityError:
                # Task already exists — return its current state without overwriting
                existing = self.get_task(task_id)
                if existing:
                    return existing
            logger.debug(f"Created task record for {task_id}")

            return {
                "task_id": task_id,
                "task_query": task_query,
                "sql_query": "",
                "sql_result": "",
                "status": "running",
                "user_feedback": "",
                "created_at": now,
                "updated_at": now,
            }

        except DatusException:
            raise
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_FAILED, message=f"Failed to create task {task_id}: {str(e)}") from e

    def update_task(self, task_id: str, sql_query: str = None, sql_result: str = None, status: str = None) -> bool:
        """Update task information."""
        try:
            updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            data: Dict[str, Any] = {}
            if sql_query is not None:
                data["sql_query"] = sql_query
            if sql_result is not None:
                data["sql_result"] = sql_result
            if status is not None:
                data["status"] = status

            if not data:
                return False

            data["updated_at"] = updated_at

            count = self._table.update(data, where={"task_id": task_id})
            if count > 0:
                logger.debug(f"Updated task {task_id}")
                return True
            return False
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_FAILED, message=f"Failed to update task {task_id}: {str(e)}") from e

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task information."""
        try:
            rows = self._table.query(TaskRecord, where={"task_id": task_id})
            if rows:
                return _task_to_dict(rows[0])
            return None
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_FAILED, message=f"Failed to get task {task_id}: {str(e)}") from e

    def delete_task(self, task_id: str) -> bool:
        """Delete a task record."""
        try:
            count = self._table.delete(where={"task_id": task_id})
            if count > 0:
                logger.debug(f"Deleted task {task_id}")
                return True
            return False
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_FAILED, message=f"Failed to delete task {task_id}: {str(e)}") from e

    def cleanup_old_tasks(self, hours: int = 24) -> int:
        """Clean up old task records."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            cutoff_str = cutoff_time.isoformat().replace("+00:00", "Z")

            count = self._table.delete(where=[("created_at", WhereOp.LT, cutoff_str)])
            if count > 0:
                logger.info(f"Cleaned up {count} old tasks")
            return count
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_FAILED, message=f"Failed to cleanup old tasks: {str(e)}") from e
