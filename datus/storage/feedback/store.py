# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Feedback storage implementation using pluggable RDB backend.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from datus_storage_base.rdb.base import ColumnDef, IndexDef, TableDefinition

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_FEEDBACK_TABLE = TableDefinition(
    table_name="feedback",
    columns=[
        ColumnDef(name="id", col_type="INTEGER", primary_key=True, autoincrement=True),
        ColumnDef(name="task_id", col_type="TEXT", nullable=False),
        ColumnDef(name="status", col_type="TEXT", nullable=False),
        ColumnDef(name="created_at", col_type="TEXT", nullable=False),
    ],
    indices=[
        IndexDef(name="idx_task_id", columns=["task_id"]),
    ],
    constraints=["UNIQUE(task_id)"],
)


@dataclass
class FeedbackRecord:
    """Typed record for the feedback table."""

    task_id: str = ""
    status: str = ""
    created_at: str = ""
    id: Optional[int] = None


class FeedbackStore:
    """RDB-backed storage for user feedback data."""

    def __init__(self, project: str = ""):
        """Initialize the feedback store scoped to *project* (defaults to active).

        Args:
            project: Project identifier used for backend isolation. When
                empty, the active ``DatusPathManager.project_name`` is used.
        """
        from datus.storage.backend_holder import create_rdb_for_store
        from datus.utils.path_manager import get_path_manager

        resolved = project or get_path_manager().project_name
        self._rdb = create_rdb_for_store("feedback", resolved)
        self._table = self._rdb.ensure_table(_FEEDBACK_TABLE)

    def record_feedback(self, task_id: str, status: str) -> Dict[str, Any]:
        """Record user feedback for a task."""
        try:
            recorded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            self._table.upsert(
                FeedbackRecord(task_id=task_id, status=status, created_at=recorded_at),
                conflict_columns=["task_id"],
            )

            logger.info(f"Recorded feedback for task {task_id}: {status}")
            return {"task_id": task_id, "status": status, "recorded_at": recorded_at}

        except Exception as e:
            raise DatusException(
                ErrorCode.STORAGE_FAILED, message=f"Failed to record feedback for task {task_id}: {str(e)}"
            ) from e

    def get_feedback(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback for a specific task."""
        try:
            rows = self._table.query(FeedbackRecord, where={"task_id": task_id})
            if rows:
                row = rows[0]
                return {"task_id": row.task_id, "status": row.status, "recorded_at": row.created_at}
            return None
        except Exception as e:
            raise DatusException(
                ErrorCode.STORAGE_FAILED, message=f"Failed to get feedback for task {task_id}: {str(e)}"
            ) from e

    def get_all_feedback(self) -> list[Dict[str, Any]]:
        """Get all recorded feedback."""
        try:
            rows = self._table.query(FeedbackRecord, order_by=["-created_at"])
            return [{"task_id": row.task_id, "status": row.status, "recorded_at": row.created_at} for row in rows]
        except Exception as e:
            raise DatusException(ErrorCode.STORAGE_FAILED, message=f"Failed to get all feedback: {str(e)}") from e

    def delete_feedback(self, task_id: str) -> bool:
        """Delete feedback for a specific task."""
        try:
            count = self._table.delete(where={"task_id": task_id})
            if count > 0:
                logger.info(f"Deleted feedback for task {task_id}")
                return True
            return False
        except Exception as e:
            raise DatusException(
                ErrorCode.STORAGE_FAILED, message=f"Failed to delete feedback for task {task_id}: {str(e)}"
            ) from e
