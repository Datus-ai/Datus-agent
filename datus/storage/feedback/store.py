# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Feedback storage implementation using SQLite.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class FeedbackStore:
    """SQLite-based storage for user feedback data."""

    def __init__(self, db_path: str):
        """Initialize the feedback store.

        Args:
            db_path: Path to the directory where the SQLite database will be stored
        """
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.db_file = os.path.join(db_path, "feedback.db")
        self._ensure_table()

    def _ensure_table(self):
        """Ensure the feedback table exists in the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(task_id)
                    )
                """
                )

                # Create index for faster task_id lookups
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_task_id ON feedback(task_id)
                """
                )

                conn.commit()
                logger.debug(f"Feedback table ensured in {self.db_file}")
        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_STORE_FAILED, message=f"Failed to create feedback table: {str(e)}"
            ) from e

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise DatusException(ErrorCode.TOOL_STORE_FAILED, message=f"Database connection error: {str(e)}") from e
        finally:
            if conn:
                conn.close()

    def record_feedback(self, task_id: str, status: str) -> Dict[str, Any]:
        """Record user feedback for a task.

        Args:
            task_id: The task ID to record feedback for
            status: The feedback status ("success" or "failed")

        Returns:
            Dictionary containing the recorded feedback data

        Raises:
            DatusException: If the feedback recording fails
        """
        try:
            recorded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Use INSERT OR REPLACE to handle duplicate task_ids
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO feedback (task_id, status, created_at)
                    VALUES (?, ?, ?)
                """,
                    (task_id, status, recorded_at),
                )

                conn.commit()

                logger.info(f"Recorded feedback for task {task_id}: {status}")

                return {"task_id": task_id, "status": status, "recorded_at": recorded_at}

        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_STORE_FAILED, message=f"Failed to record feedback for task {task_id}: {str(e)}"
            ) from e

    def get_feedback(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback for a specific task.

        Args:
            task_id: The task ID to get feedback for

        Returns:
            Dictionary containing the feedback data, or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT task_id, status, created_at
                    FROM feedback
                    WHERE task_id = ?
                """,
                    (task_id,),
                )

                row = cursor.fetchone()
                if row:
                    return {"task_id": row[0], "status": row[1], "recorded_at": row[2]}
                return None

        except Exception as e:
            logger.error(f"Failed to get feedback for task {task_id}: {str(e)}")
            return None

    def get_all_feedback(self) -> list[Dict[str, Any]]:
        """Get all recorded feedback.

        Returns:
            List of dictionaries containing all feedback data
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT task_id, status, created_at
                    FROM feedback
                    ORDER BY created_at DESC
                """
                )

                rows = cursor.fetchall()
                return [{"task_id": row[0], "status": row[1], "recorded_at": row[2]} for row in rows]

        except Exception as e:
            logger.error(f"Failed to get all feedback: {str(e)}")
            return []

    def delete_feedback(self, task_id: str) -> bool:
        """Delete feedback for a specific task.

        Args:
            task_id: The task ID to delete feedback for

        Returns:
            True if feedback was deleted, False if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM feedback WHERE task_id = ?", (task_id,))
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"Deleted feedback for task {task_id}")
                    return True
                return False

        except Exception as e:
            logger.error(f"Failed to delete feedback for task {task_id}: {str(e)}")
            return False
