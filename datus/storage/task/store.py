# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Task storage implementation using SQLite.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class TaskStore:
    """SQLite-based storage for task and feedback data."""

    def __init__(self, db_path: str):
        """Initialize the task store.

        Args:
            db_path: Path to the directory where the SQLite database will be stored
        """
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.db_file = os.path.join(db_path, "task.db")
        self._ensure_table()

    def _ensure_table(self):
        """Ensure the feedback and tasks tables exist in the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Create unified tasks table with user feedback
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        task_query TEXT NOT NULL,
                        sql_query TEXT DEFAULT '',
                        sql_result TEXT DEFAULT '',
                        status TEXT DEFAULT 'running',
                        user_feedback TEXT DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(task_id)
                    )
                """
                )

                # Create index for faster lookups
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_tasks_task_id ON tasks(task_id)
                """
                )

                conn.commit()
                logger.debug(f"Tasks table ensured in {self.db_file}")
        except Exception as e:
            raise DatusException(ErrorCode.TOOL_STORE_FAILED, message=f"Failed to create tasks table: {str(e)}") from e

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
            Dictionary containing the updated task data

        Raises:
            DatusException: If the feedback recording fails
        """
        try:
            updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Update user_feedback field for the task
                cursor.execute(
                    """
                    UPDATE tasks SET user_feedback = ?, updated_at = ?
                    WHERE task_id = ?
                """,
                    (status, updated_at, task_id),
                )

                if cursor.rowcount == 0:
                    raise DatusException(ErrorCode.TOOL_STORE_FAILED, message=f"Task {task_id} not found")

                conn.commit()

                # Get the updated task data
                cursor.execute(
                    """
                    SELECT task_id, task_query, sql_query, sql_result, status, user_feedback, created_at, updated_at
                    FROM tasks
                    WHERE task_id = ?
                    """,
                    (task_id,),
                )

                row = cursor.fetchone()
                if row:
                    logger.info(f"Recorded feedback for task {task_id}: {status}")
                    return {
                        "task_id": row[0],
                        "task_query": row[1],
                        "sql_query": row[2],
                        "sql_result": row[3],
                        "status": row[4],
                        "user_feedback": row[5],
                        "created_at": row[6],
                        "recorded_at": row[7],  # Use updated_at as recorded_at for compatibility
                    }
                else:
                    raise DatusException(
                        ErrorCode.TOOL_STORE_FAILED, message=f"Failed to retrieve updated task {task_id}"
                    )

        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_STORE_FAILED, message=f"Failed to record feedback for task {task_id}: {str(e)}"
            ) from e

    def get_feedback(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback for a specific task.

        Args:
            task_id: The task ID to get feedback for

        Returns:
            Dictionary containing the task data with feedback, or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT task_id, task_query, sql_query, sql_result, status, user_feedback, created_at, updated_at
                    FROM tasks
                    WHERE task_id = ? AND user_feedback != ''
                """,
                    (task_id,),
                )

                row = cursor.fetchone()
                if row:
                    return {
                        "task_id": row[0],
                        "task_query": row[1],
                        "sql_query": row[2],
                        "sql_result": row[3],
                        "status": row[4],
                        "user_feedback": row[5],
                        "created_at": row[6],
                        "recorded_at": row[7],
                    }
                return None

        except Exception as e:
            logger.error(f"Failed to get feedback for task {task_id}: {str(e)}")
            return None

    def get_all_feedback(self) -> list[Dict[str, Any]]:
        """Get all recorded feedback.

        Returns:
            List of dictionaries containing all tasks with feedback
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT task_id, task_query, sql_query, sql_result, status, user_feedback, created_at, updated_at
                    FROM tasks
                    WHERE user_feedback != ''
                    ORDER BY updated_at DESC
                """
                )

                rows = cursor.fetchall()
                return [
                    {
                        "task_id": row[0],
                        "task_query": row[1],
                        "sql_query": row[2],
                        "sql_result": row[3],
                        "status": row[4],
                        "user_feedback": row[5],
                        "created_at": row[6],
                        "recorded_at": row[7],
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to get all feedback: {str(e)}")
            return []

    def delete_feedback(self, task_id: str) -> bool:
        """Clear feedback for a specific task.

        Args:
            task_id: The task ID to clear feedback for

        Returns:
            True if feedback was cleared, False if not found
        """
        try:
            updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE tasks SET user_feedback = '', updated_at = ?
                    WHERE task_id = ? AND user_feedback != ''
                    """,
                    (updated_at, task_id),
                )
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"Cleared feedback for task {task_id}")
                    return True
                return False

        except Exception as e:
            logger.error(f"Failed to clear feedback for task {task_id}: {str(e)}")
            return False

    # Task management methods
    def create_task(self, task_id: str, task_query: str) -> Dict[str, Any]:
        """Create a new task record.

        Args:
            task_id: The task ID
            task_query: The original user task/query

        Returns:
            Dictionary containing the created task data

        Raises:
            DatusException: If the task creation fails
        """
        try:
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO tasks (task_id, task_query, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (task_id, task_query, now, now),
                )

                conn.commit()
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

        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_STORE_FAILED, message=f"Failed to create task {task_id}: {str(e)}"
            ) from e

    def update_task(self, task_id: str, sql_query: str = None, sql_result: str = None, status: str = None) -> bool:
        """Update task information.

        Args:
            task_id: The task ID to update
            sql_query: The generated SQL query (optional)
            sql_result: The SQL execution result (optional)
            status: The task status (optional)

        Returns:
            True if task was updated, False if not found
        """
        try:
            updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Build dynamic update query based on provided parameters
                updates = []
                params = []

                if sql_query is not None:
                    updates.append("sql_query = ?")
                    params.append(sql_query)

                if sql_result is not None:
                    updates.append("sql_result = ?")
                    params.append(sql_result)

                if status is not None:
                    updates.append("status = ?")
                    params.append(status)

                updates.append("updated_at = ?")
                params.append(updated_at)
                params.append(task_id)

                if updates:
                    query = f"UPDATE tasks SET {', '.join(updates)} WHERE task_id = ?"
                    cursor.execute(query, params)
                    conn.commit()

                    if cursor.rowcount > 0:
                        logger.debug(f"Updated task {task_id}")
                        return True

                return False

        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {str(e)}")
            return False

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task information.

        Args:
            task_id: The task ID to get

        Returns:
            Dictionary containing the task data, or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT task_id, task_query, sql_query, sql_result, status, user_feedback, created_at, updated_at
                    FROM tasks
                    WHERE task_id = ?
                    """,
                    (task_id,),
                )

                row = cursor.fetchone()
                if row:
                    return {
                        "task_id": row[0],
                        "task_query": row[1],
                        "sql_query": row[2],
                        "sql_result": row[3],
                        "status": row[4],
                        "user_feedback": row[5],
                        "created_at": row[6],
                        "updated_at": row[7],
                    }
                return None

        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {str(e)}")
            return None

    def delete_task(self, task_id: str) -> bool:
        """Delete a task record.

        Args:
            task_id: The task ID to delete

        Returns:
            True if task was deleted, False if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
                conn.commit()

                if cursor.rowcount > 0:
                    logger.debug(f"Deleted task {task_id}")
                    return True
                return False

        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {str(e)}")
            return False

    def cleanup_old_tasks(self, hours: int = 24) -> int:
        """Clean up old task records.

        Args:
            hours: Delete tasks older than this many hours

        Returns:
            Number of tasks deleted
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            cutoff_str = cutoff_time.isoformat().replace("+00:00", "Z")

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tasks WHERE created_at < ?", (cutoff_str,))
                conn.commit()

                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old tasks")

                return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old tasks: {str(e)}")
            return 0
