# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Session management wrapper for LLM models using OpenAI Agents Python session approach."""

import os
import sqlite3
from typing import Any, Dict

from agents import SQLiteSession

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ExtendedSQLiteSession(SQLiteSession):
    """Extended SQLite session that includes total_tokens column in agent_sessions table."""

    def _init_db_for_connection(self, conn: sqlite3.Connection) -> None:
        """Initialize the database schema with total_tokens column."""
        # Create sessions table with total_tokens column
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.sessions_table} (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_tokens INTEGER DEFAULT 0
            )
        """
        )

        # Create messages table (unchanged from parent)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.messages_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES {self.sessions_table} (session_id)
                    ON DELETE CASCADE
            )
        """
        )

        # Create index (unchanged from parent)
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self.messages_table}_session_id
            ON {self.messages_table} (session_id, created_at)
        """
        )

        conn.commit()


class SessionManager:
    """
    Manages sessions for multi-turn conversations across LLM models.

    Internally uses SQLiteSession from OpenAI Agents Python for robust session handling,
    but exposes a simple external interface that hides the complexity.
    """

    def __init__(self):
        """
        Initialize the session manager.

        Sessions are stored in {agent.home}/sessions directory.
        This path is fixed and cannot be configured.
        Configure agent.home in agent.yml to change the root directory.
        """
        from datus.utils.path_manager import get_path_manager

        self.session_dir = str(get_path_manager().sessions_dir)
        os.makedirs(self.session_dir, exist_ok=True)
        self._sessions: Dict[str, ExtendedSQLiteSession] = {}

    def get_session(self, session_id: str) -> ExtendedSQLiteSession:
        """
        Get or create a session with the given ID.

        Args:
            session_id: Unique identifier for the session

        Returns:
            ExtendedSQLiteSession instance for the given session ID
        """
        if session_id not in self._sessions:
            # Create session database path
            db_path = os.path.join(self.session_dir, f"{session_id}.db")
            self._sessions[session_id] = ExtendedSQLiteSession(session_id, db_path=db_path)
            # logger.debug(f"Created new session: {session_id} at {db_path}")

        return self._sessions[session_id]

    def create_session(self, session_id: str) -> ExtendedSQLiteSession:
        """
        Create a new session or get existing one.

        Args:
            session_id: Unique identifier for the session

        Returns:
            ExtendedSQLiteSession instance
        """
        return self.get_session(session_id)

    def clear_session(self, session_id: str) -> None:
        """
        Clear all conversation history for a session.

        Args:
            session_id: Session ID to clear
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear_session()
            logger.debug(f"Cleared session: {session_id}")
        else:
            logger.warning(f"Attempted to clear non-existent session: {session_id}")

    def delete_session(self, session_id: str) -> None:
        """
        Delete a session and its database file.

        Args:
            session_id: Session ID to delete
        """
        if session_id in self._sessions:
            # Close the session
            self._sessions.pop(session_id)

            # Delete the database file if it exists
            db_path = os.path.join(self.session_dir, f"{session_id}.db")
            if os.path.exists(db_path):
                os.remove(db_path)
                logger.debug(f"Deleted session database: {db_path}")

            logger.debug(f"Deleted session: {session_id}")
        else:
            logger.warning(f"Attempted to delete non-existent session: {session_id}")

    def list_sessions(self, limit: int = None, sort_by_modified: bool = False) -> list[str]:
        """
        List available session IDs.

        Args:
            limit: Maximum number of sessions to return (None for all)
            sort_by_modified: If True, sort by file modification time (newest first). Defaults to False.

        Returns:
            List of session IDs sorted by modification time (newest first) if sort_by_modified is True
        """
        # Check for existing database files
        session_ids = []
        if os.path.exists(self.session_dir):
            if sort_by_modified:
                # Get files with modification times
                files_with_mtime = []
                for filename in os.listdir(self.session_dir):
                    if filename.endswith(".db"):
                        filepath = os.path.join(self.session_dir, filename)
                        try:
                            mtime = os.path.getmtime(filepath)
                            session_id = filename[:-3]  # Remove .db extension
                            files_with_mtime.append((session_id, mtime))
                        except OSError:
                            continue

                # Sort by modification time (newest first) and extract session IDs
                files_with_mtime.sort(key=lambda x: x[1], reverse=True)
                session_ids = [sid for sid, _ in files_with_mtime]

                # Apply limit if specified
                if limit is not None:
                    session_ids = session_ids[:limit]
            else:
                for filename in os.listdir(self.session_dir):
                    if filename.endswith(".db"):
                        session_id = filename[:-3]  # Remove .db extension
                        session_ids.append(session_id)

                        # Apply limit if specified
                        if limit is not None and len(session_ids) >= limit:
                            break

        return session_ids

    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists and has actual data.

        Args:
            session_id: Session ID to check

        Returns:
            True if session exists and has data, False otherwise
        """
        # Check if database file exists first (avoid listing all sessions)
        db_path = os.path.join(self.session_dir, f"{session_id}.db")
        if not os.path.exists(db_path):
            return False

        # Check if the session has actual data (messages or session record)
        try:
            import sqlite3

            with sqlite3.connect(db_path, timeout=5.0) as conn:
                cursor = conn.cursor()

                # Check if session has any messages
                cursor.execute("SELECT COUNT(*) FROM agent_messages WHERE session_id = ?", (session_id,))
                message_count = cursor.fetchone()[0]

                if message_count > 0:
                    return True

                # Check if session has a record in agent_sessions
                cursor.execute("SELECT COUNT(*) FROM agent_sessions WHERE session_id = ?", (session_id,))
                session_count = cursor.fetchone()[0]

                return session_count > 0

        except Exception as e:
            logger.debug(f"Error checking session existence for {session_id}: {e}")
            return False

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get information about a session.

        Args:
            session_id: Session ID to get info for

        Returns:
            Dictionary with session information including timestamps, file size, etc.
        """
        db_path = os.path.join(self.session_dir, f"{session_id}.db")

        # Check if database file exists first
        if not os.path.exists(db_path):
            return {"exists": False}

        # Get basic file information
        file_info = {}
        try:
            if os.path.exists(db_path):
                stat = os.stat(db_path)
                file_info = {
                    "file_size": stat.st_size,
                    "file_modified": stat.st_mtime,
                }
        except Exception as e:
            logger.debug(f"Could not get file info for {db_path}: {e}")

        # Get all session data from database in efficient queries
        try:
            import json
            import sqlite3

            with sqlite3.connect(db_path, timeout=5.0) as conn:
                cursor = conn.cursor()

                # Get session metadata with COALESCE for backward compatibility
                cursor.execute(
                    """
                    SELECT created_at, updated_at, COALESCE(total_tokens, 0) as total_tokens
                    FROM agent_sessions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                session_row = cursor.fetchone()

                if session_row:
                    session_metadata = {
                        "created_at": session_row[0],
                        "updated_at": session_row[1],
                        "total_tokens": session_row[2] or 0,
                    }
                else:
                    session_metadata = {"total_tokens": 0}

                # Get message statistics in one query
                cursor.execute(
                    """
                    SELECT COUNT(*) as message_count, MAX(created_at) as latest_message_at
                    FROM agent_messages
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                message_stats = cursor.fetchone()
                if message_stats:
                    session_metadata.update(
                        {
                            "message_count": message_stats[0] or 0,
                            "item_count": message_stats[0] or 0,  # Same as message_count
                            "latest_message_at": message_stats[1],
                        }
                    )

                # Get latest user message (need to check all messages to find the most recent user message)
                cursor.execute(
                    """
                    SELECT message_data, created_at
                    FROM agent_messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    """,
                    (session_id,),
                )
                all_messages = cursor.fetchall()

                latest_user_message = None
                latest_user_message_at = None

                # Find the latest user message by scanning through all messages
                for message_data, created_at in all_messages:
                    try:
                        message_json = json.loads(message_data)
                        role = message_json.get("role", "")

                        # Find latest user message
                        if role == "user" and latest_user_message is None:
                            content = message_json.get("content", "")
                            latest_user_message = content
                            latest_user_message_at = created_at
                            break  # Found the latest user message, no need to continue

                    except (json.JSONDecodeError, TypeError):
                        # Skip malformed messages
                        continue

                session_metadata.update(
                    {
                        "latest_user_message": latest_user_message,
                        "latest_user_message_at": latest_user_message_at,
                    }
                )

        except Exception as e:
            logger.debug(f"Could not get session metadata for {session_id}: {e}")
            # Return basic info even if database query fails
            session_metadata = {"total_tokens": 0, "message_count": 0, "item_count": 0}

        return {
            "exists": True,
            "session_id": session_id,
            "db_path": db_path,
            **file_info,
            **session_metadata,
        }

    def close_all_sessions(self) -> None:
        """Close all active sessions."""
        for session_id in list(self._sessions.keys()):
            self._sessions.pop(session_id)
            # SQLiteSession doesn't have an explicit close method,
            # but removing it from our dict should handle cleanup
            logger.debug(f"Closed session: {session_id}")

    def update_session_tokens(self, session_id: str, total_tokens: int) -> None:
        """
        Update the total token count for a session in the SQLite database.

        Args:
            session_id: Session ID to update
            total_tokens: Current total token count for the session
        """
        if not self.session_exists(session_id):
            logger.warning(f"Attempted to update tokens for non-existent session: {session_id}")
            return

        try:
            import sqlite3

            db_path = os.path.join(self.session_dir, f"{session_id}.db")

            with sqlite3.connect(db_path, timeout=5.0) as conn:
                cursor = conn.cursor()

                # Check if total_tokens column exists, add it if missing (backward compatibility)
                cursor.execute("PRAGMA table_info(agent_sessions)")
                columns = [row[1] for row in cursor.fetchall()]

                if "total_tokens" not in columns:
                    logger.info(f"Adding total_tokens column to existing session: {session_id}")
                    cursor.execute("ALTER TABLE agent_sessions ADD COLUMN total_tokens INTEGER DEFAULT 0")
                    conn.commit()

                # Update the token count in the agent_sessions table
                cursor.execute(
                    "UPDATE agent_sessions SET total_tokens = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                    (total_tokens, session_id),
                )

                if cursor.rowcount == 0:
                    # Session doesn't exist in the table, create it
                    cursor.execute(
                        "INSERT OR REPLACE INTO agent_sessions (session_id, total_tokens, created_at, updated_at)"
                        " VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                        (session_id, total_tokens),
                    )

                conn.commit()
                logger.debug(f"Updated session {session_id} with {total_tokens} tokens in SQLite")

        except Exception as e:
            logger.error(f"Failed to update session tokens for {session_id}: {e}")
