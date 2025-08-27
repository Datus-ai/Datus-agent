"""Session management wrapper for LLM models using OpenAI Agents Python session approach."""

import os
from typing import Any, Dict, Optional

from agents import SQLiteSession
import sqlite3

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

    def __init__(self, session_dir: Optional[str] = None):
        """
        Initialize the session manager.

        Args:
            session_dir: Directory to store session databases. If None, uses default location.
        """
        self.session_dir = session_dir or os.path.expanduser("~/.datus/sessions")
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

    def list_sessions(self) -> list[str]:
        """
        List all available session IDs.

        Returns:
            List of session IDs
        """
        # Check for existing database files
        session_ids = []
        if os.path.exists(self.session_dir):
            for filename in os.listdir(self.session_dir):
                if filename.endswith(".db"):
                    session_id = filename[:-3]  # Remove .db extension
                    session_ids.append(session_id)

        return session_ids

    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists and has actual data.

        Args:
            session_id: Session ID to check

        Returns:
            True if session exists and has data, False otherwise
        """
        if session_id not in self.list_sessions():
            return False
            
        # Check if the session has actual data (messages or session record)
        try:
            import sqlite3
            db_path = os.path.join(self.session_dir, f"{session_id}.db")
            with sqlite3.connect(db_path) as conn:
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
        if not self.session_exists(session_id):
            return {"exists": False}

        try:
            session = self.get_session(session_id)
            if session is None:
                logger.warning(f"Failed to create/get session for {session_id}")
                return {"exists": False}
        except Exception as e:
            logger.debug(f"Error creating/getting session {session_id}: {e}")
            return {"exists": False}
            
        db_path = os.path.join(self.session_dir, f"{session_id}.db")

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

        # Handle async get_items() call synchronously
        import asyncio

        items = None

        try:
            if session is None:
                logger.debug(f"Session is None for {session_id}, skipping items retrieval")
                items = None
            else:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to run in thread
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, session.get_items())
                        items = future.result()
                else:
                    items = loop.run_until_complete(session.get_items())
        except RuntimeError:
            # No event loop, create new one
            try:
                if session is not None:
                    items = asyncio.run(session.get_items())
                else:
                    items = None
            except Exception as e:
                logger.debug(f"Failed to get items for session {session_id}: {e}")
                items = None
        except Exception as e:
            logger.debug(f"Failed to get items for session {session_id}: {e}")
            items = None

        # Get session metadata from database
        session_metadata = {}
        try:
            import sqlite3
            import json

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Get session metadata including actual token count
                # First check if total_tokens column exists
                cursor.execute("PRAGMA table_info(agent_sessions)")
                columns = [row[1] for row in cursor.fetchall()]
                has_tokens_column = 'total_tokens' in columns
                
                if has_tokens_column:
                    cursor.execute(
                        "SELECT created_at, updated_at, total_tokens FROM agent_sessions WHERE session_id = ?",
                        (session_id,),
                    )
                    session_row = cursor.fetchone()
                    if session_row:
                        session_metadata = {
                            "created_at": session_row[0],
                            "updated_at": session_row[1],
                            "total_tokens": session_row[2] if session_row[2] is not None else 0,
                        }
                else:
                    # Old schema without total_tokens column
                    cursor.execute(
                        "SELECT created_at, updated_at FROM agent_sessions WHERE session_id = ?",
                        (session_id,),
                    )
                    session_row = cursor.fetchone()
                    if session_row:
                        session_metadata = {
                            "created_at": session_row[0],
                            "updated_at": session_row[1],
                            "total_tokens": 0,  # Default for old sessions
                        }

                # Get message count and latest message timestamp
                cursor.execute(
                    "SELECT COUNT(*), MAX(created_at) FROM agent_messages WHERE session_id = ?", (session_id,)
                )
                message_row = cursor.fetchone()
                if message_row:
                    session_metadata.update(
                        {
                            "message_count": message_row[0],
                            "latest_message_at": message_row[1],
                        }
                    )

                # Get latest user message only (no need for token estimation anymore)
                cursor.execute(
                    "SELECT message_data, created_at FROM agent_messages WHERE session_id = ? ORDER BY created_at DESC",
                    (session_id,),
                )
                all_messages = cursor.fetchall()

                # Find latest user message
                latest_user_message = None
                latest_user_message_at = None

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

                # Add latest user message metadata
                session_metadata.update(
                    {
                        "latest_user_message": latest_user_message,
                        "latest_user_message_at": latest_user_message_at,
                    }
                )

        except Exception as e:
            logger.debug(f"Could not get session metadata for {session_id}: {e}")

        return {
            "exists": True,
            "session_id": session_id,
            "item_count": len(items) if items is not None else 0,
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

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Check if total_tokens column exists, add it if missing (backward compatibility)
                cursor.execute("PRAGMA table_info(agent_sessions)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'total_tokens' not in columns:
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
                        "INSERT OR REPLACE INTO agent_sessions (session_id, total_tokens, created_at, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                        (session_id, total_tokens),
                    )

                conn.commit()
                logger.debug(f"Updated session {session_id} with {total_tokens} tokens in SQLite")

        except Exception as e:
            logger.error(f"Failed to update session tokens for {session_id}: {e}")
