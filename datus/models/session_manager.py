"""Session management wrapper for LLM models using OpenAI Agents Python session approach."""

import os
from typing import Any, Dict, Optional

from agents import SQLiteSession

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


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
        self._sessions: Dict[str, SQLiteSession] = {}

    def get_session(self, session_id: str) -> SQLiteSession:
        """
        Get or create a session with the given ID.

        Args:
            session_id: Unique identifier for the session

        Returns:
            SQLiteSession instance for the given session ID
        """
        if session_id not in self._sessions:
            # Create session database path
            db_path = os.path.join(self.session_dir, f"{session_id}.db")
            self._sessions[session_id] = SQLiteSession(session_id, db_path=db_path)
            logger.debug(f"Created new session: {session_id} at {db_path}")

        return self._sessions[session_id]

    def create_session(self, session_id: str) -> SQLiteSession:
        """
        Create a new session or get existing one.

        Args:
            session_id: Unique identifier for the session

        Returns:
            SQLiteSession instance
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
        Check if a session exists.

        Args:
            session_id: Session ID to check

        Returns:
            True if session exists, False otherwise
        """
        return session_id in self.list_sessions()

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

        session = self.get_session(session_id)
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

        try:
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
            items = asyncio.run(session.get_items())

        # Get session metadata from database
        session_metadata = {}
        try:
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Get session metadata
                cursor.execute(
                    "SELECT created_at, updated_at FROM agent_sessions WHERE session_id = ?", 
                    (session_id,)
                )
                session_row = cursor.fetchone()
                if session_row:
                    session_metadata = {
                        "created_at": session_row[0],
                        "updated_at": session_row[1],
                    }
                
                # Get message count and latest message timestamp
                cursor.execute(
                    "SELECT COUNT(*), MAX(created_at) FROM agent_messages WHERE session_id = ?", 
                    (session_id,)
                )
                message_row = cursor.fetchone()
                if message_row:
                    session_metadata.update({
                        "message_count": message_row[0],
                        "latest_message_at": message_row[1],
                    })
                    
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
