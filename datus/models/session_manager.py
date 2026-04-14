# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Session management wrapper for LLM models using OpenAI Agents Python session approach."""

import ast
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agents.extensions.memory import AdvancedSQLiteSession

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.async_utils import run_async
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger
from datus.utils.message_utils import extract_user_input

logger = get_logger(__name__)

if TYPE_CHECKING:
    from datus.utils.path_manager import DatusPathManager


class SessionManager:
    """
    Manages sessions for multi-turn conversations across LLM models.

    Internally uses SQLiteSession from OpenAI Agents Python for robust session handling,
    but exposes a simple external interface that hides the complexity.
    """

    def __init__(
        self,
        session_dir: Optional[str] = None,
        scope: Optional[str] = None,
        *,
        path_manager: Optional["DatusPathManager"] = None,
        agent_config: Optional[Any] = None,
    ):
        """
        Initialize the session manager.

        Args:
            session_dir: Optional custom session directory path. When provided,
                sessions are stored in this directory (used by SaaS backend for
                per-project session isolation). When None, falls back to the
                default {agent.home}/sessions path.
            scope: Optional scope name for session directory isolation.
                When provided, sessions are stored under {session_dir}/{scope}/.
                When None or empty, sessions are stored directly in {session_dir}/
                (backward compatible with previous behavior).
                Only alphanumerics, hyphens, and underscores are allowed.
        """
        if session_dir and str(session_dir).strip():
            self.session_dir = str(session_dir)
        else:
            from datus.utils.path_manager import get_path_manager

            self.session_dir = str(get_path_manager(path_manager=path_manager, agent_config=agent_config).sessions_dir)

        # Apply scope subdirectory only when explicitly provided
        if scope and scope.strip():
            resolved_scope = scope.strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]+", resolved_scope):
                raise DatusException(
                    ErrorCode.COMMON_VALIDATION_FAILED,
                    message=f"Invalid scope: {resolved_scope!r}. "
                    "Scope may only contain alphanumerics, hyphens, and underscores.",
                )
            self.session_dir = os.path.join(self.session_dir, resolved_scope)
        os.makedirs(self.session_dir, exist_ok=True)
        self._sessions: Dict[str, AdvancedSQLiteSession] = {}

    # Shared pattern for validating session IDs.
    # Allows alphanumerics, hyphens, underscores, and dots.
    _SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        """Validate that a session ID is safe for use in file paths.

        Allows only alphanumerics, hyphens, underscores, and dots.
        Raises ValueError if the session ID contains unsafe characters.
        """
        if not SessionManager._SESSION_ID_RE.fullmatch(session_id):
            raise ValueError(
                f"Invalid session ID: {session_id!r}. "
                "Session IDs may only contain alphanumerics, hyphens, underscores, and dots."
            )
        return session_id

    def get_session(self, session_id: str) -> AdvancedSQLiteSession:
        """
        Get or create a session with the given ID.

        Args:
            session_id: Unique identifier for the session

        Returns:
            AdvancedSQLiteSession instance for the given session ID
        """
        self._validate_session_id(session_id)
        if session_id not in self._sessions:
            db_path = os.path.join(self.session_dir, f"{session_id}.db")
            session = AdvancedSQLiteSession(
                session_id=session_id,
                db_path=db_path,
                create_tables=True,
            )
            self._sessions[session_id] = session
            return session

        return self._sessions[session_id]

    def create_session(self, session_id: str) -> AdvancedSQLiteSession:
        """
        Create a new session or get existing one.

        Args:
            session_id: Unique identifier for the session

        Returns:
            AdvancedSQLiteSession instance
        """
        return self.get_session(session_id)

    def clear_session(self, session_id: str) -> None:
        """
        Clear all conversation history for a session.

        Args:
            session_id: Session ID to clear
        """
        # Load session from disk if not in memory
        session = self.get_session(session_id) if self.session_exists(session_id) else self._sessions.get(session_id)
        if session:
            run_async(session.clear_session())
            logger.debug(f"Cleared session: {session_id}")
        else:
            logger.warning(f"Attempted to clear non-existent session: {session_id}")

    def delete_session(self, session_id: str) -> None:
        """
        Delete a session and its database file.

        Args:
            session_id: Session ID to delete
        """
        self._validate_session_id(session_id)
        # Remove from in-memory cache if present
        self._sessions.pop(session_id, None)

        # Delete the database file and SQLite WAL/SHM files if they exist on disk
        db_path = os.path.join(self.session_dir, f"{session_id}.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            for suffix in ("-shm", "-wal"):
                wal_path = db_path + suffix
                if os.path.exists(wal_path):
                    os.remove(wal_path)
            logger.debug(f"Deleted session: {session_id}")
        else:
            logger.warning(f"Attempted to delete non-existent session: {session_id}")

    def rewind_session(
        self,
        source_session_id: str,
        up_to_user_turn: int,
        include_assistant_response: bool = True,
    ) -> str:
        """
        Create a new session by copying messages up to a given user turn from an existing session.

        Args:
            source_session_id: The session to copy from
            up_to_user_turn: Keep messages up to and including this user turn number (1-based)
            include_assistant_response: If True, also include the assistant response after the last user turn

        Returns:
            The new session ID
        """
        self._validate_session_id(source_session_id)
        if up_to_user_turn < 1:
            raise ValueError("up_to_user_turn must be >= 1")
        # Extract node type and generate new session ID
        if "_session_" in source_session_id:
            node_type = source_session_id.rsplit("_session_", 1)[0]
        else:
            node_type = "chat"
        new_session_id = f"{node_type}_session_{uuid.uuid4().hex[:8]}"

        source_db_path = os.path.join(self.session_dir, f"{source_session_id}.db")
        if not os.path.exists(source_db_path):
            raise FileNotFoundError(f"Source session database not found: {source_session_id}")

        # Read source messages ordered by creation time
        with sqlite3.connect(source_db_path, timeout=5.0) as src_conn:
            cursor = src_conn.cursor()
            cursor.execute(
                "SELECT id, session_id, message_data, created_at FROM agent_messages "
                "WHERE session_id = ? ORDER BY created_at, id",
                (source_session_id,),
            )
            rows = cursor.fetchall()

        # Determine the truncation boundary
        user_turn_count = 0
        cutoff_index = len(rows)  # default: keep all

        for i, (_, _, message_data, _) in enumerate(rows):
            try:
                msg = json.loads(message_data)
            except (json.JSONDecodeError, TypeError):
                continue
            if msg.get("role") == "user":
                user_turn_count += 1
                if user_turn_count > up_to_user_turn:
                    # This user message starts the next turn beyond the requested range
                    cutoff_index = i
                    break

        # If include_assistant_response is False, cut right after the user turn's own message
        if not include_assistant_response and user_turn_count >= up_to_user_turn:
            # Walk backwards from cutoff to find the end of the user turn's user message
            target_count = 0
            for i, (_, _, message_data, _) in enumerate(rows):
                try:
                    msg = json.loads(message_data)
                except (json.JSONDecodeError, TypeError):
                    continue
                if msg.get("role") == "user":
                    target_count += 1
                    if target_count == up_to_user_turn:
                        # Include this user message, but nothing after it
                        cutoff_index = i + 1
                        break

        kept_rows = rows[:cutoff_index]
        if not kept_rows:
            raise ValueError(f"No messages to keep for turn {up_to_user_turn}")

        # Create the new session database
        new_db_path = os.path.join(self.session_dir, f"{new_session_id}.db")
        new_session = AdvancedSQLiteSession(session_id=new_session_id, db_path=new_db_path, create_tables=True)
        # Store in cache
        self._sessions[new_session_id] = new_session

        # Read turn_usage rows for kept turns from source DB
        turn_usage_rows = []
        with sqlite3.connect(source_db_path, timeout=5.0) as src_conn:
            try:
                cursor = src_conn.cursor()
                cursor.execute(
                    "SELECT branch_id, user_turn_number, requests, input_tokens, "
                    "output_tokens, total_tokens, input_tokens_details, "
                    "output_tokens_details, created_at "
                    "FROM turn_usage WHERE session_id = ? AND user_turn_number <= ?",
                    (source_session_id, up_to_user_turn),
                )
                turn_usage_rows = cursor.fetchall()
            except sqlite3.OperationalError:
                # turn_usage table may not exist in older databases
                pass

        # Insert session record, messages, and turn_usage into the new DB
        with sqlite3.connect(new_db_path, timeout=5.0) as new_conn:
            new_conn.execute(
                "INSERT OR IGNORE INTO agent_sessions (session_id) VALUES (?)",
                (new_session_id,),
            )
            for _, _, message_data, created_at in kept_rows:
                new_conn.execute(
                    "INSERT INTO agent_messages (session_id, message_data, created_at) VALUES (?, ?, ?)",
                    (new_session_id, message_data, created_at),
                )
            for usage_row in turn_usage_rows:
                new_conn.execute(
                    "INSERT OR IGNORE INTO turn_usage "
                    "(session_id, branch_id, user_turn_number, requests, input_tokens, "
                    "output_tokens, total_tokens, input_tokens_details, "
                    "output_tokens_details, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (new_session_id, *usage_row),
                )
            new_conn.commit()

        logger.info(
            f"Rewound session {source_session_id} to turn {up_to_user_turn} -> new session {new_session_id} "
            f"({len(kept_rows)} messages copied)"
        )
        return new_session_id

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
        self._validate_session_id(session_id)
        # Check if database file exists first (avoid listing all sessions)
        db_path = os.path.join(self.session_dir, f"{session_id}.db")
        if not os.path.exists(db_path):
            return False

        # Check if the session has actual data (messages or session record)
        try:
            with sqlite3.connect(db_path, timeout=5.0) as conn:
                cursor = conn.cursor()

                # Check if session has any messages
                cursor.execute(
                    "SELECT COUNT(*) FROM agent_messages WHERE session_id = ?",
                    (session_id,),
                )
                message_count = cursor.fetchone()[0]

                if message_count > 0:
                    return True

                # Check if session has a record in agent_sessions
                cursor.execute(
                    "SELECT COUNT(*) FROM agent_sessions WHERE session_id = ?",
                    (session_id,),
                )
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
        self._validate_session_id(session_id)
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
            with sqlite3.connect(db_path, timeout=5.0) as conn:
                cursor = conn.cursor()

                # Get session metadata
                cursor.execute(
                    """
                    SELECT created_at, updated_at
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
                    }
                else:
                    session_metadata = {}

                # Aggregate total_tokens from turn_usage table
                try:
                    cursor.execute(
                        "SELECT COALESCE(SUM(total_tokens), 0) FROM turn_usage WHERE session_id = ?",
                        (session_id,),
                    )
                    session_metadata["total_tokens"] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    session_metadata["total_tokens"] = 0

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

                        # Find latest user message (extract original user input from structured content)
                        if role == "user" and latest_user_message is None:
                            content = extract_user_input(message_json.get("content", ""))
                            latest_user_message = content
                            latest_user_message_at = created_at
                            break  # Found the latest user message, no need to continue

                    except (json.JSONDecodeError, TypeError):
                        # Skip malformed messages
                        continue

                # Find the first user message (by ASC order)
                first_user_message = None
                first_user_message_at = None
                for message_data, created_at in reversed(all_messages):
                    try:
                        message_json = json.loads(message_data)
                        role = message_json.get("role", "")
                        if role == "user":
                            content = extract_user_input(message_json.get("content", ""))
                            first_user_message = content
                            first_user_message_at = created_at
                            break
                    except (json.JSONDecodeError, TypeError):
                        continue

                session_metadata.update(
                    {
                        "latest_user_message": latest_user_message,
                        "latest_user_message_at": latest_user_message_at,
                        "first_user_message": first_user_message,
                        "first_user_message_at": first_user_message_at,
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

    def get_detailed_usage(self, session_id: str) -> Dict[str, Any]:
        """Query turn_usage table and return aggregated + per-turn token usage."""
        self._validate_session_id(session_id)
        db_path = os.path.join(self.session_dir, f"{session_id}.db")
        empty_result = {
            "total": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cached_tokens": 0},
            "turns": [],
            "turn_count": 0,
        }
        if not os.path.exists(db_path):
            return empty_result

        total = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
        turns: List[Dict[str, Any]] = []

        try:
            with sqlite3.connect(db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT user_turn_number, requests, input_tokens, output_tokens, "
                    "total_tokens, input_tokens_details, output_tokens_details, created_at "
                    "FROM turn_usage WHERE session_id = ? ORDER BY user_turn_number",
                    (session_id,),
                )
                for row in cursor.fetchall():
                    turn_number, requests, inp, out, tot, inp_details, out_details, created_at = row
                    total["requests"] += requests or 0
                    total["input_tokens"] += inp or 0
                    total["output_tokens"] += out or 0
                    total["total_tokens"] += tot or 0

                    # Parse JSON detail fields
                    inp_detail_dict = {}
                    out_detail_dict = {}
                    if inp_details:
                        try:
                            inp_detail_dict = json.loads(inp_details)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if out_details:
                        try:
                            out_detail_dict = json.loads(out_details)
                        except (json.JSONDecodeError, TypeError):
                            pass

                    cached = inp_detail_dict.get("cached_tokens", 0)
                    total["cached_tokens"] += cached or 0

                    turns.append(
                        {
                            "turn_number": turn_number,
                            "requests": requests or 0,
                            "input_tokens": inp or 0,
                            "output_tokens": out or 0,
                            "total_tokens": tot or 0,
                            "input_tokens_details": inp_detail_dict,
                            "output_tokens_details": out_detail_dict,
                            "created_at": created_at,
                        }
                    )
        except sqlite3.OperationalError:
            logger.debug(f"turn_usage table not found for session {session_id}")

        return {"total": total, "turns": turns, "turn_count": len(turns)}

    @staticmethod
    def _parse_final_output(actions: List[ActionHistory], current_assistant_group: Dict) -> Optional[ActionHistory]:
        """Try to parse sql/output from the last assistant action's messages and update assistant group.

        Searches *actions* in reverse for the last ASSISTANT action and attempts
        to extract structured JSON (sql/output).  When JSON extraction fails
        (e.g. chat agent producing plain markdown), the raw text is used as
        ``content`` so it can be rendered as markdown during resume.
        """
        # Find last assistant action (may not be the very last action)
        last_assistant = None
        for action in reversed(actions):
            if action.role == ActionRole.ASSISTANT:
                last_assistant = action
                break

        if not last_assistant or not last_assistant.messages:
            return None

        result_json = llm_result2json(last_assistant.messages)
        if isinstance(result_json, str):
            # Plain string output — use directly as content
            current_assistant_group["content"] = result_json
            return None
        if isinstance(result_json, dict) and (
            "sql" in result_json or "output" in result_json or "response" in result_json
        ):
            output = {}
            if "sql" in result_json:
                output["sql"] = result_json["sql"]
            # Treat "response" as alias for "output" (prefer "response" if present)
            content_value = result_json.get("response") or result_json.get("output", "")
            output["response"] = content_value
            current_assistant_group["content"] = content_value
            current_assistant_group["sql"] = result_json.get("sql", "")
            # Create final action
            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="chat_response",
                messages="Chat interaction completed successfully",
                input_data={},
                output_data=output,
                status=ActionStatus.SUCCESS,
            )
            return final_action

        # Non-JSON output (e.g. chat agent markdown) — use raw text as content
        current_assistant_group["content"] = last_assistant.messages
        return None

    def get_session_messages(self, session_id: str) -> List[Dict]:
        """
        Get all messages from a session stored in SQLite, aggregating consecutive assistant messages.

        Args:
            session_id: Session ID to load messages from

        Returns:
            List of message dictionaries with role, content, timestamp, SQL, and progress
        """
        messages = []

        # Validate session_id to prevent path traversal
        if not self._SESSION_ID_RE.fullmatch(session_id):
            logger.warning(f"Invalid session_id format (potential path traversal): {session_id}")
            return messages

        # Build path with pathlib and resolve to absolute path
        sessions_dir = Path(self.session_dir)
        db_path = (sessions_dir / f"{session_id}.db").resolve()

        # Ensure resolved path is within sessions directory
        try:
            db_path.relative_to(sessions_dir.resolve())
        except ValueError:
            logger.warning(f"Session path outside of sessions directory (path traversal attempt): {db_path}")
            return messages

        if not db_path.exists():
            logger.warning(f"Session database not found: {db_path}")
            return messages

        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT message_data, created_at
                    FROM agent_messages
                    WHERE session_id = ?
                    ORDER BY created_at, id
                    """,
                    (session_id,),
                )

                # Aggregate consecutive assistant messages
                current_assistant_group = None
                assistant_progress = []
                current_actions = []  # Collect ActionHistory objects for detailed view

                for message_data, created_at in cursor.fetchall():
                    try:
                        message_json = json.loads(message_data)
                        role = message_json.get("role", "")
                        msg_type = message_json.get("type", "")

                        # Handle user messages
                        if role == "user":
                            # Before adding user message, flush any pending assistant group
                            if current_assistant_group:
                                final_action = self._parse_final_output(current_actions, current_assistant_group)
                                if final_action:
                                    current_actions.append(final_action)

                                # Add collected actions and progress to the assistant group
                                if current_actions:
                                    current_assistant_group["actions"] = current_actions.copy()
                                if assistant_progress:
                                    current_assistant_group["progress_messages"] = assistant_progress.copy()

                                messages.append(current_assistant_group)
                                current_assistant_group = None
                                assistant_progress = []
                                current_actions = []

                            # Add user message (extract original user input from structured content)
                            content = extract_user_input(message_json.get("content", ""))
                            messages.append(
                                {"role": "user", "content": content, "timestamp": created_at, "created_at": created_at}
                            )
                            continue

                        # Handle function calls (tool calls)
                        if msg_type == "function_call":
                            tool_name = message_json.get("name", "unknown")
                            arguments = message_json.get("arguments", "{}")

                            # Initialize assistant group if needed
                            if not current_assistant_group:
                                current_assistant_group = {
                                    "role": "assistant",
                                    "content": "",
                                    "timestamp": created_at,
                                    "created_at": created_at,
                                }

                            # Parse arguments
                            try:
                                args_dict = json.loads(arguments) if arguments else {}
                                args_str = str(args_dict)[:60]
                                assistant_progress.append(f"✓ Tool call: {tool_name}({args_str})")
                            except (json.JSONDecodeError, ValueError, TypeError):
                                args_dict = {}
                                assistant_progress.append(f"✓ Tool call: {tool_name}")

                            # Create ActionHistory for tool call (use original call_id from SDK)
                            action = ActionHistory(
                                action_id=message_json.get("call_id", str(uuid.uuid4())),
                                role=ActionRole.TOOL,
                                messages=f"Tool call: {tool_name}",
                                action_type=tool_name,
                                input={"function_name": tool_name, "arguments": arguments},
                                output=None,  # Will be filled by next function_call_output
                                status=ActionStatus.PROCESSING,
                                start_time=datetime.fromisoformat(created_at) if created_at else datetime.now(),
                            )
                            current_actions.append(action)
                            continue

                        # Handle function outputs (tool results)
                        if msg_type == "function_call_output":
                            # Create a new SUCCESS action for the tool output
                            if current_actions:
                                last_action = current_actions[-1]

                                # Extract output directly from message_json
                                output_text = message_json.get("output", "")

                                # Try to parse as Python literal (the output is stored as string repr of dict)
                                output_data = {}
                                if output_text:
                                    try:
                                        # Try ast.literal_eval first (safer than eval)
                                        output_data = ast.literal_eval(output_text)
                                    except (ValueError, SyntaxError):
                                        # If that fails, try json.loads
                                        try:
                                            output_data = json.loads(output_text)
                                        except json.JSONDecodeError:
                                            # Last resort: store as string
                                            output_data = {"result": output_text}

                                # Create a new SUCCESS action, prefix with "complete_" like openai_compatible.py
                                call_id = message_json.get("call_id", last_action.action_id)
                                success_action = ActionHistory(
                                    action_id="complete_" + call_id,
                                    role=ActionRole.TOOL,
                                    messages=f"Tool result: {last_action.action_type}",
                                    action_type=last_action.action_type,
                                    input=last_action.input,
                                    output=output_data,
                                    status=ActionStatus.SUCCESS,
                                    start_time=last_action.start_time,
                                    end_time=datetime.fromisoformat(created_at) if created_at else datetime.now(),
                                )
                                current_actions.append(success_action)
                            continue

                        # Handle assistant messages (thinking and final output)
                        if role == "assistant":
                            # Assistant message - aggregate consecutive ones
                            content_array = message_json.get("content", [])

                            for item in content_array:
                                if not isinstance(item, dict):
                                    continue

                                item_type = item.get("type", "")
                                text = item.get("text", "")

                                if item_type == "output_text" and text:
                                    # Initialize assistant group if needed
                                    if not current_assistant_group:
                                        current_assistant_group = {
                                            "role": "assistant",
                                            "content": "",
                                            "timestamp": created_at,
                                            "created_at": created_at,
                                        }

                                    # Add to progress
                                    assistant_progress.append(f"💭Thinking: {text}")

                                    # Create ActionHistory for thinking (use response_id from provider)
                                    response_id = message_json.get("provider_data", {}).get(
                                        "response_id", message_json.get("id", str(uuid.uuid4()))
                                    )
                                    thinking_action = ActionHistory(
                                        action_id=response_id,
                                        role=ActionRole.ASSISTANT,
                                        messages=text,
                                        action_type="thinking",
                                        input=None,
                                        output={"raw_output": text},
                                        status=ActionStatus.SUCCESS,
                                        start_time=(
                                            datetime.fromisoformat(created_at) if created_at else datetime.now()
                                        ),
                                        end_time=(datetime.fromisoformat(created_at) if created_at else datetime.now()),
                                    )
                                    current_actions.append(thinking_action)

                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug(f"Skipping malformed message: {e}")
                        continue

                # Flush any remaining assistant group
                if current_assistant_group:
                    final_action = self._parse_final_output(current_actions, current_assistant_group)
                    if final_action:
                        current_actions.append(final_action)

                    if not current_assistant_group.get("content"):
                        current_assistant_group["content"] = "Processing completed"
                    if assistant_progress:
                        current_assistant_group["progress_messages"] = assistant_progress
                    if current_actions:
                        current_assistant_group["actions"] = current_actions.copy()
                    messages.append(current_assistant_group)

        except Exception as e:
            logger.exception(f"Failed to load session messages for {session_id}: {e}")

        return messages

    def close_all_sessions(self) -> None:
        """Close all active sessions."""
        for session_id in list(self._sessions.keys()):
            self._sessions.pop(session_id)
            # SQLiteSession doesn't have an explicit close method,
            # but removing it from our dict should handle cleanup
            logger.debug(f"Closed session: {session_id}")
