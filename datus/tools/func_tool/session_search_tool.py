# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Session search tool for the gen_skill subagent.

Searches historical session databases for skill usage records,
tool call patterns, and errors to support skill optimization.
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

from datus.tools.func_tool.base import FuncToolResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SessionSearchTool:
    """Tool that searches session history for skill usage patterns.

    Scans session SQLite databases in the sessions directory to find
    sessions where a specific skill was loaded, and extracts tool call
    records for analysis.
    """

    def __init__(self, sessions_dir: Optional[str] = None):
        self.sessions_dir = sessions_dir

    def _resolve_sessions_dir(self) -> str:
        if self.sessions_dir:
            return os.path.expanduser(self.sessions_dir)
        # Default: ~/.datus/sessions/
        return os.path.expanduser("~/.datus/sessions")

    def search_skill_usage(self, skill_name: str, max_sessions: int = 10) -> FuncToolResult:
        """Search session history for usage of a specific skill.

        Finds sessions where the given skill was loaded via load_skill,
        and returns tool call patterns, errors, and summaries.

        Args:
            skill_name: Name of the skill to search for (e.g., "bitcoin-analysis")
            max_sessions: Maximum number of matching sessions to return (default 10)

        Returns:
            FuncToolResult with matching sessions and their tool call records.
        """
        sessions_dir = self._resolve_sessions_dir()
        if not os.path.isdir(sessions_dir):
            return FuncToolResult(success=0, error=f"Sessions directory not found: {sessions_dir}")

        db_files = sorted(Path(sessions_dir).glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not db_files:
            return FuncToolResult(result={"matches": [], "total_sessions_scanned": 0})

        matches = []
        scanned = 0

        for db_path in db_files:
            if len(matches) >= max_sessions:
                break
            scanned += 1
            try:
                match = self._search_single_session(str(db_path), skill_name)
                if match:
                    matches.append(match)
            except Exception as e:
                logger.debug(f"Error scanning session {db_path.name}: {e}")
                continue

        return FuncToolResult(
            result={
                "skill_name": skill_name,
                "matches": matches,
                "total_sessions_scanned": scanned,
                "total_matches": len(matches),
            }
        )

    @staticmethod
    def _escape_like(value: str) -> str:
        """Escape LIKE metacharacters to prevent unintended pattern matching."""
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def _search_single_session(self, db_path: str, skill_name: str) -> Optional[dict]:
        """Search a single session DB for skill usage."""
        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=2)
            cursor = conn.cursor()

            # Search for messages containing the skill name (escape LIKE metacharacters)
            escaped_name = self._escape_like(skill_name)
            cursor.execute(
                "SELECT id, message_data FROM agent_messages WHERE message_data LIKE ? ESCAPE '\\' ORDER BY id LIMIT 500",
                (f"%{escaped_name}%",),
            )
            rows = cursor.fetchall()
            if not rows:
                return None

            # Extract tool calls and errors from matching messages
            tool_calls = []
            errors = []
            skill_loaded = False
            total_messages = 0

            # Stream messages with cursor instead of loading all into memory
            cursor.execute("SELECT id, message_data FROM agent_messages ORDER BY id LIMIT 1000")
            for msg_id, msg_data in cursor:
                total_messages += 1
                try:
                    msg = json.loads(msg_data)
                except (json.JSONDecodeError, TypeError):
                    continue

                msg_type = msg.get("type", "")

                # Detect skill loading
                if "load_skill" in str(msg_data) and skill_name in str(msg_data):
                    skill_loaded = True

                # Collect tool calls (function_call type)
                if msg_type == "function_call" or "arguments" in msg:
                    name = msg.get("name", "")
                    tool_calls.append(
                        {
                            "id": msg_id,
                            "tool": name,
                            "args_preview": str(msg.get("arguments", ""))[:200],
                        }
                    )

                # Collect tool results with actual errors — parse output dict first
                if msg_type == "function_call_output":
                    output_raw = msg.get("output", "")
                    is_error = False
                    if isinstance(output_raw, dict):
                        is_error = output_raw.get("success") == 0 or (
                            output_raw.get("error") is not None and output_raw.get("error") != "None"
                        )
                    else:
                        output_str = str(output_raw)
                        is_error = "'success': 0" in output_str or (
                            "'error':" in output_str and "'error': None" not in output_str
                        )
                    if is_error:
                        errors.append(
                            {
                                "id": msg_id,
                                "output_preview": str(output_raw)[:300],
                            }
                        )

            if not skill_loaded:
                return None

            # Get session metadata
            cursor.execute("SELECT session_id, created_at FROM agent_sessions LIMIT 1")
            session_row = cursor.fetchone()
            session_id = session_row[0] if session_row else Path(db_path).stem
            created_at = session_row[1] if session_row else ""

            return {
                "session_id": session_id,
                "db_file": Path(db_path).name,
                "created_at": created_at,
                "total_messages": total_messages,
                "tool_calls": tool_calls,
                "errors": errors,
                "tool_call_count": len(tool_calls),
                "error_count": len(errors),
                "tools_used": sorted(set(tc["tool"] for tc in tool_calls if tc["tool"])),
            }
        except sqlite3.Error as e:
            logger.debug(f"SQLite error reading {db_path}: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def available_tools(self):
        """Return the search_skill_usage FunctionTool."""
        from datus.tools.func_tool.base import trans_to_function_tool

        return [trans_to_function_tool(self.search_skill_usage)]
