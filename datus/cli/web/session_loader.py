# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Session loading and management for web interface.

Handles loading chat sessions from SQLite database, including:
- Message aggregation
- Progress tracking reconstruction
- Tool call parsing
"""

import json
import re
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import structlog

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

logger = structlog.get_logger(__name__)


class SessionLoader:
    """Loads and reconstructs chat sessions from SQLite storage."""

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
        # Only allow alphanumeric, underscore, hyphen, and dot
        if not re.match(r"^[A-Za-z0-9_.-]+$", session_id):
            logger.warning(f"Invalid session_id format (potential path traversal): {session_id}")
            return messages

        # Build path with pathlib and resolve to absolute path
        from datus.utils.path_manager import get_path_manager

        sessions_dir = get_path_manager().sessions_dir
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
                    ORDER BY created_at
                    """,
                    (session_id,),
                )

                # Aggregate consecutive assistant messages
                current_assistant_group = None
                assistant_progress = []
                current_actions = []  # Collect ActionHistory objects for detailed view
                last_timestamp = None

                for message_data, created_at in cursor.fetchall():
                    try:
                        message_json = json.loads(message_data)
                        role = message_json.get("role", "")
                        msg_type = message_json.get("type", "")

                        # Handle user messages
                        if role == "user":
                            # Before adding user message, flush any pending assistant group
                            if current_assistant_group:
                                # Add collected actions to the assistant group
                                if current_actions:
                                    current_assistant_group["actions"] = current_actions.copy()
                                messages.append(current_assistant_group)
                                current_assistant_group = None
                                assistant_progress = []
                                current_actions = []

                            # Add user message
                            messages.append(
                                {"role": "user", "content": message_json.get("content", ""), "timestamp": created_at}
                            )
                            continue

                        # Handle function calls (tool calls)
                        if msg_type == "function_call":
                            tool_name = message_json.get("name", "unknown")
                            arguments = message_json.get("arguments", "{}")

                            # Initialize assistant group if needed
                            if not current_assistant_group:
                                current_assistant_group = {"role": "assistant", "content": "", "timestamp": created_at}
                                last_timestamp = created_at

                            # Parse arguments
                            try:
                                args_dict = json.loads(arguments) if arguments else {}
                                args_str = str(args_dict)[:60]
                                assistant_progress.append(f"✓ Tool call: {tool_name}({args_str})")
                            except (json.JSONDecodeError, ValueError, TypeError):
                                args_dict = {}
                                assistant_progress.append(f"✓ Tool call: {tool_name}")

                            # Create ActionHistory for tool call
                            action = ActionHistory(
                                action_id=str(uuid.uuid4()),
                                role=ActionRole.TOOL,
                                messages=f"Tool call: {tool_name}",
                                action_type=tool_name,
                                input={"function_name": tool_name, **args_dict},
                                output=None,  # Will be filled by next function_call_output
                                status=ActionStatus.PROCESSING,
                                start_time=datetime.fromisoformat(created_at) if created_at else datetime.now(),
                            )
                            current_actions.append(action)
                            continue

                        # Handle function outputs (tool results)
                        if msg_type == "function_call_output":
                            # Update the last action with output
                            if current_actions:
                                last_action = current_actions[-1]

                                # Extract output directly from message_json
                                output_text = message_json.get("output", "")

                                # Try to parse as Python literal (the output is stored as string repr of dict)
                                output_data = {}
                                if output_text:
                                    try:
                                        # Try ast.literal_eval first (safer than eval)
                                        import ast

                                        output_data = ast.literal_eval(output_text)
                                    except (ValueError, SyntaxError):
                                        # If that fails, try json.loads
                                        try:
                                            output_data = json.loads(output_text)
                                        except json.JSONDecodeError:
                                            # Last resort: store as string
                                            output_data = {"result": output_text}

                                last_action.output = output_data
                                last_action.status = ActionStatus.SUCCESS
                                last_action.end_time = (
                                    datetime.fromisoformat(created_at) if created_at else datetime.now()
                                )
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
                                    # Check if this is the final SQL output
                                    if text.strip().startswith("{"):
                                        try:
                                            output_json = json.loads(text)
                                            if "sql" in output_json and "output" in output_json:
                                                # This is the final output - finalize the group
                                                if not current_assistant_group:
                                                    current_assistant_group = {
                                                        "role": "assistant",
                                                        "content": "",
                                                        "timestamp": created_at,
                                                    }

                                                current_assistant_group["content"] = output_json["output"]
                                                current_assistant_group["sql"] = output_json["sql"]
                                                current_assistant_group["progress_messages"] = assistant_progress.copy()
                                                current_assistant_group["timestamp"] = last_timestamp or created_at

                                                # Add collected actions
                                                if current_actions:
                                                    current_assistant_group["actions"] = current_actions.copy()

                                                messages.append(current_assistant_group)
                                                current_assistant_group = None
                                                assistant_progress = []
                                                current_actions = []
                                                continue
                                        except json.JSONDecodeError:
                                            pass

                                    # This is a thinking/progress message
                                    if not current_assistant_group:
                                        current_assistant_group = {
                                            "role": "assistant",
                                            "content": "",
                                            "timestamp": created_at,
                                        }
                                        last_timestamp = created_at

                                    # Add to progress
                                    assistant_progress.append(f"💭Thinking: {text}")

                                    # Create ActionHistory for thinking
                                    thinking_action = ActionHistory(
                                        action_id=str(uuid.uuid4()),
                                        role=ActionRole.ASSISTANT,
                                        messages=text,
                                        action_type="thinking",
                                        input=None,
                                        output=None,
                                        status=ActionStatus.SUCCESS,
                                        start_time=datetime.fromisoformat(created_at) if created_at else datetime.now(),
                                        end_time=datetime.fromisoformat(created_at) if created_at else datetime.now(),
                                    )
                                    current_actions.append(thinking_action)

                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug(f"Skipping malformed message: {e}")
                        continue

                # Flush any remaining assistant group
                if current_assistant_group:
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

    def get_current_session_id(self, cli) -> Optional[str]:
        """
        Get the current session ID from the active chat node.

        Args:
            cli: DatusCLI instance

        Returns:
            Session ID if available, None otherwise
        """
        if cli and cli.chat_commands:
            # Prefer current_node over chat_node (for subagent support)
            node = cli.chat_commands.current_node or cli.chat_commands.chat_node
            if node:
                return node.session_id
        return None
