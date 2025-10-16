"""
Session loading and management for web interface.

Handles loading chat sessions from SQLite database, including:
- Message aggregation
- Progress tracking reconstruction
- Tool call parsing
"""

import json
import os
import sqlite3
from typing import Dict, List, Optional

import structlog

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
        db_path = os.path.join(os.path.expanduser("~/.datus/sessions"), f"{session_id}.db")

        if not os.path.exists(db_path):
            logger.warning(f"Session database not found: {db_path}")
            return messages

        try:
            with sqlite3.connect(db_path) as conn:
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
                                messages.append(current_assistant_group)
                                current_assistant_group = None
                                assistant_progress = []

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

                            # Add tool call to progress
                            try:
                                args_preview = json.loads(arguments) if arguments else {}
                                args_str = str(args_preview)[:60]
                                assistant_progress.append(f"✓ Tool call: {tool_name}({args_str})")
                            except (json.JSONDecodeError, ValueError, TypeError):
                                assistant_progress.append(f"✓ Tool call: {tool_name}")
                            continue

                        # Handle function outputs (tool results)
                        if msg_type == "function_call_output":
                            # These are captured but not displayed separately
                            # The output is implied by the next thinking message
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

                                                messages.append(current_assistant_group)
                                                current_assistant_group = None
                                                assistant_progress = []
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

                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug(f"Skipping malformed message: {e}")
                        continue

                # Flush any remaining assistant group
                if current_assistant_group:
                    if not current_assistant_group.get("content"):
                        current_assistant_group["content"] = "Processing completed"
                    if assistant_progress:
                        current_assistant_group["progress_messages"] = assistant_progress
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
