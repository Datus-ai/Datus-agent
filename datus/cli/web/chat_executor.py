# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Chat execution and streaming for web interface.

Handles:
- Streaming chat execution
- Action formatting for display
- SQL and response extraction
"""

import asyncio
from typing import List, Optional, Tuple

import structlog

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

logger = structlog.get_logger(__name__)


class ChatExecutor:
    """Executes chat commands with streaming support."""

    def __init__(self):
        """Initialize ChatExecutor."""
        self.last_actions = []  # Store last execution's actions

    def execute_chat_stream(self, user_message: str, cli, current_subagent: Optional[str] = None):
        """Execute chat command with streaming support - reuses chat_commands logic."""
        if not cli or not cli.chat_commands:
            yield "Error: Please load configuration first!"
            return

        try:
            # Get node and input using chat_commands logic
            at_tables, at_metrics, at_sqls = cli.at_completer.parse_at_context(user_message)

            # Reuse chat_commands node management
            need_new_node = cli.chat_commands._should_create_new_node(current_subagent)

            # Disable compact in web mode to avoid blocking
            if need_new_node:
                current_node = cli.chat_commands._create_new_node(current_subagent)
                cli.chat_commands.current_node = current_node
                cli.chat_commands.current_subagent_name = current_subagent if current_subagent else None
                if not current_subagent:
                    cli.chat_commands.chat_node = current_node
            else:
                current_node = cli.chat_commands.current_node

            # Create input using shared method from chat_commands
            node_input, _ = cli.chat_commands.create_node_input(
                user_message, current_node, at_tables, at_metrics, at_sqls, plan_mode=False
            )

            # Stream execution with deduplication
            incremental_actions = []
            seen_thinking_content = set()  # Track unique thinking content (without prefix)
            last_message = None  # Track last message to avoid consecutive duplicates

            async def collect_actions():
                """Collect all actions from the stream"""
                nonlocal last_message

                current_node.input = node_input
                async for action in current_node.execute_stream(cli.actions):
                    incremental_actions.append(action)
                    formatted = self.format_action_for_stream(action)

                    # Skip empty messages
                    if not formatted:
                        continue

                    # Deduplicate: skip if same as last message
                    if formatted == last_message:
                        continue

                    # For thinking messages, check content without emoji and prefix
                    if formatted.startswith("💭Thinking:"):
                        # Extract actual content without "💭Thinking: "
                        thinking_content = formatted[11:].strip()  # Remove "💭Thinking: "

                        # Skip if we've seen this exact thinking content before
                        if thinking_content in seen_thinking_content:
                            continue

                        seen_thinking_content.add(thinking_content)

                    last_message = formatted
                    yield formatted

            # Execute async generator with proper event loop handling
            loop = asyncio.new_event_loop()

            try:
                current_node.input = node_input

                async def run_stream():
                    """Wrapper to iterate the async generator to completion"""
                    try:
                        async for action in current_node.execute_stream(cli.actions):
                            incremental_actions.append(action)
                            yield action
                        self.last_actions = incremental_actions
                    except StopAsyncIteration:
                        pass

                async_gen = run_stream()
                while True:
                    try:
                        result = loop.run_until_complete(async_gen.__anext__())
                        yield result
                    except StopAsyncIteration:
                        break

                # Store collected actions for caller to access
            finally:
                loop.close()

        except Exception as e:
            logger.exception(f"Execution error: {e}")
            yield f"Error: {str(e)}"

    def format_action_for_stream(self, action: ActionHistory) -> str:
        """Format ActionHistory for streaming display with details."""
        if action.role == ActionRole.TOOL:
            function_name = action.function_name() or "unknown"

            # Extract input parameters for display
            input_preview = ""
            if action.input and isinstance(action.input, dict):
                # Show key parameters (limit to first 100 chars)
                params = {k: v for k, v in action.input.items() if k != "function_name"}
                if params:
                    param_str = str(params)[:100]
                    if len(str(params)) > 100:
                        param_str += "..."
                    input_preview = f" ({param_str})"

            if action.status == ActionStatus.SUCCESS:
                # Show output preview for successful tools
                output_preview = ""
                if action.output:
                    if isinstance(action.output, dict):
                        # Show first key-value or length
                        if "result" in action.output:
                            result_str = str(action.output["result"])[:80]
                            output_preview = (
                                f" → {result_str}..." if len(str(action.output["result"])) > 80 else f" → {result_str}"
                            )
                        elif len(action.output) > 0:
                            first_key = list(action.output.keys())[0]
                            output_preview = f" → {first_key}: ..."
                    else:
                        output_str = str(action.output)[:80]
                        output_preview = f" → {output_str}..." if len(str(action.output)) > 80 else f" → {output_str}"

                return f"✓Tool call: {function_name}{input_preview}{output_preview}"
            elif action.status == ActionStatus.PROCESSING:
                return f"⟳Tool call: {function_name}{input_preview}..."
            else:
                return f"✗Tool call: {function_name}{input_preview}"
        elif action.role == ActionRole.ASSISTANT and action.messages:
            # Show LLM thinking with brief preview
            message = action.messages.strip()

            # Remove "Thinking: " prefix added by openai_compatible.py
            if message.startswith("Thinking: "):
                message = message[10:].strip()  # Remove "Thinking: "

            # Skip empty or very generic messages
            if not message or message.lower() in ["thinking...", "processing...", ""]:
                return ""

            # Truncate long messages
            if len(message) > 100:
                message = message[:100] + "..."

            return f"💭Thinking: {message}"
        return ""

    def extract_sql_and_response(self, actions: List[ActionHistory], cli) -> Tuple[Optional[str], Optional[str]]:
        """Extract SQL and clean response from actions using existing logic."""
        if not actions:
            return None, None

        final_action = actions[-1]
        if not (
            final_action.output
            and isinstance(final_action.output, dict)
            and final_action.status == ActionStatus.SUCCESS
        ):
            return None, None

        sql = final_action.output.get("sql")
        response = final_action.output.get("response")

        # Handle None response
        if response is None:
            return sql, None

        # Handle dict response (already parsed)
        if isinstance(response, dict):
            return sql, response.get("raw_output", str(response))

        # Only process string responses
        if not isinstance(response, str):
            return sql, str(response)

        # Extract SQL and output using ChatCommands
        extracted_sql, extracted_output = None, response
        if cli and cli.chat_commands:
            extracted_sql, extracted_output = cli.chat_commands._extract_sql_and_output_from_content(response)
            sql = sql or extracted_sql

        # Determine clean output
        if sql:
            return sql, extracted_output or response

        if isinstance(extracted_output, dict):
            return None, extracted_output.get("raw_output", str(extracted_output))

        # Try to parse response as Python literal (only for strings)
        try:
            import ast

            response_dict = ast.literal_eval(response)
            if isinstance(response_dict, dict):
                return None, response_dict.get("raw_output", response)
        except (ValueError, SyntaxError):
            pass

        return None, response
