import asyncio
import json
import uuid
from datetime import date, datetime
from typing import Any, AsyncGenerator, Dict, Optional

from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
from pydantic import AnyUrl

from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.models.mcp_utils import multiple_mcp_servers
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import extract_json_str
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def classify_api_error(error: Exception) -> tuple[ErrorCode, bool]:
    """Classify API errors and return error code and whether it's retryable."""
    error_msg = str(error).lower()

    if isinstance(error, APIError):
        # Handle specific HTTP status codes and error types
        if any(indicator in error_msg for indicator in ["overloaded", "529"]):
            return ErrorCode.MODEL_OVERLOADED, True
        elif any(indicator in error_msg for indicator in ["rate limit", "429"]):
            return ErrorCode.MODEL_RATE_LIMIT, True
        elif any(indicator in error_msg for indicator in ["401", "unauthorized", "authentication"]):
            return ErrorCode.MODEL_AUTHENTICATION_ERROR, False
        elif any(indicator in error_msg for indicator in ["403", "forbidden", "permission"]):
            return ErrorCode.MODEL_PERMISSION_ERROR, False
        elif any(indicator in error_msg for indicator in ["404", "not found"]):
            return ErrorCode.MODEL_NOT_FOUND, False
        elif any(indicator in error_msg for indicator in ["413", "too large", "request size"]):
            return ErrorCode.MODEL_REQUEST_TOO_LARGE, False
        elif any(indicator in error_msg for indicator in ["500", "internal", "server error"]):
            return ErrorCode.MODEL_API_ERROR, True
        elif any(indicator in error_msg for indicator in ["400", "bad request", "invalid"]):
            return ErrorCode.MODEL_INVALID_RESPONSE, False

    if isinstance(error, RateLimitError):
        return ErrorCode.MODEL_RATE_LIMIT, True

    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return ErrorCode.MODEL_CONNECTION_ERROR, True

    # Default to general request failure
    return ErrorCode.MODEL_REQUEST_FAILED, False


class JSONParsingMixin:
    """Mixin for JSON parsing functionality."""

    def parse_json_response(self, response_text: str, fallback_method=None) -> Dict:
        """Parse JSON response with fallback handling."""
        try:
            return json.loads(extract_json_str(response_text), strict=False)
        except json.JSONDecodeError:
            if fallback_method:
                try:
                    return fallback_method(response_text)
                except Exception:
                    pass

            # Generic fallback
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Failed to parse JSON response: {response_text}")
            return {
                "error": "Failed to parse JSON response",
                "raw_response": response_text,
            }


class MCPMixin:
    """Mixin for MCP (Model Context Protocol) functionality."""

    def setup_json_encoder(self):
        """Setup custom JSON encoder for special types."""

        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        # Note: This is a workaround for custom JSON encoding
        # In production, consider using a more robust approach
        if hasattr(json, "_default_encoder"):
            json._default_encoder = CustomJSONEncoder()

    async def generate_with_mcp_base(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: Any,
        max_turns: int = 10,
        async_model_factory=None,
        **kwargs,
    ) -> Dict:
        """Base MCP generation method."""
        self.setup_json_encoder()

        try:
            async with multiple_mcp_servers(mcp_servers) as connected_servers:
                logger.debug("MCP servers started successfully")

                if async_model_factory:
                    async_model = async_model_factory(**kwargs)
                else:
                    raise ValueError("async_model_factory is required")

                agent = Agent(
                    name=kwargs.pop("agent_name", "MCP_Agent"),
                    instructions=instruction,
                    mcp_servers=list(connected_servers.values()),
                    output_type=output_type,
                    model=async_model,
                )
                logger.debug(f"Agent created with name: {agent.name}")

                result = await Runner.run(agent, input=prompt, max_turns=max_turns)
                logger.debug("Agent execution completed")

                return {
                    "content": result.final_output,
                    "sql_contexts": extract_sql_contexts(result),
                }
        except Exception as e:
            logger.error(f"Error in MCP execution: {str(e)}")
            raise


class StreamingMixin:
    """Mixin for streaming functionality."""

    async def generate_with_mcp_stream_base(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        action_history_manager: Optional[ActionHistoryManager] = None,
        async_model_factory=None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Base streaming MCP generation method."""
        if action_history_manager is None:
            action_history_manager = ActionHistoryManager()

        # Setup JSON encoder
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        # Note: This is a workaround for custom JSON encoding
        # In production, consider using a more robust approach
        if hasattr(json, "_default_encoder"):
            json._default_encoder = CustomJSONEncoder()

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                async with multiple_mcp_servers(mcp_servers) as connected_servers:
                    if async_model_factory:
                        async_model = async_model_factory(**kwargs)
                    else:
                        raise ValueError("async_model_factory is required")

                    agent = Agent(
                        name=kwargs.pop("agent_name", "MCP_Agent"),
                        instructions=instruction,
                        mcp_servers=list(connected_servers.values()),
                        output_type=str,  # Force to str for compatibility
                        model=async_model,
                    )

                    result = Runner.run_streamed(agent, input=prompt, max_turns=max_turns)

                    while not result.is_complete:
                        async for event in result.stream_events():
                            if not hasattr(event, "type") or event.type != "run_item_stream_event":
                                continue

                            if not (hasattr(event, "item") and hasattr(event.item, "type")):
                                continue

                            action = None
                            item_type = event.item.type

                            if item_type == "tool_call_item":
                                action = self._process_tool_call_start(event, action_history_manager)
                            elif item_type == "tool_call_output_item":
                                action = self._process_tool_call_complete(event, action_history_manager)
                            elif item_type == "message_output_item":
                                action = self._process_message_output(event, action_history_manager)

                            if action:
                                yield action

                    # If we reach here, streaming completed successfully
                    break

            except (APIError, RateLimitError, APIConnectionError, APITimeoutError) as e:
                error_code, is_retryable = classify_api_error(e)

                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"API error (attempt {attempt + 1}/{max_retries + 1}): {error_code.code} - "
                        f"{error_code.desc}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"API error after {attempt + 1} attempts: {error_code.code} - {error_code.desc}")
                    raise DatusException(error_code)

            except Exception as e:
                logger.error(f"Error in streaming MCP execution: {str(e)}")
                raise

    def _process_tool_call_start(self, event, action_history_manager: ActionHistoryManager) -> Optional[ActionHistory]:
        """Process tool_call_item events."""
        raw_item = event.item.raw_item
        call_id = getattr(raw_item, "call_id", None)
        function_name = getattr(raw_item, "name", None)
        arguments = getattr(raw_item, "arguments", None)

        # Check if action with this call_id already exists
        if call_id and action_history_manager.find_action_by_id(call_id):
            return None

        action = ActionHistory(
            action_id=str(call_id or uuid.uuid4()),
            role=ActionRole.TOOL,
            messages="MCP call",
            action_type=function_name or "unknown",
            input={"function_name": function_name, "arguments": arguments, "call_id": call_id},
            status=ActionStatus.PROCESSING,
        )
        action_history_manager.add_action(action)
        return action

    def _process_tool_call_complete(
        self, event, action_history_manager: ActionHistoryManager
    ) -> Optional[ActionHistory]:
        """Process tool_call_output_item events."""
        call_id = getattr(event.item.raw_item, "call_id", None)
        matching_action = action_history_manager.find_action_by_id(call_id) if call_id else None

        if not matching_action:
            # Try to match by the most recent PROCESSING action as fallback
            processing_actions = [a for a in action_history_manager.actions if a.status == ActionStatus.PROCESSING]
            if processing_actions:
                matching_action = processing_actions[-1]  # Get the most recent
            else:
                return None

        output_data = {
            "call_id": call_id,
            "success": True,
            "raw_output": event.item.output,
        }

        action_history_manager.update_action_by_id(
            matching_action.action_id, output=output_data, end_time=datetime.now(), status=ActionStatus.SUCCESS
        )

        # Don't return the action to avoid duplicate yield
        return None

    def _process_message_output(self, event, action_history_manager: ActionHistoryManager) -> Optional[ActionHistory]:
        """Process message_output_item events."""
        if not (hasattr(event.item, "raw_item") and hasattr(event.item.raw_item, "content")):
            return None

        content = event.item.raw_item.content
        if not content:
            return None

        logger.debug(f"Processing message output: {content}")
        # Extract text content
        if isinstance(content, list) and content:
            text_content = content[0].text if hasattr(content[0], "text") else str(content[0])
        else:
            text_content = str(content)

        # Create action with raw content
        if len(text_content) > 0:
            action = ActionHistory(
                action_id=str(uuid.uuid4()),
                role=ActionRole.ASSISTANT,
                messages=f"Thinking: {text_content}",
                action_type="message",
                input={},
                output={
                    "success": True,
                    "raw_output": text_content,
                },
                status=ActionStatus.SUCCESS,
            )
            action.end_time = datetime.now()
            action_history_manager.add_action(action)
        else:
            action = None
            logger.debug(f"No text content found in message output: {content}")
        return action
