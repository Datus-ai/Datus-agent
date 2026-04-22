# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Codex model implementation using OAuth authentication and Responses API."""

import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from agents import Agent, ModelSettings, Runner, SQLiteSession, Tool
from agents.exceptions import MaxTurnsExceeded
from agents.mcp import MCPServerStdio

from datus.auth.oauth_config import CODEX_API_BASE_URL
from datus.auth.oauth_manager import OAuthManager
from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.models.mcp_utils import multiple_mcp_servers
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Fallback values when providers.yml / OpenRouter cache do not cover a codex slug.
_CODEX_DEFAULT_CONTEXT_LENGTH = 192000
_CODEX_DEFAULT_MAX_TOKENS = 16384


class CodexModel(LLMBaseModel):
    """Access OpenAI Codex models via OAuth tokens and the Responses API.

    Unlike standard OpenAI models, Codex uses:
    - OAuth token authentication (ChatGPT subscription) instead of API keys
    - The Responses API format (POST /responses) instead of Chat Completions
    - A different base URL (chatgpt.com/backend-api/codex)
    """

    def __init__(self, model_config: ModelConfig, **kwargs):
        super().__init__(model_config)
        self.model_name = model_config.model
        self._base_url = model_config.base_url or CODEX_API_BASE_URL
        self.oauth_manager = OAuthManager()
        self._config_api_key = self._resolve_config_api_key(model_config.api_key)
        self._client = None
        self._async_client = None

    @staticmethod
    def _resolve_config_api_key(api_key: str | None) -> str | None:
        """Return the config api_key if it looks like a real token, else None."""
        if not api_key or not api_key.strip():
            return None
        # Skip unresolved env-var placeholders like ${CODEX_OAUTH_TOKEN}
        if api_key.startswith("${") and api_key.endswith("}"):
            return None
        if api_key.startswith("<MISSING:"):
            return None
        return api_key

    def _get_access_token(self) -> str:
        """Return a valid access token: config api_key first, then OAuthManager."""
        if self._config_api_key:
            return self._config_api_key
        return self.oauth_manager.get_access_token()

    def _get_client(self):
        """Get sync OpenAI client, creating or refreshing token as needed."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._get_access_token(),
                base_url=self._base_url,
            )
        else:
            # Ensure token is current on cached client
            self._client.api_key = self._get_access_token()
        return self._client

    def _get_async_client(self):
        """Get async OpenAI client, creating or refreshing token as needed."""
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(
                api_key=self._get_access_token(),
                base_url=self._base_url,
            )
        else:
            # Ensure token is current on cached client
            self._async_client.api_key = self._get_access_token()
        return self._async_client

    def _get_responses_model(self):
        """Create an OpenAIResponsesModel for use with the Agent SDK."""
        from agents.models.openai_responses import OpenAIResponsesModel

        async_client = self._get_async_client()
        return OpenAIResponsesModel(model=self.model_name, openai_client=async_client)

    def _refresh_client_token(self):
        """Refresh the token on existing clients."""
        token = self._get_access_token()
        if self._client is not None:
            self._client.api_key = token
        if self._async_client is not None:
            self._async_client.api_key = token

    @staticmethod
    def _consume_stream_text(stream) -> str:
        """Consume a streaming response and return the full output text."""
        collected = []
        for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "response.completed":
                response = getattr(event, "response", None)
                if response:
                    return getattr(response, "output_text", "")
            elif event_type == "response.output_text.delta":
                collected.append(getattr(event, "delta", ""))
        return "".join(collected)

    @staticmethod
    def _convert_prompt_to_input(prompt: Any) -> list:
        """Convert prompt to Responses API input format.

        The Codex Responses API requires input as a list of message dicts
        with 'role' and 'content' keys.
        """
        if isinstance(prompt, list):
            return prompt
        # Wrap string/other types into a user message list
        return [{"role": "user", "content": str(prompt)}]

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """Generate a response via the Codex Responses API.

        Args:
            prompt: Input prompt (string or messages list)
            enable_thinking: Not supported for Codex models (ignored)
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        self._refresh_client_token()
        input_data = self._convert_prompt_to_input(prompt)

        create_kwargs = {
            "model": self.model_name,
            "input": input_data,
            "store": False,
            "stream": True,
        }
        instructions = kwargs.get("instructions")
        if instructions:
            create_kwargs["instructions"] = instructions

        try:
            stream = self._get_client().responses.create(**create_kwargs)
            return self._consume_stream_text(stream)
        except Exception as e:
            from openai import AuthenticationError

            if isinstance(e, AuthenticationError):
                logger.info("Got 401, refreshing OAuth token and retrying...")
                self.oauth_manager.refresh_tokens()
                self._refresh_client_token()
                try:
                    stream = self._get_client().responses.create(**create_kwargs)
                    return self._consume_stream_text(stream)
                except AuthenticationError as retry_e:
                    raise DatusException(
                        ErrorCode.MODEL_AUTHENTICATION_ERROR,
                        message_args={"error_detail": f"Codex auth failed after token refresh: {retry_e}"},
                    ) from retry_e
            raise DatusException(
                ErrorCode.MODEL_REQUEST_FAILED,
                message=f"Codex generate failed: {e}",
            ) from e

    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """Generate a JSON-structured response via the Codex Responses API.

        Args:
            prompt: Input prompt (string or messages list)
            **kwargs: May contain 'output_schema' for structured output

        Returns:
            Parsed JSON response as a dictionary
        """
        import json

        self._refresh_client_token()
        input_data = self._convert_prompt_to_input(prompt)

        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "input": input_data,
            "store": False,
            "stream": True,
        }

        instructions = kwargs.get("instructions")
        if instructions:
            create_kwargs["instructions"] = instructions

        output_schema = kwargs.get("output_schema")
        if output_schema:
            create_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "schema": output_schema,
                }
            }
        else:
            create_kwargs["text"] = {"format": {"type": "json_object"}}

        try:
            stream = self._get_client().responses.create(**create_kwargs)
            return json.loads(self._consume_stream_text(stream))
        except json.JSONDecodeError as e:
            raise DatusException(
                ErrorCode.MODEL_INVALID_RESPONSE,
                message_args={"error_detail": f"Invalid JSON from Codex: {e}"},
            ) from e
        except Exception as e:
            from openai import AuthenticationError

            if isinstance(e, AuthenticationError):
                logger.info("Got 401 in generate_with_json_output, refreshing OAuth token and retrying...")
                self.oauth_manager.refresh_tokens()
                self._refresh_client_token()
                try:
                    stream = self._get_client().responses.create(**create_kwargs)
                    return json.loads(self._consume_stream_text(stream))
                except AuthenticationError as retry_e:
                    raise DatusException(
                        ErrorCode.MODEL_AUTHENTICATION_ERROR,
                        message_args={"error_detail": f"Codex auth failed after token refresh: {retry_e}"},
                    ) from retry_e
                except json.JSONDecodeError as json_e:
                    raise DatusException(
                        ErrorCode.MODEL_INVALID_RESPONSE,
                        message_args={"error_detail": f"Invalid JSON from Codex after retry: {json_e}"},
                    ) from json_e
            raise DatusException(
                ErrorCode.MODEL_REQUEST_FAILED,
                message=f"Codex generate_with_json_output failed: {e}",
            ) from e

    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        instruction: str = "",
        output_type: type = str,
        max_turns: int = 10,
        session: Optional[SQLiteSession] = None,
        **kwargs,
    ) -> Dict:
        """Generate response with tool support via the Codex Responses API."""
        self._refresh_client_token()
        responses_model = self._get_responses_model()

        async with multiple_mcp_servers(mcp_servers or {}) as connected_servers:
            agent_kwargs: Dict[str, Any] = {
                "name": kwargs.get("agent_name", "codex_agent"),
                "instructions": instruction,
                "output_type": output_type,
                "model": responses_model,
                "model_settings": ModelSettings(store=False, include_usage=True),
            }
            if connected_servers:
                agent_kwargs["mcp_servers"] = list(connected_servers.values())
            if tools:
                agent_kwargs["tools"] = tools
            if kwargs.get("hooks"):
                agent_kwargs["hooks"] = kwargs["hooks"]

            agent = Agent(**agent_kwargs)

            async def _run_streamed_to_completion(a, p):
                """Run agent via streaming (Codex API requires stream=True) and return the result."""
                result = Runner.run_streamed(a, input=p, max_turns=max_turns, session=session)
                while not result.is_complete:
                    async for _ in result.stream_events():
                        pass
                return result

            try:
                result = await _run_streamed_to_completion(agent, prompt)
            except MaxTurnsExceeded as e:
                raise DatusException(ErrorCode.MODEL_MAX_TURNS_EXCEEDED, message_args={"max_turns": max_turns}) from e
            except Exception as e:
                from openai import AuthenticationError

                if isinstance(e, AuthenticationError):
                    logger.info("Got 401 in generate_with_tools, refreshing OAuth token...")
                    self.oauth_manager.refresh_tokens()
                    self._refresh_client_token()
                    # Don't retry the full run — tool calls may have had side effects.
                    # Raise a retriable error so the caller can decide.
                    raise DatusException(
                        ErrorCode.MODEL_AUTHENTICATION_ERROR,
                        message_args={"error_detail": "Codex auth failed; token refreshed, please retry"},
                    ) from e
                raise

            if session and hasattr(session, "store_run_usage"):
                try:
                    await session.store_run_usage(result)
                except Exception as e:
                    logger.warning(f"Failed to store run usage: {e}")

            usage_info = self._extract_usage_info(result)

            return {
                "content": result.final_output,
                "sql_contexts": extract_sql_contexts(result),
                "usage": usage_info,
                "model": self.model_name,
                "turns_used": getattr(result, "turn_count", 0),
            }

    async def generate_with_tools_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        instruction: str = "",
        output_type: type = str,
        max_turns: int = 10,
        session: Optional[SQLiteSession] = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        hooks=None,
        interrupt_controller=None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate response with streaming and tool support via the Codex Responses API."""
        if action_history_manager is None:
            action_history_manager = ActionHistoryManager()

        self._refresh_client_token()
        responses_model = self._get_responses_model()

        async with multiple_mcp_servers(mcp_servers or {}) as connected_servers:
            agent_kwargs: Dict[str, Any] = {
                "name": kwargs.get("agent_name", "codex_agent"),
                "instructions": instruction,
                "output_type": output_type,
                "model": responses_model,
                "model_settings": ModelSettings(store=False, include_usage=True),
            }
            if connected_servers:
                agent_kwargs["mcp_servers"] = list(connected_servers.values())
            if tools:
                agent_kwargs["tools"] = tools
            if hooks:
                agent_kwargs["hooks"] = hooks

            agent = Agent(**agent_kwargs)

            result = Runner.run_streamed(agent, input=prompt, max_turns=max_turns, session=session)

            # Stream events and yield ActionHistory objects
            temp_tool_calls = {}  # {call_id: {"tool_name": ..., "arguments": ...}}
            early_assistant_yielded = False
            thinking_stream_id: Optional[str] = None
            thinking_accumulated = ""

            try:
                while not result.is_complete:
                    if interrupt_controller and interrupt_controller.is_interrupted:
                        from datus.cli.execution_state import ExecutionInterrupted

                        raise ExecutionInterrupted("Interrupted by user")

                    async for event in result.stream_events():
                        if interrupt_controller and interrupt_controller.is_interrupted:
                            from datus.cli.execution_state import ExecutionInterrupted

                            raise ExecutionInterrupted("Interrupted by user")

                        # Handle assistant text from raw response events (streaming)
                        if hasattr(event, "type") and event.type == "raw_response_event":
                            raw_data = getattr(event, "data", None)
                            raw_type = getattr(raw_data, "type", None) if raw_data else None

                            # Stream text delta
                            if raw_type == "response.output_text.delta":
                                delta_text = getattr(raw_data, "delta", None)
                                if delta_text:
                                    if thinking_stream_id is None:
                                        thinking_stream_id = f"thinking_stream_{uuid.uuid4().hex[:8]}"
                                    thinking_accumulated += delta_text
                                    delta_action = ActionHistory(
                                        action_id=thinking_stream_id,
                                        role=ActionRole.ASSISTANT,
                                        messages="",
                                        action_type="thinking_delta",
                                        input={},
                                        output={"delta": delta_text, "accumulated": thinking_accumulated},
                                        status=ActionStatus.PROCESSING,
                                    )
                                    yield delta_action
                                continue

                            # Content part done: emit completed thinking action
                            if raw_type == "response.content_part.done":
                                if thinking_accumulated.strip():
                                    full_text = thinking_accumulated.strip()
                                    text_split = full_text if len(full_text) <= 200 else f"{full_text[:200]}..."
                                    action = ActionHistory(
                                        action_id=thinking_stream_id or f"assistant_{uuid.uuid4().hex[:8]}",
                                        role=ActionRole.ASSISTANT,
                                        messages=f"Thinking: {text_split}",
                                        action_type="response",
                                        input={},
                                        output={"raw_output": full_text, "is_thinking": len(temp_tool_calls) > 0},
                                        status=ActionStatus.SUCCESS,
                                    )
                                    action_history_manager.add_action(action)
                                    yield action
                                    early_assistant_yielded = True
                                thinking_stream_id = None
                                thinking_accumulated = ""
                                continue

                            # Fallback: response.output_item.done type="message"
                            if raw_type == "response.output_item.done":
                                raw_item = getattr(raw_data, "item", None)
                                if raw_item and getattr(raw_item, "type", None) == "message":
                                    if not early_assistant_yielded:
                                        content_list = getattr(raw_item, "content", [])
                                        text_parts = [
                                            getattr(p, "text", "")
                                            for p in (content_list or [])
                                            if getattr(p, "text", None)
                                        ]
                                        full_text = "\n".join(text_parts).strip()
                                        if full_text:
                                            action = ActionHistory(
                                                action_id=f"assistant_{uuid.uuid4().hex[:8]}",
                                                role=ActionRole.ASSISTANT,
                                                messages=full_text[:200] + ("..." if len(full_text) > 200 else ""),
                                                action_type="response",
                                                input={},
                                                output={"raw_output": full_text},
                                                status=ActionStatus.SUCCESS,
                                            )
                                            action_history_manager.add_action(action)
                                            yield action
                                            early_assistant_yielded = True
                            continue

                        if not hasattr(event, "type") or event.type != "run_item_stream_event":
                            continue
                        if not (hasattr(event, "item") and hasattr(event.item, "type")):
                            continue

                        item_type = event.item.type

                        # Handle tool call events
                        if item_type == "tool_call_item":
                            raw_item = getattr(event.item, "raw_item", None)
                            if raw_item:
                                # Normalize access: raw_item can be dict (Responses API) or object
                                if isinstance(raw_item, dict):
                                    tool_name = raw_item.get("name", "unknown")
                                    call_id = raw_item.get("call_id")
                                    arguments = raw_item.get("arguments", "{}")
                                else:
                                    tool_name = getattr(raw_item, "name", "unknown")
                                    call_id = getattr(raw_item, "call_id", None)
                                    arguments = getattr(raw_item, "arguments", "{}")
                                if not call_id:
                                    call_id = f"tool_{uuid.uuid4().hex[:8]}"
                                args_str = str(arguments)[:80]

                                temp_tool_calls[call_id] = {
                                    "tool_name": tool_name,
                                    "arguments": arguments,
                                    "args_display": args_str,
                                }

                                action = ActionHistory(
                                    action_id=call_id,
                                    role=ActionRole.TOOL,
                                    messages=f"Tool call: {tool_name}('{args_str}...')",
                                    action_type=tool_name,
                                    input={"function_name": tool_name, "arguments": arguments},
                                    output={},
                                    status=ActionStatus.PROCESSING,
                                )
                                action_history_manager.add_action(action)
                                yield action

                        elif item_type == "tool_call_output_item":
                            raw_item = getattr(event.item, "raw_item", None)
                            output_content = getattr(event.item, "output", "")
                            if raw_item:
                                # raw_item can be a dict (Responses API) or Pydantic model (Chat Completions)
                                if isinstance(raw_item, dict):
                                    call_id = raw_item.get("call_id")
                                else:
                                    call_id = getattr(raw_item, "call_id", None)
                                if not call_id:
                                    call_id = f"tool_{uuid.uuid4().hex[:8]}"

                                # Match back to stored tool call for name
                                tool_info = temp_tool_calls.pop(call_id, None)
                                tool_name = tool_info["tool_name"] if tool_info else "unknown"
                                args_display = tool_info["args_display"] if tool_info else ""
                                arguments = tool_info["arguments"] if tool_info else "{}"

                                action = ActionHistory(
                                    action_id=f"complete_{call_id}",
                                    role=ActionRole.TOOL,
                                    messages=f"Tool call: {tool_name}('{args_display}...')",
                                    action_type=tool_name,
                                    input={"function_name": tool_name, "arguments": arguments},
                                    output={"success": True, "raw_output": output_content},
                                    status=ActionStatus.SUCCESS,
                                )
                                action_history_manager.add_action(action)
                                yield action
            except MaxTurnsExceeded as e:
                raise DatusException(ErrorCode.MODEL_MAX_TURNS_EXCEEDED, message_args={"max_turns": max_turns}) from e
            except Exception as e:
                from datus.cli.execution_state import ExecutionInterrupted

                if isinstance(e, ExecutionInterrupted):
                    raise

                from openai import AuthenticationError

                if isinstance(e, AuthenticationError):
                    logger.info("Got 401 in generate_with_tools_stream, refreshing OAuth token...")
                    self.oauth_manager.refresh_tokens()
                    self._refresh_client_token()
                    raise DatusException(
                        ErrorCode.MODEL_AUTHENTICATION_ERROR,
                        message_args={"error_detail": "Codex auth failed; token refreshed, please retry"},
                    ) from e
                raise DatusException(
                    ErrorCode.MODEL_REQUEST_FAILED,
                    message=f"Codex streaming failed: {e}",
                ) from e

            # Store per-turn usage in turn_usage table
            if session and hasattr(session, "store_run_usage"):
                try:
                    await session.store_run_usage(result)
                except Exception as e:
                    logger.warning(f"Failed to store run usage: {e}")

            # Final summary after streaming completes
            has_final_output = hasattr(result, "final_output")
            final_output = result.final_output if has_final_output else None
            usage_info = self._extract_usage_info(result)
            final_action = ActionHistory(
                action_id=f"final_{uuid.uuid4().hex[:8]}",
                role=ActionRole.ASSISTANT,
                messages=str(final_output)[:200] if final_output is not None else "",
                action_type="final_response",
                input={},
                output={
                    "raw_output": str(final_output) if final_output is not None else "",
                    "sql_contexts": extract_sql_contexts(result) if has_final_output else [],
                    "usage": usage_info,
                },
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

    def _extract_usage_info(self, result) -> dict:
        """Extract usage info from Agent SDK result for token accounting."""
        if not (hasattr(result, "context_wrapper") and hasattr(result.context_wrapper, "usage")):
            return {}
        usage = result.context_wrapper.usage

        def _int(val, default=0):
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        input_tokens = _int(getattr(usage, "input_tokens", 0))
        output_tokens = _int(getattr(usage, "output_tokens", 0))
        total_tokens = _int(getattr(usage, "total_tokens", 0))

        cached_tokens = 0
        if hasattr(usage, "input_tokens_details") and usage.input_tokens_details:
            cached_tokens = _int(getattr(usage.input_tokens_details, "cached_tokens", 0))

        reasoning_tokens = 0
        if hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
            reasoning_tokens = _int(getattr(usage.output_tokens_details, "reasoning_tokens", 0))

        last_call_input_tokens = 0
        try:
            if hasattr(usage, "request_usage_entries") and usage.request_usage_entries:
                last_call_input_tokens = _int(getattr(usage.request_usage_entries[-1], "input_tokens", 0))
        except (IndexError, TypeError):
            pass

        cache_hit_rate = round(cached_tokens / input_tokens, 3) if input_tokens > 0 else 0

        context_usage_ratio = 0
        max_context = self.context_length()
        if max_context and total_tokens > 0:
            context_usage_ratio = round(total_tokens / max_context, 3)

        return {
            "requests": _int(getattr(usage, "requests", 0)),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_hit_rate": cache_hit_rate,
            "context_usage_ratio": context_usage_ratio,
            "last_call_input_tokens": last_call_input_tokens,
        }

    def token_count(self, prompt: str) -> int:
        """Estimate token count using a simple heuristic."""
        return len(prompt) // 4

    def _lookup_spec(self, field: str) -> Optional[int]:
        """Look up ``field`` in providers.yml ``model_specs``.

        Codex-gateway models are real OpenAI models (gpt-5.x, o3, etc.) so
        their specs come from the same catalog — no codex-only filter.

        Lookup order:
        1. Exact match on model_name
        2. OpenRouter-style slug match (``*/model_name``) — catches entries
           from the OpenRouter cache that carry a provider prefix
        3. Longest prefix match from providers.yml
        """
        from datus.models.openai_compatible import _load_model_specs

        specs = _load_model_specs()

        exact = specs.get(self.model_name)
        if isinstance(exact, dict) and isinstance(exact.get(field), int):
            return exact[field]

        suffix = f"/{self.model_name}"
        for spec_model, spec in specs.items():
            if not isinstance(spec, dict) or not isinstance(spec.get(field), int):
                continue
            if spec_model.endswith(suffix):
                return spec[field]

        best_key: Optional[str] = None
        for spec_model, spec in specs.items():
            if not isinstance(spec, dict) or not isinstance(spec.get(field), int):
                continue
            if self.model_name.startswith(spec_model) and (best_key is None or len(spec_model) > len(best_key)):
                best_key = spec_model
        if best_key is None:
            return None
        return specs[best_key][field]

    def max_tokens(self) -> Optional[int]:
        """Return the max output tokens for the current model."""
        value = self._lookup_spec("max_tokens")
        return value if value is not None else _CODEX_DEFAULT_MAX_TOKENS

    def context_length(self) -> Optional[int]:
        """Return the context length for the current model."""
        value = self._lookup_spec("context_length")
        return value if value is not None else _CODEX_DEFAULT_CONTEXT_LENGTH
