# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Claude Model - Anthropic Claude model implementation.

Inherits from OpenAICompatibleModel and adds Claude-specific features:
- Prompt caching via Anthropic's native API
- Optional native Anthropic API support (use_native_api config)
- Claude-specific model specifications
"""

import copy
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import anthropic
import httpx
from agents import Agent, RunContextWrapper, Usage
from agents.mcp import MCPServerStdio
from agents.tool_context import ToolContext
from agents.usage import InputTokensDetails, OutputTokensDetails, RequestUsage

from datus.configuration.agent_config import ModelConfig
from datus.models.mcp_utils import multiple_mcp_servers
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.ssl_utils import is_ssl_cert_verification_error
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


@dataclass
class _ToolResultPart:
    """A single part of a tool result (matches MCP tool result format)."""

    text: str


@dataclass
class _ToolResult:
    """Lightweight stand-in for MCP CallToolResult (`.content[0].text`)."""

    content: List[_ToolResultPart] = field(default_factory=list)


def wrap_prompt_cache(messages):
    """Wrap messages with Anthropic prompt cache control.

    Adds cache_control to the last content block for efficient prompt caching.
    """
    messages_copy = copy.deepcopy(messages)
    msg_size = len(messages_copy)
    content = messages_copy[msg_size - 1]["content"]
    cnt_size = len(content)
    if isinstance(content, list):
        content[cnt_size - 1]["cache_control"] = {"type": "ephemeral"}

    return messages_copy


def convert_tools_for_anthropic(mcp_tools):
    """Convert MCP tools to Anthropic tool format.

    Args:
        mcp_tools: List of MCP tools

    Returns:
        List of tools in Anthropic format with cache control
    """
    anthropic_tools = []

    for tool in mcp_tools:
        anthropic_tool = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema,
        }

        # Rename inputSchema's 'properties' to match Anthropic's convention if needed
        if "properties" in anthropic_tool["input_schema"]:
            for _, prop_value in anthropic_tool["input_schema"]["properties"].items():
                if "description" not in prop_value and "desc" in prop_value:
                    prop_value["description"] = prop_value.pop("desc")

        if hasattr(tool, "annotations") and tool.annotations:
            anthropic_tool["annotations"] = tool.annotations

        anthropic_tools.append(anthropic_tool)

    # Add tool cache to last tool (if any tools exist)
    if anthropic_tools:
        anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
    return anthropic_tools


class ClaudeModel(OpenAICompatibleModel):
    """
    Claude model implementation inheriting from OpenAICompatibleModel.

    Supports both:
    - LiteLLM-based API (default, via parent class)
    - Native Anthropic API (when use_native_api=True, enables prompt caching)
    """

    # Beta headers aligned with OpenClaw's current Anthropic OAuth path.
    # Keep this in sync with the OpenClaw PI_AI_OAUTH_ANTHROPIC_BETAS set when
    # using Claude Code setup-tokens (sk-ant-oat01-...).
    OAUTH_BETA_HEADERS = [
        "claude-code-20250219",
        "oauth-2025-04-20",
        "interleaved-thinking-2025-05-14",
        "prompt-caching-scope-2026-01-05",
    ]

    # Claude Code client headers — required for subscription tokens to be accepted.
    # These mimic the official Claude CLI client identity. The version string is
    # cosmetic (Anthropic validates via anthropic-beta + x-app, not user-agent version).
    # Update the version periodically to stay current if desired.
    OAUTH_CLIENT_HEADERS = {
        "user-agent": "claude-cli/2.1.75 (external, cli)",
        "x-app": "cli",
        "anthropic-dangerous-direct-browser-access": "true",
    }

    OAUTH_SYSTEM_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

    def __init__(self, model_config: ModelConfig, **kwargs):
        # Initialize parent class (handles LiteLLM adapter, OpenAI client, etc.)
        super().__init__(model_config, **kwargs)

        # Claude-specific: check if we should use native Anthropic API
        self.use_native_api = getattr(model_config, "use_native_api", False)

        # Detect OAuth subscription token via auth_type config (canonical source)
        self._is_oauth_token = getattr(model_config, "auth_type", "api_key") == "subscription"

        # OAuth tokens must use native API to avoid LiteLLM's x-api-key interference
        if self._is_oauth_token:
            self.use_native_api = True

        # Initialize native Anthropic client (always available for prompt caching)
        self._init_anthropic_client()

    def _get_api_key(self) -> str:
        """Get Anthropic API key from config or environment."""
        if self.model_config.auth_type == "subscription":
            from datus.auth.claude_credential import get_claude_subscription_token

            token, _source = get_claude_subscription_token(self.model_config.api_key)
            return token
        api_key = self.model_config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise DatusException(ErrorCode.MODEL_AUTHENTICATION_ERROR)
        return api_key

    def _get_base_url(self) -> Optional[str]:
        """Get Anthropic base URL from config."""
        return self.model_config.base_url or "https://api.anthropic.com"

    def _init_anthropic_client(self):
        """Initialize native Anthropic client for prompt caching and native API support."""
        # Optional proxy configuration
        proxy_url = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
        self.proxy_client = None
        self.async_proxy_client = None

        # SSL verification (e.g. a private gateway CA) resolved by the parent
        # __init__. The native client takes no `verify` argument, so we must pass a
        # custom http_client. Only do so when a proxy is set or ssl_verify is
        # configured; otherwise leave http_client=None to preserve default behavior
        # (the Anthropic SDK's httpx client honors the standard SSL_CERT_FILE env var).
        verify = getattr(self, "ssl_verify", None)
        if proxy_url or verify is not None:
            verify_kwargs = {} if verify is None else {"verify": verify}
            proxy_kwargs = {"proxy": httpx.Proxy(url=proxy_url)} if proxy_url else {}
            self.proxy_client = httpx.Client(
                transport=httpx.HTTPTransport(**verify_kwargs, **proxy_kwargs),
                timeout=60.0,
            )
            self.async_proxy_client = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(**verify_kwargs, **proxy_kwargs),
                timeout=60.0,
            )

        # Build headers: merge config default_headers with OAuth headers if needed
        extra_headers = dict(self.default_headers) if self.default_headers else {}
        if self._is_oauth_token:
            extra_headers["anthropic-beta"] = ",".join(self.OAUTH_BETA_HEADERS)
            extra_headers.update(self.OAUTH_CLIENT_HEADERS)
            logger.debug("Using OAuth subscription token — injecting beta + client headers")

        if self._is_oauth_token:
            # Use auth_token (Bearer auth) instead of api_key (x-api-key) for OAuth tokens
            self.anthropic_client = anthropic.Anthropic(
                auth_token=self.api_key,
                api_key=None,
                base_url=self.base_url if self.base_url else None,
                http_client=self.proxy_client,
                default_headers=extra_headers or None,
            )
            self.async_anthropic_client = anthropic.AsyncAnthropic(
                auth_token=self.api_key,
                api_key=None,
                base_url=self.base_url if self.base_url else None,
                http_client=self.async_proxy_client,
                default_headers=extra_headers or None,
            )
            # The SDK falls back to the ANTHROPIC_API_KEY env var when api_key=None,
            # and auth_headers merges both credentials — a stale env key would make
            # requests carry X-Api-Key alongside the Bearer token and get rejected.
            self.anthropic_client.api_key = None
            self.async_anthropic_client.api_key = None
        else:
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None,
                http_client=self.proxy_client,
                default_headers=extra_headers or None,
            )
            self.async_anthropic_client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None,
                http_client=self.async_proxy_client,
                default_headers=extra_headers or None,
            )
            # Symmetric guard: the SDK falls back to the ANTHROPIC_AUTH_TOKEN env
            # var when auth_token is not given, which would add a spurious
            # Authorization: Bearer header next to X-Api-Key.
            self.anthropic_client.auth_token = None
            self.async_anthropic_client.auth_token = None

        # Wrap with LangSmith if available. Wrap sync and async clients
        # independently so a failure on the async wrap (e.g. older langsmith
        # without AsyncAnthropic support) does not also drop sync tracing.
        try:
            from langsmith.wrappers import wrap_anthropic
        except ImportError:
            logger.debug("No langsmith wrapper available")
        else:
            try:
                self.anthropic_client = wrap_anthropic(self.anthropic_client)
            except Exception as e:
                logger.warning(f"Failed to wrap sync anthropic client with langsmith: {e}")
            try:
                self.async_anthropic_client = wrap_anthropic(self.async_anthropic_client)
            except Exception as e:
                logger.warning(f"Failed to wrap async anthropic client with langsmith: {e}")

        logger.debug(f"Initialized Claude model: {self.model_name}, use_native_api={self.use_native_api}")

    def _inject_oauth_headers(self, kwargs: dict) -> dict:
        """Inject OAuth beta + client headers into kwargs for LiteLLM calls if using subscription token."""
        if self._is_oauth_token:
            existing = kwargs.get("extra_headers", {})
            kwargs["extra_headers"] = {
                **existing,
                "anthropic-beta": ",".join(self.OAUTH_BETA_HEADERS),
                "Authorization": f"Bearer {self.api_key}",
                **self.OAUTH_CLIENT_HEADERS,
            }
        return kwargs

    def _build_system_param(self, system_message: str = "") -> Any:
        """Build Anthropic system param, injecting Claude Code identity for OAuth tokens."""
        if self._is_oauth_token:
            system_blocks: list[dict[str, str]] = [
                {
                    "type": "text",
                    "text": self.OAUTH_SYSTEM_IDENTITY,
                }
            ]
            if system_message:
                system_blocks.append(
                    {
                        "type": "text",
                        "text": system_message,
                    }
                )
            return system_blocks

        return system_message if system_message else anthropic.NOT_GIVEN

    @optional_traceable(name="anthropic_messages_create", run_type="llm")
    def _anthropic_messages_create(self, **kwargs):
        """Call the correct Anthropic Messages endpoint for the current auth mode."""
        if self._is_oauth_token:
            return self.anthropic_client.beta.messages.create(**kwargs)
        return self.anthropic_client.messages.create(**kwargs)

    @optional_traceable(name="anthropic_messages_stream", run_type="llm")
    def _anthropic_messages_stream(self, **kwargs):
        """Return an async context manager that streams Anthropic Messages events.

        Routes to ``beta.messages.stream`` for OAuth subscription tokens (which
        require the OAuth beta headers and Bearer auth) and to
        ``messages.stream`` for standard API-key auth. Caller enters via
        ``async with self._anthropic_messages_stream(...) as stream:``.

        Note: ``optional_traceable`` here records the call returning the
        context manager. For ``messages.stream`` langsmith's ``wrap_anthropic``
        additionally instruments per-event iteration; ``beta.messages.stream``
        is not wrapped upstream, so this decorator is the only tracing the
        OAuth path gets.
        """
        if self.async_anthropic_client is None:
            raise DatusException(
                ErrorCode.MODEL_AUTHENTICATION_ERROR,
                "Async Anthropic client is not initialized; streaming unavailable",
            )
        if self._is_oauth_token:
            return self.async_anthropic_client.beta.messages.stream(**kwargs)
        return self.async_anthropic_client.messages.stream(**kwargs)

    def _diagnose_oauth_401(self, original_error: Exception) -> None:
        """Diagnose a 401 error for OAuth subscription tokens and raise a specific exception.

        Checks whether the token is expired (actionable: re-run setup-token) or
        rejected for other reasons (revoked, subscription inactive, corrupted).
        Only acts when ``_is_oauth_token`` is True; otherwise returns silently so
        the caller can re-raise the original error unchanged.
        """
        if not self._is_oauth_token:
            return

        expires_at = None

        # Check credentials file first
        credentials_path = Path.home() / ".claude" / ".credentials.json"
        if credentials_path.exists():
            try:
                data = json.loads(credentials_path.read_text(encoding="utf-8"))
                expires_at = data.get("claudeAiOauth", {}).get("expiresAt")
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        # Fall back to Keychain if no expiry found from file
        if expires_at is None:
            try:
                from datus.auth.claude_credential import _read_keychain_credentials

                keychain_data = _read_keychain_credentials()
                if keychain_data:
                    expires_at = keychain_data.get("claudeAiOauth", {}).get("expiresAt")
            except Exception:
                pass

        if expires_at and int(expires_at) / 1000 < time.time():
            logger.warning("Claude subscription token has expired (expiresAt check)")
            raise DatusException(ErrorCode.CLAUDE_SUBSCRIPTION_TOKEN_EXPIRED) from original_error

        # Token is not expired (or no expiry info) — something else is wrong
        logger.warning("Claude subscription token rejected (401) but token is not expired")
        raise DatusException(ErrorCode.CLAUDE_SUBSCRIPTION_AUTH_FAILED) from original_error

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """Generate response using LiteLLM (default) or native Anthropic API.

        Default uses LiteLLM path for consistent api_key/base_url handling across
        all code paths (generate, generate_with_json_output, generate_with_tools_stream).
        Set use_native_api=True in model config to use native Anthropic client instead.

        Args:
            prompt: The input prompt (str or list of messages)
            enable_thinking: Enable thinking mode (not supported by Claude, ignored)
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        if not self.use_native_api:
            # ``top_p`` suppression for Anthropic now lives in the parent
            # ``OpenAICompatibleModel._generate_operation`` and is gated on
            # ``litellm_adapter.provider == LLMProvider.CLAUDE`` — that gate
            # fires for every ClaudeModel instance regardless of how the
            # config's ``type`` field is wired, so the previous
            # ``kwargs["top_p"] = None`` ceremony here is now redundant.
            # OAuth header injection stays Claude-specific.
            self._inject_oauth_headers(kwargs)
            try:
                return super().generate(prompt, enable_thinking=enable_thinking, **kwargs)
            except DatusException as e:
                if self._is_oauth_token and e.code == ErrorCode.MODEL_AUTHENTICATION_ERROR:
                    self._diagnose_oauth_401(e)
                raise

        # Native Anthropic client path (only when use_native_api=True)
        # Build messages
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        # Extract system message if present
        system_message = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        try:
            response = self._anthropic_messages_create(
                model=self.model_name,
                messages=filtered_messages,
                system=self._build_system_param(system_message),
                max_tokens=kwargs.get("max_tokens") or self.max_tokens() or 20480,
                temperature=kwargs.get("temperature", anthropic.NOT_GIVEN),
            )

            if response.content:
                return response.content[0].text
            return ""

        except anthropic.AuthenticationError as e:
            self._diagnose_oauth_401(e)  # raises specific DatusException for OAuth tokens
            raise
        except Exception as e:
            if is_ssl_cert_verification_error(e):
                raise DatusException(ErrorCode.MODEL_SSL_CERT_ERROR) from e
            logger.error(f"Error generating with Anthropic: {str(e)}")
            raise

    async def _generate_with_mcp_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        func_tools: Optional[List[Any]] = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        interrupt_controller=None,
        session: Optional[Any] = None,
        hooks=None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Async generator: native Anthropic API with real-time tool call ActionHistory.

        Yields ActionHistory objects for each tool call (PROCESSING then SUCCESS/FAILURE),
        and a final ASSISTANT action containing the result dict.

        ``hooks`` carries the node's composed run hooks. The native loop is not
        driven by the openai-agents Runner, so the SDK never fires
        ``on_llm_end`` — we drive :class:`TokenUsageHook` manually after every
        Anthropic API response so the CLI status bar / API ``usage`` events
        update per LLM call instead of only at turn end.
        """
        # Custom JSON encoder for special types
        self._setup_custom_json_encoder()

        logger.debug(f"Using native Anthropic API with prompt caching, model: {self.model_name}")
        try:
            all_tools = []

            # Use context manager to manage multiple MCP servers
            async with multiple_mcp_servers(mcp_servers) as connected_servers:
                # Get all tools and build tool-name-to-server mapping once
                tool_server_map = {}  # tool_name -> connected_server
                mcp_tool_objs = {}  # tool_name -> SDK tool object (carries .name for hooks)
                for server_name, connected_server in connected_servers.items():
                    try:
                        agent = Agent(name="mcp-tools-agent")
                        run_context = RunContextWrapper(context=None, usage=Usage())
                        mcp_tools = await connected_server.list_tools(run_context, agent)
                        for tool in mcp_tools:
                            if tool.name in tool_server_map:
                                logger.warning(
                                    f"Duplicate MCP tool name '{tool.name}' from server '{server_name}', "
                                    f"overwriting previous mapping"
                                )
                            tool_server_map[tool.name] = connected_server
                            mcp_tool_objs[tool.name] = tool
                        all_tools.extend(mcp_tools)
                        logger.info(f"Retrieved {len(mcp_tools)} tools from {server_name}")

                    except Exception as e:
                        logger.error(f"Error getting tools from {server_name}: {str(e)}")
                        continue

                # Shared placeholder agent for hook lifecycle calls. The native
                # loop is not driven by the openai-agents Runner, so we fire the
                # composed ``hooks`` (permission / compact / KB-sync / token-usage)
                # ourselves; the hooks barely read ``agent`` so one placeholder is
                # sufficient for every callback.
                hook_agent = Agent(name="claude-native-agent")

                logger.info(f"Retrieved {len(all_tools)} total tools from MCP servers")

                tools = convert_tools_for_anthropic(all_tools)

                # Convert and merge function tools (Agent SDK FunctionTool objects)
                func_tool_map = {}
                if func_tools:
                    for ft in func_tools:
                        tools.append(
                            {
                                "name": ft.name,
                                "description": ft.description or "",
                                "input_schema": ft.params_json_schema,
                            }
                        )
                        func_tool_map[ft.name] = ft
                    # Re-apply cache control on last tool
                    if tools:
                        for t in tools:
                            t.pop("cache_control", None)
                        tools[-1]["cache_control"] = {"type": "ephemeral"}
                # Load prior turns from the session so multi-turn chat works.
                # Native Anthropic loop is not driven by openai-agents Runner, so
                # we replay session history into ``messages`` ourselves.
                # ``instruction`` is already carried by ``_build_system_param``;
                # do NOT re-embed it in the user message here, otherwise persisted
                # turns would carry duplicated system text.
                messages: List[Dict[str, Any]] = []
                if session is not None:
                    try:
                        prior_items = await session.get_items()
                        if prior_items:
                            messages.extend(prior_items)
                    except Exception as e:
                        logger.warning(f"Failed to load session history; starting fresh: {e}")
                # Anthropic ``text`` blocks must be a single string. The signature
                # inherits ``prompt: Union[str, List[Dict[str, str]]]`` from the
                # base class for legacy callers; defensively normalise list-shaped
                # inputs so a future caller can't slip an invalid block past us.
                prompt_text = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
                user_turn_message = {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}],
                }
                messages.append(user_turn_message)
                tool_call_cache = {}
                sql_contexts = []
                final_content = ""
                # Accumulate token usage across all turns
                cumulative_input_tokens = 0
                cumulative_output_tokens = 0
                cache_creation_tokens = 0
                cache_read_tokens = 0
                last_call_input_tokens = 0

                # Drive the composed run hooks through their standard lifecycle
                # so the native loop behaves exactly like the SDK Runner path.
                # ``run_ctx`` carries a real SDK ``Usage`` accumulator that
                # ``TokenUsageHook.on_llm_end`` reads via ``_extract_usage_info``
                # (inherited from OpenAICompatibleModel) — no bespoke per-hook
                # plumbing. ``on_start`` resets each hook's per-turn baseline so
                # the first response reports its full usage as a delta.
                run_ctx = RunContextWrapper(context=None, usage=Usage())
                await self._invoke_hook(hooks, "on_start", run_ctx, hook_agent)

                # Execute conversation loop
                turn = -1
                for turn in range(max_turns):
                    if interrupt_controller and interrupt_controller.is_interrupted:
                        from datus.cli.execution_state import ExecutionInterrupted

                        raise ExecutionInterrupted("Interrupted by user")

                    logger.debug(f"Turn {turn + 1}/{max_turns}")

                    request_kwargs = dict(
                        model=self.model_name,
                        system=self._build_system_param(instruction),
                        messages=wrap_prompt_cache(messages),
                        tools=tools,
                        max_tokens=kwargs.get("max_tokens") or self.max_tokens() or 20480,
                        temperature=kwargs.get("temperature", anthropic.NOT_GIVEN),
                    )

                    if self.async_anthropic_client is not None:
                        # Streaming path: yield text deltas as ``thinking_delta``
                        # ActionHistory in real time (parity with
                        # OpenAICompatibleModel.generate_with_tools_stream), then
                        # rehydrate the final ``Message`` to reuse the existing
                        # tool-use / usage / session logic below unchanged.
                        thinking_stream_id: Optional[str] = None
                        thinking_accumulated = ""
                        async with self._anthropic_messages_stream(**request_kwargs) as stream:
                            async for event in stream:
                                if interrupt_controller and interrupt_controller.is_interrupted:
                                    from datus.cli.execution_state import ExecutionInterrupted

                                    raise ExecutionInterrupted("Interrupted by user")

                                event_type = getattr(event, "type", None)
                                if event_type == "content_block_start":
                                    block_start = getattr(event, "content_block", None)
                                    if block_start is not None and getattr(block_start, "type", None) == "text":
                                        thinking_stream_id = f"thinking_stream_{uuid.uuid4().hex[:8]}"
                                        thinking_accumulated = ""
                                elif event_type == "content_block_delta":
                                    delta = getattr(event, "delta", None)
                                    if delta is None or getattr(delta, "type", None) != "text_delta":
                                        continue
                                    delta_text = getattr(delta, "text", "") or ""
                                    if not delta_text:
                                        continue
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
                                elif event_type == "content_block_stop":
                                    # Mirror OpenAICompatibleModel / codex_model: when
                                    # a text block ends, emit a paired terminal
                                    # ``response`` SUCCESS action sharing the delta
                                    # stream id. Without it the CLI's
                                    # ``_print_completed_action`` dedup path never
                                    # runs, so ``_finalize_markdown_stream`` never
                                    # clears the pinned live region — the last
                                    # paragraph stays visible while ``__exit__``
                                    # also flushes it to scrollback, producing a
                                    # visible duplicate of the closing paragraph.
                                    if thinking_accumulated.strip():
                                        full_text = thinking_accumulated.strip()
                                        text_preview = full_text if len(full_text) <= 200 else f"{full_text[:200]}..."
                                        yield ActionHistory(
                                            action_id=thinking_stream_id or f"assistant_{uuid.uuid4().hex[:8]}",
                                            role=ActionRole.ASSISTANT,
                                            messages=f"Thinking: {text_preview}",
                                            action_type="response",
                                            input={},
                                            output={"raw_output": full_text, "is_thinking": False},
                                            status=ActionStatus.SUCCESS,
                                        )
                                    thinking_stream_id = None
                                    thinking_accumulated = ""
                            response = await stream.get_final_message()
                    else:
                        # Fallback non-streaming path (kept so existing tests that
                        # mock ``anthropic_client.messages.create`` still work, and
                        # for environments where the async client failed to init).
                        response = self._anthropic_messages_create(**request_kwargs)

                    # Track token usage from this turn.
                    #
                    # Anthropic reports cached input SEPARATELY from ``input_tokens``:
                    # the latter is only the fresh, non-cached input, while
                    # ``cache_read_input_tokens`` / ``cache_creation_input_tokens`` are
                    # distinct fields. The real input processed on a call is
                    # ``input_tokens + cache_read + cache_creation``. OpenAI's
                    # ``input_tokens`` already folds cached tokens in, so we add the
                    # cache components back here to keep reporting parity — otherwise
                    # heavy prompt caching (Claude Code subscription) collapses the
                    # reported input to a handful of tokens and inflates the
                    # cache-hit-rate / context-usage ratios into nonsense.
                    if hasattr(response, "usage") and response.usage:
                        call_input = getattr(response.usage, "input_tokens", 0)
                        call_cache_read = getattr(response.usage, "cache_read_input_tokens", 0)
                        call_cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0)
                        call_total_input = call_input + call_cache_read + call_cache_creation
                        cumulative_input_tokens += call_total_input
                        cumulative_output_tokens += getattr(response.usage, "output_tokens", 0)
                        cache_creation_tokens += call_cache_creation
                        cache_read_tokens += call_cache_read
                        last_call_input_tokens = call_total_input

                        # Drive the per-LLM-call usage update through the standard
                        # ``on_llm_end`` hook so the CLI status bar / API ``usage``
                        # events refresh mid-turn. The native loop bypasses the SDK
                        # Runner, so we mutate the shared ``run_ctx.usage`` with an
                        # SDK ``Usage`` snapshot and fire the hook exactly as the
                        # Runner would after each model response.
                        run_ctx.usage = self._build_sdk_usage(
                            requests=turn + 1,
                            cumulative_input_tokens=cumulative_input_tokens,
                            cumulative_output_tokens=cumulative_output_tokens,
                            cache_read_tokens=cache_read_tokens,
                            last_call_input_tokens=last_call_input_tokens,
                        )
                        await self._invoke_hook(hooks, "on_llm_end", run_ctx, hook_agent, response)

                    message = response.content

                    # If no tool calls, conversation is complete
                    if not any(block.type == "tool_use" for block in message):
                        final_content = "\n".join([block.text for block in message if block.type == "text"])
                        logger.debug("No tool calls, conversation completed")
                        break

                    for block in message:
                        if block.type == "tool_use":
                            if interrupt_controller and interrupt_controller.is_interrupted:
                                from datus.cli.execution_state import ExecutionInterrupted

                                raise ExecutionInterrupted("Interrupted by user")

                            logger.debug(f"Executing tool: {block.name}")
                            args_str = json.dumps(block.input, ensure_ascii=False)[:80]

                            # Yield PROCESSING action for real-time tool call display
                            start_action = ActionHistory(
                                action_id=block.id,
                                role=ActionRole.TOOL,
                                messages=f"Tool call: {block.name}('{args_str}...')",
                                action_type=block.name,
                                input={"function_name": block.name, "arguments": block.input},
                                output={},
                                status=ActionStatus.PROCESSING,
                            )
                            if action_history_manager is not None:
                                action_history_manager.add_action(start_action)
                            yield start_action

                            # Build the per-tool context and resolve a tool object
                            # exposing ``.name`` BEFORE running the tool, then fire
                            # ``on_tool_start`` so the composed permission hook can
                            # gate execution exactly like the SDK Runner path. This
                            # MUST sit before the execution try/except blocks below:
                            # a ``PermissionDeniedException`` has to propagate out of
                            # the generator (mirroring the SDK, where on_tool_start
                            # raising aborts the run) instead of being swallowed and
                            # downgraded into a tool_result error by them.
                            tool_args = json.dumps(block.input, ensure_ascii=False)
                            tool_ctx = ToolContext(
                                context=None,
                                usage=Usage(),
                                tool_name=block.name,
                                tool_call_id=block.id,
                                tool_arguments=tool_args,
                            )
                            tool_obj = (
                                func_tool_map.get(block.name)
                                or mcp_tool_objs.get(block.name)
                                or SimpleNamespace(name=block.name)
                            )
                            await self._invoke_hook(hooks, "on_tool_start", tool_ctx, hook_agent, tool_obj)

                            tool_executed = False
                            # Raw structured result handed to ``on_tool_end``; GenerationHooks
                            # inspects the FuncToolResult dict / result text to sync the KB.
                            hook_result: Any = None

                            # Try function tools first
                            if block.name in func_tool_map:
                                try:
                                    ft = func_tool_map[block.name]
                                    # Reuse ``tool_ctx`` (a ToolContext, not a bare
                                    # RunContextWrapper) so the tool's ``tool_call_id``
                                    # matches this block's ``action_id`` (``block.id``).
                                    # The ``task`` tool reads ``tool_call_id`` to link
                                    # every sub-agent action's ``parent_action_id`` to the
                                    # wrapping task action; without it the CLI renderer
                                    # cannot anchor the sub-agent group and mis-renders it
                                    # as separate ``<node>_request`` / ``<node>_response``
                                    # blocks.
                                    result_val = await ft.on_invoke_tool(tool_ctx, tool_args)
                                    hook_result = result_val
                                    # Ensure result is a string (Anthropic API requires string content)
                                    result_str = result_val if isinstance(result_val, str) else json.dumps(result_val)
                                    # Wrap in object matching MCP tool result format
                                    func_result = _ToolResult(content=[_ToolResultPart(text=result_str)])
                                    tool_call_cache[block.id] = func_result
                                    tool_executed = True
                                except Exception as e:
                                    logger.error(f"Error executing function tool {block.name}: {str(e)}")

                            # Fall back to MCP servers via pre-built mapping
                            if not tool_executed:
                                target_server = tool_server_map.get(block.name)
                                if target_server:
                                    try:
                                        tool_result = await target_server.call_tool(
                                            tool_name=block.name,
                                            arguments=dict(block.input)
                                            if isinstance(block.input, dict)
                                            else block.input,
                                        )
                                        tool_call_cache[block.id] = tool_result
                                        hook_result = tool_result.content[0].text if tool_result.content else ""
                                        tool_executed = True
                                    except Exception as e:
                                        logger.error(f"Error executing tool {block.name}: {str(e)}")

                            if not tool_executed:
                                logger.error(f"Tool {block.name} could not be executed")

                            # Yield SUCCESS/FAILURE action for real-time tool call display
                            result_text = ""
                            if block.id in tool_call_cache:
                                result_text = tool_call_cache[block.id].content[0].text
                            result_summary = (
                                self._format_tool_result(result_text, block.name) if tool_executed else "Failed"
                            )
                            tool_output = {
                                "success": tool_executed,
                                "raw_output": result_text,
                                "summary": result_summary,
                                "status_message": result_summary,
                            }
                            # Keep the structured records (raw_output is stringified
                            # for the Anthropic message). Unwrap the FuncToolResult
                            # envelope so benchmark trajectory evaluation can read
                            # source_context_id provenance.
                            structured_result = None
                            if isinstance(hook_result, dict):
                                structured_result = hook_result.get("result", hook_result)
                            elif isinstance(hook_result, list):
                                structured_result = hook_result
                            if isinstance(structured_result, (dict, list)):
                                tool_output["result"] = structured_result
                            complete_action = ActionHistory(
                                action_id=f"complete_{block.id}",
                                role=ActionRole.TOOL,
                                messages=f"Tool call: {block.name}('{args_str}...')",
                                action_type=block.name,
                                input={"function_name": block.name, "arguments": block.input},
                                output=tool_output,
                                status=ActionStatus.SUCCESS if tool_executed else ActionStatus.FAILED,
                            )
                            complete_action.end_time = datetime.now()
                            if action_history_manager is not None:
                                action_history_manager.add_action(complete_action)
                            yield complete_action

                            # Fire ``on_tool_end`` so the compact / KB-sync hooks run
                            # per tool completion, mirroring the SDK Runner loop. On
                            # failure pass a ``{"success": 0}`` marker so consumers that
                            # inspect the result treat it as a failed call.
                            await self._invoke_hook(
                                hooks,
                                "on_tool_end",
                                tool_ctx,
                                hook_agent,
                                tool_obj,
                                hook_result if tool_executed else {"success": 0, "error": result_summary},
                            )

                    # Build assistant message content from all blocks
                    content = []
                    tool_use_blocks = []
                    for block in message:
                        if block.type == "text":
                            content.append({"type": "text", "text": block.text})
                        elif block.type == "tool_use":
                            content.append(
                                {
                                    "type": "tool_use",
                                    "id": block.id,
                                    "name": block.name,
                                    "input": block.input,
                                }
                            )
                            tool_use_blocks.append(block)

                    if content:
                        messages.append({"role": "assistant", "content": content})

                    for block in tool_use_blocks:
                        if block.id in tool_call_cache:
                            sql_result = tool_call_cache[block.id].content[0].text
                            # Use "Error" to determine execution success
                            if "Error" not in sql_result and block.name == "read_query":
                                sql_query = block.input.get("query") or block.input.get("sql", "")
                                sql_context = SQLContext(
                                    sql_query=sql_query,
                                    sql_return=sql_result,
                                    row_count=None,
                                )
                                sql_contexts.append(sql_context)
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": block.id,
                                            "content": sql_result,
                                        }
                                    ],
                                }
                            )
                        else:
                            error_message = f"Tool {block.name} execution failed"
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": block.id,
                                            "content": error_message,
                                        }
                                    ],
                                }
                            )

                logger.debug("Agent execution completed")
                usage_info = self._build_native_usage_info(
                    requests=turn + 1,
                    cumulative_input_tokens=cumulative_input_tokens,
                    cumulative_output_tokens=cumulative_output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                    last_call_input_tokens=last_call_input_tokens,
                )
                logger.debug(f"Native API cumulative token usage: {usage_info}")

                final_action = ActionHistory(
                    action_id=f"final_{uuid.uuid4().hex[:8]}",
                    role=ActionRole.ASSISTANT,
                    messages=str(final_content)[:200],
                    action_type="final_response",
                    input={},
                    output={
                        "raw_output": final_content,
                        "sql_contexts": sql_contexts,
                        "usage": usage_info,
                    },
                    status=ActionStatus.SUCCESS,
                )
                if action_history_manager is not None:
                    action_history_manager.add_action(final_action)

                # Persist this turn into the session so the next turn replays it
                # via ``session.get_items()``. Mirror what openai-agents Runner
                # would do via SQLiteSession.add_items, but driven by us since
                # the native Anthropic loop bypasses Runner.run.
                #
                # When the loop exits via ``max_turns`` exhaustion while still
                # tool-calling, ``final_content`` stays "" — Anthropic rejects
                # empty assistant text blocks on replay
                # (``messages.{i}.content.{j}.text: text content blocks must be
                # non-empty``), which would poison the session. Skip persistence
                # in that case so the next turn starts from a clean slate.
                if session is not None and final_content:
                    try:
                        assistant_turn_message = {
                            "role": "assistant",
                            "content": [{"type": "text", "text": final_content}],
                        }
                        await session.add_items([user_turn_message, assistant_turn_message])
                        # Persist the turn's token usage into the durable
                        # ``turn_usage`` table. The native loop never calls
                        # ``Runner.run`` (which is what normally triggers
                        # ``store_run_usage``), so the CLI status bar's
                        # cumulative total would stay at 0 without this. Must
                        # run AFTER ``add_items`` so the SDK derives the right
                        # ``user_turn_number`` from the freshly inserted rows.
                        await self._store_native_turn_usage(session, usage_info)
                    except Exception as e:
                        logger.warning(f"Failed to persist session history for native Claude turn: {e}")
                elif session is not None:
                    logger.warning(
                        "Skipping native Claude session persist: turn ended without final text "
                        "(max_turns=%s exhausted while tool-calling).",
                        max_turns,
                    )

                # Close out the composed hooks' lifecycle, completing parity with
                # the SDK Runner path (current ``on_end`` implementations are
                # no-ops, but the native loop now drives the full interface).
                await self._invoke_hook(hooks, "on_end", run_ctx, hook_agent, final_content)

                yield final_action

        except anthropic.AuthenticationError as e:
            self._diagnose_oauth_401(e)
            raise
        except Exception as e:
            if is_ssl_cert_verification_error(e):
                raise DatusException(ErrorCode.MODEL_SSL_CERT_ERROR) from e
            logger.error(f"Error in _generate_with_mcp_stream: {str(e)}")
            raise

    def _build_native_usage_info(
        self,
        *,
        requests: int,
        cumulative_input_tokens: int,
        cumulative_output_tokens: int,
        cache_read_tokens: int,
        cache_creation_tokens: int,
        last_call_input_tokens: int,
    ) -> dict:
        """Build the standardized usage dict for the native Anthropic loop.

        Shared by the mid-turn per-call updates and the final action so the
        cumulative numbers reported to the status bar / SSE never drift from
        what is attached to the final assistant action. ``cached_tokens``
        maps to Anthropic's ``cache_read_input_tokens`` (the portion served
        from cache), matching :meth:`OpenAICompatibleModel._extract_usage_info`.

        ``cumulative_input_tokens`` here already includes the cache_read /
        cache_creation components (folded in by the caller) so that
        ``input_tokens``, ``total_tokens``, ``cache_hit_rate`` and
        ``context_usage_ratio`` share OpenAI's "input includes cached" semantics.
        """
        total_tokens = cumulative_input_tokens + cumulative_output_tokens
        cached_tokens = cache_read_tokens
        context_length = self.context_length()
        return {
            "requests": requests,
            "input_tokens": cumulative_input_tokens,
            "output_tokens": cumulative_output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "reasoning_tokens": 0,
            "cache_hit_rate": (round(cached_tokens / cumulative_input_tokens, 3) if cumulative_input_tokens > 0 else 0),
            "context_usage_ratio": (
                round(total_tokens / context_length, 3) if context_length and total_tokens > 0 else 0
            ),
            "last_call_input_tokens": last_call_input_tokens,
        }

    def _build_sdk_usage(
        self,
        *,
        requests: int,
        cumulative_input_tokens: int,
        cumulative_output_tokens: int,
        cache_read_tokens: int,
        last_call_input_tokens: int,
    ) -> Usage:
        """Build an SDK ``Usage`` snapshot from the native loop's counters.

        The native Anthropic loop bypasses the openai-agents Runner, so it must
        feed :class:`TokenUsageHook` (via ``on_llm_end``) the same shape the
        Runner would. ``TokenUsageHook._emit`` reads ``context.usage`` and calls
        :meth:`OpenAICompatibleModel._extract_usage_info` (inherited here), which
        consumes ``input_tokens`` / ``output_tokens`` / ``total_tokens``,
        ``input_tokens_details.cached_tokens`` and ``request_usage_entries[-1]``
        for the last-call input. ``cumulative_input_tokens`` already folds in the
        cache_read / cache_creation components (done by the caller) so the OpenAI
        "input includes cached" semantics carry through cache-hit-rate and
        context-usage-ratio derivation.
        """
        total_tokens = cumulative_input_tokens + cumulative_output_tokens
        return Usage(
            requests=requests,
            input_tokens=cumulative_input_tokens,
            input_tokens_details=InputTokensDetails(cached_tokens=cache_read_tokens),
            output_tokens=cumulative_output_tokens,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=total_tokens,
            # ``_extract_usage_info`` derives ``last_call_input_tokens`` from the
            # final request entry's ``input_tokens`` — the real context-window
            # occupancy of the most recent LLM call.
            request_usage_entries=[
                RequestUsage(
                    input_tokens=last_call_input_tokens,
                    output_tokens=0,
                    total_tokens=last_call_input_tokens,
                    input_tokens_details=InputTokensDetails(cached_tokens=cache_read_tokens),
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                )
            ],
        )

    async def _invoke_hook(self, hooks, method_name: str, *args) -> None:
        """Drive one composed-hook lifecycle method from the native loop.

        ``hooks`` may be a bare ``AgentHooks`` or a ``CompositeHooks`` wrapping
        several; either way we call ``getattr(hooks, method_name)`` once and let
        ``CompositeHooks`` fan out to its children. ``PermissionDeniedException``
        MUST propagate so a denied tool aborts the run, mirroring the SDK path
        (where ``on_tool_start`` raising surfaces as a ``UserError``). Every
        other exception is logged and swallowed so a best-effort hook (compact,
        KB sync, token usage) can never crash the agent loop.
        """
        if hooks is None:
            return
        method = getattr(hooks, method_name, None)
        if method is None:
            return
        # Local import keeps the module import graph light and matches the
        # file's convention of importing permission types lazily.
        from datus.tools.permission.permission_hooks import PermissionDeniedException

        try:
            await method(*args)
        except PermissionDeniedException:
            raise
        except Exception:  # noqa: BLE001 — best-effort hooks never break the loop
            logger.debug("Hook '%s' raised; ignoring", method_name, exc_info=True)

    async def _store_native_turn_usage(self, session, usage_info: dict) -> None:
        """Persist the native turn's cumulative usage into ``turn_usage``.

        Constructs an SDK :class:`Usage` and feeds it through the session's
        ``store_run_usage`` via a minimal result shim so the durable schema
        and turn-numbering match the Runner-driven models exactly.
        """
        if session is None or not hasattr(session, "store_run_usage"):
            return
        try:
            usage = Usage(
                requests=int(usage_info.get("requests", 0) or 0),
                input_tokens=int(usage_info.get("input_tokens", 0) or 0),
                output_tokens=int(usage_info.get("output_tokens", 0) or 0),
                total_tokens=int(usage_info.get("total_tokens", 0) or 0),
                input_tokens_details=InputTokensDetails(cached_tokens=int(usage_info.get("cached_tokens", 0) or 0)),
                output_tokens_details=OutputTokensDetails(
                    reasoning_tokens=int(usage_info.get("reasoning_tokens", 0) or 0)
                ),
            )

            class _NativeUsageResult:
                """Minimal ``RunResult`` stand-in exposing ``context_wrapper.usage``."""

                def __init__(self, _usage):
                    self.context_wrapper = RunContextWrapper(context=None, usage=_usage)

            await session.store_run_usage(_NativeUsageResult(usage))
        except Exception as e:
            logger.warning(f"Failed to store native Claude turn usage: {e}")

    async def generate_with_mcp(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        func_tools: Optional[List[Any]] = None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        session: Optional[Any] = None,
        **kwargs,
    ) -> Dict:
        """Non-streaming wrapper: consumes _generate_with_mcp_stream and returns result dict."""
        result: Dict = {"content": "", "sql_contexts": []}
        async for action in self._generate_with_mcp_stream(
            prompt=prompt,
            mcp_servers=mcp_servers,
            instruction=instruction,
            output_type=output_type,
            max_turns=max_turns,
            func_tools=func_tools,
            action_history_manager=action_history_manager,
            session=session,
            **kwargs,
        ):
            if action.role == ActionRole.ASSISTANT and action.action_type == "final_response":
                result = {
                    "content": action.output.get("raw_output", ""),
                    "sql_contexts": action.output.get("sql_contexts", []),
                }
        return result

    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: Optional[List[Any]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        instruction: str = "",
        output_type: type = str,
        strict_json_schema: bool = True,
        max_turns: int = 10,
        session=None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        hooks=None,
        **kwargs,
    ) -> Dict:
        """Generate response with tool support.

        Routes to native Anthropic API when use_native_api=True and mcp_servers provided,
        otherwise uses parent class LiteLLM implementation.
        """
        # Use native Anthropic API when configured (required for OAuth subscription tokens
        # since LiteLLM sends x-api-key which is incompatible with Bearer auth)
        if self.use_native_api and (mcp_servers or self._is_oauth_token):
            return await self.generate_with_mcp(
                prompt=prompt,
                mcp_servers=mcp_servers or {},
                instruction=instruction,
                output_type=output_type,
                max_turns=max_turns,
                func_tools=tools,
                action_history_manager=action_history_manager,
                session=session,
                **kwargs,
            )

        # Use parent class LiteLLM implementation
        self._inject_oauth_headers(kwargs)
        try:
            return await super().generate_with_tools(
                prompt=prompt,
                tools=tools,
                mcp_servers=mcp_servers,
                instruction=instruction,
                output_type=output_type,
                strict_json_schema=strict_json_schema,
                max_turns=max_turns,
                session=session,
                action_history_manager=action_history_manager,
                hooks=hooks,
                **kwargs,
            )
        except DatusException as e:
            if self._is_oauth_token and e.code == ErrorCode.MODEL_AUTHENTICATION_ERROR:
                self._diagnose_oauth_401(e)
            raise

    async def generate_with_tools_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        tools: Optional[List[Any]] = None,
        instruction: str = "",
        output_type: type = str,
        strict_json_schema: bool = True,
        max_turns: int = 10,
        session=None,
        action_history_manager: Optional[ActionHistoryManager] = None,
        hooks=None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate response with streaming and tool support.

        Routes to native Anthropic API for OAuth subscription tokens,
        otherwise uses parent class LiteLLM implementation.
        """
        # For OAuth tokens, use native path (LiteLLM sends x-api-key which is incompatible)
        # Directly iterate the async generator for real-time tool call display
        if self.use_native_api and self._is_oauth_token:
            if action_history_manager is None:
                action_history_manager = ActionHistoryManager()
            async for action in self._generate_with_mcp_stream(
                prompt=prompt,
                mcp_servers=mcp_servers or {},
                instruction=instruction,
                output_type=output_type,
                max_turns=max_turns,
                func_tools=tools,
                action_history_manager=action_history_manager,
                interrupt_controller=kwargs.pop("interrupt_controller", None),
                session=session,
                hooks=hooks,
                **kwargs,
            ):
                yield action
            return

        self._inject_oauth_headers(kwargs)
        try:
            async for action in super().generate_with_tools_stream(
                prompt=prompt,
                mcp_servers=mcp_servers,
                tools=tools,
                instruction=instruction,
                output_type=output_type,
                strict_json_schema=strict_json_schema,
                max_turns=max_turns,
                session=session,
                action_history_manager=action_history_manager,
                hooks=hooks,
                **kwargs,
            ):
                yield action
        except DatusException as e:
            if self._is_oauth_token and e.code == ErrorCode.MODEL_AUTHENTICATION_ERROR:
                self._diagnose_oauth_401(e)
            raise

    async def aclose(self):
        """Async cleanup of resources."""
        # Close parent class resources
        # Note: Parent class doesn't have aclose, but we keep this for future compatibility

        if hasattr(self, "proxy_client") and self.proxy_client:
            try:
                self.proxy_client.close()
                logger.debug("Proxy client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing proxy client: {e}")

        if hasattr(self, "async_proxy_client") and self.async_proxy_client:
            try:
                await self.async_proxy_client.aclose()
                logger.debug("Async proxy client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing async proxy client: {e}")

        if hasattr(self, "anthropic_client") and hasattr(self.anthropic_client, "close"):
            try:
                self.anthropic_client.close()
                logger.debug("Anthropic client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing anthropic client: {e}")

        if hasattr(self, "async_anthropic_client") and self.async_anthropic_client is not None:
            try:
                await self.async_anthropic_client.close()
                logger.debug("Async anthropic client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing async anthropic client: {e}")

    def close(self):
        """Synchronous close for backward compatibility."""
        if hasattr(self, "proxy_client") and self.proxy_client:
            try:
                self.proxy_client.close()
            except Exception as e:
                logger.warning(f"Error closing proxy client: {e}")

        if hasattr(self, "anthropic_client") and hasattr(self.anthropic_client, "close"):
            try:
                self.anthropic_client.close()
            except Exception as e:
                logger.warning(f"Error closing anthropic client: {e}")

        # Async resources can't be awaited from sync ``close()``; warn so callers
        # know to invoke ``aclose()`` instead of leaking connections/threads.
        if getattr(self, "async_anthropic_client", None) is not None or getattr(self, "async_proxy_client", None):
            logger.debug("Async anthropic/proxy clients require aclose(); sync close() cannot release them")

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        try:
            self.close()
        except Exception as e:
            logger.warning(f"Error in ClaudeModel destructor: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
