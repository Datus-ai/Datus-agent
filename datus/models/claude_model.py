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
from datus.utils.constants import SQLType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import parse_sql_type
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


def _extract_document_text(node, depth: int = 0) -> str:
    """Best-effort recursive pull of readable text from a web_fetch document block.

    Anthropic's ``web_fetch_result`` wraps the fetched page as a document block
    whose text can sit at ``.source.data`` (text media type) or in a nested
    ``.content[].text`` list, depending on the page. We probe both, bounded by
    depth so a pathological structure can't recurse without end.
    """
    if node is None or depth > 6:
        return ""

    def _get(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    # Direct text payloads.
    text = _get(node, "text")
    if isinstance(text, str) and text:
        return text
    source = _get(node, "source")
    if source is not None:
        data = _get(source, "data")
        if isinstance(data, str) and data and (_get(source, "type") in (None, "text")):
            return data
    # Recurse into a nested content list / single child.
    child = _get(node, "content")
    if isinstance(child, list):
        return "\n".join(filter(None, (_extract_document_text(c, depth + 1) for c in child)))
    if child is not None:
        return _extract_document_text(child, depth + 1)
    return ""


def summarize_web_tool_result(res_block, query: str = ""):
    """Normalize an Anthropic ``web_search_tool_result`` / ``web_fetch_tool_result``.

    Returns ``(summary, canonical_result)`` where ``summary`` is the compact
    one-liner (SSE ``shortDesc`` / TUI compact line) and ``canonical_result`` is
    the provider-agnostic schema from :mod:`datus.schemas.web_result` — so a
    Claude server-tool call renders identically to the local Tavily / httpx
    backends and the OpenAI hosted path.
    """
    from datus.schemas.web_result import (
        normalize_fetch_result,
        normalize_search_results,
        web_fetch_short_summary,
        web_search_short_summary,
    )

    content = getattr(res_block, "content", None) if res_block is not None else None

    # web_search: ``content`` is a list of result blocks (title/url/page_age).
    if isinstance(content, list):
        items = []
        for item in content:
            items.append(
                {
                    "title": getattr(item, "title", None) or "",
                    "url": getattr(item, "url", None) or "",
                    "snippet": "",  # Anthropic returns encrypted_content; no readable snippet.
                    "age": getattr(item, "page_age", None),
                }
            )
        canonical = normalize_search_results(query, items)
        return web_search_short_summary(canonical), canonical

    # web_fetch: ``content`` is a single document-bearing result block.
    if content is not None:
        url = getattr(content, "url", None) or getattr(res_block, "url", None) or ""
        doc = getattr(content, "content", None)
        title = getattr(doc, "title", None) or getattr(content, "title", None) or ""
        text = _extract_document_text(doc)
        canonical = normalize_fetch_result(url=url, title=title, content=text, truncated=False)
        return web_fetch_short_summary(canonical), canonical

    return "completed", {}


# Anthropic's message *input* schema for replayed server-side blocks is stricter
# than the *output* schema: it accepts only the keys below per block type. The
# streamed response (rehydrated via ``stream.get_final_message()``) attaches
# output-only fields such as ``citations`` to these blocks — and because the SDK
# block models use ``extra="allow"``, ``model_dump`` carries them through. Replaying
# them verbatim trips a 400 (e.g. ``server_tool_use.citations: Extra inputs are not
# permitted``), which aborts the turn before it can be persisted — so the next turn
# also "forgets" the conversation. Whitelist the input-permitted keys to avoid this.
_SERVER_BLOCK_INPUT_KEYS: Dict[str, tuple[str, ...]] = {
    "server_tool_use": ("type", "id", "name", "input"),
    "web_search_tool_result": ("type", "tool_use_id", "content"),
    "web_fetch_tool_result": ("type", "tool_use_id", "content"),
}


def sanitize_server_block_for_replay(block: Any) -> Optional[Dict[str, Any]]:
    """Dump a server-side content block to the Anthropic message *input* schema.

    Returns a dict containing only the keys Anthropic accepts when the block is
    replayed inside an assistant message, or ``None`` if the block cannot be
    serialized. Drops output-only fields (notably ``citations``) that would
    otherwise raise ``Extra inputs are not permitted``.
    """
    try:
        dumped = block.model_dump(mode="json")
    except Exception:
        logger.debug("Skipping non-serializable content block: %s", getattr(block, "type", "?"))
        return None
    allowed = _SERVER_BLOCK_INPUT_KEYS.get(dumped.get("type"))
    if allowed is None:
        # Unknown server block type — fall back to verbatim dump (best effort).
        return dumped
    return {k: dumped[k] for k in allowed if k in dumped}


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
        # Enables the server-side ``web_fetch_20250910`` tool on the native path.
        # Harmless when web_fetch isn't requested — Anthropic ignores betas for
        # tools absent from the request. (web_search_20250305 is GA, no beta.)
        "web-fetch-2025-09-10",
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

    def supports_builtin_web_search(self) -> bool:
        # Anthropic web_search_20250305 (GA) runs server-side via the native path,
        # which is available for both OAuth and API-key auth. Prefer it over Tavily.
        return True

    def supports_builtin_web_fetch(self) -> bool:
        # web_fetch is served by the LOCAL httpx backend (``web_tool.web_fetch``)
        # rather than Anthropic's server-side ``web_fetch_20250910``. The hosted
        # tool emits ``server_tool_use`` / ``web_fetch_tool_result`` blocks that
        # must be replayed verbatim next turn; the rehydrated blocks carry an
        # output-only ``citations`` field the message *input* schema rejects
        # (400 ``server_tool_use.citations: Extra inputs are not permitted``),
        # which aborts the turn before it can be persisted. The local backend
        # returns a plain ``tool_result`` instead, sidestepping that entirely.
        # (web_search stays server-side — see ``sanitize_server_block_for_replay``.)
        return False

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
            # OAuth client already carries the OAuth + web-fetch betas in its
            # default headers; beta.messages is required for that path.
            return self.async_anthropic_client.beta.messages.stream(**kwargs)
        # API-key path: web_search_20250305 is GA, but web_fetch_20250910 is beta
        # and must go through beta.messages with the web-fetch beta header (the
        # API-key client carries no default betas).
        tools = kwargs.get("tools") or []
        if any(isinstance(t, dict) and t.get("type") == "web_fetch_20250910" for t in tools):
            extra = dict(kwargs.pop("extra_headers", {}) or {})
            existing_beta = extra.get("anthropic-beta")
            extra["anthropic-beta"] = ",".join(filter(None, [existing_beta, "web-fetch-2025-09-10"]))
            return self.async_anthropic_client.beta.messages.stream(extra_headers=extra, **kwargs)
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

                # Append Anthropic server-side web tools when the node opts in.
                # These run server-side (the results come back as
                # web_search_tool_result / web_fetch_tool_result content blocks),
                # so they are NOT registered in func_tool_map.
                builtin_web = kwargs.get("builtin_web_tools") or {}
                if builtin_web.get("web_search"):
                    tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 5})
                if builtin_web.get("web_fetch") and self.supports_builtin_web_fetch():
                    tools.append({"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 5})

                # Re-apply cache control so the ephemeral marker lands on the final
                # tool, whatever its source (MCP / func / server-side web).
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
                # Index of this turn's first NEW message in ``messages`` (prior
                # session history was just replayed above, so anything before
                # this index is already persisted). The end-of-turn persistence
                # stores ``messages[turn_start_index:]`` — the full structured
                # delta including every ``tool_use`` / ``tool_result`` — instead
                # of only the final text.
                turn_start_index = len(messages)
                messages.append(user_turn_message)
                tool_call_cache = {}
                sql_contexts = []
                final_content = ""
                # Guards the failure-path persistence below: the success path
                # sets this once it has committed the turn, so the ``except``
                # handler never double-writes a turn it already saved.
                turn_persisted = False
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

                    # Track server-side web tool calls emitted in real time during
                    # streaming so the post-stream scan (and the non-streaming
                    # fallback) doesn't double-emit them.
                    emitted_server_ids: set = set()
                    server_tool_names: dict = {}
                    server_tool_queries: dict = {}
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
                                    bs_type = getattr(block_start, "type", None) if block_start is not None else None
                                    if bs_type == "text":
                                        thinking_stream_id = f"thinking_stream_{uuid.uuid4().hex[:8]}"
                                        thinking_accumulated = ""
                                    elif bs_type == "server_tool_use":
                                        # Server-side web tool call — surface it in real
                                        # time (interleaved where it happens) instead of
                                        # only after the stream completes.
                                        sid = getattr(block_start, "id", None) or f"srv_{uuid.uuid4().hex[:8]}"
                                        sname = getattr(block_start, "name", None) or "web_search"
                                        server_tool_names[sid] = sname
                                        emitted_server_ids.add(sid)
                                        s_input = getattr(block_start, "input", {}) or {}
                                        if isinstance(s_input, dict):
                                            server_tool_queries[sid] = str(
                                                s_input.get("query") or s_input.get("url") or ""
                                            )
                                        yield ActionHistory(
                                            action_id=sid,
                                            role=ActionRole.TOOL,
                                            messages=f"Tool call: {sname}",
                                            action_type=sname,
                                            input={"function_name": sname, "arguments": s_input},
                                            output={},
                                            status=ActionStatus.PROCESSING,
                                        )
                                    elif bs_type in ("web_search_tool_result", "web_fetch_tool_result"):
                                        tid = getattr(block_start, "tool_use_id", None)
                                        sname = server_tool_names.get(tid, "web_search")
                                        summary, canonical = summarize_web_tool_result(
                                            block_start, query=server_tool_queries.get(tid, "")
                                        )
                                        complete = ActionHistory(
                                            action_id=f"complete_{tid}",
                                            role=ActionRole.TOOL,
                                            messages=f"Tool call: {sname}",
                                            action_type=sname,
                                            input={"function_name": sname, "arguments": {}},
                                            output={
                                                "success": True,
                                                "raw_output": {"success": True, "result": canonical},
                                                "summary": summary,
                                                "status_message": summary,
                                            },
                                            status=ActionStatus.SUCCESS,
                                        )
                                        complete.end_time = datetime.now()
                                        yield complete
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

                    # Surface server-side tool calls (Anthropic web_search /
                    # web_fetch) as TOOL ActionHistory. The streaming path already
                    # emits these in real time (tracked in ``emitted_server_ids``);
                    # this post-stream pass only covers the non-streaming fallback,
                    # so it skips ids already emitted to avoid duplicates.
                    server_results = {
                        getattr(b, "tool_use_id", None): b
                        for b in message
                        if getattr(b, "type", None) in ("web_search_tool_result", "web_fetch_tool_result")
                    }
                    for block in message:
                        if getattr(block, "type", None) != "server_tool_use":
                            continue
                        if block.id in emitted_server_ids:
                            continue
                        block_input = getattr(block, "input", {}) or {}
                        s_args = json.dumps(block_input, ensure_ascii=False)[:80]
                        q = (
                            block_input.get("query") or block_input.get("url") or ""
                            if isinstance(block_input, dict)
                            else ""
                        )
                        summary, canonical = summarize_web_tool_result(server_results.get(block.id), query=q)
                        start_action = ActionHistory(
                            action_id=block.id,
                            role=ActionRole.TOOL,
                            messages=f"Tool call: {block.name}('{s_args}...')",
                            action_type=block.name,
                            input={"function_name": block.name, "arguments": block_input},
                            output={},
                            status=ActionStatus.PROCESSING,
                        )
                        if action_history_manager is not None:
                            action_history_manager.add_action(start_action)
                        yield start_action
                        complete_action = ActionHistory(
                            action_id=f"complete_{block.id}",
                            role=ActionRole.TOOL,
                            messages=f"Tool call: {block.name}('{s_args}...')",
                            action_type=block.name,
                            input={"function_name": block.name, "arguments": block_input},
                            output={
                                "success": True,
                                "raw_output": {"success": True, "result": canonical},
                                "summary": summary,
                                "status_message": summary,
                            },
                            status=ActionStatus.SUCCESS,
                        )
                        complete_action.end_time = datetime.now()
                        if action_history_manager is not None:
                            action_history_manager.add_action(complete_action)
                        yield complete_action

                    # If no tool calls, conversation is complete
                    if not any(block.type == "tool_use" for block in message):
                        final_content = "\n".join([block.text for block in message if block.type == "text"])
                        # Persist the final assistant message — including any
                        # sanitized ``server_tool_use`` / ``web_search_tool_result``
                        # blocks from a ``web_search``-only turn — before breaking.
                        # The completion branch exits before the assistant-content
                        # append below, so without this the native web-tool replay
                        # history would be reduced to plain ``final_content``.
                        final_replay_content: List[Dict[str, Any]] = []
                        for block in message:
                            if block.type == "text":
                                final_replay_content.append({"type": "text", "text": block.text})
                            else:
                                sanitized = sanitize_server_block_for_replay(block)
                                if sanitized is not None:
                                    final_replay_content.append(sanitized)
                        if final_replay_content:
                            messages.append({"role": "assistant", "content": final_replay_content})
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
                        else:
                            # Preserve server-side blocks (server_tool_use /
                            # web_search_tool_result / web_fetch_tool_result) so a mixed
                            # turn (server web tool + local tool_use) keeps a valid
                            # assistant message when replayed to Anthropic next turn.
                            # Sanitize to the input schema to drop output-only fields
                            # (e.g. ``citations``) the API rejects on replay.
                            sanitized = sanitize_server_block_for_replay(block)
                            if sanitized is not None:
                                content.append(sanitized)

                    if content:
                        messages.append({"role": "assistant", "content": content})

                    for block in tool_use_blocks:
                        if block.id in tool_call_cache:
                            sql_result = tool_call_cache[block.id].content[0].text
                            # Only seed SQL context from successful read-only
                            # execute_sql calls. Parse the tool payload to detect
                            # failure instead of a brittle ``"Error" in text``
                            # substring check, which both misses structured
                            # ``{"success": 0, "error": ...}`` failures and can
                            # reject valid result text that merely contains "Error".
                            sql_query = block.input.get("query") or block.input.get("sql", "")
                            tool_failed = False
                            try:
                                parsed_result = json.loads(sql_result) if isinstance(sql_result, str) else sql_result
                            except (TypeError, json.JSONDecodeError):
                                parsed_result = None
                            if isinstance(parsed_result, dict):
                                tool_failed = parsed_result.get("success") == 0 or bool(parsed_result.get("error"))
                            elif isinstance(sql_result, str):
                                tool_failed = sql_result.lstrip().lower().startswith("error")
                            if (
                                not tool_failed
                                and block.name == "execute_sql"
                                and parse_sql_type(sql_query, "")
                                in (SQLType.SELECT, SQLType.METADATA_SHOW, SQLType.EXPLAIN)
                            ):
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
                        # Persist the FULL structured turn — every assistant
                        # ``tool_use`` block and its matching ``tool_result``
                        # reply — not just ``final_content``. ``messages`` from
                        # ``turn_start_index`` on is exactly the sequence
                        # Anthropic accepted during the loop, so replaying it on
                        # resume is safe and keeps the tool-calling history
                        # intact. Storing only the final text previously dropped
                        # every tool call/result, leaving resumed turns with
                        # broken history (and tool calls leaking into text).
                        turn_items = self._replay_safe_turn_items(
                            messages[turn_start_index:], user_turn_message, final_content
                        )
                        await session.add_items(turn_items)
                        turn_persisted = True
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
            await self._persist_failed_turn(locals(), e)
            self._diagnose_oauth_401(e)
            raise
        except Exception as e:
            await self._persist_failed_turn(locals(), e)
            if is_ssl_cert_verification_error(e):
                raise DatusException(ErrorCode.MODEL_SSL_CERT_ERROR) from e
            logger.error(f"Error in _generate_with_mcp_stream: {str(e)}")
            raise

    async def _persist_failed_turn(self, frame_locals: Dict[str, Any], error: Exception) -> None:
        """Best-effort save of an interrupted turn so the next turn isn't amnesiac.

        When the native loop raises mid-turn (e.g. a 400 on replay, a network
        drop, or an interrupt), the normal end-of-loop persistence never runs, so
        neither the user message nor any partial reply reaches the session — and
        the user's *next* turn "forgets" what was just asked. This commits the
        user message plus a placeholder/partial assistant message (kept non-empty
        and role-alternating so Anthropic accepts the replay) before the original
        error propagates. It must never mask that error: all failures here are
        swallowed.
        """
        session = frame_locals.get("session")
        if session is None or frame_locals.get("turn_persisted"):
            return
        user_turn_message = frame_locals.get("user_turn_message")
        if not user_turn_message:
            return
        try:
            partial = (frame_locals.get("final_content") or "").strip()
            if not partial:
                # On a mid-stream failure ``final_content`` is still empty, but
                # the partial text the user already saw lives in
                # ``thinking_accumulated`` — prefer it over the placeholder.
                partial = (frame_locals.get("thinking_accumulated") or "").strip()
            assistant_text = partial or f"[Turn interrupted before completion: {type(error).__name__}]"
            # Preserve any completed tool_use/tool_result rounds from this turn
            # too, not just a placeholder — ``_replay_safe_turn_items`` drops a
            # trailing dangling ``tool_use`` (interrupted before its result) and
            # appends ``assistant_text`` so the history ends on an assistant
            # message Anthropic will accept on replay.
            messages = frame_locals.get("messages")
            turn_start_index = frame_locals.get("turn_start_index")
            if isinstance(messages, list) and isinstance(turn_start_index, int):
                turn_items = self._replay_safe_turn_items(
                    messages[turn_start_index:], user_turn_message, assistant_text
                )
            else:
                turn_items = [
                    user_turn_message,
                    {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
                ]
            await session.add_items(turn_items)
            logger.info("Persisted interrupted native Claude turn so history survives the failure.")
        except Exception as persist_err:
            logger.warning(f"Failed to persist interrupted native Claude turn: {persist_err}")

    @staticmethod
    def _replay_safe_turn_items(
        turn_items: List[Dict[str, Any]],
        user_turn_message: Dict[str, Any],
        fallback_text: str,
    ) -> List[Dict[str, Any]]:
        """Trim one native-loop turn to a replay-safe item list for the session.

        ``turn_items`` is the slice of the native loop's ``messages`` added
        during a single user turn: the user message, then alternating assistant
        (``text`` + ``tool_use``) and user (``tool_result``) messages, ending —
        on the success path — with a clean assistant text message.

        Anthropic rejects replay of (a) an assistant ``tool_use`` block with no
        matching ``tool_result`` and (b) a history that ends on a
        user/``tool_result`` message (it would collide with the next turn's
        user message). This drops any trailing *dangling* ``tool_use`` (only the
        tail can dangle — the loop appends each ``tool_result`` immediately after
        its ``tool_use``) and guarantees the sequence ends on an assistant
        message, appending a non-empty ``fallback_text`` placeholder when needed
        so the turn is never lost. Completed tool rounds are preserved verbatim.
        """
        items = [m for m in turn_items if isinstance(m, dict)]

        def _tool_use_ids(msg: Dict[str, Any]) -> List[Any]:
            content = msg.get("content")
            if not isinstance(content, list):
                return []
            return [b.get("id") for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]

        answered = set()
        for m in items:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        answered.add(b.get("tool_use_id"))

        # Drop a trailing assistant turn whose tool calls were never answered
        # (interrupted between emitting the tool_use and recording its result).
        while items:
            last = items[-1]
            if last.get("role") == "assistant" and any(tid not in answered for tid in _tool_use_ids(last)):
                items.pop()
                continue
            break

        if not items:
            items = [user_turn_message]
        if items[-1].get("role") != "assistant":
            items.append({"role": "assistant", "content": [{"type": "text", "text": fallback_text or "[empty turn]"}]})
        return items

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
        # Route to the native Anthropic path for (a) OAuth subscription tokens
        # (LiteLLM sends x-api-key which is incompatible with Bearer auth), or
        # (b) whenever vendor-native web tools are requested — server tools can
        # only be expressed as raw tool dicts on the native Messages API, not via
        # the agents-SDK LiteLLM tool list. Non-web calls (e.g. compaction
        # summarization, which passes no builtin_web_tools) stay on LiteLLM.
        _builtin_web_tools = kwargs.get("builtin_web_tools") or {}
        _want_builtin_web = bool(
            (_builtin_web_tools.get("web_search") and self.supports_builtin_web_search())
            or (_builtin_web_tools.get("web_fetch") and self.supports_builtin_web_fetch())
        )
        if (self.use_native_api and self._is_oauth_token) or _want_builtin_web:
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
