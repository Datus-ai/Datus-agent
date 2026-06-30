# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""Capability-method contract tests for vendor-native web tools.

These methods drive the node-layer backend choice (local function tool vs.
vendor-hosted tool), so the truth table is the critical contract. We exercise
the methods as unbound functions against a stand-in ``self`` to avoid the OAuth
/ SDK setup a full model instance needs. The actual hosted-tool injection is
exercised by nightly tests against real APIs.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from datus.models.base import LLMBaseModel
from datus.models.claude_model import ClaudeModel
from datus.models.codex_model import CodexModel
from datus.models.openai_model import OpenAIModel

_NOW = datetime(2026, 1, 1, 0, 0, 0)


def test_base_defaults_false():
    s = Mock()
    assert LLMBaseModel.supports_builtin_web_search(s) is False
    assert LLMBaseModel.supports_builtin_web_fetch(s) is False


def test_codex_search_yes_only_on_official_endpoint():
    # OpenAI Responses exposes hosted web_search but no hosted fetch — and the
    # hosted tool is only honored by the official ChatGPT Codex host. A custom
    # base_url (third-party relay) must fall back to the local backend.
    official = Mock()
    official._base_url = "https://chatgpt.com/backend-api/codex"
    assert CodexModel.supports_builtin_web_search(official) is True
    assert CodexModel.supports_builtin_web_fetch(official) is False

    proxy = Mock()
    proxy._base_url = "https://my-proxy.example.com/codex"
    assert CodexModel.supports_builtin_web_search(proxy) is False
    assert CodexModel.supports_builtin_web_fetch(proxy) is False


def test_openai_search_yes_only_on_official_endpoint():
    # The hosted web_search tool is only accepted by the Responses API, which
    # ``get_agents_sdk_model`` routes to ONLY for the canonical OpenAI host.
    # supports_builtin_web_search must mirror that route exactly, otherwise the
    # node injects WebSearchTool into a LiteLLM ChatCompletions request and the
    # converter raises ``Hosted tools are not supported with the ChatCompletions
    # API``.
    official = Mock()
    official.litellm_adapter._is_official_openai.return_value = True
    assert OpenAIModel.supports_builtin_web_search(official) is True
    assert OpenAIModel.supports_builtin_web_fetch(official) is False

    # provider=openai but a custom base_url (vLLM / OpenRouter relay / Coding
    # Plan proxy) → LiteLLM ChatCompletions path → hosted tool must be OFF.
    proxy = Mock()
    proxy.litellm_adapter._is_official_openai.return_value = False
    assert OpenAIModel.supports_builtin_web_search(proxy) is False
    assert OpenAIModel.supports_builtin_web_fetch(proxy) is False


def test_claude_search_server_fetch_local():
    # web_search runs server-side (Anthropic web_search_20250305, GA) — but only
    # on the official api.anthropic.com host. Third-party Claude-compatible
    # proxies (e.g. kimi_coding at api.kimi.com/coding) share type:claude yet
    # return malformed hosted-tool blocks, so they fall back to the local
    # backend. web_fetch is ALWAYS served by the LOCAL httpx backend instead of
    # the hosted web_fetch_20250910: the hosted tool emits server_tool_use blocks
    # whose rehydrated form carries an output-only ``citations`` field the
    # message input schema rejects on replay (400), so we keep fetch local.
    official = Mock()
    official._get_base_url.return_value = "https://api.anthropic.com"
    assert ClaudeModel.supports_builtin_web_search(official) is True
    assert ClaudeModel.supports_builtin_web_fetch(official) is False

    proxy = Mock()
    proxy._get_base_url.return_value = "https://api.kimi.com/coding/"
    assert ClaudeModel.supports_builtin_web_search(proxy) is False
    assert ClaudeModel.supports_builtin_web_fetch(proxy) is False


def test_describe_hosted_tool_item_web_search_dict():
    from datus.models.openai_compatible import describe_hosted_tool_item

    name, call_id, args = describe_hosted_tool_item(
        {"type": "web_search_call", "id": "ws_1", "action": {"query": "duckdb news"}}
    )
    assert name == "web_search"
    assert call_id == "ws_1"
    assert "duckdb news" in args


def test_describe_hosted_tool_item_object():
    from openai.types.responses import ResponseFunctionWebSearch
    from openai.types.responses.response_function_web_search import ActionSearch

    from datus.models.openai_compatible import describe_hosted_tool_item

    item = ResponseFunctionWebSearch(
        id="ws_2", type="web_search_call", status="completed", action=ActionSearch(type="search", query="q")
    )
    name, call_id, args = describe_hosted_tool_item(item)
    assert name == "web_search"
    assert call_id == "ws_2"


def test_summarize_web_tool_result_extracts_titles():
    from types import SimpleNamespace

    from datus.models.claude_model import summarize_web_tool_result

    block = SimpleNamespace(
        content=[
            SimpleNamespace(title="DuckDB 1.5 released", url="https://duckdb.org/x", page_age="2 days ago"),
            SimpleNamespace(title="Release notes", url="https://github.com/duckdb", page_age=None),
        ]
    )
    summary, canonical = summarize_web_tool_result(block, query="duckdb release")
    # Compact summary lists result titles.
    assert summary.startswith("2 web results:")
    assert "DuckDB 1.5 released" in summary
    # Canonical schema carries the full structured results.
    assert canonical["query"] == "duckdb release"
    assert canonical["result_count"] == 2
    assert canonical["results"][0]["title"] == "DuckDB 1.5 released"
    assert canonical["results"][0]["url"] == "https://duckdb.org/x"
    assert canonical["results"][0]["age"] == "2 days ago"


def test_summarize_web_tool_result_web_fetch_extracts_content():
    from types import SimpleNamespace

    from datus.models.claude_model import summarize_web_tool_result

    # web_fetch result: content is a single document block (not a list).
    block = SimpleNamespace(
        content=SimpleNamespace(
            url="https://duckdb.org/notes",
            content=SimpleNamespace(
                title="Release Notes",
                source=SimpleNamespace(type="text", data="Full page body text here."),
            ),
        )
    )
    summary, canonical = summarize_web_tool_result(block)
    assert canonical["url"] == "https://duckdb.org/notes"
    assert canonical["content"] == "Full page body text here."
    assert canonical["char_count"] == len("Full page body text here.")
    assert "chars" in summary


def test_summarize_web_tool_result_handles_missing():
    from datus.models.claude_model import summarize_web_tool_result

    assert summarize_web_tool_result(None) == ("completed", {})


def test_sanitize_server_block_strips_citations_from_server_tool_use():
    """A rehydrated server_tool_use block carries an output-only ``citations``
    field (extra='allow'); replaying it verbatim trips a 400. The sanitizer must
    keep only the input-schema keys."""
    from anthropic.types import ServerToolUseBlock

    from datus.models.claude_model import sanitize_server_block_for_replay

    block = ServerToolUseBlock(id="srv_1", name="web_search", input={"query": "eth"}, type="server_tool_use")
    # Simulate the field the streaming accumulator attaches.
    block.__pydantic_extra__["citations"] = None
    assert "citations" in block.model_dump(mode="json")  # precondition

    out = sanitize_server_block_for_replay(block)
    assert out == {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "eth"}}
    assert "citations" not in out


def test_sanitize_server_block_whitelists_web_search_tool_result():
    """web_search_tool_result keeps only type/tool_use_id/content on replay."""
    from types import SimpleNamespace

    from datus.models.claude_model import sanitize_server_block_for_replay

    block = SimpleNamespace(
        model_dump=lambda mode="json": {
            "type": "web_search_tool_result",
            "tool_use_id": "srv_1",
            "content": [{"type": "web_search_result", "title": "t", "url": "https://x"}],
            "cache_control": {"type": "ephemeral"},  # extra output-only key
        }
    )
    out = sanitize_server_block_for_replay(block)
    assert set(out) == {"type", "tool_use_id", "content"}
    assert out["tool_use_id"] == "srv_1"


def test_sanitize_server_block_unknown_type_falls_back_to_verbatim():
    """An unrecognized block type is dumped verbatim (best effort, no data loss)."""
    from types import SimpleNamespace

    from datus.models.claude_model import sanitize_server_block_for_replay

    dumped = {"type": "some_future_block", "foo": "bar"}
    block = SimpleNamespace(model_dump=lambda mode="json": dumped)
    assert sanitize_server_block_for_replay(block) == dumped


def test_sanitize_server_block_returns_none_when_unserializable():
    """A block whose model_dump raises is skipped (returns None) rather than
    aborting the whole assistant-message rebuild."""
    from types import SimpleNamespace

    from datus.models.claude_model import sanitize_server_block_for_replay

    def _boom(mode: str = "json") -> dict[str, object]:
        raise ValueError("not serializable")

    block = SimpleNamespace(type="server_tool_use", model_dump=_boom)
    assert sanitize_server_block_for_replay(block) is None


def test_codex_hosted_completion_attaches_citations():
    """Deferred hosted web_search completion carries the canonical result built
    from the assistant message's url_citation annotations."""
    from types import SimpleNamespace

    from datus.models.codex_model import CodexModel

    pending = [
        {
            "call_id": "ws_1",
            "tool_name": "web_search",
            "arguments": "{}",
            "args_str": "",
            "query": "duckdb latest",
            "sources": [{"title": "", "url": "https://fallback", "snippet": ""}],
        }
    ]
    result = SimpleNamespace(
        to_input_list=lambda: [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "...",
                        "annotations": [{"type": "url_citation", "title": "DuckDB", "url": "https://duckdb.org"}],
                    }
                ],
            }
        ]
    )
    actions = CodexModel._build_hosted_search_completions(pending, result)
    assert len(actions) == 1
    done = actions[0]
    assert done.action_id == "complete_ws_1"
    assert done.action_type == "web_search"
    canonical = done.output["raw_output"]["result"]
    assert canonical["query"] == "duckdb latest"
    # url_citations win over the action.sources fallback.
    assert canonical["results"] == [{"title": "DuckDB", "url": "https://duckdb.org", "snippet": "", "age": None}]
    assert done.output["summary"] == "1 web result: DuckDB"


def test_codex_hosted_completion_falls_back_to_sources():
    """With no citations, the call's own action.sources URLs become the results."""
    from types import SimpleNamespace

    from datus.models.codex_model import CodexModel

    pending = [
        {
            "call_id": "ws_2",
            "tool_name": "web_search",
            "arguments": "{}",
            "args_str": "",
            "query": "q",
            "sources": [{"title": "", "url": "https://only-source", "snippet": ""}],
        }
    ]
    result = SimpleNamespace(to_input_list=lambda: [])  # no assistant citations
    actions = CodexModel._build_hosted_search_completions(pending, result)
    canonical = actions[0].output["raw_output"]["result"]
    assert canonical["results"] == [{"title": "", "url": "https://only-source", "snippet": "", "age": None}]


def test_describe_hosted_tool_item_ignores_function_call():
    from datus.models.openai_compatible import describe_hosted_tool_item

    # Regular function tools are handled by the normal path, not this helper.
    assert (
        describe_hosted_tool_item({"type": "function_call", "name": "read_query", "call_id": "c1", "arguments": "{}"})
        is None
    )


@pytest.mark.asyncio
async def test_openai_flush_pending_hosted_searches_uses_citations():
    """OpenAI deferred hosted web_search completion builds the canonical result
    from the assistant message's url_citation annotations."""
    from types import SimpleNamespace

    from datus.models.openai_compatible import OpenAICompatibleModel

    pending = [
        {
            "call_id": "ws_1",
            "tool_name": "web_search",
            "arguments": "{}",
            "args_str": "",
            "query": "duckdb latest",
            "sources": [{"title": "", "url": "https://fallback", "snippet": ""}],
            "start_time": _NOW,
        }
    ]
    result = SimpleNamespace(
        to_input_list=lambda: [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [{"type": "url_citation", "title": "DuckDB", "url": "https://duckdb.org"}],
                    }
                ],
            }
        ]
    )
    actions = [a async for a in OpenAICompatibleModel._flush_pending_hosted_searches(pending, result, None)]
    assert len(actions) == 1
    done = actions[0]
    assert done.action_id == "complete_ws_1"
    canonical = done.output["raw_output"]["result"]
    assert canonical["query"] == "duckdb latest"
    assert canonical["results"] == [{"title": "DuckDB", "url": "https://duckdb.org", "snippet": "", "age": None}]


@pytest.mark.asyncio
async def test_openai_flush_pending_hosted_searches_falls_back_to_sources():
    """With no assistant citations, the call's own action.sources become results."""
    from types import SimpleNamespace

    from datus.models.openai_compatible import OpenAICompatibleModel

    pending = [
        {
            "call_id": "ws_2",
            "tool_name": "web_search",
            "arguments": "{}",
            "args_str": "",
            "query": "q",
            "sources": [{"title": "", "url": "https://only-source", "snippet": ""}],
            "start_time": _NOW,
        }
    ]
    result = SimpleNamespace(to_input_list=lambda: [])
    actions = [a async for a in OpenAICompatibleModel._flush_pending_hosted_searches(pending, result, None)]
    canonical = actions[0].output["raw_output"]["result"]
    assert canonical["results"] == [{"title": "", "url": "https://only-source", "snippet": "", "age": None}]


def test_describe_hosted_tool_item_ignores_non_web_hosted_calls():
    from datus.models.openai_compatible import describe_hosted_tool_item

    # Only web_search_call is normalized through the hosted-search path. Other
    # hosted *_call items must NOT be routed here (they would be force-fit into
    # the web_search canonical schema by the deferred completion logic).
    for itype in ("file_search_call", "image_generation_call", "code_interpreter_call", "computer_call"):
        assert describe_hosted_tool_item({"type": itype, "id": "x"}) is None


def _user_msg(text: str = "explore eth tables") -> dict[str, object]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


@pytest.mark.asyncio
async def test_persist_failed_turn_saves_user_and_placeholder():
    """A turn that raises before normal persistence still commits the user
    message (plus a non-empty placeholder assistant) so the next turn isn't
    amnesiac. Role-alternating + non-empty text keeps the replay API-valid."""
    from unittest.mock import AsyncMock

    session = AsyncMock()
    frame_locals = {
        "session": session,
        "turn_persisted": False,
        "user_turn_message": _user_msg(),
        "final_content": "",
    }
    await ClaudeModel._persist_failed_turn(Mock(), frame_locals, RuntimeError("boom"))

    session.add_items.assert_awaited_once()
    items = session.add_items.await_args.args[0]
    assert [m["role"] for m in items] == ["user", "assistant"]
    # Placeholder assistant text is non-empty (Anthropic rejects empty text).
    assert items[1]["content"][0]["text"].strip()
    assert "RuntimeError" in items[1]["content"][0]["text"]


@pytest.mark.asyncio
async def test_persist_failed_turn_uses_partial_content():
    """When the turn produced partial assistant text before failing, that text
    is preserved rather than replaced by the error placeholder."""
    from unittest.mock import AsyncMock

    session = AsyncMock()
    frame_locals = {
        "session": session,
        "turn_persisted": False,
        "user_turn_message": _user_msg(),
        "final_content": "partial answer so far",
    }
    await ClaudeModel._persist_failed_turn(Mock(), frame_locals, RuntimeError("boom"))
    items = session.add_items.await_args.args[0]
    assert items[1]["content"][0]["text"] == "partial answer so far"


@pytest.mark.asyncio
async def test_persist_failed_turn_uses_thinking_accumulated_when_final_empty():
    """On a mid-stream failure ``final_content`` is empty but the visible text
    lives in ``thinking_accumulated`` — prefer it over the placeholder."""
    from unittest.mock import AsyncMock

    session = AsyncMock()
    frame_locals = {
        "session": session,
        "turn_persisted": False,
        "user_turn_message": _user_msg(),
        "final_content": "",
        "thinking_accumulated": "streamed-but-not-finalized",
    }
    await ClaudeModel._persist_failed_turn(Mock(), frame_locals, RuntimeError("boom"))
    items = session.add_items.await_args.args[0]
    assert items[1]["content"][0]["text"] == "streamed-but-not-finalized"


@pytest.mark.asyncio
async def test_persist_failed_turn_preserves_completed_tool_rounds():
    """When ``messages``/``turn_start_index`` are present, the interrupted turn is
    rebuilt via ``_replay_safe_turn_items`` so completed tool rounds survive."""
    from unittest.mock import AsyncMock

    user = _user_msg()
    session = AsyncMock()
    frame_locals = {
        "session": session,
        "turn_persisted": False,
        "user_turn_message": user,
        "final_content": "",
        "messages": [user, _tool_use_msg("t1"), _tool_result_msg("t1")],
        "turn_start_index": 0,
    }
    # ``_persist_failed_turn`` calls ``self._replay_safe_turn_items`` — wire the
    # real static method onto the stand-in self so the rebuild actually runs.
    fake_self = Mock()
    fake_self._replay_safe_turn_items = ClaudeModel._replay_safe_turn_items
    await ClaudeModel._persist_failed_turn(fake_self, frame_locals, RuntimeError("boom"))
    items = session.add_items.await_args.args[0]
    # Completed tool round preserved, ends on an appended assistant placeholder.
    assert _tool_use_msg("t1") in items
    assert items[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_persist_failed_turn_skips_when_already_persisted():
    """The success path already committed the turn; the failure path must not
    double-write."""
    from unittest.mock import AsyncMock

    session = AsyncMock()
    frame_locals = {
        "session": session,
        "turn_persisted": True,
        "user_turn_message": _user_msg(),
        "final_content": "done",
    }
    await ClaudeModel._persist_failed_turn(Mock(), frame_locals, RuntimeError("boom"))
    session.add_items.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_failed_turn_noop_without_session_or_user_message():
    """No session (e.g. one-shot call) or a failure before the user message was
    built → nothing to persist, and no crash."""
    from unittest.mock import AsyncMock

    # No session.
    await ClaudeModel._persist_failed_turn(Mock(), {"session": None}, RuntimeError("x"))

    # Session present but exception fired before user_turn_message existed.
    session = AsyncMock()
    await ClaudeModel._persist_failed_turn(Mock(), {"session": session, "turn_persisted": False}, RuntimeError("x"))
    session.add_items.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_failed_turn_swallows_persist_errors():
    """The fallback must never mask the original error: a failing add_items is
    logged and swallowed, not raised."""
    from unittest.mock import AsyncMock

    session = AsyncMock()
    session.add_items.side_effect = RuntimeError("db locked")
    frame_locals = {
        "session": session,
        "turn_persisted": False,
        "user_turn_message": _user_msg(),
        "final_content": "",
    }
    # Must complete without re-raising (would otherwise mask the real error)...
    await ClaudeModel._persist_failed_turn(Mock(), frame_locals, ValueError("original"))
    # ...while still having attempted the persist exactly once.
    session.add_items.assert_awaited_once()


def _tool_use_msg(tool_id="toolu_1"):
    return {"role": "assistant", "content": [{"type": "tool_use", "id": tool_id, "name": "execute_sql", "input": {}}]}


def _tool_result_msg(tool_id="toolu_1"):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}]}


def test_replay_safe_turn_items_preserves_completed_rounds():
    """A finished turn (user → tool_use → tool_result → assistant text) is kept
    verbatim and ends on the assistant message."""
    user = _user_msg()
    final = {"role": "assistant", "content": [{"type": "text", "text": "done"}]}
    turn = [user, _tool_use_msg(), _tool_result_msg(), final]
    items = ClaudeModel._replay_safe_turn_items(turn, user, "fallback")
    assert items == turn
    assert items[-1]["role"] == "assistant"


def test_replay_safe_turn_items_drops_dangling_tool_use():
    """A trailing assistant tool_use with no matching tool_result is dropped, and
    a non-empty fallback assistant message is appended."""
    user = _user_msg()
    turn = [user, _tool_use_msg("toolu_x")]
    items = ClaudeModel._replay_safe_turn_items(turn, user, "interrupted")
    assert len(items) == 2
    assert items[0] == user
    assert items[1] == {"role": "assistant", "content": [{"type": "text", "text": "interrupted"}]}


def test_replay_safe_turn_items_appends_fallback_when_ends_on_user():
    """A history that ends on a user/tool_result message gets an assistant
    fallback appended so it never collides with the next turn's user message."""
    user = _user_msg()
    answered_use = _tool_use_msg("toolu_1")
    result = _tool_result_msg("toolu_1")
    items = ClaudeModel._replay_safe_turn_items([user, answered_use, result], user, "placeholder")
    assert items[-1] == {"role": "assistant", "content": [{"type": "text", "text": "placeholder"}]}
    # The answered tool_use round is preserved (not dropped).
    assert answered_use in items


def test_replay_safe_turn_items_empty_falls_back_to_user_plus_assistant():
    """When everything is dropped, the user message plus a fallback assistant
    message are restored so the turn is never lost."""
    user = _user_msg()
    items = ClaudeModel._replay_safe_turn_items([_tool_use_msg("toolu_x")], user, "")
    assert items[0] == user
    assert items[-1] == {"role": "assistant", "content": [{"type": "text", "text": "[empty turn]"}]}


def test_replay_safe_turn_items_skips_non_dict_entries():
    """Non-dict entries in the raw turn slice are filtered out."""
    user = _user_msg()
    final = {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
    items = ClaudeModel._replay_safe_turn_items([user, "garbage", None, final], user, "fb")
    assert items == [user, final]


def test_replay_safe_turn_items_tolerates_non_list_assistant_content():
    """A trailing assistant message whose content is a plain string (not a block
    list) is treated as having no dangling tool_use and kept as-is."""
    user = _user_msg()
    final = {"role": "assistant", "content": "plain string reply"}
    items = ClaudeModel._replay_safe_turn_items([user, final], user, "fb")
    assert items == [user, final]
