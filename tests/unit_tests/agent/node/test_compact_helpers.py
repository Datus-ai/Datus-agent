# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Targeted unit tests for AgenticNode compact helpers.

These cover the dispatch / trigger / cutoff / history-dump methods that the
end-to-end ``compact()`` tests in test_compact_minor exercise indirectly.
Splitting them out keeps each test focused on a single branch — the
``_decide_compact_mode`` priority order, the ``_user_turn_count_from_session``
gate (reads session items so the resume case is handled), and the user-turn
cutoff resolver.
"""

import json
from typing import AsyncGenerator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.compact_archive import ToolArchive
from datus.configuration.agent_config import CompactConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.token_usage import TokenUsage


class _Node(AgenticNode):
    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        yield  # pragma: no cover

    def get_node_name(self) -> str:
        return "test_chat"


def _build_node(tmp_path):
    with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
        node = _Node.__new__(_Node)
    node.agent_config = None
    node._agent_config_ref = None
    node._pinned_model = None
    node._node_model_name = None
    node.session_id = "sid_test"
    node.actions = []
    node.running_turn_usage = None
    node._compact_cfg = CompactConfig()
    node._compacted_until = 0
    node._archive = ToolArchive(project_name="proj", session_id="sid_test", base_dir=tmp_path / "data")
    node._compact_lock = None
    node._session = None
    return node


class TestEnsureCompactState:
    """Lazy attr init for test harnesses that bypass __init__."""

    def test_populates_defaults_when_missing(self):
        with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
            node = _Node.__new__(_Node)
        node.agent_config = None
        # NONE of the compact attrs exist.
        assert not hasattr(node, "_compact_cfg")
        node._ensure_compact_state()
        assert isinstance(node._compact_cfg, CompactConfig)
        assert node._compacted_until == 0
        assert node._archive is None
        assert node._compact_lock is None

    def test_preserves_existing_values(self):
        with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
            node = _Node.__new__(_Node)
        node.agent_config = None
        node._compact_cfg = CompactConfig()
        node._compacted_until = 7
        node._archive = None
        node._compact_lock = None
        # Should NOT clobber existing values.
        node._ensure_compact_state()
        assert node._compacted_until == 7


class TestDecideCompactMode:
    @pytest.mark.asyncio
    async def test_returns_noop_when_both_disabled(self, tmp_path):
        node = _build_node(tmp_path)
        node._compact_cfg.major.enabled = False
        node._compact_cfg.minor.enabled = False
        assert await node._decide_compact_mode() == "noop"

    @pytest.mark.asyncio
    async def test_picks_major_when_token_ratio_exceeds_threshold(self, tmp_path):
        node = _build_node(tmp_path)
        with patch.object(_Node, "_history_token_ratio_sync", return_value=0.95):
            assert await node._decide_compact_mode() == "major"

    @pytest.mark.asyncio
    async def test_picks_minor_when_user_turns_exceed_keep_window(self, tmp_path):
        node = _build_node(tmp_path)
        node._compact_cfg.minor.keep_recent_user_turns = 2
        with patch.object(_Node, "_history_token_ratio_sync", return_value=0.1):
            with patch.object(_Node, "_user_turn_count_from_session", new=AsyncMock(return_value=5)):
                assert await node._decide_compact_mode() == "minor"

    @pytest.mark.asyncio
    async def test_returns_noop_when_user_turn_count_below_window(self, tmp_path):
        node = _build_node(tmp_path)
        node._compact_cfg.minor.keep_recent_user_turns = 5
        with patch.object(_Node, "_history_token_ratio_sync", return_value=0.1):
            with patch.object(_Node, "_user_turn_count_from_session", new=AsyncMock(return_value=3)):
                assert await node._decide_compact_mode() == "noop"

    @pytest.mark.asyncio
    async def test_returns_noop_when_user_turn_count_equals_window(self, tmp_path):
        """``count == N`` means nothing is older than the kept window — no-op."""
        node = _build_node(tmp_path)
        node._compact_cfg.minor.keep_recent_user_turns = 4
        with patch.object(_Node, "_history_token_ratio_sync", return_value=0.1):
            with patch.object(_Node, "_user_turn_count_from_session", new=AsyncMock(return_value=4)):
                assert await node._decide_compact_mode() == "noop"

    @pytest.mark.asyncio
    async def test_ratio_exception_falls_back_to_zero(self, tmp_path):
        """A buggy ratio calc must not crash the dispatcher — minor / noop
        branches still get a fair shot.
        """
        node = _build_node(tmp_path)
        with patch.object(_Node, "_history_token_ratio_sync", side_effect=RuntimeError("boom")):
            with patch.object(_Node, "_user_turn_count_from_session", new=AsyncMock(return_value=0)):
                # Ratio defaults to 0.0 → below major threshold → noop.
                assert await node._decide_compact_mode() == "noop"

    @pytest.mark.asyncio
    async def test_mid_turn_skips_minor(self, tmp_path):
        """Within a turn (``mid_turn=True``) the user-turn-count minor gate is
        skipped — that count cannot change between tool calls, so minor is left
        to the turn-start (``pre_user_turn``) check."""
        node = _build_node(tmp_path)
        node._compact_cfg.minor.keep_recent_user_turns = 2
        with patch.object(_Node, "_history_token_ratio_sync", return_value=0.1):
            with patch.object(_Node, "_user_turn_count_from_session", new=AsyncMock(return_value=99)):
                # mid-turn: skip minor even though the count is well over the window
                assert await node._decide_compact_mode(mid_turn=True) == "noop"
                # turn start (default): the same state does pick minor
                assert await node._decide_compact_mode() == "minor"

    @pytest.mark.asyncio
    async def test_mid_turn_still_allows_major(self, tmp_path):
        """major still fires mid-turn — its token-ratio gate genuinely changes
        as the turn progresses."""
        node = _build_node(tmp_path)
        with patch.object(_Node, "_history_token_ratio_sync", return_value=0.95):
            assert await node._decide_compact_mode(mid_turn=True) == "major"


class TestUserTurnCountFromSession:
    """``_user_turn_count_from_session`` counts ``role == "user"`` items in
    the active session — same source as ``_resolve_user_turn_cutoff`` so the
    dispatcher and the worker agree on the eligibility window even after a
    resume that left ``self.actions`` empty.
    """

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_session(self, tmp_path):
        node = _build_node(tmp_path)
        node._session = None
        node.session_id = ""
        assert await node._user_turn_count_from_session() == 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_get_items_fails(self, tmp_path):
        node = _build_node(tmp_path)
        node._session = MagicMock()
        node._session.get_items = AsyncMock(side_effect=RuntimeError("db locked"))
        assert await node._user_turn_count_from_session() == 0

    @pytest.mark.asyncio
    async def test_counts_user_role_items(self, tmp_path):
        node = _build_node(tmp_path)
        node._session = MagicMock()
        node._session.get_items = AsyncMock(
            return_value=[
                {"type": "message", "role": "user", "content": "q1"},
                {"type": "function_call", "name": "f"},
                {"type": "message", "role": "user", "content": "q2"},
                {"type": "message", "role": "assistant", "content": "a"},
                "non-dict-item",  # robustness: skipped
                {"type": "message", "role": "user", "content": "q3"},
            ]
        )
        assert await node._user_turn_count_from_session() == 3

    @pytest.mark.asyncio
    async def test_materializes_session_when_id_set(self, tmp_path):
        """Resume path: ``session_id`` is set but ``_session`` is None until
        ``_get_or_create_session`` runs. The dispatcher must trigger the
        materialization itself so the gate doesn't miss user turns held in
        the on-disk session before the first execute call.
        """
        node = _build_node(tmp_path)
        node._session = None
        node.session_id = "sid_resume"
        materialized = MagicMock()
        materialized.get_items = AsyncMock(return_value=[{"role": "user", "content": "q"}])

        def _create():
            node._session = materialized

        node._get_or_create_session = MagicMock(side_effect=_create)
        assert await node._user_turn_count_from_session() == 1
        node._get_or_create_session.assert_called_once()


class TestHistoryTokenRatioSync:
    def test_zero_when_no_context_length(self, tmp_path):
        node = _build_node(tmp_path)
        assert node._history_token_ratio_sync() == 0.0

    def test_zero_when_no_actions(self, tmp_path):
        node = _build_node(tmp_path)
        node._pinned_model = MagicMock()
        node._pinned_model.context_length.return_value = 1000
        assert node._history_token_ratio_sync() == 0.0

    def test_reads_last_call_input_tokens(self, tmp_path):
        node = _build_node(tmp_path)
        node._pinned_model = MagicMock()
        node._pinned_model.context_length.return_value = 1000
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat",
            messages="ok",
            input_data={},
            output_data={"usage": {"last_call_input_tokens": 700, "input_tokens": 500}},
            status=ActionStatus.SUCCESS,
        )
        node.actions.append(action)
        # 700/1000 = 0.7 — prefer last_call_input_tokens over input_tokens.
        assert node._history_token_ratio_sync() == 0.7

    def test_falls_back_to_input_tokens(self, tmp_path):
        node = _build_node(tmp_path)
        node._pinned_model = MagicMock()
        node._pinned_model.context_length.return_value = 1000
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat",
            messages="ok",
            input_data={},
            output_data={"usage": {"input_tokens": 400}},
            status=ActionStatus.SUCCESS,
        )
        node.actions.append(action)
        assert node._history_token_ratio_sync() == 0.4

    def test_stops_at_user_action_boundary(self, tmp_path):
        """The scan walks back from the latest action and stops at the
        previous user action — so an old usage record before the current
        turn never bleeds into the ratio.
        """
        node = _build_node(tmp_path)
        node._pinned_model = MagicMock()
        node._pinned_model.context_length.return_value = 1000
        old_assistant = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat",
            messages="old",
            input_data={},
            output_data={"usage": {"input_tokens": 999}},
            status=ActionStatus.SUCCESS,
        )
        user = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type="chat",
            messages="hi",
            input_data={},
            output_data={},
            status=ActionStatus.SUCCESS,
        )
        node.actions.extend([old_assistant, user])
        assert node._history_token_ratio_sync() == 0.0

    def test_prefers_running_turn_usage_over_actions(self, tmp_path):
        """Mid-turn: the live ``running_turn_usage`` snapshot wins over the
        (stale, prior-turn) ``self.actions`` scan. This is what lets a major
        compact fire mid-turn rather than one turn late.
        """
        node = _build_node(tmp_path)
        node._pinned_model = MagicMock()
        node._pinned_model.context_length.return_value = 1000
        # A stale action that would yield 0.999 if the actions scan ran.
        node.actions.append(
            ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="chat",
                messages="stale",
                input_data={},
                output_data={"usage": {"last_call_input_tokens": 999}},
                status=ActionStatus.SUCCESS,
            )
        )
        node.running_turn_usage = TokenUsage(session_total_tokens=300, context_length=1000)
        # 300/1000 from the live snapshot, NOT 999/1000 from actions.
        assert node._history_token_ratio_sync() == 0.3

    def test_running_turn_usage_context_length_falls_back_to_model(self, tmp_path):
        """A snapshot without its own ``context_length`` uses the node model's."""
        node = _build_node(tmp_path)
        node._pinned_model = MagicMock()
        node._pinned_model.context_length.return_value = 2000
        node.running_turn_usage = TokenUsage(session_total_tokens=500, context_length=0)
        assert node._history_token_ratio_sync() == 0.25

    def test_running_turn_usage_falls_back_to_input_tokens(self, tmp_path):
        """When ``session_total_tokens`` is 0, the snapshot's ``input_tokens``
        is used as the live occupancy signal.
        """
        node = _build_node(tmp_path)
        node._pinned_model = MagicMock()
        node._pinned_model.context_length.return_value = 1000
        node.running_turn_usage = TokenUsage(session_total_tokens=0, input_tokens=600, context_length=1000)
        assert node._history_token_ratio_sync() == 0.6

    def test_empty_running_turn_usage_does_not_mask_actions_fallback(self, tmp_path):
        """A zero-token snapshot must not short-circuit the actions fallback —
        the scan still surfaces the most recent usable usage record.
        """
        node = _build_node(tmp_path)
        node._pinned_model = MagicMock()
        node._pinned_model.context_length.return_value = 1000
        node.running_turn_usage = TokenUsage(session_total_tokens=0, input_tokens=0, context_length=1000)
        node.actions.append(
            ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="chat",
                messages="ok",
                input_data={},
                output_data={"usage": {"last_call_input_tokens": 700}},
                status=ActionStatus.SUCCESS,
            )
        )
        assert node._history_token_ratio_sync() == 0.7


class TestResolveUserTurnCutoff:
    """The cutoff is the item-index that separates the eligible-to-archive
    region from the kept window. It is anchored on the position of the
    ``keep_recent_user_turns``-th most-recent ``role == "user"`` message.
    """

    def _u(self):
        return {"type": "message", "role": "user", "content": "q"}

    def _fc(self):
        return {"type": "function_call", "name": "f", "arguments": "x", "call_id": "c"}

    def _fco(self):
        return {"type": "function_call_output", "output": "y", "call_id": "c"}

    def test_returns_minus_one_when_too_few_user_turns(self, tmp_path):
        """With ``keep_recent_user_turns=3`` and 2 user messages there is
        nothing older than the kept window — no-op signal is ``-1``.
        """
        node = _build_node(tmp_path)
        node._compact_cfg.minor.keep_recent_user_turns = 3
        items = [self._u(), self._fc(), self._fco(), self._u(), self._fc(), self._fco()]
        assert node._resolve_user_turn_cutoff(items) == -1

    def test_returns_minus_one_when_exactly_n_user_turns(self, tmp_path):
        """``len(user_indices) == N`` is still "nothing older than the kept
        window" — strictly greater is required to produce a cutoff.
        """
        node = _build_node(tmp_path)
        node._compact_cfg.minor.keep_recent_user_turns = 2
        items = [self._u(), self._fc(), self._u(), self._fc()]  # 2 user turns
        assert node._resolve_user_turn_cutoff(items) == -1

    def test_returns_nth_user_turn_index_when_enough(self, tmp_path):
        """With 4 user turns and ``N=2``, the cutoff is the index of the
        2nd-most-recent user message — i.e. the 3rd user message overall.
        Items before that index belong to user turns 0 and 1, both stale.
        """
        node = _build_node(tmp_path)
        node._compact_cfg.minor.keep_recent_user_turns = 2
        items = []
        for _ in range(4):
            items.append(self._u())
            items.extend([self._fc(), self._fco()])
        # User-message positions: 0, 3, 6, 9 → cutoff = items[-2] of those = 6.
        cutoff = node._resolve_user_turn_cutoff(items)
        assert cutoff == 6
        # Sanity: items[cutoff:] still contains exactly N user messages.
        assert sum(1 for it in items[cutoff:] if it.get("role") == "user") == 2

    def test_returns_minus_one_when_n_is_zero_or_negative(self, tmp_path):
        """A misconfigured ``N <= 0`` is treated as "disabled" — no items
        ever pass the cutoff, which matches the safe-fail intent.
        """
        node = _build_node(tmp_path)
        node._compact_cfg.minor.keep_recent_user_turns = 0
        assert node._resolve_user_turn_cutoff([self._u(), self._fc(), self._fco()]) == -1

    def test_robust_against_non_dict_items(self, tmp_path):
        """Items must be dicts to be considered for the user-role check;
        defensive coding because old session schemas occasionally carried
        non-dict entries.
        """
        node = _build_node(tmp_path)
        node._compact_cfg.minor.keep_recent_user_turns = 1
        items = ["not a dict", None, self._u(), self._fc(), self._u(), self._fc()]
        # 2 user turns present → cutoff is the latest user index (4).
        assert node._resolve_user_turn_cutoff(items) == 4


class TestDumpSessionHistoryJsonl:
    @pytest.mark.asyncio
    async def test_writes_one_item_per_line(self, tmp_path):
        node = _build_node(tmp_path)
        items = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        node._session = MagicMock()
        node._session.get_items = AsyncMock(return_value=items)
        path = await node._dump_session_history_jsonl()
        assert path is not None and path.exists()
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == items[0]
        assert json.loads(lines[1]) == items[1]

    @pytest.mark.asyncio
    async def test_returns_none_when_no_session(self, tmp_path):
        node = _build_node(tmp_path)
        node._session = None
        assert await node._dump_session_history_jsonl() is None

    @pytest.mark.asyncio
    async def test_returns_none_when_get_items_fails(self, tmp_path):
        node = _build_node(tmp_path)
        node._session = MagicMock()
        node._session.get_items = AsyncMock(side_effect=RuntimeError("db broke"))
        # Should swallow the error rather than break the whole major pass.
        assert await node._dump_session_history_jsonl() is None


class TestCompactDisplayInjection:
    """``compact()`` injects compact_progress/compact_summary display actions
    for EVERY major path (hook_major, pre_user_turn, cli_manual) — so the CLI
    feedback is driven from one place."""

    @pytest.mark.asyncio
    async def test_major_injects_progress_then_summary(self, tmp_path):
        node = _build_node(tmp_path)
        node.action_bus = MagicMock()
        node._major_compact = AsyncMock(
            return_value={
                "mode": "major",
                "success": True,
                "summary": "S",
                "summary_token": 7,
                "history_jsonl": "/h",
            }
        )
        result = await node.compact(mode="major", reason="test")
        assert result["success"]
        assert node.action_bus.put.call_count == 2
        progress = node.action_bus.put.call_args_list[0].args[0]
        summary = node.action_bus.put.call_args_list[1].args[0]
        assert progress.action_type == "compact_progress"
        assert summary.action_type == "compact_summary"
        assert summary.action_id == progress.action_id  # shared id
        assert summary.output["summary"] == "S"
        assert summary.output["summary_token"] == 7
        assert summary.output["history_jsonl"] == "/h"

    @pytest.mark.asyncio
    async def test_pre_user_turn_auto_major_injects_display(self, tmp_path):
        """The turn-start ``_auto_compact`` (mode=auto, reason=pre_user_turn)
        path must also display when it resolves to major."""
        node = _build_node(tmp_path)
        node.action_bus = MagicMock()
        node._decide_compact_mode = AsyncMock(return_value="major")
        node._major_compact = AsyncMock(return_value={"mode": "major", "success": True, "summary": "S"})
        ran = await node._auto_compact()
        assert ran is True
        types = [c.args[0].action_type for c in node.action_bus.put.call_args_list]
        assert types == ["compact_progress", "compact_summary"]

    @pytest.mark.asyncio
    async def test_major_failure_emits_terminal_with_empty_summary(self, tmp_path):
        node = _build_node(tmp_path)
        node.action_bus = MagicMock()
        node._major_compact = AsyncMock(return_value={"mode": "major", "success": False})
        result = await node.compact(mode="major", reason="test")
        assert result["success"] is False
        # progress + a terminal summary with empty payload, so the renderer can
        # clear the pinned hint without drawing a panel.
        assert node.action_bus.put.call_count == 2
        progress, terminal = (c.args[0] for c in node.action_bus.put.call_args_list)
        assert progress.action_type == "compact_progress"
        assert terminal.action_type == "compact_summary"
        assert terminal.output["summary"] == ""

    @pytest.mark.asyncio
    async def test_progress_injected_before_blocking_summary_call(self, tmp_path):
        node = _build_node(tmp_path)
        node.action_bus = MagicMock()
        seen = {}

        async def _major(*, reason):
            seen["puts_before"] = node.action_bus.put.call_count
            return {"mode": "major", "success": True, "summary": "S"}

        node._major_compact = AsyncMock(side_effect=_major)
        await node.compact(mode="major", reason="test")
        assert seen["puts_before"] == 1  # progress already out before the blocking call

    @pytest.mark.asyncio
    async def test_minor_does_not_inject_display(self, tmp_path):
        node = _build_node(tmp_path)
        node.action_bus = MagicMock()
        node._minor_compact = AsyncMock(return_value={"mode": "minor", "success": True})
        await node.compact(mode="minor", reason="test")
        node.action_bus.put.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_does_not_inject_display(self, tmp_path):
        node = _build_node(tmp_path)
        node.action_bus = MagicMock()
        node._compact_cfg.major.enabled = False
        node._compact_cfg.minor.enabled = False
        await node.compact(mode="auto", reason="test")
        node.action_bus.put.assert_not_called()
