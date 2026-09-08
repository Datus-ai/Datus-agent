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
from pathlib import Path
from typing import AsyncGenerator, Optional
from unittest.mock import ANY, AsyncMock, MagicMock, PropertyMock, patch

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.compact_archive import ToolArchive
from datus.agent.node.context_rewriter import estimate_items_tokens
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


class TestDecideMidTurnCompactMode:
    """Per-model-call dispatcher used by ``MidTurnCompactor`` inside a turn.

    The ratio it receives already folds in the tool outputs appended since the
    last model call and the output headroom, so the decision is a pure
    threshold comparison with two independent kill switches.
    """

    def test_major_at_or_above_major_threshold(self, tmp_path):
        node = _build_node(tmp_path)
        assert node._decide_mid_turn_compact_mode(0.9) == "major"
        assert node._decide_mid_turn_compact_mode(0.97) == "major"

    def test_minor_in_the_band_between_the_two_thresholds(self, tmp_path):
        node = _build_node(tmp_path)
        assert node._decide_mid_turn_compact_mode(0.75) == "minor"
        assert node._decide_mid_turn_compact_mode(0.89) == "minor"

    def test_noop_below_minor_threshold(self, tmp_path):
        node = _build_node(tmp_path)
        assert node._decide_mid_turn_compact_mode(0.5) == "noop"
        assert node._decide_mid_turn_compact_mode(0.0) == "noop"

    def test_major_mid_turn_switch_falls_through_to_minor(self, tmp_path):
        """With ``major.mid_turn_enabled=False`` a 95% ratio still gets the
        cheap archive stage — the switch disables the LLM summary only."""
        node = _build_node(tmp_path)
        node._compact_cfg.major.mid_turn_enabled = False
        assert node._decide_mid_turn_compact_mode(0.95) == "minor"

    def test_minor_mid_turn_switch_leaves_major_alone(self, tmp_path):
        node = _build_node(tmp_path)
        node._compact_cfg.minor.mid_turn_enabled = False
        assert node._decide_mid_turn_compact_mode(0.8) == "noop"
        assert node._decide_mid_turn_compact_mode(0.95) == "major"

    def test_disabled_passes_still_respect_enabled_flags(self, tmp_path):
        node = _build_node(tmp_path)
        node._compact_cfg.major.enabled = False
        node._compact_cfg.minor.enabled = False
        assert node._decide_mid_turn_compact_mode(0.99) == "noop"

    def test_custom_minor_threshold_is_honoured(self, tmp_path):
        node = _build_node(tmp_path)
        node._compact_cfg.minor.mid_turn_token_threshold = 0.5
        assert node._decide_mid_turn_compact_mode(0.5) == "minor"
        assert node._decide_mid_turn_compact_mode(0.49) == "noop"


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
        node.running_turn_usage = TokenUsage(session_total_tokens=700, context_length=1000)
        # 700/1000 = 0.7 — prefer last_call_input_tokens over input_tokens.
        assert node._history_token_ratio_sync() == 0.7

    def test_cumulative_input_is_not_an_occupancy_fallback(self, tmp_path):
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
        assert node._history_token_ratio_sync() == 0

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

    def test_running_spend_does_not_replace_unknown_occupancy(self, tmp_path):
        """When ``session_total_tokens`` is 0, the snapshot's ``input_tokens``
        is used as the live occupancy signal.
        """
        node = _build_node(tmp_path)
        node._pinned_model = MagicMock()
        node._pinned_model.context_length.return_value = 1000
        node.running_turn_usage = TokenUsage(session_total_tokens=0, input_tokens=600, context_length=1000)
        assert node._history_token_ratio_sync() == 0

    def test_unknown_snapshot_does_not_revive_stale_actions(self, tmp_path):
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
        assert node._history_token_ratio_sync() == 0


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


class TestCompactMidTurn:
    """``compact_mid_turn``: two-stage rewrite of the transcript a running turn
    is about to send. Driven by ``MidTurnCompactor`` before each model call."""

    @staticmethod
    def _node(tmp_path, *, context_length=1000, max_tokens=100, keep_recent=2):
        node = _build_node(tmp_path)
        model = MagicMock()
        model.context_length.return_value = context_length
        model.max_tokens.return_value = max_tokens
        model.summarize_items = AsyncMock(
            return_value={"content": "## Summary\nprogress", "usage": {"output_tokens": 12}}
        )
        node._pinned_model = model
        node._session = MagicMock(replace_items=AsyncMock())
        node._session_manager = MagicMock()
        node._session_manager.checkpoint_turn.return_value = "CHECKPOINT"
        # ``compact_mid_turn`` falls back to the node's system prompt when the
        # rewriter passes none; the bypassed-__init__ node cannot render it.
        node._get_system_prompt = lambda: "SYS"
        node.action_bus = MagicMock()
        node.mid_turn_rewrite_checkpoint = None
        node._compact_cfg.minor.archive_threshold = 100
        # Short preview so the 300-char fixture outputs really shrink when archived
        # (``_build_node`` pre-creates the archive, so rebuild it with the new width).
        node._compact_cfg.minor.archive_preview_chars = 50
        node._archive = ToolArchive(
            project_name="proj", session_id="sid_test", base_dir=tmp_path / "data", preview_chars=50
        )
        node._compact_cfg.minor.keep_recent_tool_results = keep_recent
        return node

    @staticmethod
    def _items(n_outputs=4, size=1200):  # realistic tool outputs: above the archive threshold
        items = [{"role": "user", "content": "analyse refunds"}]
        for i in range(n_outputs):
            items.append({"type": "function_call", "call_id": f"c{i}", "name": "execute_sql", "arguments": "{}"})
            items.append({"type": "function_call_output", "call_id": f"c{i}", "output": f"{i}:" + "r" * size})
        return items

    @pytest.mark.asyncio
    async def test_noop_below_the_minor_threshold(self, tmp_path):
        node = self._node(tmp_path)
        items = self._items()
        result = await node.compact_mid_turn(items, item_format="responses", base_tokens=100, tail_start=len(items))
        assert result["mode"] == "noop" and result["success"] is True
        assert result["items"] is items
        node._session.replace_items.assert_not_awaited()
        node._pinned_model.summarize_items.assert_not_awaited()
        node.action_bus.put.assert_not_called()

    @pytest.mark.asyncio
    async def test_output_reserve_and_tail_estimate_count_towards_the_ratio(self, tmp_path):
        """base 500 + reserve 100 = 0.38 of a 1600 window (noop); with the
        ~670-token tail of appended tool rounds it lands in the archive band.
        After archiving, the ratio is the measured total minus what the markers
        saved (not a fresh estimate of the items alone), so it drops but stays
        well above zero and below the major threshold."""
        node = self._node(tmp_path, keep_recent=10, context_length=1600)
        items = self._items(2)
        noop = await node.compact_mid_turn(items, item_format="responses", base_tokens=500, tail_start=len(items))
        assert noop["mode"] == "noop"
        # Same base, but the tool rounds are not yet counted in ``base_tokens``.
        decided = await node.compact_mid_turn(items, item_format="responses", base_tokens=500, tail_start=1)
        assert decided["mode"] == "noop"  # in the band, but nothing archivable (keep_recent=10)
        node._pinned_model.summarize_items.assert_not_awaited()
        node._compact_cfg.minor.keep_recent_tool_results = 0
        minor = await node.compact_mid_turn(items, item_format="responses", base_tokens=500, tail_start=1)
        assert minor["mode"] == "minor"

    @pytest.mark.asyncio
    async def test_minor_stage_archives_older_outputs_and_persists_the_view(self, tmp_path):
        node = self._node(tmp_path)
        node.running_turn_usage = TokenUsage(requests=1, session_total_tokens=700, context_length=1000)
        items = self._items(4)
        result = await node.compact_mid_turn(items, item_format="responses", base_tokens=700, tail_start=len(items))

        assert result["mode"] == "minor" and result["success"] is True
        assert result["archived_count"] == 2
        view = result["items"]
        outputs = [it for it in view if it.get("type") == "function_call_output"]
        assert outputs[0]["output"].startswith("[DATUS_ARCHIVED]")
        assert outputs[1]["output"].startswith("[DATUS_ARCHIVED]")
        assert outputs[2] is items[6] and outputs[3] is items[8]
        # No LLM call, no display panel for the archive stage.
        node._pinned_model.summarize_items.assert_not_awaited()
        node.action_bus.put.assert_not_called()
        # Persisted exactly what is returned.
        node._session.replace_items.assert_awaited_once_with(view)
        # Live occupancy and rollback boundary follow the rewrite.
        assert node.running_turn_usage.session_total_tokens == 0
        node._session_manager.checkpoint_turn.assert_called_once_with("sid_test")
        assert node.mid_turn_rewrite_checkpoint == "CHECKPOINT"
        node._session_manager.delete_system_prompt_snapshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_major_stage_summarizes_the_archived_view(self, tmp_path):
        node = self._node(tmp_path)
        # The measured input (2400 of 1000) is far over the window; archiving the two
        # oldest 3000-char outputs saves ~1400 tokens, which still leaves it over.
        node.running_turn_usage = TokenUsage(requests=1, session_total_tokens=2400, context_length=1000)
        items = self._items(4, size=3000)
        request = items[0]
        result = await node.compact_mid_turn(
            items,
            item_format="responses",
            base_tokens=2400,
            tail_start=len(items),
            instruction="SYS",
            turn_request=request,
        )

        assert result["mode"] == "major" and result["success"] is True
        assert result["archived_count"] == 2
        assert result["summary"] == "## Summary\nprogress"
        assert result["summary_token"] == 12
        # The summary call saw the *archived* view, never the raw items or the session.
        summarized = node._pinned_model.summarize_items.await_args.args[0]
        assert summarized is not items
        assert any(
            it.get("type") == "function_call_output" and it["output"].startswith("[DATUS_ARCHIVED]")
            for it in summarized
        )
        kwargs = node._pinned_model.summarize_items.await_args.kwargs
        assert kwargs["item_format"] == "responses"
        assert kwargs["instruction"] == "SYS"
        assert "in the middle of a task" in kwargs["prompt"]
        # Rewritten view: this turn's request, the summary, the resume nudge.
        view = result["items"]
        assert view[0] is request
        assert view[1]["role"] == "assistant" and "## Summary" in view[1]["content"][0]["text"]
        # The JSONL recovery pointer is appended host-side to the summary.
        assert view[1]["content"][0]["text"].endswith(f"`read_file({result['history_jsonl']!r})`")
        assert view[2]["role"] == "user" and view[2]["content"][0]["text"].startswith("[DATUS_COMPACT_RESUME]")
        assert len(view) == 3
        node._session.replace_items.assert_awaited_once_with(view)
        # Full pre-compaction transcript dumped for recovery, one item per line.
        assert result["history_jsonl"]
        dumped = Path(result["history_jsonl"]).read_text(encoding="utf-8").splitlines()
        assert len(dumped) == len(items)
        assert json.loads(dumped[2])["output"] == items[2]["output"]  # original, un-archived text
        # Bookkeeping shared with the turn-start major pass.
        assert node._compacted_until == 0
        node._session_manager.delete_system_prompt_snapshot.assert_called_once_with("sid_test")
        types = [c.args[0].action_type for c in node.action_bus.put.call_args_list]
        assert types == ["compact_progress", "compact_summary"]
        assert node.action_bus.put.call_args_list[1].args[0].output["summary"] == "## Summary\nprogress"

    @pytest.mark.asyncio
    async def test_summary_failure_keeps_the_archive_result_and_reports_the_error(self, tmp_path):
        node = self._node(tmp_path)
        node._pinned_model.summarize_items = AsyncMock(side_effect=RuntimeError("llm down"))
        items = self._items(4, size=3000)
        result = await node.compact_mid_turn(items, item_format="responses", base_tokens=2400, tail_start=len(items))

        assert result["mode"] == "minor" and result["success"] is True
        assert result["major_error"] == "llm down"
        assert result["archived_count"] == 2
        node._session.replace_items.assert_awaited_once_with(result["items"])
        assert not any(it.get("role") == "assistant" for it in result["items"])
        # The pinned progress hint is still cleared by a terminal action.
        types = [c.args[0].action_type for c in node.action_bus.put.call_args_list]
        assert types == ["compact_progress", "compact_summary"]
        assert node.action_bus.put.call_args_list[1].args[0].output["summary"] == ""

    @pytest.mark.asyncio
    async def test_summary_failure_with_nothing_archived_leaves_the_session_alone(self, tmp_path):
        node = self._node(tmp_path, keep_recent=10)
        node._pinned_model.summarize_items = AsyncMock(side_effect=RuntimeError("llm down"))
        items = self._items(4)
        result = await node.compact_mid_turn(items, item_format="responses", base_tokens=950, tail_start=len(items))

        assert result["mode"] == "noop" and result["success"] is True
        assert result["major_error"] == "llm down"
        assert result["items"] is items
        node._session.replace_items.assert_not_awaited()
        node._session.replace_items.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_persistence_failure_returns_the_original_items(self, tmp_path):
        node = self._node(tmp_path)
        node._session.replace_items = AsyncMock(side_effect=RuntimeError("disk"))
        items = self._items(4, size=3000)
        result = await node.compact_mid_turn(items, item_format="responses", base_tokens=950, tail_start=len(items))

        assert result["success"] is False
        assert result["items"] is items
        assert node.mid_turn_rewrite_checkpoint is None
        node._session_manager.delete_system_prompt_snapshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_context_window_is_noop(self, tmp_path):
        node = self._node(tmp_path, context_length=None)
        items = self._items()
        result = await node.compact_mid_turn(items, item_format="responses", base_tokens=10_000, tail_start=0)
        assert result["mode"] == "noop"
        node._pinned_model.summarize_items.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mid_turn_switches_disable_each_stage(self, tmp_path):
        node = self._node(tmp_path)
        node._compact_cfg.major.mid_turn_enabled = False
        node._compact_cfg.minor.mid_turn_enabled = False
        items = self._items(4, size=3000)
        result = await node.compact_mid_turn(items, item_format="responses", base_tokens=950, tail_start=len(items))
        assert result["mode"] == "noop"
        assert result["items"] is items

    @pytest.mark.asyncio
    async def test_anthropic_transcript_gets_anthropic_view(self, tmp_path):
        node = self._node(tmp_path, keep_recent=10)
        request = {"role": "user", "content": [{"type": "text", "text": "analyse refunds"}]}
        items = [
            request,
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "execute_sql", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r" * 300}]},
        ]
        result = await node.compact_mid_turn(
            items, item_format="anthropic", base_tokens=950, tail_start=len(items), turn_request=request
        )
        assert result["mode"] == "major"
        assert node._pinned_model.summarize_items.await_args.kwargs["item_format"] == "anthropic"
        view = result["items"]
        assert view[0] is request
        assert view[1] == {"role": "assistant", "content": [{"type": "text", "text": ANY}]}
        assert view[2]["role"] == "user" and view[2]["content"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_dump_writes_explicit_items(self, tmp_path):
        node = _build_node(tmp_path)
        items = self._items(2)
        path = await node._dump_session_history_jsonl(items)
        assert path.parent == node._archive.dir
        assert path.name.startswith("history_") and path.suffix == ".jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == items


class TestCompactMidTurnWithRealSession:
    @pytest.mark.asyncio
    async def test_session_equals_the_returned_view(self, tmp_path):
        """After the rewrite, SQLite holds exactly the list the model is sent."""
        from datus.models.sqlite_session import DatusSQLiteSession as AdvancedSQLiteSession

        session = AdvancedSQLiteSession(session_id="sid_test", db_path=str(tmp_path / "s.db"), create_tables=True)
        node = TestCompactMidTurn._node(tmp_path)
        node._session = session
        items = TestCompactMidTurn._items(4, size=3000)
        await session.add_items(items)

        result = await node.compact_mid_turn(
            items, item_format="responses", base_tokens=950, tail_start=1, instruction="SYS", turn_request=items[0]
        )
        assert result["mode"] == "major"
        stored = await session.get_items()
        assert stored == result["items"]
        assert not any(it.get("type") == "function_call" for it in stored)
        session.close()


class TestOccupancyAfterSessionRewrite:
    """Every compact mode shrinks the persisted history, so the occupancy meter
    has to follow it down.

    ``running_turn_usage`` is only refreshed by ``TokenUsageHook.on_llm_end``,
    and a compact's own summarization call runs with no hooks — so without an
    explicit reset the meter keeps describing the pre-compact context. That
    figure is what ``_history_token_ratio_sync`` reads, so
    ``_decide_compact_mode`` picks major again on the session that was just
    compacted, paying for an LLM summarization every turn. The reset lives in
    ``_replace_session_items``, the single choke point every mode rewrites
    through, so no mode can skip it.
    """

    WINDOW = 100_000
    PRE_COMPACT = 95_000
    TURN_OUTPUT = 2_000

    @classmethod
    def _node(cls, tmp_path):
        node = _build_node(tmp_path)
        model = MagicMock()
        model.context_length.return_value = cls.WINDOW
        model.summarize_items = AsyncMock(return_value={"content": "## Summary\nrecap", "usage": {"output_tokens": 40}})
        node._pinned_model = model
        node._session = MagicMock(replace_items=AsyncMock())
        node._session_manager = MagicMock()
        node._get_system_prompt = lambda: "SYS"
        node.running_turn_usage = TokenUsage(
            requests=3,
            input_tokens=cls.PRE_COMPACT,
            output_tokens=cls.TURN_OUTPUT,
            total_tokens=cls.PRE_COMPACT + cls.TURN_OUTPUT,
            session_total_tokens=cls.PRE_COMPACT,
            context_usage_ratio=cls.PRE_COMPACT / cls.WINDOW,
            context_length=cls.WINDOW,
        )
        return node

    @staticmethod
    def _history(n_turns=6, size=4000):
        items = []
        for i in range(n_turns):
            items.append({"role": "user", "content": f"q{i}"})
            items.append({"type": "function_call", "call_id": f"c{i}", "name": "execute_sql", "arguments": "{}"})
            items.append({"type": "function_call_output", "call_id": f"c{i}", "output": "r" * size})
        return items

    @pytest.mark.asyncio
    async def test_major_compact_drops_the_occupancy_to_the_rewritten_session(self, tmp_path):
        node = self._node(tmp_path)
        node._session.get_items = AsyncMock(return_value=self._history())

        result = await node.compact(mode="major", reason="cli_manual")

        assert result["success"] is True
        assert node.running_turn_usage.session_total_tokens == 0
        assert node.running_turn_usage.session_total_tokens < self.PRE_COMPACT
        assert node._history_token_ratio_sync() < node._compact_cfg.major.token_threshold

    @pytest.mark.asyncio
    async def test_minor_compact_drops_the_occupancy_to_the_archived_view(self, tmp_path):
        """The archive pass moves tool output to disk, which shrinks the
        session just as much — the meter must not keep over-reporting the
        pre-archive size for the rest of the session."""
        node = self._node(tmp_path)
        node._compact_cfg.minor.archive_threshold = 100
        node._compact_cfg.minor.keep_recent_user_turns = 2
        node._session.get_items = AsyncMock(return_value=self._history())

        result = await node.compact(mode="minor", reason="pre_user_turn")

        assert result["success"] is True and result["archived_count"] > 0
        assert node.running_turn_usage.session_total_tokens == 0
        assert node.running_turn_usage.session_total_tokens < self.PRE_COMPACT

    @pytest.mark.asyncio
    async def test_a_compacted_session_no_longer_decides_on_major(self, tmp_path):
        """The loop itself: the same dispatcher that chose major before the
        compact must not choose it again straight afterwards."""
        node = self._node(tmp_path)
        node._compact_cfg.minor.enabled = False  # isolate the token-threshold branch
        node._session.get_items = AsyncMock(return_value=self._history())

        assert await node._decide_compact_mode() == "major"
        result = await node.compact(mode="major", reason="pre_user_turn")

        assert result["success"] is True
        assert await node._decide_compact_mode() == "noop"

    @pytest.mark.asyncio
    async def test_failed_atomic_rewrite_preserves_the_measurement(self, tmp_path):
        """A failed transaction keeps both the history and its old measurement."""
        node = self._node(tmp_path)
        node._session.get_items = AsyncMock(return_value=self._history())
        node._session.replace_items = AsyncMock(side_effect=RuntimeError("disk full"))
        result = await node.compact(mode="major", reason="cli_manual")
        assert result["success"] is False
        assert node.running_turn_usage.session_total_tokens == self.PRE_COMPACT
        assert node._history_token_ratio_sync() >= node._compact_cfg.major.token_threshold

    @pytest.mark.asyncio
    async def test_post_compact_occupancy_is_persisted_for_the_next_process(self, tmp_path):
        """``restore_context_state`` re-hydrates this file on resume, so a
        pre-compact figure left on disk outlives the process and fires a major
        compact before the new one's first LLM call — where the in-memory
        mirror alone cannot save it."""
        from datus.storage.session_state import ContextState

        node = self._node(tmp_path)
        state_path = tmp_path / "agent_state.json"
        node._agent_state_file = lambda: state_path
        node._session.get_items = AsyncMock(return_value=self._history())

        result = await node.compact(mode="major", reason="cli_manual")
        assert result["success"] is True

        on_disk = ContextState.load(state_path)
        assert on_disk.last_call_input_tokens == 0
        # The in-memory mirror agrees, so a between-turns status-bar read and a
        # post-restart read cannot disagree.
        assert node._restored_context_used == on_disk.last_call_input_tokens

    @pytest.mark.asyncio
    async def test_the_snapshot_ratio_agrees_with_the_new_occupancy(self, tmp_path):
        """``context_usage_ratio`` describes the same quantity as
        ``session_total_tokens``; left stale, a reader that picks it (the
        natural field name for a compact gate) reopens the loop."""
        node = self._node(tmp_path)
        node._session.get_items = AsyncMock(return_value=self._history())

        result = await node.compact(mode="major", reason="cli_manual")
        assert result["success"] is True

        usage = node.running_turn_usage
        assert usage.context_usage_ratio < self.PRE_COMPACT / self.WINDOW
        assert usage.context_usage_ratio == pytest.approx(usage.session_total_tokens / usage.context_length, abs=1e-3)
        # A compact reclaims context, it does not refund the turn's spend.
        assert usage.total_tokens == self.PRE_COMPACT + self.TURN_OUTPUT
        assert usage.output_tokens == self.TURN_OUTPUT

    @pytest.mark.asyncio
    async def test_bookkeeping_failure_does_not_fail_a_successful_rewrite(self, tmp_path):
        """The refresh runs once the history is already rewritten, so nothing
        it touches may turn a successful compact into a reported failure.
        ``self.context_length`` resolves ``self.model``, which raises when
        model resolution fails."""
        node = self._node(tmp_path)
        node._session.get_items = AsyncMock(return_value=self._history())

        with patch.object(type(node), "context_length", new_callable=PropertyMock) as ctx:
            ctx.side_effect = RuntimeError("model resolution failed")
            result = await node.compact(mode="major", reason="cli_manual")

        assert result["success"] is True
        node._session.replace_items.assert_awaited_once()


class TestMeasuredContextContracts:
    @staticmethod
    def _node(tmp_path):
        from datus.models.session_manager import SessionManager

        node = TestOccupancyAfterSessionRewrite._node(tmp_path)
        node._session_manager = SessionManager(session_dir=str(tmp_path / "sessions"))
        node._session = node.session_manager.get_session(node.session_id)
        node._agent_state_file = lambda: tmp_path / "state.json"
        return node

    @pytest.mark.asyncio
    async def test_compact_invalidates_measured_usage_until_next_response(self, tmp_path):
        """A rewritten history has no measured occupancy, including after resume."""
        from datus.agent.node.token_usage_hook import TokenUsageHook
        from datus.cli.status_bar import StatusBarProvider

        node = self._node(tmp_path)
        await node._session.add_items(TestOccupancyAfterSessionRewrite._history())
        node.persist_context_state(95_000)
        result = await node.compact(mode="major")
        assert result["success"] is True
        assert node.running_turn_usage.session_total_tokens == 0
        assert node.running_turn_usage.total_tokens == 97_000
        bar = StatusBarProvider.__new__(StatusBarProvider)
        bar._current_node = lambda: node
        assert bar._resolve_context_used() == 0
        node.running_turn_usage = None
        node.restore_context_state()
        assert await node._count_session_tokens() == 0
        TokenUsageHook(node)._publish(
            {"requests": 1, "input_tokens": 1200, "last_call_input_tokens": 1200, "output_tokens": 20}
        )
        assert bar._resolve_context_used() == 1200
        assert node.running_turn_usage.context_usage_ratio == 0.012
        node._session.close()

    @pytest.mark.asyncio
    async def test_rewrite_updates_memory_without_a_state_file(self, tmp_path):
        """A missing state path must not leave pre-rewrite occupancy in memory."""
        node = self._node(tmp_path)
        node.persist_context_state(95_000)
        node._agent_state_file = lambda: None
        node.running_turn_usage = None
        await node._replace_session_items([{"role": "assistant", "content": "recap"}])
        assert node._history_token_ratio_sync() == 0
        node._session.close()

    @pytest.mark.asyncio
    async def test_rewrite_publishes_a_safe_rollback_boundary(self, tmp_path):
        """ESC keeps the rewritten history and removes only subsequent messages."""
        node = self._node(tmp_path)
        await node._session.add_items(TestOccupancyAfterSessionRewrite._history())
        old_checkpoint = node.session_manager.checkpoint_turn(node.session_id)
        result = await node.compact(mode="major", reason="pre_user_turn")
        assert result["success"] is True
        rewritten = await node._session.get_items()
        await node._session.add_items([{"role": "user", "content": "cancel this"}])
        checkpoint = getattr(node, "mid_turn_rewrite_checkpoint", None) or old_checkpoint
        node.session_manager.rollback_turn(node.session_id, checkpoint)
        assert await node._session.get_items() == rewritten
        node._session.close()

    def test_response_ratio_uses_last_call_instead_of_cumulative_spend(self, tmp_path):
        """Repeated model calls do not inflate context occupancy with billing totals."""
        from datus.agent.node.token_usage_hook import TokenUsageHook

        node = self._node(tmp_path)
        TokenUsageHook(node)._publish(
            {
                "requests": 3,
                "input_tokens": 180000,
                "last_call_input_tokens": 95000,
                "total_tokens": 185000,
                "context_usage_ratio": 1.85,
            }
        )
        assert node.running_turn_usage.context_usage_ratio == 0.95
        node._session.close()

    @pytest.mark.asyncio
    async def test_first_request_checks_and_rewrites_large_input(self, tmp_path):
        """The first call uses the same capacity check as later model calls."""
        from datus.agent.node.context_rewriter import MidTurnCompactor

        node = self._node(tmp_path)
        node._compact_cfg.minor.enabled = False
        items = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "x" * 400000},
            {"role": "user", "content": "current request"},
        ]
        await node._session.add_items(items)
        rewriter = MidTurnCompactor(node)
        view = await rewriter.rewrite_sdk_input(items)
        assert estimate_items_tokens(view) < estimate_items_tokens(items)
        assert any(item.get("content") == "current request" for item in view)
        node._session.close()


@pytest.mark.asyncio
async def test_cancelled_rewrite_finishes_bookkeeping_before_esc_rollback(tmp_path):
    """Cancellation cannot race a background commit with the old ESC checkpoint."""
    import asyncio
    import sqlite3
    import threading

    from datus.models.sqlite_session import DatusSQLiteSession

    entered = threading.Event()
    release = threading.Event()

    def pause_insert():
        entered.set()
        release.wait(timeout=2)
        return 0

    class BlockingSession(DatusSQLiteSession):
        def _get_connection(self):
            conn = super()._get_connection()
            conn.create_function("pause_insert", 0, pause_insert)
            return conn

    node = TestMeasuredContextContracts._node(tmp_path)
    await node._session.add_items([{"role": "user", "content": "old"}])
    original_session = node._session
    db_path = tmp_path / "sessions" / f"{node.session_id}.db"
    node._session = BlockingSession(session_id=node.session_id, db_path=db_path, create_tables=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TRIGGER pause_rewrite BEFORE INSERT ON message_structure BEGIN SELECT pause_insert(); END")
    rewritten = [{"role": "assistant", "content": "preserve this recap"}]
    task = asyncio.create_task(node._replace_session_items(rewritten))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert node._session_rewritten_this_turn is True
    assert node.get_context_usage().last_call_input_tokens == 0
    node.session_manager.rollback_turn(node.session_id, node.mid_turn_rewrite_checkpoint)
    assert await node._session.get_items() == rewritten
    node._session.close()
    original_session.close()
