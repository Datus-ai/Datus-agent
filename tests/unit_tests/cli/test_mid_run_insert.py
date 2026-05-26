# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the mid-run user-insert surface.

Covers the small-but-load-bearing CLI / TUI pieces that bridge the
in-memory :class:`PendingInputQueue` to the running agent: the
auto-continuation loop in :meth:`ChatCommands.execute_chat_command`, the
``_active_pending_input_queue`` accessor wired into ``DatusApp``, and the
``DatusApp`` queue-preview + Enter helpers that drive the pinned
preview box above the status bar.

CI tier: zero external dependencies — these tests construct lightweight
stubs (no real prompt_toolkit Application, no real CLI bootstrap)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datus.cli.execution_state import InterruptController, PendingInputQueue

# ---------------------------------------------------------------------------
# ChatCommands.execute_chat_command — auto-continuation loop
# ---------------------------------------------------------------------------


class _StubChatCommands:
    """Subset of :class:`ChatCommands` needed to drive the loop.

    We instantiate the real class is overkill here — the loop only reads
    ``self.current_node`` / ``self.console`` and calls ``_execute_chat``.
    A minimal stub keeps the test focused on the loop logic itself.
    """

    from datus.cli.chat_commands import ChatCommands as _Real

    execute_chat_command = _Real.execute_chat_command

    def __init__(self, queue_residuals):
        """``queue_residuals`` is a list of lists: residuals[i] is the
        contents of the queue *after* the i-th ``_execute_chat`` call."""
        self.console = MagicMock()
        self.current_node = SimpleNamespace(
            pending_input_queue=PendingInputQueue(),
            interrupt_controller=InterruptController(),
        )
        self._residuals = list(queue_residuals)
        self._calls = []

    def _execute_chat(self, message, plan_mode=False, subagent_name=None, interactive=True):
        self._calls.append(message)
        # Refresh the queue contents to simulate what would have been
        # pushed by the TUI / API while the previous call was running.
        next_residual = self._residuals.pop(0) if self._residuals else []
        # The production code drains the queue between iterations, so the
        # stub fully resets here and re-pushes whatever the scenario
        # wants the next iteration to see.
        self.current_node.pending_input_queue.clear()
        for text in next_residual:
            self.current_node.pending_input_queue.push(text)


class TestExecuteChatCommandAutoContinuation:
    """Behavioural contract for the final-turn auto-continuation loop."""

    def test_single_turn_when_queue_stays_empty(self):
        """Happy path: no mid-run insertions → exactly one ``_execute_chat`` call."""
        cc = _StubChatCommands(queue_residuals=[[]])

        cc.execute_chat_command("first prompt")

        assert cc._calls == ["first prompt"]

    def test_residual_after_run_triggers_one_extra_call(self):
        """Queue with content after the run → loop drains and runs again."""
        cc = _StubChatCommands(queue_residuals=[["also stats"], []])

        cc.execute_chat_command("first prompt")

        # First call uses the original message, second call uses the
        # drained residual — order preserved verbatim.
        assert cc._calls == ["first prompt", "also stats"]

    def test_multiple_residual_items_are_joined_with_double_newline(self):
        """Several queued lines collapse into a single follow-up message."""
        cc = _StubChatCommands(queue_residuals=[["one", "two", "three"], []])

        cc.execute_chat_command("orig")

        assert cc._calls == ["orig", "one\n\ntwo\n\nthree"]

    def test_loop_drains_every_residual_with_no_cap(self):
        """No continuation cap: the loop keeps spawning runs until the
        queue is empty, no matter how many residual rounds appear."""
        rounds = 7
        # ``rounds`` consecutive non-empty residuals followed by an empty
        # one. Without a cap the loop runs exactly ``rounds + 1`` times.
        residuals = [[f"msg-{i}"] for i in range(rounds)] + [[]]
        cc = _StubChatCommands(queue_residuals=residuals)

        cc.execute_chat_command("orig")

        assert cc._calls == ["orig"] + [f"msg-{i}" for i in range(rounds)]
        # No warning printed — the cap is gone.
        cc.console.print.assert_not_called()

    def test_interrupt_clears_queue_and_stops_loop(self):
        """An interrupted run drains the queue and breaks the loop —
        residual mid-run text is dropped because the user explicitly
        cancelled."""
        cc = _StubChatCommands(queue_residuals=[["should-be-cleared"]])

        # Simulate ESC during the first call.
        def _execute_with_interrupt(message, **_kw):
            cc._calls.append(message)
            cc.current_node.pending_input_queue.clear()
            for text in ["should-be-cleared"]:
                cc.current_node.pending_input_queue.push(text)
            cc.current_node.interrupt_controller.interrupt()

        cc._execute_chat = _execute_with_interrupt
        cc.execute_chat_command("orig")

        assert cc._calls == ["orig"]
        assert len(cc.current_node.pending_input_queue) == 0

    def test_no_current_node_breaks_loop_immediately(self):
        """If the implementation never created a node (degenerate path)
        the loop must still terminate after the first call."""
        cc = _StubChatCommands(queue_residuals=[[]])
        # Wipe the node after the call to simulate the no-node case.
        original_execute = cc._execute_chat

        def _execute_then_clear(message, **kw):
            original_execute(message, **kw)
            cc.current_node = None

        cc._execute_chat = _execute_then_clear
        cc.execute_chat_command("orig")

        assert cc._calls == ["orig"]


# ---------------------------------------------------------------------------
# DatusCLI._active_pending_input_queue accessor
# ---------------------------------------------------------------------------


class TestActivePendingInputQueueAccessor:
    """``DatusCLI._active_pending_input_queue`` exposes the running node's
    queue to the TUI without pulling the whole CLI graph into the
    keybinding closure."""

    @staticmethod
    def _call(cli):
        from datus.cli.repl import DatusCLI

        # Methods are descriptors that yield plain functions on class
        # access. Invoke directly with our stub bound as ``self``.
        return DatusCLI._active_pending_input_queue(cli)

    def test_returns_queue_when_node_has_one(self):
        queue = PendingInputQueue()
        chat_commands = SimpleNamespace(
            current_node=SimpleNamespace(pending_input_queue=queue),
            current_streaming_ctx=object(),
        )
        cli = SimpleNamespace(chat_commands=chat_commands)

        assert self._call(cli) is queue

    def test_returns_none_when_no_chat_commands(self):
        cli = SimpleNamespace()
        assert self._call(cli) is None

    def test_returns_none_when_no_current_node(self):
        cli = SimpleNamespace(chat_commands=SimpleNamespace(current_node=None, current_streaming_ctx=object()))
        assert self._call(cli) is None

    def test_returns_none_when_node_has_no_queue_attribute(self):
        chat_commands = SimpleNamespace(current_node=SimpleNamespace(), current_streaming_ctx=object())
        cli = SimpleNamespace(chat_commands=chat_commands)
        assert self._call(cli) is None

    def test_returns_none_when_not_streaming(self):
        """Stale ``current_node`` between chat turns must not leak its queue.

        Without an active streamed turn (``current_streaming_ctx is None``),
        a slash command or other worker task could otherwise push input into
        a queue that the next chat run would surprisingly flush.
        """
        queue = PendingInputQueue()
        chat_commands = SimpleNamespace(
            current_node=SimpleNamespace(pending_input_queue=queue),
            current_streaming_ctx=None,
        )
        cli = SimpleNamespace(chat_commands=chat_commands)
        assert self._call(cli) is None


# ---------------------------------------------------------------------------
# DatusApp helpers — queue preview + Enter enqueue
# ---------------------------------------------------------------------------


def _build_app(pending_input_provider=None):
    from datus.cli.tui.app import DatusApp

    return DatusApp(
        status_tokens_fn=lambda: [],
        dispatch_fn=lambda _: None,
        pending_input_provider=pending_input_provider,
    )


class TestQueuePreviewVisible:
    """``DatusApp._queue_preview_visible`` drives the
    ``ConditionalContainer`` that owns the preview rows. When it returns
    False the box collapses to zero rows; when True the user sees their
    pending mid-run injections above the status bar."""

    def test_no_provider_hides_box(self):
        app = _build_app(pending_input_provider=None)
        assert app._queue_preview_visible() is False

    def test_provider_returning_none_hides_box(self):
        app = _build_app(pending_input_provider=lambda: None)
        assert app._queue_preview_visible() is False

    def test_empty_queue_hides_box(self):
        queue = PendingInputQueue()
        app = _build_app(pending_input_provider=lambda: queue)
        assert app._queue_preview_visible() is False

    def test_non_empty_queue_shows_box(self):
        queue = PendingInputQueue()
        queue.push("追加一条")
        app = _build_app(pending_input_provider=lambda: queue)
        assert app._queue_preview_visible() is True

    def test_object_without_len_hides_box(self):
        """Defensive: if the provider returns something that does not
        support ``len()`` (e.g. a half-built stub during a hot reload),
        we hide rather than crash the layout."""
        app = _build_app(pending_input_provider=lambda: object())
        assert app._queue_preview_visible() is False


class TestQueuePreviewTokens:
    """``DatusApp._queue_preview_tokens`` renders the formatted rows."""

    def test_no_provider_returns_empty(self):
        app = _build_app(pending_input_provider=None)
        assert app._queue_preview_tokens() == []

    def test_provider_returning_none_returns_empty(self):
        app = _build_app(pending_input_provider=lambda: None)
        assert app._queue_preview_tokens() == []

    def test_empty_queue_returns_empty(self):
        queue = PendingInputQueue()
        app = _build_app(pending_input_provider=lambda: queue)
        assert app._queue_preview_tokens() == []

    def test_renders_header_and_items_in_order(self):
        queue = PendingInputQueue()
        queue.push("first")
        queue.push("second")
        app = _build_app(pending_input_provider=lambda: queue)

        tokens = app._queue_preview_tokens()

        # First token is the header carrying the count.
        assert tokens[0][0] == "class:queue-preview.header"
        assert "queued for agent (2)" in tokens[0][1]
        # Then one item row per queued message, in FIFO order.
        item_texts = [text for cls, text in tokens[1:] if cls == "class:queue-preview.item"]
        assert len(item_texts) == 2
        assert "1. first" in item_texts[0]
        assert "2. second" in item_texts[1]

    def test_long_text_is_truncated(self):
        queue = PendingInputQueue()
        long_text = "x" * 200
        queue.push(long_text)
        app = _build_app(pending_input_provider=lambda: queue)

        tokens = app._queue_preview_tokens()
        item_row = next(text for cls, text in tokens if cls == "class:queue-preview.item")
        # Hard cap (77 chars + ellipsis) keeps the box at a predictable width.
        assert "..." in item_row
        assert len(item_row) < 90

    def test_overflow_collapses_to_more_row(self):
        queue = PendingInputQueue()
        from datus.cli.tui.app import DatusApp

        for i in range(DatusApp._QUEUE_PREVIEW_MAX_LINES + 3):
            queue.push(f"item-{i}")
        app = _build_app(pending_input_provider=lambda: queue)

        tokens = app._queue_preview_tokens()
        # MAX rendered rows + 1 header + 1 overflow row.
        item_rows = [text for cls, text in tokens if cls == "class:queue-preview.item"]
        assert len(item_rows) == DatusApp._QUEUE_PREVIEW_MAX_LINES + 1
        assert "(+3 more)" in item_rows[-1]

    def test_snapshot_attribute_error_returns_empty(self):
        """Defensive: a stub provider that returns something without
        ``snapshot()`` must not crash the renderer."""
        app = _build_app(pending_input_provider=lambda: object())
        assert app._queue_preview_tokens() == []


class TestEnqueuePendingInput:
    """``DatusApp._enqueue_pending_input`` is the production code-path
    exercised by the TUI Enter key while ``_agent_running`` is set."""

    @staticmethod
    def _make_buffer(text=""):
        # prompt_toolkit's Buffer is heavy; a small stub with the two
        # attributes the helper touches is sufficient.
        return SimpleNamespace(text=text, reset=MagicMock())

    def test_returns_false_when_no_provider(self):
        app = _build_app(pending_input_provider=None)
        buf = self._make_buffer("hello")
        fake_app = SimpleNamespace(invalidate=MagicMock())

        assert app._enqueue_pending_input(buf, fake_app) is False
        # Buffer untouched; no invalidate call.
        buf.reset.assert_not_called()
        fake_app.invalidate.assert_not_called()

    def test_returns_false_when_provider_returns_none(self):
        app = _build_app(pending_input_provider=lambda: None)
        buf = self._make_buffer("hello")
        fake_app = SimpleNamespace(invalidate=MagicMock())

        assert app._enqueue_pending_input(buf, fake_app) is False

    def test_blank_text_is_not_queued(self):
        queue = PendingInputQueue()
        app = _build_app(pending_input_provider=lambda: queue)
        buf = self._make_buffer("   \t  ")
        fake_app = SimpleNamespace(invalidate=MagicMock())

        assert app._enqueue_pending_input(buf, fake_app) is False
        assert len(queue) == 0
        buf.reset.assert_not_called()

    def test_happy_path_pushes_stripped_text_and_resets_buffer(self):
        queue = PendingInputQueue()
        app = _build_app(pending_input_provider=lambda: queue)
        buf = self._make_buffer("  also stats  ")
        fake_app = SimpleNamespace(invalidate=MagicMock())

        assert app._enqueue_pending_input(buf, fake_app) is True

        # Pushed in stripped form so the model isn't fed ragged whitespace.
        assert queue.snapshot() == ["also stats"]
        buf.reset.assert_called_once()
        fake_app.invalidate.assert_called_once()

    def test_handles_missing_app_for_invalidate(self):
        """The helper is also called from contexts where the
        :class:`Application` reference is not yet wired (early tests,
        bare construction). It must not raise."""
        queue = PendingInputQueue()
        app = _build_app(pending_input_provider=lambda: queue)
        buf = self._make_buffer("real text")

        assert app._enqueue_pending_input(buf, None) is True
        assert queue.snapshot() == ["real text"]


# ---------------------------------------------------------------------------
# Broker emission exception path inside the model-layer input filter
# ---------------------------------------------------------------------------


class TestFilterBrokerEmissionFailure:
    """When ``broker.emit_user_insert`` raises, the filter must still
    return the augmented :class:`ModelInputData` — broker emission is
    best-effort, the model layer of the run is the contract that has to
    survive a misbehaving display side."""

    @pytest.mark.asyncio
    async def test_broker_emit_exception_does_not_drop_model_input(self):
        from agents.run import CallModelData, ModelInputData

        # _make_model lives in the openai_compatible test module.
        from tests.unit_tests.models.test_openai_compatible import _make_model

        model = _make_model()
        queue = PendingInputQueue()
        queue.push("追加：see this anyway")

        bad_broker = MagicMock()
        bad_broker.emit_user_insert = MagicMock(side_effect=RuntimeError("display down"))

        rc = model._build_run_config(pending_input_queue=queue, session=None, interaction_broker=bad_broker)
        baseline = ModelInputData(
            input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            instructions="be brief",
        )
        data = CallModelData(model_data=baseline, agent=MagicMock(), context=None)

        result = await rc.call_model_input_filter(data)

        # Filter still injected the queued message into the model input.
        assert len(result.input) == 2
        assert result.input[-1]["content"][0]["text"] == "追加：see this anyway"
        # Emission was attempted exactly once.
        bad_broker.emit_user_insert.assert_called_once_with("追加：see this anyway")
