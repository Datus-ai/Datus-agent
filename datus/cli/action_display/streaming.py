# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""InlineStreamingContext: unified processing pipeline for streaming and sync modes."""

import asyncio
import io
import sys
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from datus.cli.action_display.markdown_stream import MarkdownStreamBuffer
from datus.cli.action_display.renderers import _truncate_middle
from datus.schemas.action_history import SUBAGENT_COMPLETE_ACTION_TYPE, ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.action_display.display import ActionHistoryDisplay
    from datus.cli.execution_state import InteractionBroker
    from datus.cli.tui.live_display_state import LiveDisplayLine, LiveDisplayState

logger = get_logger(__name__)

# Blinking dot animation frames for PROCESSING status
_BLINK_FRAMES = ["○", "●"]

# In compact mode, only show the last N subagent actions in the Live overlay
_SUBAGENT_ROLLING_WINDOW_SIZE = 2


def _tail_has_incomplete_table(text: str) -> bool:
    """Should tail rendering fall back to plain text to avoid a broken table?

    Rich's ``Markdown`` renderer needs at least a header row, a separator
    row, and one body row to produce a box-drawing table; when only the
    header (and optionally the separator) have been streamed, Rich emits a
    corrupted half-grid. Once a body row is in, Rich renders a proper box
    even mid-stream, so we deliberately *stop* falling back at that point
    to honour the "live markdown" expectation — partial trailing cells are
    accepted as visual noise that disappears the moment the next row
    arrives.

    Returns True (use plain text) only when:

    * The tail contains at least one pipe line.
    * The tail is not terminated by a blank line (``\\n\\n``).
    * The pipe-line count is still ≤ 2 (i.e. header-only or header+separator).
    """
    if not text or "|" not in text:
        return False
    if text.endswith("\n\n"):
        return False
    pipe_lines = [line for line in text.splitlines() if "|" in line.strip()]
    if not pipe_lines:
        return False
    return len(pipe_lines) <= 2


class InlineStreamingContext:
    """Context manager for flat inline streaming display (Claude Code style).

    Actions are printed permanently to the console as they arrive/complete.
    PROCESSING tools are shown with a blinking dot animation via Rich Live.
    Completed actions are printed permanently and never refreshed in-place.

    Supports two execution modes:
    - Async (default): background thread with Live animation for streaming.
    - Sync (``sync_mode=True``): one-shot processing without threads/Live,
      used by ``render_action_history``, resume, and Ctrl+O reprint.
    """

    def __init__(
        self,
        actions_list: List[ActionHistory],
        display_instance: "ActionHistoryDisplay",
        history_turns: Optional[List[Tuple[str, List[ActionHistory]]]] = None,
        current_user_message: str = "",
        sync_mode: bool = False,
        interaction_broker: Optional["InteractionBroker"] = None,
        live_state: Optional["LiveDisplayState"] = None,
        streaming_deltas: Optional[List[ActionHistory]] = None,
    ):
        self.actions = actions_list
        # Parallel queue holding ``thinking_delta`` actions only. Populated by
        # the caller (``chat_commands.run_chat_stream``) as deltas stream in
        # and cleared on paired terminal response actions. The streaming
        # context consumes from here via ``_delta_processed_index`` so
        # ``self.actions`` stays delta-free and structural.
        self._deltas: List[ActionHistory] = streaming_deltas if streaming_deltas is not None else []
        self._delta_processed_index: int = 0
        self.display = display_instance
        self._history_turns: List[Tuple[str, List[ActionHistory]]] = history_turns or []
        self._current_user_message = current_user_message
        self._sync_mode = sync_mode
        self._processed_index = 0
        self._tick = 0
        self._stop_event = threading.Event()
        self._refresh_thread: Optional[threading.Thread] = None
        # WARNING: key-binding handlers on the prompt_toolkit loop must NOT
        # acquire ``_print_lock``. The refresh daemon holds it while calling
        # ``run_in_terminal_sync``; if a key handler tried to grab it on the
        # main loop thread we would deadlock.
        self._print_lock = threading.Lock()
        self._live: Optional[Live] = None
        self._paused = False
        self._subagent_groups: Dict[Optional[str], Dict] = {}
        self._completed_group_ids: set = set()
        self._subagent_live: Optional[Live] = None
        self._verbose = False
        self._verbose_frozen = False  # True = in frozen verbose mode, no real-time processing
        self._verbose_toggle_event = threading.Event()
        self._broker = interaction_broker
        self._input_collector: Optional[Callable[[ActionHistory, Console], Optional[List[List[str]]]]] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._clear_header_callback: Optional[Callable[[], None]] = None
        # TUI path: when provided, all subagent/processing rolling-window
        # rendering is pushed into this shared state (painted by DatusApp's
        # own layout) instead of Rich ``Live``, eliminating cursor-fights
        # with ``patch_stdout(raw=True)``.
        self._live_state: Optional["LiveDisplayState"] = live_state
        self._tui_mode = live_state is not None
        # Streaming markdown state (TUI mode only). ``_markdown_buffer`` holds
        # the unstable tail rendered into the pinned region; stable segments
        # are flushed above via ``_print_markdown_to_scrollback``. A single
        # buffer is reused across turns (cleared by ``_finalize_markdown_stream``
        # and ``__exit__``) to avoid per-delta allocations.
        self._markdown_buffer: Optional[MarkdownStreamBuffer] = MarkdownStreamBuffer() if self._tui_mode else None
        self._markdown_stream_has_streamed: bool = False
        # Current PROCESSING action whose blink frame occupies the pinned
        # region (TUI mode). ``_repaint_live`` reads this together with
        # ``_tick`` to animate the frame.
        self._processing_action: Optional[ActionHistory] = None
        # ``action_id`` of every thinking_delta we've processed in the current
        # turn. The paired terminal ASSISTANT SUCCESS action from
        # ``openai_compatible`` / ``codex_model`` reuses the same id
        # (``thinking_stream_id``), so matching on id — not on
        # ``action_type`` suffix — catches the full matrix of ``"response"``
        # and ``"*_response"`` names that otherwise leaks a duplicate body
        # into the scrollback.
        self._markdown_active_stream_ids: Set[str] = set()
        # Stream ids whose paired terminal action has already been de-duped
        # (so any later reprint — e.g. Ctrl+O — does not double-flush).
        self._markdown_stream_consumed_ids: Set[str] = set()
        # Latched for the rest of the current turn once we finalize a
        # streaming response. Any *subsequent* main-agent ASSISTANT SUCCESS
        # action in the same turn is an agent-node wrapper re-emission of
        # the same body (e.g. ``chat_response`` from
        # :mod:`chat_agentic_node` on top of the underlying
        # ``openai_compatible.py`` ``"response"`` action) and must be
        # dropped. Reset on ``__enter__``.
        self._turn_finalized: bool = False
        # Persistent flag: once we have flushed any non-empty main-agent
        # body to the scrollback in this streaming context, the outer
        # ``chat_commands._render_final_response`` pipeline should skip its
        # ``_display_markdown_response`` step or the user sees the same
        # body twice. Unlike ``_turn_finalized`` this flag is *not* cleared
        # per-turn — the accumulator-only flow only produces one body per
        # context, so one-shot latching is exactly right.
        self._stream_body_finalized: bool = False

    @property
    def live(self) -> Optional[Live]:
        """Compatibility property: expose the current mini-Live (if any)."""
        return self._live

    def stop_display(self) -> None:
        """Stop display for INTERACTION pause."""
        self._paused = True
        self._stop_processing_live()

    def restart_display(self) -> None:
        """Resume display after INTERACTION pause."""
        self._paused = False

    def toggle_verbose(self) -> None:
        """Toggle verbose mode (called from Ctrl+O key callback)."""
        self._verbose_toggle_event.set()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the asyncio event loop for broker.submit calls from daemon thread."""
        self._event_loop = loop

    def set_input_collector(self, collector: Callable[[ActionHistory, Console], Optional[List[List[str]]]]) -> None:
        """Set the synchronous input collector callback for INTERACTION actions.

        The collector returns ``List[List[str]]`` answers, or None if aborted.
        """
        self._input_collector = collector

    def set_clear_header_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Register a callback invoked at the top of the screen after Ctrl+O clears it.

        Used to reprint the CLI banner so it remains the first thing on screen
        after a verbose-mode toggle redraw.
        """
        self._clear_header_callback = callback

    # -- sync mode entry point ---------------------------------------------

    def run_sync(self) -> None:
        """Synchronous mode: process all actions without threads or Live animation.

        Used by render_action_history, resume, Ctrl+O reprint, display_action_list.
        """
        self._process_actions_sync()

    def _process_actions_sync(self) -> None:
        """Synchronous version of _process_actions.

        Differences from async _process_actions:
        1. PROCESSING TOOL actions are skipped (SUCCESS version follows).
        2. No Live animation is started.
        3. No thread, runs on the calling thread.
        INTERACTION actions are skipped (only shown during live interaction).
        ``thinking_delta`` actions never reach ``self.actions`` in the new
        split-queue design — no explicit guard needed.
        """
        while self._processed_index < len(self.actions):
            action = self.actions[self._processed_index]

            # INTERACTION actions: skip in history replay (only shown during live interaction)
            if action.role == ActionRole.INTERACTION:
                self._processed_index += 1
                continue

            # Skip PROCESSING tool entries — SUCCESS/FAILED version follows
            if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                self._processed_index += 1
                continue

            # subagent_complete closes a group
            if action.action_type == SUBAGENT_COMPLETE_ACTION_TYPE:
                group_key = action.parent_action_id
                self._processed_index += 1
                self._end_subagent_group_sync(group_key, action)
                continue

            # Sub-agent action (depth > 0)
            if action.depth > 0:
                group_key = action.parent_action_id
                if group_key not in self._subagent_groups:
                    self._start_subagent_group_sync(action, group_key)
                self._update_subagent_display_sync(action, group_key)
                self._processed_index += 1
                continue

            # Legacy sub-agent group closing via depth transition
            if None in self._subagent_groups:
                self._processed_index += 1
                self._end_subagent_group_sync(None, action)
                continue

            # Normal main-agent action
            self._print_completed_action(action)
            self._processed_index += 1

    def _start_subagent_group_sync(self, first_action: ActionHistory, group_key: Optional[str] = None) -> None:
        """Create sub-agent group state (sync mode — no Live display)."""
        subagent_type = first_action.action_type or "subagent"
        self._subagent_groups[group_key] = {
            "start_time": first_action.start_time,
            "tool_count": 0,
            "subagent_type": subagent_type,
            "first_action": first_action,
            "actions": [],
        }

    def _update_subagent_display_sync(self, action: ActionHistory, group_key: Optional[str] = None) -> None:
        """Buffer sub-agent action (sync mode — no Live update)."""
        group = self._subagent_groups.get(group_key)
        if group is None:
            return
        if action.role == ActionRole.TOOL:
            group["tool_count"] += 1
        if action.role == ActionRole.USER:
            return
        group["actions"].append(action)

    def _end_subagent_group_sync(self, group_key: Optional[str], end_action: ActionHistory) -> None:
        """End sub-agent group (sync mode — print permanently, no screen clear)."""
        group = self._subagent_groups.pop(group_key, None)
        if group is None:
            return

        if group_key is not None:
            self._completed_group_ids.add(group_key)

        # In sync mode, always print expanded (non-collapsed) group.
        # Collapse is handled at render_action_history level (caller decides).
        first_action = group.get("first_action")
        if first_action:
            self.display.renderer.print_renderables(
                self.display.console, self.display.renderer.render_subagent_header(first_action, self._verbose)
            )
        for buffered in group.get("actions", []):
            self.display.renderer.print_renderables(
                self.display.console, self.display.renderer.render_subagent_action(buffered, self._verbose)
            )
        done_text = self.display.renderer.render_subagent_done(group["tool_count"], group["start_time"], end_action)
        self.display.console.print(done_text)

    # -- context manager (async mode) --------------------------------------

    def __enter__(self):
        self._processed_index = 0
        self._delta_processed_index = 0
        self._stop_event.clear()
        self._paused = False
        # Fresh turn: forget which response we've already finalized so the
        # next stream's first paired completion triggers the full dedupe
        # + reprint cycle again.
        self._turn_finalized = False

        # Register with display instance
        self.display._current_context = self

        # Start background refresh thread
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pylint: disable=unused-argument
        self._verbose_frozen = False  # Ensure unfreeze before flush
        self._stop_event.set()
        self._stop_processing_live()
        self._stop_subagent_live()

        if self._refresh_thread:
            self._refresh_thread.join(timeout=2.0)

        # Drain any deltas that arrived after the last refresh tick so
        # _finalize_markdown_stream can still flush their stable segments.
        if self._tui_mode:
            try:
                self._process_deltas()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("_process_deltas raised during exit drain: %s", exc)

        # Flush remaining actions after thread has stopped (no more concurrent access)
        self._flush_remaining_actions()

        # Drain any markdown tail left over (interrupted stream, or a final
        # response action that never arrived). Printing here preserves the
        # "where the model stopped talking" context in the scrollback.
        if self._tui_mode and self._markdown_buffer is not None and self._markdown_buffer.has_tail():
            tail = self._markdown_buffer.flush()
            if tail.strip():
                try:
                    self._print_markdown_to_scrollback(tail)
                    # Mark the body as landed so the outer
                    # ``_render_final_response`` does not paint it again.
                    self._stream_body_finalized = True
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Markdown tail flush on exit failed: %s", exc)

        # Unregister
        if self.display._current_context is self:
            self.display._current_context = None

    # -- refresh loop (daemon thread) --------------------------------------

    def _refresh_loop(self) -> None:
        """Background thread: poll actions ~4x/sec and dispatch print/Live."""
        while not self._stop_event.is_set():
            # Check for verbose toggle request (Ctrl+O)
            if self._verbose_toggle_event.is_set():
                self._verbose_toggle_event.clear()
                self._verbose = not self._verbose

                if self._verbose:
                    # Entering verbose mode: freeze — show snapshot, stop all Live displays
                    self._verbose_frozen = True
                    self._stop_processing_live()
                    self._stop_subagent_live()
                    with self._print_lock:
                        self.display.console.clear()
                        sys.stdout.write("\033[3J")
                        sys.stdout.flush()
                        if self._clear_header_callback is not None:
                            try:
                                self._clear_header_callback()
                            except Exception as exc:
                                logger.debug(
                                    "clear_header_callback raised in verbose toggle: %s",
                                    exc,
                                    exc_info=True,
                                )
                        self.display.console.print(
                            "[bold bright_black]  \u23af switched to verbose mode (frozen) \u23af[/]"
                        )
                    self._reprint_history(verbose=True, show_active_groups=True)
                    # Do NOT restart any Live displays — screen is frozen
                else:
                    # Returning to compact mode: unfreeze — resume real-time processing
                    self._verbose_frozen = False
                    self._stop_processing_live()
                    self._stop_subagent_live()
                    with self._print_lock:
                        self.display.console.clear()
                        sys.stdout.write("\033[3J")
                        sys.stdout.flush()
                        if self._clear_header_callback is not None:
                            try:
                                self._clear_header_callback()
                            except Exception as exc:
                                logger.debug(
                                    "clear_header_callback raised in compact toggle: %s",
                                    exc,
                                    exc_info=True,
                                )
                        self.display.console.print("[bold bright_black]  \u23af switched to compact mode \u23af[/]")
                    self._reprint_history(verbose=self._verbose)
                    # Restart Live for any remaining active subagent groups.
                    # In TUI mode the pinned region re-draws from LiveDisplayState
                    # on its own; explicit re-emission handles non-TUI Rich Live.
                    if self._subagent_groups:
                        with self._print_lock:
                            self._update_subagent_groups_live()

            if not self._paused and not self._verbose_frozen:
                if self._tui_mode:
                    self._process_deltas()
                self._process_actions()
                self._tick += 1
                if self._tui_mode:
                    # Periodic repaint drives the processing-frame blink
                    # animation (frame = _tick % 2) and picks up buffer
                    # tail changes that arrive between ticks. In the
                    # accumulator-only flow nothing is pushed mid-stream —
                    # the final flush happens in ``_finalize_markdown_stream``
                    # at the message boundary (caller clears the deltas
                    # queue) and as an ``__exit__`` safety drain.
                    self._repaint_live()
            self._stop_event.wait(timeout=0.25)

    # -- unified reprint history -------------------------------------------

    def _reprint_history(
        self,
        verbose: bool,
        show_active_groups: bool = False,
    ) -> None:
        """Unified reprint of all already-processed actions.

        Called when user toggles Ctrl+O mid-execution so that the
        history retroactively reflects the new display style.

        Args:
            verbose: Whether to render in verbose mode (True=expanded, False=collapsed).
            show_active_groups: If True, render active subagent groups as static output
                with an "in progress" indicator (used for verbose/frozen snapshot).
        """
        with self._print_lock:
            # 1. Render previous turns (with per-turn response rendering)
            if self._history_turns:

                def _render_turn_response(turn_actions: List[ActionHistory]) -> None:
                    # Prefer the wrapper action (e.g. ``chat_response``) since
                    # ``render_action_history`` already skips it. Fall back to
                    # plain ``response`` for providers that don't emit a
                    # wrapper — otherwise the turn ends without any visible
                    # main-agent body.
                    wrapper = None
                    plain = None
                    for a in reversed(turn_actions):
                        if (
                            a.role != ActionRole.ASSISTANT
                            or a.depth != 0
                            or a.status != ActionStatus.SUCCESS
                            or not a.action_type
                        ):
                            continue
                        if a.action_type.endswith("_response") and a.action_type != "response":
                            wrapper = a
                            break
                        if a.action_type == "response" and plain is None:
                            plain = a
                    chosen = wrapper or plain
                    if chosen is not None:
                        self.display.renderer.print_renderables(
                            self.display.console,
                            self.display.renderer.render_main_action(chosen, verbose=verbose),
                        )

                self.display.render_multi_turn_history(
                    self._history_turns, verbose=verbose, per_turn_callback=_render_turn_response
                )
            # 2. Render current turn header
            if self._current_user_message:
                self.display.renderer.print_renderables(
                    self.display.console,
                    [
                        self.display.renderer.render_user_header(self._current_user_message),
                        self.display.renderer.render_separator(),
                    ],
                )
            # 3. Render already-processed actions
            current_actions = [
                a for a in self.actions[: self._processed_index] if not (a.role == ActionRole.USER and a.depth == 0)
            ]
            self.display.render_action_history(current_actions, verbose=verbose, _show_partial_done=False)
            # 3a. Render the streaming main-agent body (from deltas accumulator)
            # so mid-run Ctrl+O doesn't drop the text the user was reading.
            # ``_response`` / plain ``response`` only arrive after deltas end,
            # so the deltas queue is the only source of truth for the
            # *in-flight* message body. We pick the last delta's
            # ``accumulated`` field because each delta carries the full
            # running text up to that point. After printing we reset the
            # markdown buffer and latch ``_stream_body_finalized`` so the
            # end-of-message ``_finalize_markdown_stream`` only commits the
            # *incremental* tail that arrives after this point — avoiding a
            # duplicate body in the scrollback.
            if self._deltas:
                if self._markdown_buffer is not None and self._markdown_buffer.has_spilled():
                    # Overflow paragraphs have already been committed to the
                    # scrollback during the stream; only the unspilled tail
                    # is still live. Print just that tail so we don't repeat
                    # the already-committed prefix.
                    remaining = self._markdown_buffer.get_tail()
                    if remaining.strip():
                        self.display.console.print(Markdown(remaining))
                    self._markdown_buffer.clear()
                    self._stream_body_finalized = True
                else:
                    last_accumulated = ""
                    for delta in reversed(self._deltas):
                        if isinstance(delta.output, dict):
                            candidate = delta.output.get("accumulated", "")
                            if isinstance(candidate, str) and candidate:
                                last_accumulated = candidate
                                break
                    if last_accumulated.strip():
                        self.display.console.print(Markdown(last_accumulated))
                        if self._markdown_buffer is not None:
                            self._markdown_buffer.clear()
                        self._stream_body_finalized = True
            # 4. Optionally render active subagent groups (verbose snapshot)
            if show_active_groups:
                for _group_key, group in self._subagent_groups.items():
                    first_action = group.get("first_action")
                    if first_action:
                        self.display.renderer.print_renderables(
                            self.display.console,
                            self.display.renderer.render_subagent_header(first_action, verbose=True),
                        )
                    for buffered in group.get("actions", []):
                        self.display.renderer.print_renderables(
                            self.display.console,
                            self.display.renderer.render_subagent_action(buffered, verbose=True),
                        )
                    # Show "in progress" indicator
                    dur_str = ""
                    if group["start_time"]:
                        dur_sec = (datetime.now() - group["start_time"]).total_seconds()
                        dur_str = f" \u00b7 {dur_sec:.1f}s"
                    self.display.console.print(
                        f"[dim]  \u23bf  \u23f3 in progress ({group['tool_count']} tool uses{dur_str})...[/dim]"
                    )

    # -- core processing (async mode) --------------------------------------

    def _process_actions(self) -> None:
        """Walk from _processed_index forward and handle each action.

        ``thinking_delta`` actions are never seen here — the caller routes them
        into a separate ``streaming_deltas`` queue consumed by
        :meth:`_process_deltas`.
        """
        while self._processed_index < len(self.actions):
            action = self.actions[self._processed_index]

            # INTERACTION actions: collect input during live interaction, skip in history
            if action.role == ActionRole.INTERACTION:
                if action.status == ActionStatus.PROCESSING and self._input_collector:
                    self._stop_processing_live()
                    self._stop_subagent_live()
                    user_input = self._input_collector(action, self.display.console)
                    if user_input is None:
                        logger.warning("Interaction input aborted by collector")
                        self._processed_index += 1
                        return
                    if not (self._broker and self._event_loop):
                        logger.error("Cannot submit interaction response: broker or event_loop is missing")
                        return
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            self._broker.submit(action.action_id, user_input), self._event_loop
                        )
                        future.result(timeout=60)
                    except Exception as e:
                        logger.error(f"Error submitting interaction response: {e}")
                        return
                    self._processed_index += 1
                    return  # Wait for SUCCESS action to arrive
                else:
                    # SUCCESS or other status: skip (interaction result not shown in history)
                    self._processed_index += 1
                    continue

            # -- subagent_complete action closes a group --
            if action.action_type == SUBAGENT_COMPLETE_ACTION_TYPE:
                group_key = action.parent_action_id
                # Advance index BEFORE ending the group so that _reprint_with_collapse
                # includes this subagent_complete action in its slice.
                self._processed_index += 1
                self._end_subagent_group_by_key(group_key, action)
                continue

            # -- Sub-agent action (depth > 0) --
            if action.depth > 0:
                # Skip PROCESSING tools first — ensures the subagent group header
                # is created from the same action as render_action_history uses.
                if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                    self._processed_index += 1
                    continue
                group_key = action.parent_action_id
                if group_key not in self._subagent_groups:
                    # New sub-agent group: stop current Live, print header
                    self._stop_processing_live()
                    self._start_subagent_group(action, group_key)
                # Update current action display
                self._update_subagent_display(action, group_key)
                self._processed_index += 1
                continue

            # -- Main agent action (depth == 0), but preceded by legacy sub-agent group --
            if None in self._subagent_groups:
                # Advance index BEFORE ending the group so that _reprint_with_collapse
                # includes this action in its slice.
                self._processed_index += 1
                self._end_subagent_group_by_key(None, action)
                continue

            # TOOL with PROCESSING -> show blinking Live
            if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                self._update_processing_live(action)
                # Don't advance index yet; wait for status change
                return

            # Completed action (non-PROCESSING) -> print permanently
            self._stop_processing_live()
            self._print_completed_action(action)
            self._processed_index += 1

    def _flush_remaining_actions(self) -> None:
        """Flush all remaining actions at exit time without waiting for status changes."""
        while self._processed_index < len(self.actions):
            action = self.actions[self._processed_index]

            # INTERACTION actions: skip in flush (only shown during live interaction)
            if action.role == ActionRole.INTERACTION:
                self._processed_index += 1
                continue

            # Skip PROCESSING tool entries — their SUCCESS version follows in the list
            if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                self._processed_index += 1
                continue

            # subagent_complete closes the group
            if action.action_type == SUBAGENT_COMPLETE_ACTION_TYPE:
                group_key = action.parent_action_id
                self._processed_index += 1
                self._end_subagent_group_by_key(group_key, action)
                continue

            # depth>0: render inside the sub-agent group
            if action.depth > 0:
                group_key = action.parent_action_id
                if group_key not in self._subagent_groups:
                    self._start_subagent_group(action, group_key)
                self._update_subagent_display(action, group_key)
                self._processed_index += 1
                continue

            self._stop_processing_live()
            self._print_completed_action(action)
            self._processed_index += 1

        # Now close any sub-agent groups that are still open (no subagent_complete arrived)
        if self._subagent_groups:
            self._stop_processing_live()
            self._stop_subagent_live()
            for group_key in list(self._subagent_groups.keys()):
                group = self._subagent_groups.pop(group_key)
                first_action = group.get("first_action")
                duration_sec = (datetime.now() - group["start_time"]).total_seconds() if group["start_time"] else 0.0
                summary = f"  \u23bf  Done ({group['tool_count']} tool uses \u00b7 {duration_sec:.1f}s)"
                if self._tui_mode:
                    # Header lives in the pinned region (just cleared) —
                    # emit ⏺ header + ⎿ Done together so the scrollback
                    # still identifies which subagent this summary is for.
                    if first_action is not None:
                        self._print_subagent_summary_to_append(first_action, summary)
                    else:
                        self._print_to_append_area(f"[dim]{summary}[/dim]")
                    continue
                if first_action:
                    self.display.renderer.print_renderables(
                        self.display.console,
                        self.display.renderer.render_subagent_header(first_action, self._verbose),
                    )
                for buffered in group.get("actions", []):
                    self.display.renderer.print_renderables(
                        self.display.console,
                        self.display.renderer.render_subagent_action(buffered, self._verbose),
                    )
                self.display.console.print(f"[dim]{summary}[/dim]")

    # -- sub-agent group display (async mode) --------------------------------

    @staticmethod
    def _truncate_middle(text: str, max_len: int = 120) -> str:
        """Truncate text in the middle if too long, keeping head and tail."""
        return _truncate_middle(text, max_len)

    # -- TUI-mode append-area helpers ---------------------------------------

    def _print_to_append_area(self, markup: str) -> None:
        """Print ``markup`` above the pinned TUI region, or direct when no app.

        Uses :func:`run_in_terminal_sync` so prompt_toolkit re-orchestrates
        the bottom area; outside TUI the helper falls through to a direct
        call (same behaviour as today).
        """
        from datus.cli.tui.console_bridge import run_in_terminal_sync

        console = self.display.console
        run_in_terminal_sync(lambda: console.print(markup))

    def _print_subagent_summary_to_append(self, first_action: ActionHistory, summary: str) -> None:
        """Emit ``⏺ header`` + ``⎿ Done(…)`` into the append area on group completion."""
        from datus.cli.tui.console_bridge import run_in_terminal_sync

        console = self.display.console
        header_renderables = self.display.renderer.render_subagent_header(first_action, verbose=False)

        def _emit() -> None:
            for r in header_renderables:
                console.print(r)
            console.print(f"[dim]{summary}[/dim]")

        run_in_terminal_sync(_emit)

    def _start_subagent_group(self, first_action: ActionHistory, group_key: Optional[str] = None) -> None:
        """Create sub-agent group and update the Live display.

        In TUI mode the group's ``⏺ header`` row is rendered as the first
        line of the pinned block by :meth:`_build_subagent_live_lines`;
        nothing is emitted to the append area here. On group completion
        :meth:`_end_subagent_group_by_key` appends a ``header + Done``
        summary to the scrollback so a permanent trace remains.
        """
        subagent_type = first_action.action_type or "subagent"

        with self._print_lock:
            self._subagent_groups[group_key] = {
                "start_time": first_action.start_time,
                "tool_count": 0,
                "subagent_type": subagent_type,
                "first_action": first_action,
                "actions": [],
            }
            self._update_subagent_groups_live()

    def _update_subagent_display(self, action: ActionHistory, group_key: Optional[str] = None) -> None:
        """Buffer sub-agent action and update the grouped Live display."""
        group = self._subagent_groups.get(group_key)
        if group is None:
            return
        if action.role == ActionRole.TOOL:
            group["tool_count"] += 1
        if action.role == ActionRole.USER:
            # Prompt already shown in header, skip
            return
        group["actions"].append(action)
        with self._print_lock:
            self._update_subagent_groups_live()

    def _end_subagent_group_by_key(self, group_key: Optional[str], end_action: ActionHistory) -> None:
        """End sub-agent group: stop Live, permanently print completed group, restart Live for remaining.

        In compact mode, triggers a full reprint with completed groups collapsed.
        In verbose mode, permanently prints the completed group (header + actions + Done).

        In TUI mode, the header was already printed on group-start (see
        :meth:`_start_subagent_group`), the rolling window is pinned state
        (cleared on group end), and we only need to append a collapsed Done
        line so the append-area scrollback keeps a permanent trace.
        """
        self._stop_processing_live()
        self._stop_subagent_live()

        group = self._subagent_groups.pop(group_key, None)
        if group is None:
            return

        if group_key is not None:
            self._completed_group_ids.add(group_key)

        end_time = end_action.end_time or datetime.now()
        duration = ""
        if group["start_time"]:
            duration_sec = (end_time - group["start_time"]).total_seconds()
            duration = f" \u00b7 {duration_sec:.1f}s"

        tool_count = group["tool_count"]
        summary = f"  \u23bf  Done ({tool_count} tool uses{duration})"

        if self._tui_mode:
            # TUI mode: emit ⏺ header + ⎿ Done into the append area as a
            # permanent scrollback block so the reader knows which subagent
            # this Done refers to. No reprint-with-collapse is performed.
            first_action = group.get("first_action")
            if first_action is not None:
                self._print_subagent_summary_to_append(first_action, summary)
            else:
                self._print_to_append_area(f"[dim]{summary}[/dim]")
            # Refresh the pinned region: either paint remaining subagent
            # groups, or fall through to markdown tail / clear when none
            # are active. Without this, a lingering snapshot of the
            # just-ended group stays visible until the next tick.
            self._repaint_live()
            return

        if not self._verbose:
            # Compact mode: clear screen and reprint with completed groups collapsed
            self._reprint_with_collapse()
        else:
            # Verbose mode: permanently print the completed group
            with self._print_lock:
                first_action = group.get("first_action")
                if first_action:
                    self.display.renderer.print_renderables(
                        self.display.console,
                        self.display.renderer.render_subagent_header(first_action, self._verbose),
                    )
                for buffered in group.get("actions", []):
                    self.display.renderer.print_renderables(
                        self.display.console,
                        self.display.renderer.render_subagent_action(buffered, self._verbose),
                    )
                self.display.console.print(f"[dim]{summary}[/dim]")

        # Restart Live for remaining active groups
        if self._subagent_groups:
            with self._print_lock:
                self._update_subagent_groups_live()

    def _reprint_with_collapse(self) -> None:
        """Clear screen and reprint all history with completed groups collapsed."""
        with self._print_lock:
            self.display.console.clear()
            sys.stdout.write("\033[3J")
            sys.stdout.flush()
        self._reprint_history(verbose=self._verbose)

    # -- subagent Live display ------------------------------------------------

    def _update_subagent_groups_live(self) -> None:
        """Start or update the subagent rolling-window display.

        In TUI mode (``self._tui_mode``), the rolling window is one input
        to the shared pinned-region painter :meth:`_repaint_live`; the
        painter resolves priority against processing-tool blink and
        streaming markdown tail. Outside TUI, the legacy Rich ``Live``
        path renders directly.
        """
        if self._tui_mode:
            self._repaint_live()
            return
        renderable = self._build_subagent_groups_renderable()
        if self._subagent_live is None:
            # ``redirect_stdout`` / ``redirect_stderr`` are set to ``False`` so
            # Rich does not install its own stdout wrapper. When the REPL runs
            # under the TUI, stdout is already patched by prompt_toolkit's
            # ``patch_stdout(raw=True)`` — double-patching would fight the
            # pinned status-bar + input. Outside the TUI the behavior is
            # unchanged: Rich writes ANSI directly to the real stdout.
            self._subagent_live = Live(
                renderable,
                console=self.display.console,
                refresh_per_second=4,
                transient=True,
                redirect_stdout=False,
                redirect_stderr=False,
                screen=False,
            )
            self._subagent_live.start()
        else:
            self._subagent_live.update(renderable)

    def _stop_subagent_live(self) -> None:
        """Stop the subagent rolling-window display."""
        if self._tui_mode:
            # The subagent groups dict is the source of truth — callers have
            # already popped completed groups before invoking this. Ask the
            # painter to repick the pinned content so processing frames or
            # markdown tails can reclaim the region.
            self._repaint_live()
            return
        with self._print_lock:
            if self._subagent_live is not None:
                try:
                    self._subagent_live.stop()
                except Exception as e:
                    logger.debug(f"Error stopping subagent live display: {e}")
                self._subagent_live = None

    def _build_subagent_groups_renderable(self) -> Group:
        """Build a Group renderable showing all active subagent groups with actions grouped."""
        items: List[Text] = []
        for _group_key, group in self._subagent_groups.items():
            first_action = group.get("first_action")
            if first_action:
                items.extend(self.display.renderer.render_subagent_header(first_action, self._verbose))
            actions_list = group.get("actions", [])
            if self._verbose:
                display_actions = actions_list
            else:
                display_actions = actions_list[-_SUBAGENT_ROLLING_WINDOW_SIZE:]
                hidden = len(actions_list) - len(display_actions)
                if hidden > 0:
                    items.append(Text.from_markup(f"[dim]  \u23bf  ... {hidden} earlier action(s) ...[/dim]"))
            for action in display_actions:
                items.extend(self.display.renderer.render_subagent_action(action, self._verbose))
        return Group(*items) if items else Group(Text(""))

    def _build_subagent_live_lines(self) -> List["LiveDisplayLine"]:
        """Stack each active subagent as its own pinned block.

        Each block = one header row (``⏺ subagent_type(goal)``) followed by
        up to ``TOOL_LINES_PER_GROUP`` rolling tool-call rows for that same
        subagent. Blocks are concatenated in group-creation order so the
        visual flow is ``header1 → tools1 → header2 → tools2 → …`` — each
        subagent's tools stay grouped under its own header instead of
        interleaving across groups. The combined total is capped by
        :data:`MAX_LIVE_LINES_TOTAL`.
        """
        from datus.cli.tui.live_display_state import TOOL_LINES_PER_GROUP, LiveDisplayLine

        result: List["LiveDisplayLine"] = []
        for _group_key, group in self._subagent_groups.items():
            first_action = group.get("first_action")
            if first_action is not None:
                header_segments = self._build_subagent_header_segments(first_action)
                if header_segments:
                    result.append(LiveDisplayLine(segments=header_segments))
            tool_texts: List[Text] = []
            for action in group.get("actions", []):
                tool_texts.extend(self.display.renderer.render_subagent_action(action, verbose=False))
            tool_texts = [t for t in tool_texts if (t.plain or "").strip()]
            tail = tool_texts[-TOOL_LINES_PER_GROUP:]
            for text in tail:
                result.append(LiveDisplayLine(segments=[("class:subagent-live", text.plain)]))
        return result

    @staticmethod
    def _build_subagent_header_segments(first_action: ActionHistory) -> List[Tuple[str, str]]:
        """Split the ⏺ header into a cyan name segment + default goal segment.

        Mirrors the field layout of
        :meth:`ActionRenderer.render_subagent_header` so the pinned-region
        header stays in lock-step with the scrollback header without
        re-parsing Rich markup back into prompt_toolkit styles.
        """
        subagent_type = first_action.action_type or "subagent"
        prompt = first_action.messages or ""
        if prompt.startswith("User: "):
            prompt = prompt[6:]
        description = ""
        if first_action.input and isinstance(first_action.input, dict):
            description = first_action.input.get("_task_description", "")
        goal = description or (_truncate_middle(prompt, max_len=200) if prompt else "")

        segments: List[Tuple[str, str]] = [("class:subagent-header-live", f"\u23fa {subagent_type}")]
        if goal:
            segments.append(("class:subagent-header-goal-live", f"({goal})"))
        return segments

    # -- pinned-region painter (TUI-mode priority multiplexer) --------------

    def _repaint_live(self) -> None:
        """Render the pinned region from current state (TUI mode only).

        Priority, high → low:

        1. ``_processing_action`` — blinking frame for a running tool.
        2. ``_subagent_groups`` — rolling header + tool-tail lines.
        3. Streaming markdown ``tail`` — the part the user is watching grow.
        4. Nothing — clear so the region collapses to zero rows.

        Each writer only updates its own state slot; this method combines
        them on every call so the three layers never overwrite each other
        out-of-order. Safe to call from the refresh daemon thread because
        :class:`LiveDisplayState` is internally locked and its
        ``invalidate_cb`` dispatches to the prompt_toolkit loop.
        """
        if not self._tui_mode or self._live_state is None:
            return
        if self._verbose_frozen:
            # Frozen snapshot owns the screen; pinned region must not repaint.
            self._live_state.clear()
            return
        from datus.cli.tui.live_display_state import LiveDisplayLine

        if self._processing_action is not None:
            frame = _BLINK_FRAMES[self._tick % len(_BLINK_FRAMES)]
            renderable = self.display.renderer.render_processing(self._processing_action, frame)
            self._live_state.set_lines([LiveDisplayLine(segments=[("class:processing-live", renderable.plain)])])
            return
        if self._subagent_groups:
            self._live_state.set_lines(self._build_subagent_live_lines())
            return
        if self._markdown_buffer is not None and self._markdown_buffer.has_tail():
            self._live_state.set_lines(self._build_markdown_tail_lines())
            return
        self._live_state.clear()

    def _build_markdown_tail_lines(self) -> List["LiveDisplayLine"]:
        """Render the current markdown tail into pinned ``LiveDisplayLine``s.

        The tail is rendered via a throwaway ``Rich.Console`` capturing ANSI
        to a string, then split per-line and converted to prompt_toolkit
        :class:`FormattedText` fragments via
        :class:`prompt_toolkit.formatted_text.ANSI`. When the rendered output
        is taller than the current pinned-row budget (derived from the live
        terminal height), only the bottom rows are kept (newest tokens live
        at the end) with a leading ``…`` row.
        """
        from prompt_toolkit.formatted_text import ANSI

        from datus.cli.tui.live_display_state import LiveDisplayLine

        assert self._markdown_buffer is not None  # noqa: S101  guarded by caller
        tail = self._markdown_buffer.get_tail()
        if not tail:
            return []
        cols = max(20, getattr(self.display.console, "width", 80) or 80)
        buf = io.StringIO()
        rc = Console(
            file=buf,
            force_terminal=True,
            color_system="truecolor",
            width=cols,
            legacy_windows=False,
        )
        # When the tail looks like a table that hasn't been terminated
        # by a blank line, Rich's ``Markdown`` renderer garbles the
        # half-formed grid (columns shift, borders break, next tokens
        # land inside the wrong cell). Fall back to plain-text rendering
        # for that case so the user sees a readable incremental table
        # instead of a corrupted one; the completed version still lands
        # in the scrollback as proper Markdown once the blank line
        # arrives and the segment becomes stable.
        if _tail_has_incomplete_table(tail):
            rc.print(tail, markup=False, highlight=False)
        else:
            try:
                rc.print(Markdown(tail))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Rich markdown render raised on tail: %s", exc)
                rc.print(tail, markup=False, highlight=False)
        ansi = buf.getvalue().rstrip("\n")
        if not ansi:
            return []
        all_lines = ansi.split("\n")
        max_rows = self._live_state.max_rows() if self._live_state else len(all_lines)
        if len(all_lines) > max_rows:
            visible = all_lines[-(max_rows - 1) :]
            lines: List[LiveDisplayLine] = [LiveDisplayLine(segments=[("class:markdown-stream-ellipsis", "…")])]
            for ansi_line in visible:
                lines.append(LiveDisplayLine(segments=list(ANSI(ansi_line).__pt_formatted_text__())))
            return lines
        return [LiveDisplayLine(segments=list(ANSI(ansi_line).__pt_formatted_text__())) for ansi_line in all_lines]

    # -- thinking_delta handling --------------------------------------------

    def _process_deltas(self) -> None:
        """Drain new ``thinking_delta`` actions from the streaming_deltas queue.

        Called on every refresh tick in TUI mode. Handles the caller's
        per-message reset (``streaming_deltas.clear()``): if the queue shrank
        below our cursor we finalize the markdown stream (flushing any
        pending tail into the scrollback) and rewind. New deltas are
        forwarded to :meth:`_handle_thinking_delta` in order.
        """
        if self._markdown_buffer is None:
            return
        current_len = len(self._deltas)
        if current_len < self._delta_processed_index:
            # Caller cleared the list — message boundary reached. Commit
            # whatever tail is still buffered so it lands in the scrollback
            # and reset our cursor.
            self._finalize_markdown_stream()
            self._delta_processed_index = 0
            current_len = len(self._deltas)
        while self._delta_processed_index < current_len:
            action = self._deltas[self._delta_processed_index]
            self._delta_processed_index += 1
            if action.depth != 0:
                continue
            self._handle_thinking_delta(action)

    def _handle_thinking_delta(self, action: ActionHistory) -> None:
        """Append a streaming text delta into the markdown buffer (TUI mode).

        Each call pipes the delta through :meth:`MarkdownStreamBuffer.append`,
        which commits a stable prefix whenever either trigger fires:
        (a) a ``"\\n\\n"`` markdown-segment boundary appeared, or (b) the
        residual tail would overflow the pinned-region budget, in which
        case everything up to the last ``"\\n"`` is handed off. Those
        committed segments are printed to the scrollback immediately so
        the user sees long responses stream upward; only the unfinished
        tail stays live in the pinned region. A ``_finalize_markdown_stream``
        at the message boundary drains whatever tail is left.
        """
        if self._markdown_buffer is None:
            return
        delta = ""
        if isinstance(action.output, dict):
            raw = action.output.get("delta", "")
            if isinstance(raw, str):
                delta = raw
        if not delta:
            return
        self._markdown_stream_has_streamed = True
        if action.action_id:
            self._markdown_active_stream_ids.add(action.action_id)
        stable_segments = self._markdown_buffer.append(delta)
        for segment in stable_segments:
            self._print_markdown_to_scrollback(segment)
        if stable_segments:
            # A portion of the body has already landed in the scrollback,
            # so the outer ``_render_final_response`` must not repaint
            # the whole response again.
            self._stream_body_finalized = True
        self._repaint_live()

    def _print_markdown_to_scrollback(self, segment: str) -> None:
        """Print a stable markdown segment above the pinned region.

        Uses :func:`run_in_terminal_sync` so prompt_toolkit keeps the pinned
        live-region + status bar + input pinned at the bottom. The helper
        must never be called while ``_print_lock`` is held — see the lock's
        class-level warning — so this method is deliberately lock-free.
        """
        if not segment.strip():
            return
        from datus.cli.tui.console_bridge import run_in_terminal_sync

        console = self.display.console
        md = Markdown(segment)

        def _emit() -> None:
            console.print(md)

        run_in_terminal_sync(_emit)

    def _finalize_markdown_stream(self) -> None:
        """Flush the accumulated body to the scrollback and reset state.

        In the accumulator-only flow this is the *only* point where the
        main-agent response lands in the Rich scrollback. Triggered by:
        ``_process_deltas`` detecting the caller's
        ``streaming_deltas.clear()`` (paired terminal action arrived),
        ``_print_completed_action`` seeing the plain ``response`` action,
        and ``__exit__`` as a safety drain.
        """
        if self._markdown_buffer is None:
            return
        tail = self._markdown_buffer.flush()
        if tail.strip():
            self._print_markdown_to_scrollback(tail)
            # Latch: the outer pipeline's ``_display_markdown_response``
            # should not repeat this body.
            self._stream_body_finalized = True
        self._markdown_stream_has_streamed = False
        self._repaint_live()

    @property
    def has_streamed_response(self) -> bool:
        """Whether this context has flushed a main-agent body to scrollback.

        Consumed by ``chat_commands._render_final_response`` so it can skip
        the one-shot ``_display_markdown_response`` step when the streaming
        path has already landed the full body.
        """
        return self._stream_body_finalized

    def _format_subagent_action_items(self, action: ActionHistory) -> List[Text]:
        """Format a single subagent action as a list of Text renderables."""
        return self.display.renderer.render_subagent_action(action, self._verbose)

    def _format_subagent_action_line(self, action: ActionHistory) -> str:
        """Format a single subagent action as a plain string (for tests)."""
        items = self._format_subagent_action_items(action)
        return "\n".join(item.plain for item in items)

    # -- completed action printing -------------------------------------------

    def _print_completed_action(self, action: ActionHistory) -> None:
        """Print a completed action permanently to the console."""
        # Skip "task" tool calls — already represented by the subagent group display
        if action.role == ActionRole.TOOL:
            fn = action.input.get("function_name", "") if action.input else ""
            if fn == "task":
                return
        # Streaming-markdown de-dup: once this turn has emitted any
        # thinking_delta, the next main-agent ASSISTANT SUCCESS action is
        # the paired completion regardless of ``action_type`` (plain
        # ``"response"`` from openai_compatible / codex vs ``"*_response"``
        # from agent nodes) or whether the ``action_id`` lines up with the
        # delta's ``thinking_stream_id`` (some providers mint a fresh id
        # on the fallback path). The match stays narrow because:
        #   * ``depth == 0`` excludes subagent responses.
        #   * ``has_streamed`` is only true after we actually painted a
        #     delta, so no non-streaming turn can be swallowed.
        #
        # After the first such finalize, ``_turn_finalized`` latches True
        # so any *further* depth-0 ASSISTANT SUCCESS action in the same
        # turn — i.e. an agent-node wrapper re-emission of the same body,
        # like ``chat_agentic_node``'s ``chat_response`` action sitting on
        # top of ``openai_compatible.py``'s ``"response"`` action — is also
        # dropped. Without this latch the wrapper's body hits
        # ``render_main_action`` and appears in the scrollback a second
        # time right below the streamed version.
        is_main_assistant_success = (
            self._tui_mode
            and action.role == ActionRole.ASSISTANT
            and action.status == ActionStatus.SUCCESS
            and action.depth == 0
        )
        # Accumulator-only dedup. Three signals can indicate the main-agent
        # body has already been (or is being) flushed to the scrollback:
        #   * ``_markdown_stream_has_streamed`` — deltas arrived but finalize
        #     hasn't run yet (this call is the paired terminal action).
        #   * ``_stream_body_finalized`` — ``_process_deltas`` already
        #     detected the caller's ``streaming_deltas.clear()`` and flushed.
        #   * ``_turn_finalized`` — a previous wrapper action in this turn
        #     already went through the finalize path.
        # Any of them means we must not re-render via ``render_main_action``.
        if is_main_assistant_success and (
            self._markdown_stream_has_streamed or self._stream_body_finalized or self._turn_finalized
        ):
            if action.action_id:
                self._markdown_stream_consumed_ids.add(action.action_id)
            self._markdown_active_stream_ids.clear()
            if self._markdown_stream_has_streamed:
                self._finalize_markdown_stream()
            self._turn_finalized = True
            return
        if self._tui_mode and action.action_id and action.action_id in self._markdown_stream_consumed_ids:
            return
        renderables = self.display.renderer.render_main_action(action, self._verbose)
        if not renderables:
            return
        if self._tui_mode:
            from datus.cli.tui.console_bridge import run_in_terminal_sync

            console = self.display.console
            renderer = self.display.renderer

            def _emit() -> None:
                renderer.print_renderables(console, renderables)

            run_in_terminal_sync(_emit)
            return
        with self._print_lock:
            self.display.renderer.print_renderables(self.display.console, renderables)

    # -- blinking Live for PROCESSING tools --------------------------------

    def _update_processing_live(self, action: ActionHistory) -> None:
        """Render the blinking frame for a PROCESSING tool.

        In TUI mode the frame is just state (``_processing_action``) that
        :meth:`_repaint_live` turns into a pinned line every tick, so the
        blink animation follows the shared 4 Hz refresh loop. Outside TUI
        the legacy Rich ``Live`` is still used.
        """
        if self._tui_mode:
            self._processing_action = action
            self._repaint_live()
            return

        frame = _BLINK_FRAMES[self._tick % len(_BLINK_FRAMES)]
        renderable = self.display.renderer.render_processing(action, frame)

        with self._print_lock:
            if self._live is None:
                # See the comment in ``_update_subagent_groups_live``: Live
                # must not install its own stdout wrapper when the TUI is
                # active, and the flags are safe for non-TUI callers too.
                self._live = Live(
                    renderable,
                    console=self.display.console,
                    refresh_per_second=4,
                    transient=True,
                    redirect_stdout=False,
                    redirect_stderr=False,
                    screen=False,
                )
                self._live.start()
            else:
                self._live.update(renderable)

    def _stop_processing_live(self) -> None:
        """Stop the current processing-tool display."""
        if self._tui_mode:
            # Drop the processing-frame state; :meth:`_repaint_live` then
            # falls through to subagent rolling lines / markdown tail /
            # clear depending on what else is active.
            self._processing_action = None
            self._repaint_live()
            return
        with self._print_lock:
            if self._live is not None:
                try:
                    self._live.stop()
                except Exception:
                    pass
                self._live = None
