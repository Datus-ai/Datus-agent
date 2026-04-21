# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""InlineStreamingContext: unified processing pipeline for streaming and sync modes."""

import asyncio
import sys
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from datus.cli.action_display.renderers import _truncate_middle
from datus.schemas.action_history import SUBAGENT_COMPLETE_ACTION_TYPE, ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.action_display.display import ActionHistoryDisplay

logger = get_logger(__name__)

# Blinking dot animation frames for PROCESSING status
_BLINK_FRAMES = ["○", "●"]

# In compact mode, only show the last N subagent actions in the Live overlay
_SUBAGENT_ROLLING_WINDOW_SIZE = 2


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
        interaction_broker: Any = None,
    ):
        self.actions = actions_list
        self.display = display_instance
        self._history_turns: List[Tuple[str, List[ActionHistory]]] = history_turns or []
        self._current_user_message = current_user_message
        self._sync_mode = sync_mode
        self._processed_index = 0
        self._tick = 0
        self._stop_event = threading.Event()
        self._refresh_thread: Optional[threading.Thread] = None
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
        self._input_collector: Optional[Callable[[ActionHistory, Console], Optional[str]]] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._clear_header_callback: Optional[Callable[[], None]] = None

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

    def recreate_live_display(self):
        """Compatibility shim: restart display after interaction."""
        self.restart_display()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the asyncio event loop for broker.submit calls from daemon thread."""
        self._event_loop = loop

    def set_input_collector(self, collector: Callable[[ActionHistory, Console], Optional[str]]) -> None:
        """Set the synchronous input collector callback for INTERACTION actions.

        The collector returns the user's choice string, or None if the interaction was aborted.
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
        """
        while self._processed_index < len(self.actions):
            action = self.actions[self._processed_index]

            # Skip thinking_delta actions in sync mode (transient streaming events)
            if action.action_type == "thinking_delta":
                self._processed_index += 1
                continue

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
        self._stop_event.clear()
        self._paused = False

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

        # Flush remaining actions after thread has stopped (no more concurrent access)
        self._flush_remaining_actions()

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
                    # Restart Live for any remaining active subagent groups
                    if self._subagent_groups:
                        with self._print_lock:
                            self._update_subagent_groups_live()

            if not self._paused and not self._verbose_frozen:
                self._process_actions()
                self._tick += 1
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
                    for a in reversed(turn_actions):
                        if (
                            a.role == ActionRole.ASSISTANT
                            and a.action_type
                            and a.action_type.endswith("_response")
                            and a.depth == 0
                            and a.status == ActionStatus.SUCCESS
                        ):
                            self.display.renderer.print_renderables(
                                self.display.console,
                                self.display.renderer.render_main_action(a, verbose=verbose),
                            )
                            break

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
        """Walk from _processed_index forward and handle each action."""
        while self._processed_index < len(self.actions):
            action = self.actions[self._processed_index]

            # Streaming thinking deltas: skip in CLI (not supported yet)
            if action.action_type == "thinking_delta":
                self._processed_index += 1
                continue

            # INTERACTION actions: collect input during live interaction, skip in history
            if action.role == ActionRole.INTERACTION:
                if action.status == ActionStatus.PROCESSING and self._input_collector:
                    self._stop_processing_live()
                    self._stop_subagent_live()
                    with self._print_lock:
                        renderables = self.display.renderer.render_interaction_request(action, self._verbose)
                        self.display.renderer.print_renderables(self.display.console, renderables)
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

            # Skip thinking_delta actions in flush (transient, already displayed via Live)
            if action.action_type == "thinking_delta":
                self._processed_index += 1
                continue

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
                duration_sec = (datetime.now() - group["start_time"]).total_seconds()
                summary = f"  \u23bf  Done ({group['tool_count']} tool uses \u00b7 {duration_sec:.1f}s)"
                self.display.console.print(f"[dim]{summary}[/dim]")

    # -- sub-agent group display (async mode) --------------------------------

    @staticmethod
    def _truncate_middle(text: str, max_len: int = 120) -> str:
        """Truncate text in the middle if too long, keeping head and tail."""
        return _truncate_middle(text, max_len)

    def _start_subagent_group(self, first_action: ActionHistory, group_key: Optional[str] = None) -> None:
        """Create sub-agent group and update the Live display."""
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
        """
        self._stop_processing_live()
        self._stop_subagent_live()

        group = self._subagent_groups.pop(group_key, None)
        if group is None:
            return

        if group_key is not None:
            self._completed_group_ids.add(group_key)

        if not self._verbose:
            # Compact mode: clear screen and reprint with completed groups collapsed
            self._reprint_with_collapse()
        else:
            # Verbose mode: permanently print the completed group
            end_time = end_action.end_time or datetime.now()
            duration = ""
            if group["start_time"]:
                duration_sec = (end_time - group["start_time"]).total_seconds()
                duration = f" \u00b7 {duration_sec:.1f}s"

            tool_count = group["tool_count"]
            summary = f"  \u23bf  Done ({tool_count} tool uses{duration})"

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
        """Start or update the Live display showing all active subagent groups."""
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
        """Stop the subagent Live display if running."""
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
        renderables = self.display.renderer.render_main_action(action, self._verbose)
        if not renderables:
            return
        with self._print_lock:
            self.display.renderer.print_renderables(self.display.console, renderables)

    # -- blinking Live for PROCESSING tools --------------------------------

    def _update_processing_live(self, action: ActionHistory) -> None:
        """Create or update the mini-Live for a PROCESSING tool."""
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
        """Stop the current mini-Live if running."""
        with self._print_lock:
            if self._live is not None:
                try:
                    self._live.stop()
                except Exception:
                    pass
                self._live = None
