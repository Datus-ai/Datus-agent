# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Public interface: ActionHistoryDisplay."""

from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from rich.console import Console

from datus.cli.action_display.renderers import ActionContentGenerator, ActionRenderer
from datus.schemas.action_history import SUBAGENT_COMPLETE_ACTION_TYPE, ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.action_display.streaming import InlineStreamingContext

logger = get_logger(__name__)


class ActionHistoryDisplay:
    """Display ActionHistory in a flat inline format (Claude Code style)"""

    def __init__(self, console: Optional[Console] = None, enable_truncation: bool = True):
        self.console = console or Console()
        self.enable_truncation = enable_truncation

        # Create content generator with truncation setting
        self.content_generator = ActionContentGenerator(enable_truncation=enable_truncation)

        # Unified renderer
        self.renderer = ActionRenderer(self.content_generator)

        # Reference to current streaming context for live control
        self._current_context: Optional["InlineStreamingContext"] = None

    # -- unified render helpers (delegating to renderer) --------------------

    def _render_subagent_header(self, action: ActionHistory, verbose: bool) -> None:
        """Print sub-agent group header (type + prompt/description)."""
        self.renderer.print_renderables(self.console, self.renderer.render_subagent_header(action, verbose))

    def _render_subagent_action(self, action: ActionHistory, verbose: bool) -> None:
        """Print a single sub-agent action line."""
        self.renderer.print_renderables(self.console, self.renderer.render_subagent_action(action, verbose))

    def _render_subagent_done(
        self, tool_count: int, start_time: Optional[datetime], next_action: ActionHistory
    ) -> None:
        """Print the Done summary line for a sub-agent group."""
        self.console.print(self.renderer.render_subagent_done(tool_count, start_time, next_action))

    def _render_subagent_collapsed(
        self,
        first_action: ActionHistory,
        tool_count: int,
        start_time: Optional[datetime],
        end_action: ActionHistory,
    ) -> None:
        """Render a completed subagent group as collapsed: header line + Done summary line."""
        self.renderer.print_renderables(
            self.console, self.renderer.render_subagent_collapsed(first_action, tool_count, start_time, end_action)
        )

    def _render_subagent_response(self, action: ActionHistory) -> None:
        """Show the subagent response value after the Done line (verbose only)."""
        self.renderer.print_renderables(self.console, self.renderer.render_subagent_response(action))

    def _render_deferred_group(
        self,
        group: dict,
        end_action: ActionHistory,
        task_tool_action: ActionHistory,
        verbose: bool,
    ) -> None:
        """Render a completed subagent group together with its task tool response."""
        for buffered in group.get("actions", []):
            if buffered is group["first_action"]:
                self._render_subagent_header(buffered, verbose)
            self._render_subagent_action(buffered, verbose)
        self._render_subagent_done(group["tool_count"], group["start_time"], end_action)
        if verbose:
            self._render_subagent_response(task_tool_action)

    def _flush_deferred_groups(
        self,
        deferred_groups: List[Tuple[dict, ActionHistory]],
        verbose: bool,
    ) -> None:
        """Render any deferred groups that never got a task tool response."""
        while deferred_groups:
            grp, end_act = deferred_groups.pop(0)
            for buffered in grp.get("actions", []):
                if buffered is grp["first_action"]:
                    self._render_subagent_header(buffered, verbose)
                self._render_subagent_action(buffered, verbose)
            self._render_subagent_done(grp["tool_count"], grp["start_time"], end_act)

    def _render_task_tool_as_subagent(self, action: ActionHistory, verbose: bool) -> None:
        """Render a standalone 'task' tool action as a subagent summary."""
        self.renderer.print_renderables(self.console, self.renderer.render_task_tool_as_subagent(action, verbose))

    def _render_main_action(self, action: ActionHistory, verbose: bool) -> None:
        """Print a depth=0 completed action."""
        self.renderer.print_renderables(self.console, self.renderer.render_main_action(action, verbose))

    def render_action_history(
        self,
        actions: List[ActionHistory],
        verbose: bool = False,
        _show_partial_done: bool = True,
    ) -> None:
        """Render a list of completed actions using the unified inline format.

        This is the single source of truth for rendering an action history list.
        Used by _reprint_history (Ctrl+O toggle), display_action_list (.resume),
        and display_inline_trace_details (post-run Ctrl+O).

        Args:
            actions: List of ActionHistory to render.
            verbose: If True, show full arguments/output (no truncation).
                     If False, compact mode with completed subagent groups collapsed.
            _show_partial_done: Internal flag. If True, print a Done summary even
                when the sub-agent group is still open. Set to False by streaming
                context when reprinting mid-execution (the Live display handles
                the running group).
        """
        if not actions:
            self.console.print("[dim]No actions to display[/dim]")
            return

        # In compact mode, collapse completed subagent groups
        do_collapse = not verbose

        # Multi-group state: key = parent_action_id (None for legacy groups)
        subagent_groups: Dict[Optional[str], dict] = {}
        pending_task_tool_skips = 0
        deferred_groups: List[Tuple[dict, ActionHistory]] = []

        for action in actions:
            # INTERACTION actions: only shown during live interaction, skip in history replay
            if action.role == ActionRole.INTERACTION:
                continue
            # Skip PROCESSING TOOL entries (only render SUCCESS/FAILED)
            if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                continue
            # Skip node final actions (e.g. chat_response) — rendered separately
            if action.role == ActionRole.ASSISTANT and action.action_type and action.action_type.endswith("_response"):
                continue

            # -- subagent_complete action closes a group --
            if action.action_type == SUBAGENT_COMPLETE_ACTION_TYPE:
                group_key = action.parent_action_id
                group = subagent_groups.pop(group_key, None)
                if group:
                    if do_collapse:
                        self._render_subagent_collapsed(
                            group["first_action"], group["tool_count"], group["start_time"], action
                        )
                    else:
                        deferred_groups.append((group, action))
                    pending_task_tool_skips += 1
                continue

            # -- sub-agent group handling --
            if action.depth > 0:
                group_key = action.parent_action_id
                if group_key not in subagent_groups:
                    subagent_groups[group_key] = {
                        "start_time": action.start_time,
                        "tool_count": 0,
                        "subagent_type": action.action_type or "subagent",
                        "first_action": action,
                        "actions": [],
                    }
                if action.role == ActionRole.TOOL:
                    subagent_groups[group_key]["tool_count"] += 1
                subagent_groups[group_key]["actions"].append(action)
                continue

            # -- leaving legacy sub-agent group (no parent_action_id) via depth transition --
            if None in subagent_groups:
                group = subagent_groups.pop(None)
                if do_collapse:
                    self._render_subagent_collapsed(
                        group["first_action"], group["tool_count"], group["start_time"], action
                    )
                else:
                    for buffered in group.get("actions", []):
                        if buffered is group["first_action"]:
                            self._render_subagent_header(buffered, verbose)
                        self._render_subagent_action(buffered, verbose)
                    self._render_subagent_done(group["tool_count"], group["start_time"], action)
                continue

            if pending_task_tool_skips > 0:
                if action.role == ActionRole.TOOL:
                    fn = action.input.get("function_name", "") if action.input else ""
                    if fn == "task":
                        if deferred_groups:
                            grp, end_act = deferred_groups.pop(0)
                            self._render_deferred_group(grp, end_act, action, verbose)
                        pending_task_tool_skips -= 1
                        continue
                    self._flush_deferred_groups(deferred_groups, verbose)

            # -- normal main-agent action --
            self._render_main_action(action, verbose)

        # Flush deferred groups that never got a task tool result
        self._flush_deferred_groups(deferred_groups, verbose)

        # If any sub-agent groups never closed (still active).
        if subagent_groups and _show_partial_done:
            for group in subagent_groups.values():
                if group.get("first_action"):
                    self._render_subagent_header(group["first_action"], verbose)
                for buffered in group.get("actions", []):
                    self._render_subagent_action(buffered, verbose)
                dur_str = ""
                if group["start_time"]:
                    dur_sec = (datetime.now() - group["start_time"]).total_seconds()
                    dur_str = f" \u00b7 {dur_sec:.1f}s"
                summary = f"  \u23bf  Done ({group['tool_count']} tool uses{dur_str})"
                self.console.print(f"[dim]{summary}[/dim]")

    def render_multi_turn_history(
        self,
        turns: List[Tuple[str, List[ActionHistory]]],
        verbose: bool = False,
        per_turn_callback: Optional[Callable[[List[ActionHistory]], None]] = None,
    ) -> None:
        """Render all historical turns, each preceded by a user-message header.

        Args:
            turns: List of (user_message, actions) tuples.
            verbose: If True, show full arguments/output.
            per_turn_callback: If provided, called after rendering each turn's action
                history with the turn's raw actions list. Useful for rendering final
                response output that render_action_history skips.
        """
        for user_message, actions in turns:
            self.renderer.print_renderables(
                self.console,
                [self.renderer.render_user_header(user_message), self.renderer.render_separator()],
            )
            non_user_actions = [a for a in actions if not (a.role == ActionRole.USER and a.depth == 0)]
            self.render_action_history(non_user_actions, verbose=verbose)
            if per_turn_callback:
                per_turn_callback(actions)
            self.console.print()

    def display_streaming_actions(
        self,
        actions: List[ActionHistory],
        history_turns: Optional[List[Tuple[str, List[ActionHistory]]]] = None,
        current_user_message: str = "",
        interaction_broker=None,
    ) -> "InlineStreamingContext":
        """Create an inline streaming display context for actions (Claude Code style)"""
        from datus.cli.action_display.streaming import InlineStreamingContext

        return InlineStreamingContext(
            actions,
            self,
            history_turns=history_turns or [],
            current_user_message=current_user_message,
            interaction_broker=interaction_broker,
        )

    def stop_live(self) -> None:
        """Stop the live display temporarily for user interaction."""
        if self._current_context:
            try:
                self._current_context.stop_display()
            except Exception as e:
                logger.debug(f"Error stopping live display: {e}")

    def restart_live(self) -> None:
        """Restart the live display after user interaction."""
        if self._current_context:
            try:
                self._current_context.restart_display()
            except Exception as e:
                logger.debug(f"Error restarting live display: {e}")


def create_action_display(console: Optional[Console] = None, enable_truncation: bool = True) -> ActionHistoryDisplay:
    """Factory function to create ActionHistoryDisplay with truncation control"""
    return ActionHistoryDisplay(console, enable_truncation)
