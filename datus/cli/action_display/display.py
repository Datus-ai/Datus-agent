# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Public interface: ActionHistoryDisplay."""

from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple

from rich.console import Console

from datus.cli.action_display.renderers import ActionContentGenerator, ActionRenderer
from datus.schemas.action_history import SUBAGENT_COMPLETE_ACTION_TYPE, ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.action_display.streaming import InlineStreamingContext
    from datus.cli.execution_state import InteractionBroker
    from datus.cli.tui.live_display_state import LiveDisplayState

logger = get_logger(__name__)


class ActionHistoryDisplay:
    """Display ActionHistory in a flat inline format (Claude Code style)"""

    def __init__(
        self,
        console: Optional[Console] = None,
        enable_truncation: bool = True,
        live_state: Optional["LiveDisplayState"] = None,
    ):
        self.console = console or Console()
        self.enable_truncation = enable_truncation
        self.live_state = live_state

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

        # Pre-scan to identify which top-level ``task`` tool calls wrap a
        # sub-agent. ``subagent_complete.parent_action_id`` carries the
        # bare ``call_id`` of the wrapping task tool, so any task tool
        # whose normalised id (after stripping the ``complete_`` prefix
        # that ``openai_compatible``/``claude_model``/``codex_model``
        # apply to the SUCCESS action — see e.g. ``openai_compatible.py``
        # ``action_id="complete_" + call_id``) appears here is a sub-agent
        # invocation and must never be rendered as a standalone task —
        # regardless of arrival order. The ActionBus may yield the outer
        # ``task`` SUCCESS before the queued ``subagent_complete`` (see
        # ``ActionBus.merge`` FIRST_COMPLETED race), and the end-of-turn
        # full-screen reprint replays whatever order it received, so
        # without this set the iterator below would render the task as
        # a standalone block AND also render the collapsed/expanded
        # sub-agent group, producing a duplicate such as
        # ``⏺ explore(...)`` followed by ``⏴ explore(...) Done``.
        subagent_complete_call_ids: Set[Optional[str]] = {
            a.parent_action_id for a in actions if a.action_type == SUBAGENT_COMPLETE_ACTION_TYPE and a.parent_action_id
        }

        def _normalize_call_id(action_id: Optional[str]) -> Optional[str]:
            """Strip the ``complete_`` prefix used by tool SUCCESS actions."""
            if action_id and action_id.startswith("complete_"):
                return action_id[len("complete_") :]
            return action_id

        # Verbose mode renders the sub-agent response after the Done line
        # using the ``task`` tool action's output. Indexing by the bare
        # ``call_id`` keeps the response paired with the right group even
        # when the SUCCESS action carries the ``complete_`` prefix or
        # arrives in a reversed order.
        task_actions_by_call_id: Dict[Optional[str], ActionHistory] = {}
        for a in actions:
            if a.role != ActionRole.TOOL or not a.action_id:
                continue
            normalized = _normalize_call_id(a.action_id)
            if normalized not in subagent_complete_call_ids:
                continue
            fn = a.input.get("function_name", "") if a.input else ""
            if fn == "task":
                task_actions_by_call_id[normalized] = a

        # Locate the trailing plain ``response`` action (depth=0). Its body is
        # the same text the wrapping ``*_response`` carries, and the wrapper's
        # body is painted externally (``_render_final_response`` for redraw;
        # ``_display_markdown_response(content)`` for resume), so rendering
        # this one too would duplicate the final answer. Mid-turn ``response``
        # actions — those emitted alongside tool calls, e.g. brief commentary
        # the model writes before invoking a tool — must NOT be filtered here:
        # the end-of-turn full-screen reprint clears scrollback, and skipping
        # them is what makes that text disappear after the redraw.
        last_plain_response_idx: Optional[int] = None
        for i in range(len(actions) - 1, -1, -1):
            a = actions[i]
            if (
                a.role == ActionRole.ASSISTANT
                and a.depth == 0
                and a.action_type == "response"
                and a.status == ActionStatus.SUCCESS
            ):
                last_plain_response_idx = i
                break

        for idx, action in enumerate(actions):
            # INTERACTION actions: only shown during live interaction, skip in history replay
            if action.role == ActionRole.INTERACTION:
                continue
            # Skip PROCESSING TOOL entries (only render SUCCESS/FAILED)
            if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                continue
            # Terminal wrapper (``chat_response`` etc.): body is painted by
            # the per-turn callback / ``_render_final_response`` pipeline.
            if (
                action.role == ActionRole.ASSISTANT
                and action.action_type
                and action.action_type.endswith("_response")
                and action.depth == 0
            ):
                continue
            # Trailing plain ``response``: duplicates the wrapping
            # ``*_response`` rendered externally — skip to avoid double paint.
            if (
                action.role == ActionRole.ASSISTANT
                and action.action_type == "response"
                and action.depth == 0
                and idx == last_plain_response_idx
            ):
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
                        # Render the expanded group together with its task
                        # tool response immediately when the task action is
                        # already known. Falling back to the deferred queue
                        # only happens for partial slices that do not yet
                        # contain the outer task SUCCESS.
                        task_action = task_actions_by_call_id.get(group_key)
                        if task_action is not None:
                            self._render_deferred_group(group, action, task_action, verbose)
                        else:
                            deferred_groups.append((group, action))
                    pending_task_tool_skips += 1
                continue

            # Order-independent skip: any task tool that wraps a sub-agent
            # (its normalised id appears as a ``subagent_complete.parent_action_id``)
            # is represented by the subagent group rendering above and must
            # not also be rendered as a standalone main-agent action. The
            # SUCCESS action carries an ``action_id`` of ``"complete_" +
            # call_id`` (set by the model layer), while the PROCESSING
            # action uses the bare ``call_id``; both must be skipped.
            if action.role == ActionRole.TOOL and action.action_id:
                normalized = _normalize_call_id(action.action_id)
                if normalized in subagent_complete_call_ids:
                    fn = action.input.get("function_name", "") if action.input else ""
                    if fn == "task":
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
        interaction_broker: Optional["InteractionBroker"] = None,
        streaming_deltas: Optional[List[ActionHistory]] = None,
    ) -> "InlineStreamingContext":
        """Create an inline streaming display context for actions (Claude Code style).

        ``streaming_deltas`` is an optional parallel queue for
        ``thinking_delta`` actions; the caller populates it as delta events
        arrive and clears it on each message boundary. The streaming context
        reads from it for pinned-region live rendering and mid-run Ctrl+O
        re-render of the main body.
        """
        from datus.cli.action_display.streaming import InlineStreamingContext

        return InlineStreamingContext(
            actions,
            self,
            history_turns=history_turns or [],
            current_user_message=current_user_message,
            interaction_broker=interaction_broker,
            live_state=self.live_state,
            streaming_deltas=streaming_deltas,
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


def create_action_display(
    console: Optional[Console] = None,
    enable_truncation: bool = True,
    live_state: Optional["LiveDisplayState"] = None,
) -> ActionHistoryDisplay:
    """Factory function to create ActionHistoryDisplay with truncation control"""
    return ActionHistoryDisplay(console, enable_truncation, live_state=live_state)
