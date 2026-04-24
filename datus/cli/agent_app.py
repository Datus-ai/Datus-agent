# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unified ``/agent`` + ``/subagent`` TUI as a single prompt_toolkit
:class:`Application`.

Merges the legacy default-agent picker (``/agent``) and the text-based
sub-agent manager (``/subagent add|list|remove|update``) into one window
with two tabs:

* **Built-in** — lists :data:`SYS_SUB_AGENTS` (minus
  :data:`HIDDEN_SYS_SUB_AGENTS`). ``Enter`` sets the highlighted agent
  as the current one; ``e`` opens a single-field form that writes a
  ``max_turns`` override to ``agent.yml -> agentic_nodes.<name>``. The
  per-node ``model`` override stays deliberately out of the UI: the
  global ``/model`` picker already owns model selection and letting
  users rebind a built-in node to an arbitrary model here would be an
  easy way to silently break it. Hand-edited ``model`` overrides in
  ``agent.yml`` are preserved on save.
* **Custom** — lists user-defined sub-agents. ``Enter`` sets as
  current; ``e`` / ``a`` / ``d`` (twice) respectively edit / add /
  delete. Wizard invocations hand control back to the caller (see
  :class:`AgentSelection`) so the existing full-screen
  :class:`SubAgentWizard` can be launched *after* this Application
  exits — mirroring the ``needs_oauth`` hand-off pattern in
  :class:`ModelApp`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    DynamicContainer,
    HSplit,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import TextArea
from rich.console import Console

from datus.cli.cli_styles import CLR_CURRENT, CLR_CURSOR, SYM_ARROW, print_error, render_tui_title_bar
from datus.configuration.agent_config import AgentConfig
from datus.utils.constants import HIDDEN_SYS_SUB_AGENTS, SYS_SUB_AGENTS
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class _Tab(Enum):
    BUILTIN = "builtin"
    CUSTOM = "custom"


_TAB_CYCLE: Tuple[_Tab, ...] = (_Tab.BUILTIN, _Tab.CUSTOM)


class _View(Enum):
    AGENT_LIST = "agent_list"
    BUILTIN_EDIT = "builtin_edit"


@dataclass
class AgentSelection:
    """Outcome of an :class:`AgentApp` run.

    ``kind`` discriminates the payload:

    * ``"set_default"`` — caller sets ``default_agent`` to ``name``
      (``"chat"`` resets).
    * ``"new_custom"`` — caller launches the sub-agent wizard for a
      brand-new entry; re-open ``AgentApp`` afterwards.
    * ``"edit_custom"`` — caller launches the wizard for an existing
      custom agent ``name``.
    * ``"delete_custom"`` — caller removes ``name`` from
      ``agent.yml -> agentic_nodes`` and re-opens the app.
    * ``"override_saved"`` — a Built-in override was persisted inside
      the app (no caller action required); the app normally stays open
      and this kind is only exposed by the test hooks.

    ``return_to_tab`` hints the caller which tab to seed when reopening.
    """

    kind: str
    name: Optional[str] = None
    return_to_tab: Optional[str] = None


class AgentApp:
    """Two-tab agent management picker.

    The caller is expected to:

    1. Wrap ``app.run()`` in ``tui_app.suspend_input()`` when the REPL
       is in TUI mode (no-op otherwise).
    2. Apply the returned :class:`AgentSelection` — launching
       :class:`SubAgentWizard` for ``edit_custom`` / ``new_custom``,
       deleting config for ``delete_custom``, or updating
       ``DatusCLI.default_agent`` for ``set_default``. The caller
       typically re-enters the app after applying non-default-set
       outcomes so the user sees the updated list.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        console: Console,
        *,
        default_agent: str = "",
        visible_custom_agents: Optional[Iterable[str]] = None,
        seed_tab: Optional[str] = None,
    ) -> None:
        self._cfg = agent_config
        self._console = console
        self._default_agent = default_agent or ""
        # ``default_agent = ""`` means "chat" in the REPL; normalize here
        # so list highlighting matches the visible row key.
        self._current_default_key = self._default_agent or "chat"
        self._visible_custom: Set[str] = set(visible_custom_agents or [])
        self._seed_tab_arg = seed_tab

        self._tab: _Tab = _Tab.BUILTIN
        self._view: _View = _View.AGENT_LIST
        self._list_cursor: int = 0
        self._list_offset: int = 0

        self._builtin_names: List[str] = self._load_builtin_names()
        self._custom_names: List[str] = self._load_custom_names()

        # Built-in edit state. Only ``max_turns`` is user-editable;
        # ``model`` is intentionally NOT exposed — the global ``/model``
        # selector owns that choice, and the two-way override would
        # otherwise let users silently break a node by pointing it at a
        # model their node_class does not support. The underlying config
        # path (``agent.agentic_nodes.<name>.model``) remains intact;
        # users can still hand-edit it if they really need to.
        self._edit_target: Optional[str] = None
        self._max_turns_input = TextArea(
            height=1,
            multiline=False,
            prompt="max_turns: ",
            focus_on_click=True,
        )

        self._pending_delete_custom: Optional[str] = None
        self._error_message: Optional[str] = None
        self._result: Optional[AgentSelection] = None

        # Leave room for title + tabs + separators + error + hint.
        term_height = shutil.get_terminal_size((120, 40)).lines
        self._max_visible: int = max(3, min(15, term_height - 8))

        self._app = self._build_application()

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def run(self) -> Optional[AgentSelection]:
        """Run the Application. Returns ``None`` on cancel."""
        self._apply_seed()
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("AgentApp crashed: %s", exc)
            print_error(self._console, f"/agent error: {exc}")
            return None

    def _apply_seed(self) -> None:
        if self._seed_tab_arg == "custom":
            self._tab = _Tab.CUSTOM
        elif self._seed_tab_arg == "builtin":
            self._tab = _Tab.BUILTIN
        self._list_cursor = self._initial_cursor_for_tab()
        self._list_offset = 0

    # ─────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────

    def _load_builtin_names(self) -> List[str]:
        names = sorted(SYS_SUB_AGENTS - HIDDEN_SYS_SUB_AGENTS)
        # ``chat`` is the pseudo-default every session starts with. Pin it to
        # the top so users can reset the default from the Built-in tab.
        return ["chat"] + names

    def _load_custom_names(self) -> List[str]:
        nodes = getattr(self._cfg, "agentic_nodes", {}) or {}
        candidates = [name for name in nodes.keys() if name not in SYS_SUB_AGENTS and name != "chat"]
        if self._visible_custom:
            candidates = [n for n in candidates if n in self._visible_custom]
        return sorted(candidates)

    def _initial_cursor_for_tab(self) -> int:
        items = self._builtin_names if self._tab == _Tab.BUILTIN else self._custom_names
        if self._current_default_key in items:
            return items.index(self._current_default_key)
        return 0

    def _current_override(self, name: str) -> Dict[str, Any]:
        nodes = getattr(self._cfg, "agentic_nodes", {}) or {}
        entry = nodes.get(name) or {}
        return {
            "model": entry.get("model"),
            "max_turns": entry.get("max_turns"),
        }

    # ─────────────────────────────────────────────────────────────────
    # Layout construction
    # ─────────────────────────────────────────────────────────────────

    def _build_application(self) -> Application:
        tab_window = Window(
            content=FormattedTextControl(self._render_tab_strip, focusable=False),
            height=1,
            style="class:agent-app.tabs",
        )

        list_window = Window(
            content=FormattedTextControl(self._render_list, focusable=True),
            always_hide_cursor=True,
            style="class:agent-app.list",
            height=Dimension(min=3),
        )

        edit_header = Window(
            FormattedTextControl(self._render_edit_header, focusable=False),
            height=Dimension(min=1, max=3),
        )
        edit_form = HSplit(
            [
                edit_header,
                self._max_turns_input,
            ]
        )

        def _body_container():
            if self._view == _View.BUILTIN_EDIT:
                return edit_form
            return list_window

        body = DynamicContainer(_body_container)

        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=1,
            style="class:agent-app.hint",
        )
        error_window = ConditionalContainer(
            content=Window(
                FormattedTextControl(lambda: [("class:agent-app.error", f"  {self._error_message or ''}")]),
                height=1,
                style="class:agent-app.error",
            ),
            filter=Condition(lambda: bool(self._error_message)),
        )
        title_bar = Window(
            content=FormattedTextControl(lambda: render_tui_title_bar("Agent Management")),
            height=1,
        )

        root = HSplit(
            [
                title_bar,
                tab_window,
                Window(height=1, char="\u2500", style="class:agent-app.separator"),
                body,
                error_window,
                Window(height=1, char="\u2500", style="class:agent-app.separator"),
                hint_window,
            ]
        )

        return Application(
            layout=Layout(root, focused_element=None),
            key_bindings=self._build_key_bindings(),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )

    # ─────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────

    def _render_tab_strip(self) -> List[Tuple[str, str]]:
        parts: List[Tuple[str, str]] = [("", "  ")]
        for tab, label in (
            (_Tab.BUILTIN, " Built-in "),
            (_Tab.CUSTOM, " Custom "),
        ):
            style = "reverse bold" if tab == self._tab else ""
            parts.append((style, label))
            parts.append(("", " "))
        parts.append(("class:agent-app.tabs-hint", "  (Tab or \u2190/\u2192 to switch)"))
        return parts

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        if self._view == _View.AGENT_LIST:
            if self._tab == _Tab.BUILTIN:
                hint = (
                    "  \u2191\u2193 navigate   Enter set as current   e edit   "
                    "Tab/\u2190\u2192 switch   Esc back   Ctrl+C cancel"
                )
            else:
                hint = (
                    "  \u2191\u2193 navigate   Enter set as current   e edit   a add   d delete   "
                    "Tab/\u2190\u2192 switch   Esc back   Ctrl+C cancel"
                )
        else:
            hint = "  Enter / Ctrl+S save   Esc back   Ctrl+C cancel"
        return [("class:agent-app.hint", hint)]

    def _render_list(self) -> List[Tuple[str, str]]:
        items = self._current_list_items()
        if not items:
            empty_msg = "  (no custom agents — press 'a' to add one)\n" if self._tab == _Tab.CUSTOM else "  (empty)\n"
            return [("class:agent-app.dim", empty_msg)]
        self._clamp_cursor(len(items))
        start, end = self._visible_slice(len(items))
        lines: List[Tuple[str, str]] = []
        if end - start < len(items):
            lines.append(("class:agent-app.scroll", f"  ({start + 1}-{end} of {len(items)})\n"))
        for i in range(start, end):
            label, style = items[i]
            if i == self._list_cursor:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append((style, f"    {label}\n"))
        return lines

    def _current_list_items(self) -> List[Tuple[str, str]]:
        if self._tab == _Tab.BUILTIN:
            return self._builtin_items()
        return self._custom_items()

    def _builtin_items(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for name in self._builtin_names:
            override = self._current_override(name)
            fragments = []
            model = override.get("model")
            if model:
                fragments.append(f"model={model}")
            max_turns = override.get("max_turns")
            if max_turns is not None:
                fragments.append(f"max_turns={max_turns}")
            suffix = f"  ({', '.join(fragments)})" if fragments else ""
            label = f"{name}{suffix}"
            is_current = name == self._current_default_key
            if is_current:
                label += "  \u2190 default"
            out.append((label, CLR_CURRENT if is_current else ""))
        return out

    def _custom_items(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for name in self._custom_names:
            override = self._current_override(name)
            fragments = []
            model = override.get("model")
            if model:
                fragments.append(f"model={model}")
            max_turns = override.get("max_turns")
            if max_turns is not None:
                fragments.append(f"max_turns={max_turns}")
            suffix = f"  ({', '.join(fragments)})" if fragments else ""
            label = f"{name}{suffix}"
            is_current = name == self._current_default_key
            if is_current:
                label += "  \u2190 default"
            out.append((label, CLR_CURRENT if is_current else ""))
        out.append(("+ Add agent\u2026", "class:agent-app.accent"))
        return out

    def _render_edit_header(self) -> List[Tuple[str, str]]:
        name = self._edit_target or ""
        current_max = self._current_override(name).get("max_turns") if name else None
        model_hint = self._current_override(name).get("model") if name else None
        lines: List[Tuple[str, str]] = [
            ("bold", f"  Override built-in agent: {name}\n"),
            (
                "class:agent-app.dim",
                "  Only ``max_turns`` is editable here. Leave blank to clear the override "
                "and fall back to the node's built-in default.\n",
            ),
        ]
        if current_max is not None:
            lines.append(
                (
                    "class:agent-app.dim",
                    f"  Current override: max_turns={current_max}\n",
                )
            )
        if model_hint:
            # Surface any hand-edited model override so the user is not
            # surprised that it silently stays in place — the UI no
            # longer lets them change it.
            lines.append(
                (
                    "class:agent-app.dim",
                    f"  (model={model_hint!r} set in agent.yml is preserved; edit the YAML directly to change it.)\n",
                )
            )
        return lines

    # ─────────────────────────────────────────────────────────────────
    # Cursor / scroll helpers (copied from ModelApp pattern)
    # ─────────────────────────────────────────────────────────────────

    def _clamp_cursor(self, total: int) -> None:
        if total <= 0:
            self._list_cursor = 0
            self._list_offset = 0
            return
        if self._list_cursor >= total:
            self._list_cursor = total - 1
        if self._list_cursor < 0:
            self._list_cursor = 0

    def _visible_slice(self, total: int) -> Tuple[int, int]:
        max_visible = self._max_visible
        if total <= max_visible:
            self._list_offset = 0
            return 0, total
        if self._list_cursor < self._list_offset:
            self._list_offset = self._list_cursor
        elif self._list_cursor >= self._list_offset + max_visible:
            self._list_offset = self._list_cursor - max_visible + 1
        start = max(0, min(self._list_offset, total - max_visible))
        return start, start + max_visible

    # ─────────────────────────────────────────────────────────────────
    # State transitions
    # ─────────────────────────────────────────────────────────────────

    def _enter_list_view(self, tab: _Tab) -> None:
        self._tab = tab
        self._view = _View.AGENT_LIST
        self._list_cursor = self._initial_cursor_for_tab()
        self._list_offset = 0
        self._error_message = None

    def _enter_builtin_edit(self, name: str) -> None:
        """Open the single-field override form for built-in ``name``.

        Only ``max_turns`` is editable. Any existing ``model`` override
        is intentionally left untouched — see the docstring on the
        ``_edit_target`` state for the rationale.
        """
        self._edit_target = name
        current_max = self._current_override(name).get("max_turns")
        self._max_turns_input.text = str(current_max) if current_max is not None else ""
        self._view = _View.BUILTIN_EDIT
        self._app.layout.focus(self._max_turns_input)
        self._error_message = None

    # ─────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────

    def _on_list_enter(self) -> None:
        """``Enter`` = set the highlighted row as the current agent.

        The Custom tab's trailing ``+ Add agent…`` sentinel is an
        exception: it isn't an agent, so Enter on that row launches the
        add wizard (consistent with the user's expectation that Enter
        "does the obvious thing").
        """
        if self._tab == _Tab.CUSTOM and self._list_cursor == len(self._custom_names):
            self._exit_with(AgentSelection(kind="new_custom", return_to_tab="custom"))
            return
        name = self._current_row_name()
        if name is None:
            return
        self._exit_with(AgentSelection(kind="set_default", name=name))

    def _on_list_edit(self) -> None:
        """``e`` = open the edit view for the highlighted row.

        Built-in rows open the two-field override form. Custom rows hand
        off to :class:`SubAgentWizard` via an ``edit_custom`` selection.
        The ``chat`` pseudo-row and the ``+ Add agent…`` sentinel have
        no "edit" semantics; the former is a no-op and the latter is
        treated as an add.
        """
        if self._tab == _Tab.BUILTIN:
            if self._list_cursor >= len(self._builtin_names):
                return
            name = self._builtin_names[self._list_cursor]
            if name == "chat":
                self._error_message = "`chat` has no configurable overrides."
                return
            self._enter_builtin_edit(name)
            return

        total = len(self._custom_names)
        if self._list_cursor == total:
            self._exit_with(AgentSelection(kind="new_custom", return_to_tab="custom"))
            return
        if 0 <= self._list_cursor < total:
            name = self._custom_names[self._list_cursor]
            self._exit_with(AgentSelection(kind="edit_custom", name=name, return_to_tab="custom"))

    def _current_row_name(self) -> Optional[str]:
        """Return the agent name at the current cursor, or ``None`` when
        the cursor lands on a non-agent row (Custom's ``+ Add`` sentinel
        or an out-of-range index)."""
        if self._tab == _Tab.BUILTIN:
            if 0 <= self._list_cursor < len(self._builtin_names):
                return self._builtin_names[self._list_cursor]
            return None
        if 0 <= self._list_cursor < len(self._custom_names):
            return self._custom_names[self._list_cursor]
        return None

    def _on_add_custom(self) -> None:
        if self._tab != _Tab.CUSTOM:
            return
        self._exit_with(AgentSelection(kind="new_custom", return_to_tab="custom"))

    def _on_delete_custom(self) -> None:
        if self._tab != _Tab.CUSTOM:
            return
        if self._list_cursor < 0 or self._list_cursor >= len(self._custom_names):
            self._pending_delete_custom = None
            return
        name = self._custom_names[self._list_cursor]
        if self._pending_delete_custom == name:
            self._pending_delete_custom = None
            self._exit_with(AgentSelection(kind="delete_custom", name=name, return_to_tab="custom"))
            return
        self._pending_delete_custom = name
        self._error_message = f"Delete `{name}`? Press d again to confirm, any other key to cancel."

    def _submit_builtin_edit(self) -> None:
        """Persist the active edit form.

        Empty ``max_turns`` / ``<inherit global>`` model map to "clear
        override"; anything else is written through
        :meth:`AgentConfig.set_agentic_node_override`. On success we stay
        inside the app — the user can see the updated row immediately
        and tweak more agents before exiting.
        """
        name = self._edit_target
        if not name:
            self._error_message = "No active agent — this should not happen."
            return
        raw_max = self._max_turns_input.text.strip()
        if raw_max:
            try:
                max_turns: Optional[int] = int(raw_max)
            except ValueError:
                self._error_message = "max_turns must be an integer."
                return
            if max_turns <= 0:
                self._error_message = "max_turns must be a positive integer (or empty to clear)."
                return
        else:
            max_turns = None

        # Preserve any hand-edited ``model`` override: the UI no longer
        # exposes that field, so re-read whatever is already in the
        # config and forward it unchanged. Without this, submitting the
        # form would silently wipe ``model`` alongside the max_turns
        # update.
        existing_model = self._current_override(name).get("model")

        try:
            self._cfg.set_agentic_node_override(name, model=existing_model, max_turns=max_turns)
        except Exception as exc:
            self._error_message = f"Failed to save override: {exc}"
            return
        self._enter_list_view(self._tab)
        if name in self._builtin_names:
            self._list_cursor = self._builtin_names.index(name)

    def _exit_with(self, selection: AgentSelection) -> None:
        self._result = selection
        self._app.exit(result=selection)

    # ─────────────────────────────────────────────────────────────────
    # Key bindings
    # ─────────────────────────────────────────────────────────────────

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        is_list = Condition(lambda: self._view == _View.AGENT_LIST)
        is_custom_list = Condition(lambda: self._view == _View.AGENT_LIST and self._tab == _Tab.CUSTOM)
        is_edit = Condition(lambda: self._view == _View.BUILTIN_EDIT)

        def _clear_pending_delete() -> None:
            self._pending_delete_custom = None

        # ── List navigation ─────────────────────────────────────────
        @kb.add("up", filter=is_list)
        def _(event):
            items = self._current_list_items()
            if not items:
                return
            self._list_cursor = (self._list_cursor - 1) % len(items)
            self._error_message = None
            _clear_pending_delete()

        @kb.add("down", filter=is_list)
        def _(event):
            items = self._current_list_items()
            if not items:
                return
            self._list_cursor = (self._list_cursor + 1) % len(items)
            self._error_message = None
            _clear_pending_delete()

        @kb.add("pageup", filter=is_list)
        def _(event):
            self._list_cursor = max(0, self._list_cursor - 10)
            self._error_message = None
            _clear_pending_delete()

        @kb.add("pagedown", filter=is_list)
        def _(event):
            items = self._current_list_items()
            self._list_cursor = min(max(0, len(items) - 1), self._list_cursor + 10)
            self._error_message = None
            _clear_pending_delete()

        @kb.add("enter", filter=is_list)
        def _(event):
            _clear_pending_delete()
            self._on_list_enter()

        @kb.add("e", filter=is_list)
        def _(event):
            _clear_pending_delete()
            self._on_list_edit()

        @kb.add("a", filter=is_custom_list)
        def _(event):
            _clear_pending_delete()
            self._on_add_custom()

        @kb.add("d", filter=is_custom_list)
        def _(event):
            # Two-press confirmation is managed inside ``_on_delete_custom``.
            self._on_delete_custom()

        @kb.add("tab", filter=is_list)
        def _(event):
            _clear_pending_delete()
            self._cycle_tab(+1)

        @kb.add("s-tab", filter=is_list)
        def _(event):
            _clear_pending_delete()
            self._cycle_tab(-1)

        @kb.add("right", filter=is_list)
        def _(event):
            _clear_pending_delete()
            self._cycle_tab(+1)

        @kb.add("left", filter=is_list)
        def _(event):
            _clear_pending_delete()
            self._cycle_tab(-1)

        @kb.add("escape", filter=is_list)
        def _(event):
            _clear_pending_delete()
            event.app.exit(result=None)

        # ── Built-in edit form ──────────────────────────────────────
        # Single-field form: the TextArea owns focus, Enter / Ctrl+S
        # submit, Esc exits. Tab is intentionally left to the default
        # TextArea behaviour (insert literal tab, which is a no-op in a
        # one-line integer input) so we don't need any focus cycling.
        @kb.add("enter", filter=is_edit)
        def _(event):
            self._submit_builtin_edit()

        @kb.add("c-s", filter=is_edit)
        def _(event):
            self._submit_builtin_edit()

        @kb.add("escape", filter=is_edit)
        def _(event):
            self._enter_list_view(self._tab)

        # ── Global cancel ───────────────────────────────────────────
        @kb.add("c-c")
        def _(event):
            event.app.exit(result=None)

        return kb

    def _cycle_tab(self, direction: int = 1) -> None:
        try:
            idx = _TAB_CYCLE.index(self._tab)
        except ValueError:
            idx = 0
        next_tab = _TAB_CYCLE[(idx + direction) % len(_TAB_CYCLE)]
        self._enter_list_view(next_tab)


__all__ = ["AgentApp", "AgentSelection"]
