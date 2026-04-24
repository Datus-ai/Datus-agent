# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Self-contained ``/skill`` browser rendered as a single prompt_toolkit Application.

Mirrors :mod:`datus.cli.model_app`:

- A single Application hosts tab switching (Installed / Marketplace / Published),
  drill-down (detail view), credential capture (login form), and
  two-press remove confirmation — so the outer :class:`~datus.cli.tui.app.DatusApp`
  only releases stdin once via :meth:`DatusApp.suspend_input`.
- All network I/O (marketplace search refresh, login HTTP POST, install
  download) runs **after** :meth:`SkillApp.run` returns, driven by
  :class:`~datus.cli.skill_commands.SkillCommands`.
- The marketplace list is pre-fetched once per Application instance and
  filtered client-side by ``/`` input; pressing ``R`` exits with
  ``kind="refresh"`` so the caller can reopen the app with a fresh fetch.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import ConditionalContainer, DynamicContainer, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import TextArea
from rich.console import Console

from datus.cli.cli_styles import CLR_CURRENT, CLR_CURSOR, SYM_ARROW, print_error, render_tui_title_bar
from datus.tools.skill_tools.skill_config import SkillMetadata
from datus.tools.skill_tools.skill_manager import SkillManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class _Tab(Enum):
    INSTALLED = "installed"
    MARKETPLACE = "marketplace"
    PUBLISHED = "published"


_TAB_CYCLE: Tuple[_Tab, ...] = (_Tab.INSTALLED, _Tab.MARKETPLACE, _Tab.PUBLISHED)


class _View(Enum):
    LIST = "list"
    DETAIL = "detail"
    LOGIN_FORM = "login_form"
    SEARCH_BAR = "search_bar"


@dataclass
class SkillSelection:
    """Outcome of a :class:`SkillApp` run.

    ``kind`` discriminates the payload:

    - ``"install"`` — ``name`` (+ optional ``version``) for a marketplace install.
    - ``"remove"``  — ``name`` of a locally-installed skill.
    - ``"update"``  — ``name`` of an installed marketplace skill to refresh.
    - ``"login"``   — ``email``, ``password``, ``marketplace_url`` collected.
    - ``"logout"``  — user asked to clear saved credentials.
    - ``"refresh"`` — caller should re-fetch marketplace results and reopen.
    - ``"cancel"``  — user dismissed the app (also returned for ``None``).
    """

    kind: str
    name: Optional[str] = None
    version: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    marketplace_url: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class SkillApp:
    """Three-tab skill browser (Installed / Marketplace / Published).

    The caller is expected to:

    1. Pre-fetch ``installed`` and ``marketplace`` lists, pass them in.
    2. Wrap :meth:`run` in ``tui_app.suspend_input()`` when the REPL is in
       TUI mode (no-op otherwise).
    3. Apply the returned :class:`SkillSelection` by calling the matching
       :class:`~datus.tools.skill_tools.skill_manager.SkillManager` method
       or by re-opening the app after a refresh / login.
    """

    def __init__(
        self,
        manager: SkillManager,
        console: Console,
        *,
        installed: Optional[List[SkillMetadata]] = None,
        marketplace: Optional[List[Dict[str, Any]]] = None,
        seed_tab: Optional[str] = None,
        seed_search: Optional[str] = None,
    ) -> None:
        self._manager = manager
        self._console = console
        self._installed: List[SkillMetadata] = list(installed or [])
        self._marketplace: List[Dict[str, Any]] = list(marketplace or [])

        # Published is synthesised from the installed set — any skill whose
        # source is ``marketplace`` is considered "mine" for MVP purposes.
        # Real ``/api/skills/mine`` wiring is left for a follow-up task.
        self._published: List[SkillMetadata] = [s for s in self._installed if (s.source or "") == "marketplace"]

        initial_tab = _Tab.INSTALLED
        if seed_tab == "marketplace":
            initial_tab = _Tab.MARKETPLACE
        elif seed_tab == "published":
            initial_tab = _Tab.PUBLISHED
        self._tab: _Tab = initial_tab
        self._view: _View = _View.LIST

        self._list_cursor: int = 0
        self._list_offset: int = 0
        self._filter_query: str = (seed_search or "").strip()
        # Pending two-press state: name currently armed for removal.
        self._pending_remove: Optional[str] = None

        self._result: Optional[SkillSelection] = None
        self._error_message: Optional[str] = None

        default_url = getattr(manager.config, "marketplace_url", "") or ""
        self._login_email = TextArea(height=1, multiline=False, prompt="Email:    ", focus_on_click=True)
        self._login_password = TextArea(
            height=1, multiline=False, prompt="Password: ", password=True, focus_on_click=True
        )
        self._login_url = TextArea(height=1, multiline=False, prompt="URL:      ", focus_on_click=True)
        self._login_url.text = default_url
        self._search_input = TextArea(height=1, multiline=False, prompt="/", focus_on_click=True)
        self._search_input.text = self._filter_query

        self._form_focus_order: List[TextArea] = []
        self._form_focus_idx: int = 0

        # title(1) + tabs(1) + 2 separators(2) + error(1) + footer(1) + scroll hint(1) = 7
        term_height = shutil.get_terminal_size((120, 40)).lines
        self._max_visible: int = max(3, min(15, term_height - 7))

        self._app = self._build_application()

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def run(self) -> Optional[SkillSelection]:
        """Run the Application. Returns ``None`` on cancel / Ctrl-C."""
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("SkillApp crashed: %s", exc)
            print_error(self._console, f"/skill error: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────

    def _build_application(self) -> Application:
        tab_window = Window(
            content=FormattedTextControl(self._render_tab_strip, focusable=False),
            height=1,
            style="class:skill-app.tabs",
        )

        list_window = Window(
            content=FormattedTextControl(self._render_list, focusable=True),
            always_hide_cursor=True,
            style="class:skill-app.list",
            height=Dimension(min=3),
        )

        detail_window = Window(
            content=FormattedTextControl(self._render_detail, focusable=False),
            always_hide_cursor=True,
            style="class:skill-app.detail",
            height=Dimension(min=3),
        )

        login_form = HSplit(
            [
                Window(
                    FormattedTextControl(self._render_login_header, focusable=False),
                    height=Dimension(min=1, max=3),
                ),
                self._login_email,
                self._login_password,
                self._login_url,
            ]
        )

        search_bar = HSplit(
            [
                Window(
                    FormattedTextControl(self._render_search_header, focusable=False),
                    height=1,
                ),
                self._search_input,
                list_window,
            ]
        )

        def _body_container():
            if self._view == _View.DETAIL:
                return detail_window
            if self._view == _View.LOGIN_FORM:
                return login_form
            if self._view == _View.SEARCH_BAR:
                return search_bar
            return list_window

        body = DynamicContainer(_body_container)

        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=1,
            style="class:skill-app.hint",
        )
        error_window = ConditionalContainer(
            content=Window(
                FormattedTextControl(lambda: [("class:skill-app.error", f"  {self._error_message or ''}")]),
                height=1,
                style="class:skill-app.error",
            ),
            filter=Condition(lambda: bool(self._error_message)),
        )

        title_bar = Window(
            content=FormattedTextControl(lambda: render_tui_title_bar("Skill Marketplace")),
            height=1,
        )

        root = HSplit(
            [
                title_bar,
                tab_window,
                Window(height=1, char="\u2500", style="class:skill-app.separator"),
                body,
                error_window,
                Window(height=1, char="\u2500", style="class:skill-app.separator"),
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
            (_Tab.INSTALLED, f" Installed ({len(self._installed)}) "),
            (_Tab.MARKETPLACE, f" Marketplace ({len(self._marketplace)}) "),
            (_Tab.PUBLISHED, f" Published ({len(self._published)}) "),
        ):
            style = "reverse bold" if tab == self._tab else ""
            parts.append((style, label))
            parts.append(("", " "))
        parts.append(("class:skill-app.tabs-hint", "  (Tab or \u2190/\u2192 to switch)"))
        return parts

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        if self._view == _View.LIST:
            if self._tab == _Tab.INSTALLED:
                hint = (
                    "  \u2191\u2193 navigate   Enter details   r remove   u update   "
                    "l login   L logout   Tab switch   q/Esc quit"
                )
            elif self._tab == _Tab.MARKETPLACE:
                hint = (
                    "  \u2191\u2193 navigate   Enter details   i install   / filter   R refresh   "
                    "l login   Tab switch   q/Esc quit"
                )
            else:
                hint = "  \u2191\u2193 navigate   Enter details   Tab switch   q/Esc quit"
        elif self._view == _View.DETAIL:
            hint = "  r remove   u update   i install   Esc back   Ctrl+C cancel"
        elif self._view == _View.LOGIN_FORM:
            hint = "  Tab next field   Enter submit   Ctrl+S submit   Esc back   Ctrl+C cancel"
        else:
            hint = "  Type to filter   Enter apply   Esc clear   Ctrl+C cancel"
        return [("class:skill-app.hint", hint)]

    def _render_list(self) -> List[Tuple[str, str]]:
        items = self._current_items()
        if not items:
            return [("class:skill-app.dim", "  (nothing to show)\n")]
        self._clamp_cursor(len(items))
        visible = self._visible_slice(len(items))
        lines: List[Tuple[str, str]] = []
        start, end = visible
        if end - start < len(items):
            lines.append(("class:skill-app.scroll", f"  ({start + 1}-{end} of {len(items)})\n"))
        for i in range(start, end):
            label, style = items[i]
            if i == self._list_cursor:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append((style, f"    {label}\n"))
        return lines

    def _render_detail(self) -> List[Tuple[str, str]]:
        row = self._current_row()
        if row is None:
            return [("class:skill-app.dim", "  (no selection)\n")]
        lines: List[Tuple[str, str]] = []
        for label, value in self._detail_fields(row):
            lines.append(("bold", f"  {label}: "))
            lines.append(("", f"{value}\n"))
        return lines

    def _render_login_header(self) -> List[Tuple[str, str]]:
        return [
            ("bold", "  Log in to Skill Marketplace\n"),
            (
                "class:skill-app.dim",
                "  Credentials are exchanged for a JWT saved under ~/.datus/marketplace_auth.json\n",
            ),
        ]

    def _render_search_header(self) -> List[Tuple[str, str]]:
        return [
            ("bold", "  Filter Marketplace\n"),
        ]

    # ─────────────────────────────────────────────────────────────────
    # Items + detail field mapping per tab
    # ─────────────────────────────────────────────────────────────────

    def _current_items(self) -> List[Tuple[str, str]]:
        if self._tab == _Tab.INSTALLED:
            return [self._format_installed_row(s) for s in self._visible_installed()]
        if self._tab == _Tab.MARKETPLACE:
            return [self._format_marketplace_row(s) for s in self._visible_marketplace()]
        if self._tab == _Tab.PUBLISHED:
            return [self._format_installed_row(s) for s in self._published]
        return []

    def _visible_installed(self) -> List[SkillMetadata]:
        if not self._filter_query:
            return self._installed
        q = self._filter_query.lower()
        return [s for s in self._installed if self._match_installed(s, q)]

    def _visible_marketplace(self) -> List[Dict[str, Any]]:
        if not self._filter_query:
            return self._marketplace
        q = self._filter_query.lower()
        return [s for s in self._marketplace if self._match_marketplace(s, q)]

    @staticmethod
    def _match_installed(skill: SkillMetadata, q: str) -> bool:
        haystack = " ".join(
            filter(
                None,
                [
                    skill.name,
                    skill.description or "",
                    " ".join(skill.tags or []),
                ],
            )
        ).lower()
        return q in haystack

    @staticmethod
    def _match_marketplace(row: Dict[str, Any], q: str) -> bool:
        haystack = " ".join(
            filter(
                None,
                [
                    str(row.get("name", "")),
                    str(row.get("description", "")),
                    " ".join(row.get("tags", []) or []),
                    str(row.get("owner", "")),
                ],
            )
        ).lower()
        return q in haystack

    def _current_row(self) -> Optional[Any]:
        if self._tab == _Tab.INSTALLED:
            rows = self._visible_installed()
        elif self._tab == _Tab.MARKETPLACE:
            rows = self._visible_marketplace()
        elif self._tab == _Tab.PUBLISHED:
            rows = self._published
        else:
            return None
        if not rows or self._list_cursor < 0 or self._list_cursor >= len(rows):
            return None
        return rows[self._list_cursor]

    @staticmethod
    def _format_installed_row(skill: SkillMetadata) -> Tuple[str, str]:
        source = skill.source or "local"
        version = skill.version or "-"
        label = f"{skill.name:<28} v{version:<10} {source}"
        style = CLR_CURRENT if source == "marketplace" else ""
        return label, style

    @staticmethod
    def _format_marketplace_row(row: Dict[str, Any]) -> Tuple[str, str]:
        name = str(row.get("name", "?"))
        version = str(row.get("latest_version", "-"))
        owner = str(row.get("owner", "-"))
        promoted = "\u2605 " if row.get("promoted") else "  "
        label = f"{promoted}{name:<28} v{version:<10} by {owner}"
        return label, ""

    @staticmethod
    def _detail_fields(row: Any) -> List[Tuple[str, str]]:
        if isinstance(row, SkillMetadata):
            return [
                ("Name", row.name),
                ("Version", row.version or "unversioned"),
                ("Source", row.source or "local"),
                ("Location", str(row.location) if row.location else "-"),
                ("Tags", ", ".join(row.tags) if row.tags else "(none)"),
                ("License", row.license or "-"),
                ("Description", row.description or "-"),
            ]
        if isinstance(row, dict):
            return [
                ("Name", str(row.get("name", ""))),
                ("Latest Version", str(row.get("latest_version", "-"))),
                ("Owner", str(row.get("owner", "-"))),
                ("Promoted", "yes" if row.get("promoted") else "no"),
                ("Usage Count", str(row.get("usage_count", 0))),
                ("Tags", ", ".join(row.get("tags") or []) or "(none)"),
                ("Description", str(row.get("description", "") or "-")),
            ]
        return []

    # ─────────────────────────────────────────────────────────────────
    # Cursor / scroll helpers
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

    def _enter_list(self, tab: _Tab) -> None:
        self._tab = tab
        self._view = _View.LIST
        self._list_cursor = 0
        self._list_offset = 0
        self._error_message = None
        self._pending_remove = None

    def _enter_detail(self) -> None:
        if self._current_row() is None:
            return
        self._view = _View.DETAIL
        self._error_message = None
        self._pending_remove = None

    def _enter_login_form(self) -> None:
        self._view = _View.LOGIN_FORM
        self._error_message = None
        self._pending_remove = None
        self._form_focus_order = [self._login_email, self._login_password, self._login_url]
        self._form_focus_idx = 0
        self._app.layout.focus(self._login_email)

    def _enter_search_bar(self) -> None:
        if self._tab != _Tab.MARKETPLACE:
            return
        self._view = _View.SEARCH_BAR
        self._error_message = None
        self._pending_remove = None
        self._form_focus_order = [self._search_input]
        self._form_focus_idx = 0
        self._app.layout.focus(self._search_input)

    # ─────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────

    def _on_install(self) -> None:
        row = self._current_row()
        if not isinstance(row, dict):
            return
        name = str(row.get("name", "")).strip()
        version = str(row.get("latest_version", "") or "").strip() or "latest"
        if not name:
            return
        self._result = SkillSelection(kind="install", name=name, version=version)
        self._app.exit(result=self._result)

    def _on_update(self) -> None:
        row = self._current_row()
        if not isinstance(row, SkillMetadata):
            return
        if (row.source or "") != "marketplace":
            self._error_message = f"`{row.name}` is not marketplace-sourced — nothing to update."
            return
        self._result = SkillSelection(kind="update", name=row.name)
        self._app.exit(result=self._result)

    def _on_remove(self) -> None:
        row = self._current_row()
        if not isinstance(row, SkillMetadata):
            return
        if self._pending_remove == row.name:
            self._pending_remove = None
            self._result = SkillSelection(kind="remove", name=row.name)
            self._app.exit(result=self._result)
            return
        self._pending_remove = row.name
        self._error_message = f"Delete `{row.name}`? Press r again to confirm, any other key to cancel."

    def _on_logout(self) -> None:
        self._result = SkillSelection(kind="logout")
        self._app.exit(result=self._result)

    def _on_refresh(self) -> None:
        self._result = SkillSelection(kind="refresh")
        self._app.exit(result=self._result)

    def _submit_login_form(self) -> None:
        email = self._login_email.text.strip()
        password = self._login_password.text
        url = self._login_url.text.strip() or getattr(self._manager.config, "marketplace_url", "") or ""
        if not email:
            self._error_message = "Email is required"
            return
        if not password:
            self._error_message = "Password is required"
            return
        self._result = SkillSelection(
            kind="login",
            email=email,
            password=password,
            marketplace_url=url,
        )
        self._app.exit(result=self._result)

    def _apply_search_filter(self) -> None:
        self._filter_query = self._search_input.text.strip()
        self._view = _View.LIST
        self._list_cursor = 0
        self._list_offset = 0
        self._error_message = None

    def _cancel_search_filter(self) -> None:
        self._search_input.text = self._filter_query
        self._view = _View.LIST
        self._error_message = None

    # ─────────────────────────────────────────────────────────────────
    # Key bindings
    # ─────────────────────────────────────────────────────────────────

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        is_list = Condition(lambda: self._view == _View.LIST)
        is_detail = Condition(lambda: self._view == _View.DETAIL)
        is_login = Condition(lambda: self._view == _View.LOGIN_FORM)
        is_search = Condition(lambda: self._view == _View.SEARCH_BAR)
        is_installed_list = Condition(lambda: self._view == _View.LIST and self._tab == _Tab.INSTALLED)
        is_marketplace_list = Condition(lambda: self._view == _View.LIST and self._tab == _Tab.MARKETPLACE)

        def _clear_pending() -> None:
            self._pending_remove = None

        # ─── LIST navigation ─────────────────────────────────────────
        @kb.add("up", filter=is_list)
        def _(event):
            items = self._current_items()
            if not items:
                return
            self._list_cursor = (self._list_cursor - 1) % len(items)
            self._error_message = None
            _clear_pending()

        @kb.add("down", filter=is_list)
        def _(event):
            items = self._current_items()
            if not items:
                return
            self._list_cursor = (self._list_cursor + 1) % len(items)
            self._error_message = None
            _clear_pending()

        @kb.add("pageup", filter=is_list)
        def _(event):
            self._list_cursor = max(0, self._list_cursor - 10)
            self._error_message = None
            _clear_pending()

        @kb.add("pagedown", filter=is_list)
        def _(event):
            items = self._current_items()
            self._list_cursor = min(max(0, len(items) - 1), self._list_cursor + 10)
            self._error_message = None
            _clear_pending()

        @kb.add("enter", filter=is_list)
        def _(event):
            _clear_pending()
            self._enter_detail()

        # ─── Tab switching ───────────────────────────────────────────
        @kb.add("tab", filter=is_list)
        def _(event):
            _clear_pending()
            self._cycle_tab(+1)

        @kb.add("s-tab", filter=is_list)
        def _(event):
            _clear_pending()
            self._cycle_tab(-1)

        @kb.add("right", filter=is_list)
        def _(event):
            _clear_pending()
            self._cycle_tab(+1)

        @kb.add("left", filter=is_list)
        def _(event):
            _clear_pending()
            self._cycle_tab(-1)

        # ─── Tab-scoped actions ──────────────────────────────────────
        @kb.add("r", filter=is_installed_list)
        def _(event):
            # ``_on_remove`` manages its own two-press flag.
            self._on_remove()

        @kb.add("u", filter=is_installed_list)
        def _(event):
            _clear_pending()
            self._on_update()

        @kb.add("i", filter=is_marketplace_list)
        def _(event):
            _clear_pending()
            self._on_install()

        @kb.add("/", filter=is_marketplace_list)
        def _(event):
            _clear_pending()
            self._enter_search_bar()

        @kb.add("R", filter=is_marketplace_list)
        def _(event):
            _clear_pending()
            self._on_refresh()

        # ─── Global list shortcuts (l/L/q/Esc) ───────────────────────
        @kb.add("l", filter=is_list)
        def _(event):
            _clear_pending()
            self._enter_login_form()

        @kb.add("L", filter=is_list)
        def _(event):
            _clear_pending()
            self._on_logout()

        @kb.add("q", filter=is_list)
        def _(event):
            event.app.exit(result=SkillSelection(kind="cancel"))

        @kb.add("escape", filter=is_list)
        def _(event):
            event.app.exit(result=SkillSelection(kind="cancel"))

        # ─── Detail view ─────────────────────────────────────────────
        @kb.add("escape", filter=is_detail)
        def _(event):
            self._view = _View.LIST

        @kb.add("r", filter=is_detail)
        def _(event):
            if self._tab == _Tab.INSTALLED or self._tab == _Tab.PUBLISHED:
                self._on_remove()

        @kb.add("u", filter=is_detail)
        def _(event):
            if self._tab == _Tab.INSTALLED:
                self._on_update()

        @kb.add("i", filter=is_detail)
        def _(event):
            if self._tab == _Tab.MARKETPLACE:
                self._on_install()

        # ─── Login form ──────────────────────────────────────────────
        @kb.add("tab", filter=is_login)
        def _(event):
            self._advance_form_focus(+1)

        @kb.add("s-tab", filter=is_login)
        def _(event):
            self._advance_form_focus(-1)

        @kb.add("down", filter=is_login)
        def _(event):
            self._advance_form_focus(+1)

        @kb.add("up", filter=is_login)
        def _(event):
            self._advance_form_focus(-1)

        @kb.add("enter", filter=is_login)
        def _(event):
            if self._form_focus_idx >= len(self._form_focus_order) - 1:
                self._submit_login_form()
            else:
                self._advance_form_focus(+1)

        @kb.add("c-s", filter=is_login)
        def _(event):
            self._submit_login_form()

        @kb.add("escape", filter=is_login)
        def _(event):
            self._view = _View.LIST
            self._error_message = None

        # ─── Search bar ──────────────────────────────────────────────
        @kb.add("enter", filter=is_search)
        def _(event):
            self._apply_search_filter()

        @kb.add("escape", filter=is_search)
        def _(event):
            self._cancel_search_filter()

        # ─── Global cancel ───────────────────────────────────────────
        @kb.add("c-c")
        def _(event):
            event.app.exit(result=None)

        return kb

    # ─────────────────────────────────────────────────────────────────
    # Focus / tab helpers
    # ─────────────────────────────────────────────────────────────────

    def _advance_form_focus(self, delta: int) -> None:
        if not self._form_focus_order:
            return
        self._form_focus_idx = (self._form_focus_idx + delta) % len(self._form_focus_order)
        self._app.layout.focus(self._form_focus_order[self._form_focus_idx])

    def _cycle_tab(self, direction: int = 1) -> None:
        try:
            idx = _TAB_CYCLE.index(self._tab)
        except ValueError:
            idx = 0
        next_tab = _TAB_CYCLE[(idx + direction) % len(_TAB_CYCLE)]
        self._enter_list(next_tab)


__all__ = ["SkillApp", "SkillSelection"]
