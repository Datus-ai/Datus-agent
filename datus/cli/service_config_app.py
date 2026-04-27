# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Self-contained ``/services`` configuration TUI.

Two-tab picker (Dashboard / Scheduler) modeled on
:class:`datus.cli.model_app.ModelApp`. The whole interaction — list,
adapter-type picker, credential form — lives inside **one**
prompt_toolkit Application so the outer :class:`DatusApp` only needs to
release ``stdin`` once via :meth:`DatusApp.suspend_input`.

Slow side effects (``pip install``, connection probe, project-default
prompt) are deliberately *not* run inside the Application. The App
returns a :class:`ServiceConfigSelection` describing the user's intent
and :class:`ServiceCommands` orchestrates the install / probe / persist
chain on the outer console, then re-enters the App so the list reflects
the new state.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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

from datus.cli.cli_styles import CLR_CURRENT, CLR_CURSOR, SYM_ARROW, render_tui_title_bar
from datus.configuration.agent_config import AgentConfig
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# Built-in adapter types per section. Picked from the documented set in
# the data-engineering quickstart guide; users are free to extend this
# list at runtime by passing ``extra_types`` to :class:`ServiceConfigApp`
# (e.g. when a private adapter is registered before the TUI opens).
_BUILTIN_TYPES: Dict[str, Tuple[str, ...]] = {
    "bi_platforms": ("superset", "grafana"),
    "schedulers": ("airflow",),
    "semantic_layer": ("metricflow",),
}


_MASKED_PLACEHOLDER = "••••••••"


class _Tab(Enum):
    DASHBOARD = "dashboard"
    SCHEDULER = "scheduler"
    SEMANTIC = "semantic"


_TAB_CYCLE: Tuple[_Tab, ...] = (_Tab.DASHBOARD, _Tab.SCHEDULER, _Tab.SEMANTIC)


_SECTION_OF: Dict[_Tab, str] = {
    _Tab.DASHBOARD: "bi_platforms",
    _Tab.SCHEDULER: "schedulers",
    _Tab.SEMANTIC: "semantic_layer",
}


_TAB_OF_SECTION: Dict[str, _Tab] = {v: k for k, v in _SECTION_OF.items()}


class _View(Enum):
    LIST = "list"
    TYPE_PICKER = "type_picker"
    FORM = "form"


@dataclass
class ServiceConfigSelection:
    """Outcome of a :class:`ServiceConfigApp` run.

    ``action`` discriminates the payload:

    - ``"save"`` — caller should ``ensure_adapter`` → ``hot_reload_adapter``
      → probe → persist + ask "set as project default?". ``payload``
      carries the YAML-shaped service dict; ``name`` carries the entry
      key.
    - ``"delete"`` — caller should drop ``services.<section>.<name>``.
    - ``"test"`` — caller should run a connection probe against the
      already-saved entry and surface the result.
    - ``"set_default"`` — flip the global ``default: true`` flag (only
      issued from the Scheduler tab).
    """

    action: str
    section: str
    name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Entry:
    """Row in the LIST view — pre-flattened from ``DashboardConfig`` /
    ``scheduler_services``. Built by :meth:`ServiceConfigApp._load_entries`."""

    name: str
    adapter_type: str
    is_default: bool  # global ``default: true`` (scheduler) or single-entry
    is_project_default: bool  # project-level pin from .datus/config.yml
    raw: Dict[str, Any]


class ServiceConfigApp:
    """Two-tab service configuration picker.

    The caller is expected to:

    1. Wrap ``app.run()`` in :meth:`DatusApp.suspend_input` when the REPL
       is in TUI mode (no-op otherwise).
    2. Apply the returned :class:`ServiceConfigSelection` via
       :class:`ServiceCommands` and re-enter the app so the LIST reflects
       the post-write state.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        console: Console,
        *,
        initial_tab: str = "dashboard",
        extra_types: Optional[Dict[str, Tuple[str, ...]]] = None,
        status_message: Optional[str] = None,
    ) -> None:
        self._cfg = agent_config
        self._console = console
        self._extra_types = extra_types or {}
        if initial_tab == "scheduler":
            self._tab: _Tab = _Tab.SCHEDULER
        elif initial_tab == "semantic":
            self._tab = _Tab.SEMANTIC
        else:
            self._tab = _Tab.DASHBOARD
        self._view: _View = _View.LIST
        self._list_cursor: int = 0
        self._list_offset: int = 0
        self._error_message: Optional[str] = None
        self._status_message: Optional[str] = status_message
        self._result: Optional[ServiceConfigSelection] = None

        # Type-picker state
        self._type_choices: List[str] = []
        self._type_cursor: int = 0

        # Form state
        self._form_section: str = "bi_platforms"
        self._form_name: str = ""
        self._form_type: str = ""
        self._form_is_edit: bool = False
        self._form_focus_order: List[TextArea] = []
        self._form_focus_idx: int = 0

        # Cached per-tab snapshots so the LIST view paints without
        # recomputing on every render.
        self._dashboard_entries: List[_Entry] = []
        self._scheduler_entries: List[_Entry] = []
        self._semantic_entries: List[_Entry] = []
        self._reload_entries()

        # Form widgets — built up-front so key bindings can reference
        # them. Only the ones relevant to ``_form_section`` get wired
        # into the focus chain at form-entry time.
        self._fld_name = TextArea(height=1, multiline=False, prompt="name:            ", focus_on_click=True)
        self._fld_api_base_url = TextArea(height=1, multiline=False, prompt="api_base_url:    ", focus_on_click=True)
        self._fld_username = TextArea(height=1, multiline=False, prompt="username:        ", focus_on_click=True)
        self._fld_password = TextArea(
            height=1, multiline=False, prompt="password:        ", password=True, focus_on_click=True
        )
        self._fld_api_key = TextArea(
            height=1, multiline=False, prompt="api_key:         ", password=True, focus_on_click=True
        )
        self._fld_datasource_ref = TextArea(height=1, multiline=False, prompt="datasource_ref:  ", focus_on_click=True)
        self._fld_bi_database = TextArea(height=1, multiline=False, prompt="bi_database_name:", focus_on_click=True)
        self._fld_dags_folder = TextArea(height=1, multiline=False, prompt="dags_folder:     ", focus_on_click=True)

        term_height = shutil.get_terminal_size((120, 40)).lines
        # title(1) + tabs(1) + 2 sep(2) + error(1) + status(1) + footer(1) = 7
        self._max_visible: int = max(3, min(15, term_height - 7))

        self._app = self._build_application()

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def run(self) -> Optional[ServiceConfigSelection]:
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("ServiceConfigApp crashed: %s", exc)
            return None

    # ─────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────

    def _reload_entries(self) -> None:
        self._dashboard_entries = self._build_dashboard_entries()
        self._scheduler_entries = self._build_scheduler_entries()
        self._semantic_entries = self._build_semantic_entries()

    def _build_dashboard_entries(self) -> List[_Entry]:
        dashboards = getattr(self._cfg, "dashboard_config", {}) or {}
        active_fn = getattr(self._cfg, "active_dashboard", None)
        active = active_fn() if callable(active_fn) else None
        out: List[_Entry] = []
        for name in sorted(dashboards.keys()):
            cfg = dashboards[name]
            adapter_type = getattr(cfg, "adapter_type", "") or name
            raw = {
                "type": adapter_type,
                "api_base_url": getattr(cfg, "api_base_url", "") or "",
                "username": getattr(cfg, "username", "") or "",
                "password": getattr(cfg, "password", "") or "",
                "api_key": getattr(cfg, "api_key", "") or "",
                "extra": getattr(cfg, "extra", {}) or {},
            }
            dataset_db = getattr(cfg, "dataset_db", None)
            if dataset_db is not None:
                raw["dataset_db"] = {
                    "datasource_ref": getattr(dataset_db, "datasource_ref", "") or "",
                    "bi_database_name": getattr(dataset_db, "bi_database_name", None),
                }
            out.append(
                _Entry(
                    name=name,
                    adapter_type=adapter_type,
                    # ``is_default`` reflects the explicit YAML flag — the
                    # single-entry shortcut is an implicit fallback handled
                    # by the resolver, not a label we want to surface.
                    is_default=bool(getattr(cfg, "default", False)),
                    is_project_default=(name == active),
                    raw=raw,
                )
            )
        return out

    def _build_scheduler_entries(self) -> List[_Entry]:
        services = getattr(self._cfg, "scheduler_services", {}) or {}
        active_fn = getattr(self._cfg, "active_scheduler", None)
        active = active_fn() if callable(active_fn) else None
        out: List[_Entry] = []
        for name in sorted(services.keys()):
            cfg = dict(services[name])
            adapter_type = str(cfg.get("type") or "").strip().lower() or name
            out.append(
                _Entry(
                    name=name,
                    adapter_type=adapter_type,
                    is_default=bool(cfg.get("default")),
                    is_project_default=(name == active),
                    raw=cfg,
                )
            )
        return out

    def _build_semantic_entries(self) -> List[_Entry]:
        # ``init_semantic_layer`` already enforces ``key == type`` and
        # resolves env vars, so iterating ``semantic_layer_configs`` is
        # safe — every entry maps a YAML key (e.g. ``metricflow``) to a
        # dict whose ``type`` field equals that key.
        services = getattr(self._cfg, "semantic_layer_configs", {}) or {}
        active_fn = getattr(self._cfg, "active_semantic", None)
        active = active_fn() if callable(active_fn) else None
        out: List[_Entry] = []
        for name in sorted(services.keys()):
            cfg = dict(services[name])
            adapter_type = str(cfg.get("type") or name).strip().lower()
            out.append(
                _Entry(
                    name=name,
                    adapter_type=adapter_type,
                    is_default=bool(cfg.get("default")),
                    is_project_default=(name == active),
                    raw=cfg,
                )
            )
        return out

    def _entries_for(self, tab: _Tab) -> List[_Entry]:
        if tab == _Tab.DASHBOARD:
            return self._dashboard_entries
        if tab == _Tab.SCHEDULER:
            return self._scheduler_entries
        return self._semantic_entries

    # ─────────────────────────────────────────────────────────────────
    # Layout construction
    # ─────────────────────────────────────────────────────────────────

    def _build_application(self) -> Application:
        title_bar = Window(
            content=FormattedTextControl(lambda: render_tui_title_bar("Datus Services")),
            height=1,
        )
        tab_window = Window(
            content=FormattedTextControl(self._render_tab_strip, focusable=False),
            height=1,
        )
        list_window = Window(
            content=FormattedTextControl(self._render_list, focusable=True),
            always_hide_cursor=True,
            height=Dimension(min=3),
        )
        type_window = Window(
            content=FormattedTextControl(self._render_type_picker, focusable=True),
            always_hide_cursor=True,
            height=Dimension(min=3),
        )
        bi_form = HSplit(
            [
                Window(FormattedTextControl(self._render_form_header, focusable=False), height=Dimension(min=1, max=3)),
                self._fld_name,
                self._fld_api_base_url,
                self._fld_username,
                self._fld_password,
                self._fld_api_key,
                self._fld_datasource_ref,
                self._fld_bi_database,
            ]
        )
        scheduler_form = HSplit(
            [
                Window(FormattedTextControl(self._render_form_header, focusable=False), height=Dimension(min=1, max=3)),
                self._fld_name,
                self._fld_api_base_url,
                self._fld_username,
                self._fld_password,
                self._fld_dags_folder,
            ]
        )

        def _body_container():
            if self._view == _View.TYPE_PICKER:
                return type_window
            if self._view == _View.FORM:
                return bi_form if self._form_section == "bi_platforms" else scheduler_form
            return list_window

        body = DynamicContainer(_body_container)

        error_window = ConditionalContainer(
            content=Window(
                FormattedTextControl(lambda: [("class:service-app.error", f"  {self._error_message or ''}")]),
                height=1,
            ),
            filter=Condition(lambda: bool(self._error_message)),
        )
        status_window = ConditionalContainer(
            content=Window(
                FormattedTextControl(lambda: [("class:service-app.status", f"  {self._status_message or ''}")]),
                height=1,
            ),
            filter=Condition(lambda: bool(self._status_message)),
        )
        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=1,
        )

        root = HSplit(
            [
                title_bar,
                tab_window,
                Window(height=1, char="\u2500"),
                body,
                error_window,
                status_window,
                Window(height=1, char="\u2500"),
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
            (_Tab.DASHBOARD, " Dashboard "),
            (_Tab.SCHEDULER, " Scheduler "),
            (_Tab.SEMANTIC, " Semantic "),
        ):
            style = "reverse bold" if tab == self._tab else ""
            parts.append((style, label))
            parts.append(("", " "))
        parts.append(("class:service-app.tabs-hint", "  (Tab or \u2190/\u2192 to switch)"))
        return parts

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        if self._view == _View.LIST:
            if self._tab == _Tab.SEMANTIC:
                # ``e edit`` is hidden on this tab — metricflow has no
                # editable fields. ``d`` / ``p`` work the same as on the
                # other tabs.
                base = (
                    "  \u2191\u2193 navigate   \u21b5 open   x delete   t test   "
                    "d global default   p project default   Tab switch   Esc cancel"
                )
            else:
                base = (
                    "  \u2191\u2193 navigate   \u21b5 open   e edit   x delete   t test   "
                    "d global default   p project default   Tab switch   Esc cancel"
                )
        elif self._view == _View.TYPE_PICKER:
            base = "  \u2191\u2193 navigate   \u21b5 select   Esc back"
        else:
            base = "  Tab next field   \u21b5 submit on last field   Ctrl+S submit   Esc back"
        return [("class:service-app.hint", base)]

    def _render_list(self) -> List[Tuple[str, str]]:
        entries = self._entries_for(self._tab)
        total_rows = len(entries) + 1  # +1 for the trailing "Add" row
        self._clamp_cursor(total_rows)
        visible = self._visible_slice(total_rows)
        lines: List[Tuple[str, str]] = []
        start, end = visible
        if end - start < total_rows:
            lines.append(("class:service-app.scroll", f"  ({start + 1}-{end} of {total_rows})\n"))
        for i in range(start, end):
            if i < len(entries):
                entry = entries[i]
                marker = "*" if entry.is_project_default else " "
                default_tag = " [default]" if entry.is_default else ""
                label = f"{marker} {entry.name:<22} {entry.adapter_type:<14}{default_tag}"
                style = CLR_CURRENT if entry.is_project_default else ""
            else:
                label = f"  + Add new {self._tab.value}\u2026"
                style = "class:service-app.accent"
            if i == self._list_cursor:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append((style, f"    {label}\n"))
        return lines

    def _render_type_picker(self) -> List[Tuple[str, str]]:
        self._type_cursor = max(0, min(self._type_cursor, len(self._type_choices) - 1))
        if self._tab == _Tab.DASHBOARD:
            section_label = "BI dashboard"
        elif self._tab == _Tab.SCHEDULER:
            section_label = "scheduler"
        else:
            section_label = "semantic layer"
        lines: List[Tuple[str, str]] = [
            ("bold", f"  Pick adapter type for new {section_label}:\n"),
        ]
        for i, type_name in enumerate(self._type_choices):
            installed = self._is_installed(self._tab_section(), type_name)
            tag = "(installed)" if installed else "(will pip install)"
            label = f"{type_name:<18} {tag}"
            if i == self._type_cursor:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append(("", f"    {label}\n"))
        return lines

    def _render_form_header(self) -> List[Tuple[str, str]]:
        verb = "Edit" if self._form_is_edit else "Create"
        section_label = "BI dashboard" if self._form_section == "bi_platforms" else "scheduler"
        type_label = self._form_type or "(?)"
        return [
            ("bold", f"  {verb} {section_label}: type={type_label}\n"),
            ("class:service-app.dim", "  Leave password blank to keep existing.\n"),
        ]

    # ─────────────────────────────────────────────────────────────────
    # Cursor helpers
    # ─────────────────────────────────────────────────────────────────

    def _clamp_cursor(self, total: int) -> None:
        if total <= 0:
            self._list_cursor = 0
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
    # Section / type helpers
    # ─────────────────────────────────────────────────────────────────

    def _tab_section(self) -> str:
        return _SECTION_OF[self._tab]

    def _is_installed(self, section: str, adapter_type: str) -> bool:
        # Imported lazily so unit tests can monkey-patch the helper.
        from datus.cli.service_adapter_installer import is_adapter_installed

        return is_adapter_installed(section, adapter_type)

    def _types_for(self, section: str) -> List[str]:
        builtin = _BUILTIN_TYPES.get(section, ())
        extra = self._extra_types.get(section, ())
        # ``dict.fromkeys`` preserves order while removing duplicates so
        # the picker is deterministic regardless of test injection.
        return list(dict.fromkeys((*builtin, *extra)))

    # ─────────────────────────────────────────────────────────────────
    # State transitions
    # ─────────────────────────────────────────────────────────────────

    def _enter_list(self) -> None:
        self._view = _View.LIST
        self._error_message = None

    def _enter_type_picker(self) -> None:
        self._type_choices = self._types_for(self._tab_section())
        self._type_cursor = 0
        self._view = _View.TYPE_PICKER
        self._error_message = None

    def _enter_form_for_new(self, adapter_type: str) -> None:
        section = self._tab_section()
        # Semantic adapters currently have no user-editable parameters;
        # creating one is a single decision (``which type?``) so we skip
        # the FORM view entirely and emit ``save`` straight from the
        # type-picker. The on-disk shape is just ``{type: <type>}`` —
        # ``init_semantic_layer`` requires the YAML key to equal ``type``.
        if section == "semantic_layer":
            self._result = ServiceConfigSelection(
                action="save",
                section=section,
                name=adapter_type,
                payload={"type": adapter_type},
            )
            self._app.exit(result=self._result)
            return
        self._form_section = section
        self._form_type = adapter_type
        self._form_name = ""
        self._form_is_edit = False
        for ta in (
            self._fld_name,
            self._fld_api_base_url,
            self._fld_username,
            self._fld_password,
            self._fld_api_key,
            self._fld_datasource_ref,
            self._fld_bi_database,
            self._fld_dags_folder,
        ):
            ta.text = ""
        self._fld_name.text = adapter_type  # convenient default
        # Switch view *before* focusing — ``Layout.focus()`` walks the
        # currently visible container tree, and DynamicContainer only
        # exposes the form's TextAreas once ``_body_container()`` returns
        # the form. Focusing while still in LIST view raises
        # ``ValueError: Window does not appear in the layout``.
        self._view = _View.FORM
        self._wire_form_focus()
        self._error_message = None

    def _enter_form_for_edit(self, entry: _Entry) -> None:
        self._form_section = self._tab_section()
        self._form_type = entry.adapter_type
        self._form_name = entry.name
        self._form_is_edit = True
        raw = entry.raw or {}
        self._fld_name.text = entry.name
        self._fld_api_base_url.text = str(raw.get("api_base_url", "") or "")
        self._fld_username.text = str(raw.get("username", "") or "")
        # Mask saved password / api_key so the user can keep them by
        # leaving the field untouched. Submit logic checks for the
        # placeholder before overwriting.
        self._fld_password.text = _MASKED_PLACEHOLDER if raw.get("password") else ""
        self._fld_api_key.text = _MASKED_PLACEHOLDER if raw.get("api_key") else ""
        if self._form_section == "bi_platforms":
            dsdb = raw.get("dataset_db") or {}
            self._fld_datasource_ref.text = str(dsdb.get("datasource_ref", "") or "")
            self._fld_bi_database.text = str(dsdb.get("bi_database_name") or "")
        else:
            self._fld_dags_folder.text = str(raw.get("dags_folder", "") or "")
        # See ``_enter_form_for_new`` — switch view before focusing.
        self._view = _View.FORM
        self._wire_form_focus()
        self._error_message = None

    def _wire_form_focus(self) -> None:
        if self._form_section == "bi_platforms":
            self._form_focus_order = [
                self._fld_name,
                self._fld_api_base_url,
                self._fld_username,
                self._fld_password,
                self._fld_api_key,
                self._fld_datasource_ref,
                self._fld_bi_database,
            ]
        else:
            self._form_focus_order = [
                self._fld_name,
                self._fld_api_base_url,
                self._fld_username,
                self._fld_password,
                self._fld_dags_folder,
            ]
        self._form_focus_idx = 0
        self._app.layout.focus(self._form_focus_order[0])

    # ─────────────────────────────────────────────────────────────────
    # Submit handlers
    # ─────────────────────────────────────────────────────────────────

    def _submit_form(self) -> None:
        name = self._fld_name.text.strip()
        if not name:
            self._error_message = "name is required"
            return
        if self._form_section == "bi_platforms":
            payload = self._build_bi_payload()
        else:
            payload = self._build_scheduler_payload()
        if payload is None:
            return  # error already set
        # Reject collisions on add (edit is allowed to keep the same name).
        if not self._form_is_edit:
            existing_names = {e.name for e in self._entries_for(self._tab)}
            if name in existing_names:
                self._error_message = f"`{name}` already exists in this tab"
                return
        self._result = ServiceConfigSelection(
            action="save",
            section=self._form_section,
            name=name,
            payload=payload,
        )
        self._app.exit(result=self._result)

    def _build_bi_payload(self) -> Optional[Dict[str, Any]]:
        api_base_url = self._fld_api_base_url.text.strip()
        if not api_base_url:
            self._error_message = "api_base_url is required"
            return None
        # password / api_key: keep existing if user left the masked
        # placeholder untouched. Empty string clears the field.
        password = self._read_secret(self._fld_password.text, edit=self._form_is_edit, key="password")
        api_key = self._read_secret(self._fld_api_key.text, edit=self._form_is_edit, key="api_key")
        datasource_ref = self._fld_datasource_ref.text.strip()
        bi_database = self._fld_bi_database.text.strip()
        payload: Dict[str, Any] = {
            "type": self._form_type,
            "api_base_url": api_base_url,
            "username": self._fld_username.text.strip(),
        }
        if password is not None:
            payload["password"] = password
        if api_key is not None:
            payload["api_key"] = api_key
        if datasource_ref or bi_database:
            dataset_db: Dict[str, Any] = {}
            if datasource_ref:
                dataset_db["datasource_ref"] = datasource_ref
            if bi_database:
                dataset_db["bi_database_name"] = bi_database
            payload["dataset_db"] = dataset_db
        return payload

    def _build_scheduler_payload(self) -> Optional[Dict[str, Any]]:
        api_base_url = self._fld_api_base_url.text.strip()
        password = self._read_secret(self._fld_password.text, edit=self._form_is_edit, key="password")
        payload: Dict[str, Any] = {
            "type": self._form_type,
            "api_base_url": api_base_url,
            "username": self._fld_username.text.strip(),
        }
        if password is not None:
            payload["password"] = password
        dags_folder = self._fld_dags_folder.text.strip()
        if dags_folder:
            payload["dags_folder"] = dags_folder
        return payload

    @staticmethod
    def _read_secret(raw: str, *, edit: bool, key: str) -> Optional[str]:
        """Translate masked / empty / new input into the saved value.

        - Edit + placeholder untouched → ``None`` (caller keeps existing).
        - Edit + empty → empty string (clears the field).
        - Add + empty → ``None`` (don't write the key at all).
        - Otherwise → raw text.
        """
        del key  # currently only used for symmetry; reserved for richer logging
        if edit and raw == _MASKED_PLACEHOLDER:
            return None
        if not edit and not raw:
            return None
        return raw

    # ─────────────────────────────────────────────────────────────────
    # Key-binding-driven actions
    # ─────────────────────────────────────────────────────────────────

    def _on_list_enter(self) -> None:
        entries = self._entries_for(self._tab)
        if self._list_cursor >= len(entries):
            self._enter_type_picker()
            return
        # Semantic entries have no editable fields — keep ENTER on the
        # "Add new" row as the only way into the type picker, and ignore
        # it on existing rows.
        if self._tab == _Tab.SEMANTIC:
            return
        self._enter_form_for_edit(entries[self._list_cursor])

    def _on_edit(self) -> None:
        if self._tab == _Tab.SEMANTIC:
            return
        entries = self._entries_for(self._tab)
        if 0 <= self._list_cursor < len(entries):
            self._enter_form_for_edit(entries[self._list_cursor])

    def _on_delete(self) -> None:
        entries = self._entries_for(self._tab)
        if 0 <= self._list_cursor < len(entries):
            entry = entries[self._list_cursor]
            self._result = ServiceConfigSelection(action="delete", section=self._tab_section(), name=entry.name)
            self._app.exit(result=self._result)

    def _on_test(self) -> None:
        entries = self._entries_for(self._tab)
        if 0 <= self._list_cursor < len(entries):
            entry = entries[self._list_cursor]
            self._result = ServiceConfigSelection(action="test", section=self._tab_section(), name=entry.name)
            self._app.exit(result=self._result)

    def _on_set_default(self) -> None:
        entries = self._entries_for(self._tab)
        if 0 <= self._list_cursor < len(entries):
            entry = entries[self._list_cursor]
            self._result = ServiceConfigSelection(
                action="set_default",
                section=self._tab_section(),
                name=entry.name,
            )
            self._app.exit(result=self._result)

    def _on_set_project_default(self) -> None:
        """Pin (or clear) the project-level default for the highlighted entry.

        Pressing ``p`` on a non-default row exits with
        ``action="set_project_default"``; pressing ``p`` on the row that
        is *already* the project default exits with the same action and
        an empty ``name`` so :class:`ServiceCommands` clears the override.
        """
        entries = self._entries_for(self._tab)
        if not (0 <= self._list_cursor < len(entries)):
            return
        entry = entries[self._list_cursor]
        target_name = "" if entry.is_project_default else entry.name
        self._result = ServiceConfigSelection(
            action="set_project_default",
            section=self._tab_section(),
            name=target_name,
        )
        self._app.exit(result=self._result)

    def _on_type_picker_enter(self) -> None:
        if not self._type_choices:
            return
        self._enter_form_for_new(self._type_choices[self._type_cursor])

    # ─────────────────────────────────────────────────────────────────
    # Key bindings
    # ─────────────────────────────────────────────────────────────────

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        is_list = Condition(lambda: self._view == _View.LIST)
        is_type = Condition(lambda: self._view == _View.TYPE_PICKER)
        is_form = Condition(lambda: self._view == _View.FORM)

        @kb.add("up", filter=is_list)
        def _(event):
            total = len(self._entries_for(self._tab)) + 1
            self._list_cursor = (self._list_cursor - 1) % total
            self._error_message = None

        @kb.add("down", filter=is_list)
        def _(event):
            total = len(self._entries_for(self._tab)) + 1
            self._list_cursor = (self._list_cursor + 1) % total
            self._error_message = None

        @kb.add("pageup", filter=is_list)
        def _(event):
            self._list_cursor = max(0, self._list_cursor - 10)

        @kb.add("pagedown", filter=is_list)
        def _(event):
            total = len(self._entries_for(self._tab)) + 1
            self._list_cursor = min(total - 1, self._list_cursor + 10)

        @kb.add("enter", filter=is_list)
        def _(event):
            self._on_list_enter()

        @kb.add("e", filter=is_list)
        def _(event):
            self._on_edit()

        @kb.add("x", filter=is_list)
        def _(event):
            self._on_delete()

        @kb.add("t", filter=is_list)
        def _(event):
            self._on_test()

        @kb.add("d", filter=is_list)
        def _(event):
            self._on_set_default()

        @kb.add("p", filter=is_list)
        def _(event):
            self._on_set_project_default()

        @kb.add("tab", filter=is_list)
        def _(event):
            self._cycle_tab(+1)

        @kb.add("s-tab", filter=is_list)
        def _(event):
            self._cycle_tab(-1)

        @kb.add("right", filter=is_list)
        def _(event):
            self._cycle_tab(+1)

        @kb.add("left", filter=is_list)
        def _(event):
            self._cycle_tab(-1)

        @kb.add("escape", filter=is_list)
        def _(event):
            event.app.exit(result=None)

        # Type-picker ----------------------------------------------------
        @kb.add("up", filter=is_type)
        def _(event):
            if self._type_choices:
                self._type_cursor = (self._type_cursor - 1) % len(self._type_choices)

        @kb.add("down", filter=is_type)
        def _(event):
            if self._type_choices:
                self._type_cursor = (self._type_cursor + 1) % len(self._type_choices)

        @kb.add("enter", filter=is_type)
        def _(event):
            self._on_type_picker_enter()

        @kb.add("escape", filter=is_type)
        def _(event):
            self._enter_list()

        # Form -----------------------------------------------------------
        @kb.add("tab", filter=is_form)
        def _(event):
            self._advance_form_focus(+1)

        @kb.add("s-tab", filter=is_form)
        def _(event):
            self._advance_form_focus(-1)

        @kb.add("down", filter=is_form)
        def _(event):
            self._advance_form_focus(+1)

        @kb.add("up", filter=is_form)
        def _(event):
            self._advance_form_focus(-1)

        @kb.add("enter", filter=is_form)
        def _(event):
            if self._form_focus_idx >= len(self._form_focus_order) - 1:
                self._submit_form()
            else:
                self._advance_form_focus(+1)

        @kb.add("c-s", filter=is_form)
        def _(event):
            self._submit_form()

        @kb.add("escape", filter=is_form)
        def _(event):
            self._enter_list()

        # Global cancel --------------------------------------------------
        @kb.add("c-c")
        def _(event):
            event.app.exit(result=None)

        return kb

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
        self._tab = _TAB_CYCLE[(idx + direction) % len(_TAB_CYCLE)]
        self._list_cursor = 0
        self._list_offset = 0
        self._error_message = None


__all__ = ["ServiceConfigApp", "ServiceConfigSelection"]
