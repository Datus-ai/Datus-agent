# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Self-contained ``/datasource`` picker rendered as a single prompt_toolkit
:class:`Application`.

Follows the same architecture as :mod:`datus.cli.model_app`:

* One Application owns ``stdin`` for the entire interaction.
* The outer TUI suspends via :meth:`DatusApp.suspend_input` once.
* Result is returned as a :class:`DatasourceSelection` dataclass.
* Config fields are collected in an inline form (``CONFIG_FORM`` view)
  with Tab/Up/Down navigation and Enter/Ctrl+S to submit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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

from datus.cli.cli_styles import CLR_CURRENT, CLR_CURSOR
from datus.configuration.agent_config import AgentConfig
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

INSTALLABLE_TYPES = (
    "clickhouse",
    "clickzetta",
    "greenplum",
    "hive",
    "mysql",
    "postgresql",
    "redshift",
    "snowflake",
    "spark",
    "starrocks",
    "trino",
)

_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_PW_PLACEHOLDER = "********"


class _View(Enum):
    DATASOURCE_LIST = "datasource_list"
    DATASOURCE_ACTIONS = "datasource_actions"
    TYPE_SELECT = "type_select"
    CONFIG_FORM = "config_form"


_ACTION_EDIT = "edit"
_ACTION_DELETE = "delete"
_ACTION_SET_DEFAULT = "set_default"
_ACTION_INSTALL = "install"

_ACTIONS: List[Tuple[str, str]] = [
    (_ACTION_EDIT, "Edit connection"),
    (_ACTION_DELETE, "Delete datasource"),
    (_ACTION_SET_DEFAULT, "Set as default"),
    (_ACTION_INSTALL, "Install / update adapter"),
]


@dataclass
class DatasourceSelection:
    """Outcome of a :class:`DatasourceApp` run.

    ``kind`` discriminates the payload:

    - ``"switch"`` — user pressed Enter on a datasource; ``name`` filled.
    - ``"add_submit"`` — form submitted for new datasource; ``db_type`` +
      ``payload`` (field dict including ``_name``) filled.
    - ``"edit_submit"`` — form submitted for edit; ``name`` + ``payload`` filled.
    - ``"delete"`` — user confirmed Delete; ``name`` filled.
    - ``"set_default"`` — user chose Set as default; ``name`` filled.
    - ``"install"`` — user chose Install adapter; ``name`` + ``db_type`` filled.
    - ``"needs_install"`` — selected type needs adapter install first;
      ``db_type`` filled.
    """

    kind: str
    name: str = ""
    db_type: str = ""
    needs_install: bool = False
    payload: Optional[Dict[str, Any]] = field(default=None)


class DatasourceApp:
    """Single-screen datasource picker with action sub-menu and config form."""

    def __init__(self, agent_config: AgentConfig, console: Console) -> None:
        self._cfg = agent_config
        self._console = console

        self._view: _View = _View.DATASOURCE_LIST
        self._list_cursor: int = 0
        self._list_offset: int = 0

        self._datasources: List[Tuple[str, str, bool]] = []
        self._current = agent_config.current_datasource or ""
        self._load_datasources()

        self._active_ds_name: str = ""
        self._active_ds_type: str = ""

        self._db_types: List[Tuple[str, str, bool]] = []

        self._pending_delete: bool = False
        self._error_message: Optional[str] = None

        # ── Form state ────────────────────────────────────────────
        self._form_db_type: str = ""
        self._form_edit_name: str = ""
        self._form_field_names: List[str] = []
        self._form_payload_keys: List[str] = []
        self._form_field_meta: List[Dict[str, Any]] = []
        self._form_textareas: List[TextArea] = []
        self._form_focus_order: List[TextArea] = []
        self._form_focus_idx: int = 0
        self._form_container: Optional[HSplit] = None

        self._app = self._build_application()

    def run(self) -> Optional[DatasourceSelection]:
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:
            logger.error("DatasourceApp crashed: %s", exc)
            self._console.print(f"[red]/datasource error:[/] {exc}")
            return None

    async def run_async(self) -> Optional[DatasourceSelection]:
        try:
            return await self._app.run_async()
        except KeyboardInterrupt:
            return None
        except Exception as exc:
            logger.error("DatasourceApp crashed: %s", exc)
            return None

    # ── Data loading ──────────────────────────────────────────────

    def _load_datasources(self) -> None:
        ds_configs = self._cfg.datasource_configs or {}
        services = getattr(self._cfg, "services", None)
        ds_objects = services.datasources if services else {}
        items: List[Tuple[str, str, bool]] = []
        for name in ds_configs:
            db_cfg = ds_objects.get(name)
            db_type = getattr(db_cfg, "type", "") if db_cfg else ""
            is_default = getattr(db_cfg, "default", False) if db_cfg else False
            items.append((name, db_type, is_default))
        self._datasources = items
        if self._current and any(name == self._current for name, _, _ in items):
            self._list_cursor = next(i for i, (n, _, _) in enumerate(items) if n == self._current)
        else:
            self._list_cursor = 0

    def _load_db_types(self) -> None:
        from datus.tools.db_tools import connector_registry

        available = connector_registry.list_available_adapters()
        installed = set(available.keys())
        not_installed = sorted(set(INSTALLABLE_TYPES) - installed)
        items: List[Tuple[str, str, bool]] = []
        for t in sorted(installed):
            items.append((t, t, True))
        for t in not_installed:
            items.append((t, f"{t} (not installed \u2014 will install datus-{t})", False))
        self._db_types = items
        default_idx = next((i for i, (k, _, _) in enumerate(items) if k == "duckdb"), 0)
        self._list_cursor = default_idx
        self._list_offset = 0

    # ── Layout ────────────────────────────────────────────────────

    def _build_application(self) -> Application:
        self._list_window = Window(
            content=FormattedTextControl(self._render_list, focusable=True),
            always_hide_cursor=True,
            height=Dimension(min=3),
        )

        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=1,
        )

        error_window = ConditionalContainer(
            content=Window(
                FormattedTextControl(lambda: [("ansired", f"  {self._error_message or ''}")]),
                height=1,
            ),
            filter=Condition(lambda: bool(self._error_message)),
        )

        root = HSplit(
            [
                DynamicContainer(self._body_container),
                error_window,
                Window(height=1, char="\u2500"),
                hint_window,
            ]
        )

        return Application(
            layout=Layout(root),
            key_bindings=self._build_key_bindings(),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )

    def _body_container(self):
        if self._view == _View.CONFIG_FORM and self._form_container is not None:
            return self._form_container
        return self._list_window

    # ── Rendering ─────────────────────────────────────────────────

    def _render_list(self) -> List[Tuple[str, str]]:
        if self._view == _View.DATASOURCE_LIST:
            return self._render_datasource_list()
        if self._view == _View.DATASOURCE_ACTIONS:
            return self._render_action_menu()
        if self._view == _View.TYPE_SELECT:
            return self._render_type_select()
        return []

    def _render_datasource_list(self) -> List[Tuple[str, str]]:
        total = len(self._datasources) + 1
        self._clamp_cursor(total)
        visible = self._visible_slice(total)
        lines: List[Tuple[str, str]] = []
        start, end = visible
        if end - start < total:
            lines.append(("ansiyellow", f"  ({start + 1}-{end} of {total})\n"))
        for i in range(start, end):
            if i < len(self._datasources):
                name, db_type, is_default = self._datasources[i]
                is_current = name == self._current
                type_label = f" ({db_type})" if db_type else ""
                suffix = "  \u2190 current" if is_current else ""
                default_marker = " *" if is_default and not is_current else ""
                label = f"{name}{type_label}{default_marker}{suffix}"
                if i == self._list_cursor:
                    lines.append((f"{CLR_CURSOR} bold", f"  \u2192 {label}\n"))
                elif is_current:
                    lines.append((CLR_CURRENT, f"    {label}\n"))
                else:
                    lines.append(("", f"    {label}\n"))
            else:
                label = "+ Add datasource\u2026"
                style = f"{CLR_CURSOR} bold" if i == self._list_cursor else "ansibrightblack"
                prefix = "  \u2192 " if i == self._list_cursor else "    "
                lines.append((style, f"{prefix}{label}\n"))
        return lines

    def _render_action_menu(self) -> List[Tuple[str, str]]:
        lines: List[Tuple[str, str]] = [
            ("bold", f"  {self._active_ds_name} ({self._active_ds_type})\n"),
            ("", "\n"),
        ]
        for i, (_, display) in enumerate(_ACTIONS):
            if i == self._list_cursor:
                lines.append((f"{CLR_CURSOR} bold", f"  \u2192 {display}\n"))
            else:
                lines.append(("", f"    {display}\n"))
        return lines

    def _render_type_select(self) -> List[Tuple[str, str]]:
        lines: List[Tuple[str, str]] = [("bold", "  Select database type:\n"), ("", "\n")]
        total = len(self._db_types)
        self._clamp_cursor(total)
        visible = self._visible_slice(total)
        start, end = visible
        if end - start < total:
            lines.append(("ansiyellow", f"  ({start + 1}-{end} of {total})\n"))
        for i in range(start, end):
            key, display, installed = self._db_types[i]
            if i == self._list_cursor:
                lines.append((f"{CLR_CURSOR} bold", f"  \u2192 {display}\n"))
            elif not installed:
                lines.append(("ansibrightblack", f"    {display}\n"))
            else:
                lines.append(("", f"    {display}\n"))
        return lines

    def _render_form_header(self) -> List[Tuple[str, str]]:
        if self._form_edit_name:
            return [("bold", f"  Edit: {self._form_edit_name} ({self._form_db_type})\n")]
        return [("bold", f"  Add Datasource ({self._form_db_type})\n")]

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        if self._view == _View.DATASOURCE_LIST:
            return [("ansibrightblack", "  \u2191\u2193 navigate   Enter switch   \u2192 actions   Esc cancel")]
        if self._view == _View.DATASOURCE_ACTIONS:
            return [("ansibrightblack", "  \u2191\u2193 navigate   Enter select   Esc/\u2190 back")]
        if self._view == _View.TYPE_SELECT:
            return [("ansibrightblack", "  \u2191\u2193 navigate   Enter select   Esc back")]
        if self._view == _View.CONFIG_FORM:
            return [("ansibrightblack", "  Tab/\u2191\u2193 navigate   Enter next/submit   Ctrl+S submit   Esc back")]
        return []

    # ── Cursor / scroll ───────────────────────────────────────────

    def _clamp_cursor(self, total: int) -> None:
        if total <= 0:
            self._list_cursor = 0
            self._list_offset = 0
            return
        self._list_cursor = max(0, min(self._list_cursor, total - 1))

    def _visible_slice(self, total: int) -> Tuple[int, int]:
        max_visible = 15
        if total <= max_visible:
            self._list_offset = 0
            return 0, total
        if self._list_cursor < self._list_offset:
            self._list_offset = self._list_cursor
        elif self._list_cursor >= self._list_offset + max_visible:
            self._list_offset = self._list_cursor - max_visible + 1
        start = max(0, min(self._list_offset, total - max_visible))
        return start, start + max_visible

    # ── State transitions ─────────────────────────────────────────

    def _enter_datasource_list(self) -> None:
        self._view = _View.DATASOURCE_LIST
        self._load_datasources()
        self._error_message = None
        self._pending_delete = False

    def _enter_action_menu(self, name: str, db_type: str) -> None:
        self._active_ds_name = name
        self._active_ds_type = db_type
        self._view = _View.DATASOURCE_ACTIONS
        self._list_cursor = 0
        self._list_offset = 0
        self._error_message = None
        self._pending_delete = False

    def _enter_type_select(self) -> None:
        self._view = _View.TYPE_SELECT
        self._load_db_types()
        self._error_message = None

    def _enter_config_form(self, db_type: str, edit_name: str = "", existing: Optional[Dict[str, Any]] = None) -> None:
        from datus.tools.db_tools import connector_registry

        available = connector_registry.list_available_adapters()
        adapter_metadata = available.get(db_type)
        if not adapter_metadata:
            self._error_message = f"Adapter '{db_type}' not available."
            return
        config_fields = adapter_metadata.get_config_fields()
        if not config_fields:
            self._error_message = f"Adapter '{db_type}' has no configuration schema."
            return

        self._form_db_type = db_type
        self._form_edit_name = edit_name
        self._form_field_names = []
        self._form_payload_keys = []
        self._form_field_meta = []
        self._form_textareas = []

        alias_map = self._build_alias_map(adapter_metadata)

        max_label_len = 0
        fields_to_add: List[Tuple[str, str, Dict[str, Any], str]] = []

        if not edit_name:
            fields_to_add.append(("_name", "_name", {"required": True, "input_type": "text"}, ""))
            max_label_len = len("Datasource name")

        for fn, fi in config_fields.items():
            if fn in ("type", "name"):
                continue
            payload_key = alias_map.get(fn, fn)
            label = fn.replace("_", " ").capitalize()
            max_label_len = max(max_label_len, len(label))
            default_val = fi.get("default", "")
            existing_key = (
                fn
                if (existing and fn in existing)
                else (payload_key if (existing and payload_key in existing) else None)
            )
            if existing and existing_key:
                is_pw = fi.get("input_type") == "password" or fn == "password"
                if is_pw:
                    default_val = _PW_PLACEHOLDER if existing[existing_key] else ""
                else:
                    default_val = existing[existing_key] or ""
            fields_to_add.append((fn, payload_key, fi, str(default_val) if default_val else ""))

        for fn, payload_key, fi, default_text in fields_to_add:
            if fn == "_name":
                display_label = "Datasource name"
            else:
                display_label = fn.replace("_", " ").capitalize()

            required = fi.get("required", False)
            req_mark = " *" if required else "  "
            padded = display_label.ljust(max_label_len)
            prompt_str = f"{padded}{req_mark}: "

            is_pw = fi.get("input_type") == "password" or fn == "password"
            ta = TextArea(
                height=1,
                multiline=False,
                prompt=prompt_str,
                password=is_pw,
                text=default_text,
                focus_on_click=True,
            )
            self._form_field_names.append(fn)
            self._form_payload_keys.append(payload_key)
            self._form_field_meta.append(fi)
            self._form_textareas.append(ta)

        header = Window(
            FormattedTextControl(self._render_form_header, focusable=False),
            height=Dimension(min=1, max=2),
        )
        self._form_container = HSplit([header] + list(self._form_textareas))
        self._form_focus_order = list(self._form_textareas)
        self._form_focus_idx = 0
        self._view = _View.CONFIG_FORM
        self._error_message = None

        if self._form_textareas:
            self._app.layout.focus(self._form_textareas[0])

    # ── Form helpers ──────────────────────────────────────────────

    @staticmethod
    def _build_alias_map(adapter_metadata) -> Dict[str, str]:
        """Return ``{pydantic_field_name: alias}`` for fields that define an alias."""
        cfg_cls = getattr(adapter_metadata, "config_class", None)
        if cfg_cls is None:
            return {}
        model_fields = getattr(cfg_cls, "model_fields", None)
        if model_fields is None:
            return {}
        mapping: Dict[str, str] = {}
        for name, fi in model_fields.items():
            alias = getattr(fi, "alias", None)
            if alias and alias != name:
                mapping[name] = alias
        return mapping

    def _advance_form_focus(self, delta: int) -> None:
        if not self._form_focus_order:
            return
        self._form_focus_idx = (self._form_focus_idx + delta) % len(self._form_focus_order)
        self._app.layout.focus(self._form_focus_order[self._form_focus_idx])

    def _submit_form(self) -> None:
        payload: Dict[str, Any] = {"type": self._form_db_type}
        for i, fn in enumerate(self._form_field_names):
            value = self._form_textareas[i].text.strip()
            meta = self._form_field_meta[i]
            pk = self._form_payload_keys[i]
            required = meta.get("required", False)

            if fn == "_name":
                if not value:
                    self._error_message = "Datasource name is required."
                    self._form_focus_idx = i
                    self._app.layout.focus(self._form_textareas[i])
                    return
                if not _SAFE_NAME_RE.match(value):
                    self._error_message = "Name may only contain letters, digits, hyphens, and underscores."
                    self._form_focus_idx = i
                    self._app.layout.focus(self._form_textareas[i])
                    return
                existing_names = {n for n, _, _ in self._datasources}
                if value in existing_names:
                    self._error_message = f"Datasource '{value}' already exists."
                    self._form_focus_idx = i
                    self._app.layout.focus(self._form_textareas[i])
                    return
                payload["_name"] = value
                continue

            if required and not value:
                label = fn.replace("_", " ").capitalize()
                self._error_message = f"{label} is required."
                self._form_focus_idx = i
                self._app.layout.focus(self._form_textareas[i])
                return

            is_pw = meta.get("input_type") == "password" or fn == "password"
            if is_pw and value == _PW_PLACEHOLDER:
                continue

            if value:
                field_type = meta.get("type", "")
                if field_type == "int" or pk == "port":
                    try:
                        int_val = int(value)
                        if pk == "port" and not (1 <= int_val <= 65535):
                            self._error_message = "Port must be between 1 and 65535."
                            self._form_focus_idx = i
                            self._app.layout.focus(self._form_textareas[i])
                            return
                        payload[pk] = int_val
                    except ValueError:
                        label = fn.replace("_", " ").capitalize()
                        self._error_message = f"{label} must be a valid integer."
                        self._form_focus_idx = i
                        self._app.layout.focus(self._form_textareas[i])
                        return
                else:
                    payload[pk] = value

        if self._form_edit_name:
            self._app.exit(result=DatasourceSelection(kind="edit_submit", name=self._form_edit_name, payload=payload))
        else:
            self._app.exit(result=DatasourceSelection(kind="add_submit", db_type=self._form_db_type, payload=payload))

    # ── Key bindings ──────────────────────────────────────────────

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        is_list = Condition(lambda: self._view in {_View.DATASOURCE_LIST, _View.TYPE_SELECT, _View.DATASOURCE_ACTIONS})
        is_ds_list = Condition(lambda: self._view == _View.DATASOURCE_LIST)
        is_action = Condition(lambda: self._view == _View.DATASOURCE_ACTIONS)
        is_form = Condition(lambda: self._view == _View.CONFIG_FORM)

        def _total() -> int:
            if self._view == _View.DATASOURCE_LIST:
                return len(self._datasources) + 1
            if self._view == _View.DATASOURCE_ACTIONS:
                return len(_ACTIONS)
            if self._view == _View.TYPE_SELECT:
                return len(self._db_types)
            return 0

        # ── List navigation ──────────────────────────────────────
        @kb.add("up", filter=is_list)
        def _(event):
            t = _total()
            if t:
                self._list_cursor = (self._list_cursor - 1) % t
            self._error_message = None
            self._pending_delete = False

        @kb.add("down", filter=is_list)
        def _(event):
            t = _total()
            if t:
                self._list_cursor = (self._list_cursor + 1) % t
            self._error_message = None
            self._pending_delete = False

        @kb.add("pageup", filter=is_list)
        def _(event):
            self._list_cursor = max(0, self._list_cursor - 10)
            self._error_message = None

        @kb.add("pagedown", filter=is_list)
        def _(event):
            self._list_cursor = min(max(0, _total() - 1), self._list_cursor + 10)
            self._error_message = None

        # ── Datasource list Enter / → ────────────────────────────
        @kb.add("enter", filter=is_ds_list)
        def _(event):
            idx = self._list_cursor
            if idx < len(self._datasources):
                name, _, _ = self._datasources[idx]
                event.app.exit(result=DatasourceSelection(kind="switch", name=name))
            else:
                self._enter_type_select()

        @kb.add("right", filter=is_ds_list)
        def _(event):
            idx = self._list_cursor
            if idx < len(self._datasources):
                name, db_type, _ = self._datasources[idx]
                self._enter_action_menu(name, db_type)

        # ── Action menu Enter ────────────────────────────────────
        @kb.add("enter", filter=is_action)
        def _(event):
            action_key, _ = _ACTIONS[self._list_cursor]
            name = self._active_ds_name
            db_type = self._active_ds_type
            if action_key == _ACTION_EDIT:
                services = getattr(self._cfg, "services", None)
                ds_objects = services.datasources if services else {}
                db_cfg = ds_objects.get(name)
                existing: Dict[str, Any] = {}
                if db_cfg:
                    existing = db_cfg.to_dict()
                    if isinstance(getattr(db_cfg, "extra", None), dict):
                        existing.update(db_cfg.extra)
                self._enter_config_form(db_type, edit_name=name, existing=existing)
            elif action_key == _ACTION_DELETE:
                if self._pending_delete:
                    event.app.exit(result=DatasourceSelection(kind="delete", name=name))
                else:
                    self._pending_delete = True
                    self._error_message = f"Delete '{name}'? Press Enter again to confirm, Esc to cancel."
            elif action_key == _ACTION_SET_DEFAULT:
                event.app.exit(result=DatasourceSelection(kind="set_default", name=name))
            elif action_key == _ACTION_INSTALL:
                event.app.exit(result=DatasourceSelection(kind="install", name=name, db_type=db_type))

        # ── Type select Enter ────────────────────────────────────
        @kb.add("enter", filter=Condition(lambda: self._view == _View.TYPE_SELECT))
        def _(event):
            if not self._db_types:
                return
            key, _, installed = self._db_types[self._list_cursor]
            if not installed:
                event.app.exit(result=DatasourceSelection(kind="needs_install", db_type=key))
            else:
                self._enter_config_form(key)

        # ── Escape / back ────────────────────────────────────────
        @kb.add("escape", filter=is_ds_list)
        def _(event):
            event.app.exit(result=None)

        @kb.add("escape", filter=is_action)
        def _(event):
            self._enter_datasource_list()

        @kb.add("left", filter=is_action)
        def _(event):
            self._enter_datasource_list()

        @kb.add("escape", filter=Condition(lambda: self._view == _View.TYPE_SELECT))
        def _(event):
            self._enter_datasource_list()

        # ── Form navigation ──────────────────────────────────────
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
            if self._form_edit_name:
                self._enter_action_menu(self._form_edit_name, self._form_db_type)
            else:
                self._enter_type_select()

        # ── Global cancel ────────────────────────────────────────
        @kb.add("c-c")
        def _(event):
            event.app.exit(result=None)

        return kb


__all__ = ["DatasourceApp", "DatasourceSelection"]
