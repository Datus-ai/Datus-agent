# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Self-contained ``/plugins`` manager rendered as one prompt_toolkit
:class:`Application`.

Mirrors :class:`datus.cli.model_app.ModelApp`: a single Application hosts the
whole flow (plugin list → profile list → profile form) so the outer
:class:`datus.cli.tui.app.DatusApp` only releases ``stdin`` once via
:meth:`run_wizard`. Two concerns are managed here:

1. **Global profile CRUD** — create / edit / delete ``agent.plugins.<plugin>
   .<profile>`` entries in agent.yml, with the form fields derived from the
   plugin manifest's ``config_schema`` (a JSON Schema; nested objects expand
   into dotted fields like ``s3.secret_access_key`` and are re-assembled into
   the nested profile shape on save) and candidate profiles validated against
   it before saving.
2. **Project activation** — toggle a plugin's ``enabled`` flag and pick which
   profiles are active, persisted to ``./.datus/config.yml``.

All mutations are applied directly to the shared :class:`AgentConfig` (and its
YAML files) as they happen, so the app simply returns when the user exits.
"""

from __future__ import annotations

import asyncio
import re
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from datus.cli.tui.wizard_host import EmbeddedWizard

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import ConditionalContainer, DynamicContainer, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor
from prompt_toolkit.widgets import TextArea
from rich.console import Console

from datus.cli.cli_styles import (
    CLR_CURRENT,
    CLR_CURSOR,
    STATUS_BAR_FG_HINT,
    SYM_ARROW,
    SYM_CHECK,
    print_error,
    render_tui_title_bar,
)
from datus.cli.plugin_service import list_plugins
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_MAX_LIST_ROWS = 12

# Placeholder text (a field's schema description shown while it is empty).
_PLACEHOLDER_STYLE = f"italic fg:{STATUS_BAR_FG_HINT}"

# A ``${ENV_VAR}`` reference (optionally ``${VAR:-default}``). Secret fields must
# carry one so a literal credential is never persisted into agent.yml.
_ENV_REF_RE = re.compile(r"\$\{[^}]+\}")


# Nested config_schema objects surface in the form as flat dotted field names
# (``s3.secret_access_key``); profiles stay nested dicts in agent.yml. These
# helpers translate between the two shapes.


def _nested_get(config: Dict[str, Any], dotted: str) -> Any:
    """Read a dotted path out of a nested dict; ``None`` when absent."""
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _nested_set(config: Dict[str, Any], dotted: str, value: Any) -> None:
    """Write a dotted path into a nested dict, creating intermediate dicts."""
    parts = dotted.split(".")
    current = config
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict values into dotted keys (free-form edit fallback)."""
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        if not isinstance(key, str):
            continue
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            flat.update(_flatten_config(value, prefix=f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


class _View(Enum):
    PLUGIN_LIST = "plugin_list"
    PROFILE_LIST = "profile_list"
    PROFILE_FORM = "profile_form"


class PluginApp:
    """Interactive plugin manager (list / profile CRUD / activation).

    The caller wraps ``app.run()`` in ``tui_app.suspend_input()`` when the REPL
    is in TUI mode (no-op otherwise); the embedded path goes through
    :meth:`build_embedded_panel` via :meth:`DatusApp.run_wizard`.
    """

    def __init__(self, agent_config, console: Console):
        self._agent_config = agent_config
        self._console = console

        self._view = _View.PLUGIN_LIST
        self._plugins = list_plugins(agent_config)
        self._plugin_cursor = 0
        self._profile_cursor = 0
        self._selected_plugin: Optional[str] = None
        self._error_message: Optional[str] = None
        self._pending_delete: Optional[str] = None

        # Form state (rebuilt on entry; fields are config_schema-driven).
        self._form_mode = "new"  # or "edit"
        self._form_specs: List[dict] = []
        self._form_name_input: Optional[TextArea] = None
        self._form_inputs: List[TextArea] = []
        self._form_focus_order: List[TextArea] = []
        self._form_focus_idx = 0

        self._on_done = None
        self._app: Optional[Application] = None
        self._list_window: Optional[Window] = None

    # ─────────────────────────────────────────────────────────────────
    # Public API (mirrors ModelApp)
    # ─────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Run as a transient ``Application(full_screen=False)`` (non-TUI)."""
        kb = self._build_key_bindings()
        root = self._build_root_container(kb)
        self._app = Application(
            layout=Layout(root, focused_element=self._list_window),
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )
        self._on_done = lambda result: self._app.exit(result=result)
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("PluginApp crashed: %s", exc)
            print_error(self._console, f"/plugins error: {exc}")
            return None
        finally:
            self._on_done = None
            self._app = None

    def build_embedded_panel(self, done_future: "asyncio.Future") -> "EmbeddedWizard":
        """Build the panel mounted in the parent :class:`DatusApp`'s bottom slot."""
        from datus.cli.tui.wizard_host import EmbeddedWizard

        self._on_done = lambda result: self._finish_via_future(done_future, result)
        kb = self._build_key_bindings()
        root = self._build_root_container(kb)
        return EmbeddedWizard(
            container=root,
            key_bindings=kb,
            first_focus=self._list_window,
            done_future=done_future,
        )

    # ─────────────────────────────────────────────────────────────────
    # Finish hooks (shared standalone/embedded)
    # ─────────────────────────────────────────────────────────────────

    def _finish(self, result=None) -> None:
        if self._on_done is None:
            return
        self._on_done(result)

    @staticmethod
    def _finish_via_future(done_future: "asyncio.Future", result) -> None:
        from datus.cli.tui.wizard_host import resolve_cancel, resolve_with

        if result is None:
            resolve_cancel(done_future)
        else:
            resolve_with(done_future, result)

    def _layout(self) -> Optional[Layout]:
        if self._app is not None:
            return self._app.layout
        try:
            from prompt_toolkit.application import get_app

            return get_app().layout
        except Exception:
            return None

    def _focus(self, target) -> None:
        layout = self._layout()
        if layout is None or target is None:
            return
        try:
            layout.focus(target)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("PluginApp focus(%r) failed: %s", target, exc)

    # ─────────────────────────────────────────────────────────────────
    # Data model
    # ─────────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """Re-read plugin/profile/activation state from the agent config."""
        self._plugins = list_plugins(self._agent_config)

    def _profiles_of(self, plugin: str) -> List[str]:
        services = getattr(self._agent_config, "plugin_services", {}) or {}
        return sorted((services.get(plugin) or {}).keys())

    def _plugin_items(self) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = []
        for info in self._plugins:
            mark = SYM_CHECK if info.active else " "
            profiles = f"{len(info.profiles)} profile(s)" if info.profiles else "no profiles"
            label = f"[{mark}] {info.name}  ({profiles})"
            style = CLR_CURRENT if info.active else "class:plugin-app.dim"
            items.append((label, style))
        return items

    def _profile_items(self) -> List[Tuple[str, str]]:
        plugin = self._selected_plugin or ""
        profiles = self._profiles_of(plugin)
        active = self._agent_config.active_plugin_profiles(plugin)  # None=all, []=inactive, or list
        items: List[Tuple[str, str]] = []
        for name in profiles:
            is_active = active is None or name in active
            mark = SYM_CHECK if is_active else " "
            items.append((f"[{mark}] {name}", CLR_CURRENT if is_active else "class:plugin-app.dim"))
        return items

    def _current_items(self) -> List[Tuple[str, str]]:
        if self._view == _View.PLUGIN_LIST:
            return self._plugin_items()
        if self._view == _View.PROFILE_LIST:
            return self._profile_items()
        return []

    def _cursor(self) -> int:
        return self._plugin_cursor if self._view == _View.PLUGIN_LIST else self._profile_cursor

    def _set_cursor(self, value: int) -> None:
        if self._view == _View.PLUGIN_LIST:
            self._plugin_cursor = value
        else:
            self._profile_cursor = value

    # ─────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────

    def _build_root_container(self, kb: KeyBindings) -> HSplit:
        self._list_window = Window(
            content=FormattedTextControl(self._render_list, focusable=True, key_bindings=kb),
            always_hide_cursor=True,
            style="class:plugin-app.list",
            height=Dimension(min=3),
        )

        def _body_container():
            if self._view == _View.PROFILE_FORM:
                return self._form_container()
            return self._list_window

        body = DynamicContainer(_body_container)

        header = Window(
            content=FormattedTextControl(self._render_header, focusable=False),
            height=Dimension(min=1, max=3),
        )
        error_window = ConditionalContainer(
            content=Window(
                FormattedTextControl(lambda: [("class:plugin-app.error", f"  {self._error_message or ''}")]),
                height=1,
                style="class:plugin-app.error",
            ),
            filter=Condition(lambda: bool(self._error_message)),
        )
        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=1,
            style="class:plugin-app.hint",
        )
        title_bar = Window(
            content=FormattedTextControl(lambda: render_tui_title_bar("Plugins")),
            height=1,
        )
        return HSplit(
            [
                title_bar,
                header,
                Window(height=1, char="─", style="class:plugin-app.separator"),
                body,
                error_window,
                Window(height=1, char="─", style="class:plugin-app.separator"),
                hint_window,
            ]
        )

    def _form_container(self):
        rows = [
            Window(
                FormattedTextControl(self._render_form_header, focusable=False),
                height=Dimension(min=1, max=3),
            )
        ]
        if self._form_name_input is not None:
            rows.append(self._form_name_input)
        rows.extend(self._form_inputs)
        return HSplit(rows)

    # ─────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────

    def _render_header(self) -> List[Tuple[str, str]]:
        if self._view == _View.PLUGIN_LIST:
            return [("bold", "  Installed plugins")]
        if self._view == _View.PROFILE_LIST:
            plugin = self._selected_plugin or ""
            enabled = self._agent_config.plugin_active(plugin)
            state = "enabled" if enabled else "disabled"
            return [
                ("bold", f"  {plugin}"),
                ("class:plugin-app.dim", f"   (project: {state})"),
            ]
        return [("bold", "  ")]

    def _render_form_header(self) -> List[Tuple[str, str]]:
        verb = "Edit" if self._form_mode == "edit" else "New"
        plugin = self._selected_plugin or ""
        return [
            ("bold", f"  {verb} profile for {plugin}\n"),
            (
                "class:plugin-app.dim",
                "  Secrets: enter a ${ENV_VAR} reference, never a literal value. Ctrl+S to save.\n",
            ),
        ]

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        if self._view == _View.PLUGIN_LIST:
            hint = "  ↑↓ navigate   Enter profiles   Space enable/disable   Esc close   Ctrl+C cancel"
        elif self._view == _View.PROFILE_LIST:
            hint = (
                "  ↑↓ navigate   Enter toggle-active   Space enable/disable plugin   "
                "n new   e edit   x delete   Esc back"
            )
        else:
            hint = "  Tab next field   Ctrl+S save   Esc back   Ctrl+C cancel"
        return [("class:plugin-app.hint", hint)]

    def _render_list(self) -> List[Tuple[str, str]]:
        items = self._current_items()
        if not items:
            empty = "  (no plugins installed — `datus plugin install <source>`)\n"
            if self._view == _View.PROFILE_LIST:
                empty = "  (no profiles configured — press 'n' to add one)\n"
            return [("class:plugin-app.dim", empty)]
        cursor = max(0, min(self._cursor(), len(items) - 1))
        self._set_cursor(cursor)
        start = 0
        end = len(items)
        if len(items) > _MAX_LIST_ROWS:
            start = max(0, min(cursor - _MAX_LIST_ROWS // 2, len(items) - _MAX_LIST_ROWS))
            end = start + _MAX_LIST_ROWS
        lines: List[Tuple[str, str]] = []
        if end - start < len(items):
            lines.append(("class:plugin-app.scroll", f"  ({start + 1}-{end} of {len(items)})\n"))
        for i in range(start, end):
            label, style = items[i]
            if i == cursor:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append((style, f"    {label}\n"))
        return lines

    # ─────────────────────────────────────────────────────────────────
    # Navigation / actions
    # ─────────────────────────────────────────────────────────────────

    def _enter_profile_list(self, plugin: str) -> None:
        self._selected_plugin = plugin
        self._view = _View.PROFILE_LIST
        self._profile_cursor = 0
        self._error_message = None
        self._pending_delete = None

    def _enter_plugin_list(self) -> None:
        self._view = _View.PLUGIN_LIST
        self._error_message = None
        self._pending_delete = None
        self._refresh()

    def _toggle_plugin_enabled(self, plugin: str) -> None:
        currently = self._agent_config.plugin_active(plugin)
        try:
            self._agent_config.set_plugin_activation(plugin, enabled=not currently)
        except Exception as exc:  # noqa: BLE001 - surface, do not crash the app
            self._error_message = f"Failed to update activation: {exc}"
            return
        self._error_message = None
        self._refresh()

    def _toggle_profile_active(self, plugin: str, profile: str) -> None:
        all_profiles = self._profiles_of(plugin)
        current = self._agent_config.active_plugin_profiles(plugin)
        materialized = set(all_profiles if current is None else current)
        if profile in materialized:
            materialized.discard(profile)
        else:
            materialized.add(profile)
        try:
            if not materialized:
                # Deselecting the last active profile disables the plugin rather
                # than persisting enabled=True with an empty pin — which
                # ``plugin_active()`` reads as active and an empty pin reads as
                # "no narrowing → all profiles", the opposite of what the user
                # just did.
                self._agent_config.set_plugin_activation(plugin, enabled=False, clear_profiles=True)
            elif materialized == set(all_profiles):
                # Back to "all profiles active" — store as None for cleanliness.
                self._agent_config.set_plugin_activation(plugin, enabled=True, clear_profiles=True)
            else:
                self._agent_config.set_plugin_activation(plugin, enabled=True, active_profiles=sorted(materialized))
        except Exception as exc:  # noqa: BLE001
            self._error_message = f"Failed to update active profiles: {exc}"
            return
        self._error_message = None
        self._refresh()

    def _open_profile_form(self, mode: str, profile: Optional[str] = None) -> None:
        plugin = self._selected_plugin or ""
        from datus.plugins.registry import plugin_config_schema

        self._form_mode = mode
        self._error_message = None
        specs = plugin_config_schema(plugin)
        existing = {}
        if mode == "edit" and profile:
            existing = dict((getattr(self._agent_config, "plugin_services", {}).get(plugin, {}) or {}).get(profile, {}))
        # Free-form fallback: no schema → surface the (flattened) keys already present.
        if not specs:
            specs = [
                {"name": key, "description": "", "required": False, "secret": False}
                for key in _flatten_config(existing)
                if key != "name"
            ] or [{"name": "value", "description": "", "required": False, "secret": False}]
        self._form_specs = specs

        # Profile-name field only when creating (name is the map key).
        if mode == "new":
            self._form_name_input = TextArea(
                multiline=False, height=1, prompt="profile name: ", style="class:plugin-app.input"
            )
        else:
            self._form_name_input = None
        self._form_profile_name = profile or ""

        self._form_inputs = []
        for spec in specs:
            name = spec["name"]
            is_secret = bool(spec.get("secret"))
            value = _nested_get(existing, name)
            if value is None and "default" in spec:
                value = spec["default"]
            prompt = f"{name}{' *' if spec.get('required') else ''}: "
            # Secret fields are masked and never pre-filled with the stored
            # value; a blank secret on edit keeps the current value.
            text = "" if is_secret else ("" if value is None else str(value))
            self._form_inputs.append(self._build_form_input(spec, text=text, prompt=prompt, is_secret=is_secret))

        self._form_focus_order = ([self._form_name_input] if self._form_name_input else []) + self._form_inputs
        self._form_focus_idx = 0
        self._view = _View.PROFILE_FORM
        self._focus(self._form_focus_order[0] if self._form_focus_order else self._list_window)

    @staticmethod
    def _build_form_input(spec: dict, *, text: str, prompt: str, is_secret: bool) -> TextArea:
        """One form field; while it is empty (no pre-filled default), its
        schema ``description`` shows as a dim placeholder that disappears as
        soon as the user types."""
        placeholder = str(spec.get("description") or "").strip()
        processors = None
        if placeholder:
            holder: Dict[str, TextArea] = {}
            processors = [
                ConditionalProcessor(
                    AfterInput(placeholder, style=_PLACEHOLDER_STYLE),
                    filter=Condition(lambda: not holder["area"].text),
                )
            ]
        area = TextArea(
            text=text,
            multiline=False,
            height=1,
            password=is_secret,
            prompt=prompt,
            style="class:plugin-app.input",
            input_processors=processors,
        )
        if placeholder:
            holder["area"] = area
        return area

    def _submit_profile_form(self) -> None:
        plugin = self._selected_plugin or ""
        if self._form_mode == "new":
            name = (self._form_name_input.text if self._form_name_input else "").strip()
            if not name:
                self._error_message = "Profile name is required."
                return
            if name in self._profiles_of(plugin):
                self._error_message = f"Profile `{name}` already exists."
                return
        else:
            name = self._form_profile_name

        existing = dict((getattr(self._agent_config, "plugin_services", {}).get(plugin, {}) or {}).get(name, {}))
        config: Dict[str, Any] = {}
        for spec, area in zip(self._form_specs, self._form_inputs):
            field_name = spec["name"]
            raw = area.text.strip()
            is_secret = bool(spec.get("secret"))
            if is_secret and raw == "" and self._form_mode == "edit":
                # Blank secret on edit keeps the previously-stored value.
                previous = _nested_get(existing, field_name)
                if previous is not None:
                    _nested_set(config, field_name, previous)
                continue
            if raw == "":
                continue
            if is_secret and not _ENV_REF_RE.search(raw):
                # A secret must reference an env var (``${ENV_VAR}``), never a
                # literal credential — those would be persisted verbatim into
                # agent.yml.
                self._error_message = f"`{field_name}` is a secret: use a ${{ENV_VAR}} reference, not a literal value."
                return
            _nested_set(config, field_name, raw)

        # Validate required fields locally, then against the config schema.
        missing = [s["name"] for s in self._form_specs if s.get("required") and _nested_get(config, s["name"]) is None]
        if missing:
            self._error_message = f"Missing required field(s): {', '.join(missing)}"
            return
        from datus.plugins.registry import plugin_validate_profile

        errors = plugin_validate_profile(plugin, config)
        if errors:
            self._error_message = errors[0]
            return

        try:
            self._agent_config.save_plugin_profile(plugin, name, config)
        except Exception as exc:  # noqa: BLE001
            self._error_message = f"Failed to save profile: {exc}"
            return
        self._enter_profile_list(plugin)
        self._focus(self._list_window)

    def _delete_profile(self, plugin: str, profile: str) -> None:
        # Two-press confirmation, mirroring ModelApp's custom-model delete.
        if self._pending_delete != profile:
            self._pending_delete = profile
            self._error_message = f"Press x again to delete profile `{profile}`."
            return
        self._pending_delete = None
        try:
            removed = self._agent_config.delete_plugin_profile(plugin, profile)
        except Exception as exc:  # noqa: BLE001
            self._error_message = f"Failed to delete profile: {exc}"
            return
        self._error_message = None if removed else f"Profile `{profile}` not found."
        self._profile_cursor = 0
        self._refresh()

    def _advance_form_focus(self, delta: int) -> None:
        if not self._form_focus_order:
            return
        self._form_focus_idx = (self._form_focus_idx + delta) % len(self._form_focus_order)
        self._focus(self._form_focus_order[self._form_focus_idx])

    # ─────────────────────────────────────────────────────────────────
    # Key bindings
    # ─────────────────────────────────────────────────────────────────

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        is_list = Condition(lambda: self._view in {_View.PLUGIN_LIST, _View.PROFILE_LIST})
        is_plugin_list = Condition(lambda: self._view == _View.PLUGIN_LIST)
        is_profile_list = Condition(lambda: self._view == _View.PROFILE_LIST)
        is_form = Condition(lambda: self._view == _View.PROFILE_FORM)

        def _clear_pending() -> None:
            self._pending_delete = None

        @kb.add("up", filter=is_list)
        def _(event):
            items = self._current_items()
            if items:
                self._set_cursor((self._cursor() - 1) % len(items))
                self._error_message = None
                _clear_pending()

        @kb.add("down", filter=is_list)
        def _(event):
            items = self._current_items()
            if items:
                self._set_cursor((self._cursor() + 1) % len(items))
                self._error_message = None
                _clear_pending()

        @kb.add("enter", filter=is_plugin_list)
        def _(event):
            _clear_pending()
            if self._plugins:
                self._enter_profile_list(self._plugins[self._plugin_cursor].name)

        @kb.add("enter", filter=is_profile_list)
        def _(event):
            _clear_pending()
            profiles = self._profiles_of(self._selected_plugin or "")
            if profiles:
                self._toggle_profile_active(self._selected_plugin or "", profiles[self._profile_cursor])

        @kb.add("space", filter=is_plugin_list)
        def _(event):
            _clear_pending()
            if self._plugins:
                self._toggle_plugin_enabled(self._plugins[self._plugin_cursor].name)

        @kb.add("space", filter=is_profile_list)
        def _(event):
            _clear_pending()
            self._toggle_plugin_enabled(self._selected_plugin or "")

        @kb.add("n", filter=is_profile_list)
        def _(event):
            _clear_pending()
            self._open_profile_form("new")

        @kb.add("e", filter=is_profile_list)
        def _(event):
            _clear_pending()
            profiles = self._profiles_of(self._selected_plugin or "")
            if profiles:
                self._open_profile_form("edit", profiles[self._profile_cursor])

        @kb.add("x", filter=is_profile_list)
        def _(event):
            profiles = self._profiles_of(self._selected_plugin or "")
            if profiles:
                self._delete_profile(self._selected_plugin or "", profiles[self._profile_cursor])

        @kb.add("escape", filter=is_plugin_list)
        def _(event):
            self._finish(None)

        @kb.add("escape", filter=is_profile_list)
        def _(event):
            self._enter_plugin_list()
            self._focus(self._list_window)

        # Form bindings ----------------------------------------------------
        @kb.add("tab", filter=is_form)
        def _(event):
            self._advance_form_focus(+1)

        @kb.add("s-tab", filter=is_form)
        def _(event):
            self._advance_form_focus(-1)

        @kb.add("c-s", filter=is_form)
        def _(event):
            self._submit_profile_form()

        @kb.add("escape", filter=is_form)
        def _(event):
            self._enter_profile_list(self._selected_plugin or "")
            self._focus(self._list_window)

        @kb.add("c-c")
        def _(event):
            self._finish(None)

        return kb


__all__ = ["PluginApp"]
