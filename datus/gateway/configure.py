# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Interactive configuration hub for Datus Gateway IM channels.

Single-window TUI: an "Add" entry at the top, followed by a live list of
configured channels. Selecting a channel opens a per-channel submenu
(toggle enabled, change verbose, reinstall deps, delete). Any save
operation is reflected in ``agent.yml`` immediately.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from getpass import getpass
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.prompt import Confirm, Prompt

from datus.cli._cli_utils import select_choice
from datus.configuration.agent_config_loader import configuration_manager
from datus.gateway.channel.registry import list_adapters, register_builtins
from datus.gateway.models import ChannelConfig, Verbose
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
console = Console()


ADAPTER_DEPS: Dict[str, List[str]] = {
    "feishu": ["lark-oapi"],
    "slack": ["slack-sdk[socket_mode]"],
}

# (field_name, prompt_text, is_secret, required)
ADAPTER_FIELDS: Dict[str, List[Tuple[str, str, bool, bool]]] = {
    "feishu": [
        ("app_id", "Feishu App ID (cli_...)", False, True),
        ("app_secret", "Feishu App Secret", True, True),
    ],
    "slack": [
        ("app_token", "Slack App Token (xapp-...)", True, True),
        ("bot_token", "Slack Bot Token (xoxb-...)", True, True),
    ],
}

_SECRET_KEY_PATTERN = re.compile(r"(secret|token|password)", re.IGNORECASE)
_ENV_PLACEHOLDER_PATTERN = re.compile(r"^\$\{[^}]+\}$")
_NAME_INVALID_CHARS = set(' \t\n/\\:*?"<>|')

_MENU_SEPARATOR = "─" * 40


def _validate_channel_name(name: str, existing: Dict[str, Any]) -> Tuple[bool, str]:
    name = (name or "").strip()
    if not name:
        return False, "Channel name cannot be empty."
    if any(ch in _NAME_INVALID_CHARS for ch in name):
        return False, 'Channel name cannot contain whitespace or any of / \\ : * ? " < > |'
    if name in existing:
        return False, f"Channel '{name}' already exists."
    return True, ""


def _redact(key: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if _ENV_PLACEHOLDER_PATTERN.match(value):
        return value
    if not _SECRET_KEY_PATTERN.search(key):
        return value
    tail = value[-4:] if len(value) > 4 else ""
    return f"***{tail}" if tail else "***"


def _select_menu(
    rows: List[Tuple[str, str, bool]],
    default_key: Optional[str] = None,
    footer: str = "  [↑↓] navigate  [Enter] select  [q/Esc] back",
) -> Optional[str]:
    """Arrow-key menu supporting mixed selectable/static rows.

    Args:
        rows: Ordered list of ``(key, display, selectable)`` tuples. Rows with
              ``selectable=False`` are shown but skipped during navigation.
        default_key: Key of the row that should be pre-selected. Falls back to
                     the first selectable row when missing.
        footer: Hint line rendered below the list.

    Returns:
        The selected row's key, or ``None`` if the user cancels (Esc / ``q`` /
        Ctrl+C) or if there are no selectable rows.
    """
    if not rows:
        return None

    total = len(rows)
    selectable_indices = [i for i, r in enumerate(rows) if r[2]]
    if not selectable_indices:
        return None

    try:
        from prompt_toolkit import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        initial = selectable_indices[0]
        if default_key is not None:
            for i, (key, _, selectable) in enumerate(rows):
                if selectable and key == default_key:
                    initial = i
                    break
        selected = [initial]

        term_height = shutil.get_terminal_size((120, 40)).lines
        max_visible = max(3, term_height - 4)
        offset = [0]
        if selected[0] >= max_visible:
            offset[0] = min(max(0, selected[0] - max_visible // 2), max(0, total - max_visible))

        def _ensure_visible() -> None:
            if selected[0] < offset[0]:
                offset[0] = selected[0]
            elif selected[0] >= offset[0] + max_visible:
                offset[0] = selected[0] - max_visible + 1

        def _next_selectable(current: int, direction: int) -> int:
            i = current
            for _ in range(total):
                i = (i + direction) % total
                if rows[i][2]:
                    return i
            return current

        def _nearest_selectable(target: int) -> int:
            if rows[target][2]:
                return target
            for delta in range(1, total):
                fwd = target + delta
                bwd = target - delta
                if 0 <= fwd < total and rows[fwd][2]:
                    return fwd
                if 0 <= bwd < total and rows[bwd][2]:
                    return bwd
            return selected[0]

        kb = KeyBindings()

        @kb.add("up")
        def _move_up(event):
            selected[0] = _next_selectable(selected[0], -1)
            _ensure_visible()

        @kb.add("down")
        def _move_down(event):
            selected[0] = _next_selectable(selected[0], 1)
            _ensure_visible()

        @kb.add("pageup")
        def _page_up(event):
            target = max(0, selected[0] - max_visible)
            selected[0] = _nearest_selectable(target)
            _ensure_visible()

        @kb.add("pagedown")
        def _page_down(event):
            target = min(total - 1, selected[0] + max_visible)
            selected[0] = _nearest_selectable(target)
            _ensure_visible()

        @kb.add("enter")
        def _confirm(event):
            event.app.exit(result=rows[selected[0]][0])

        @kb.add("c-c")
        def _cancel_ctrlc(event):
            event.app.exit(result=None)

        @kb.add("escape")
        def _cancel_escape(event):
            event.app.exit(result=None)

        @kb.add("q")
        def _cancel_q(event):
            event.app.exit(result=None)

        def _get_formatted_text():
            lines = []
            visible_end = min(offset[0] + max_visible, total)
            if total > max_visible:
                lines.append(("ansiyellow", f"  ({offset[0] + 1}-{visible_end} of {total})\n"))
            for i in range(offset[0], visible_end):
                _, display, selectable = rows[i]
                is_sel = i == selected[0]
                if is_sel:
                    lines.append(("ansicyan bold", f"  →  {display}\n"))
                elif not selectable:
                    lines.append(("ansibrightblack", f"     {display}\n"))
                else:
                    lines.append(("", f"     {display}\n"))
            if footer:
                lines.append(("ansibrightblack", f"\n{footer}\n"))
            return lines

        app = Application(
            layout=Layout(
                Window(FormattedTextControl(_get_formatted_text, show_cursor=False), always_hide_cursor=True)
            ),
            key_bindings=kb,
            full_screen=False,
        )
        return app.run()

    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Input cancelled[/]")
        return None
    except Exception as e:
        logger.error(f"Interactive menu error: {e}")
        console.print(f"[bold red]Menu error:[/] {str(e)}")
        return None


class ChannelConfigurator:
    """Interactive hub for the ``channels:`` section of ``agent.yml``."""

    def __init__(self, config_path: str = ""):
        self.config_path = config_path
        register_builtins()
        try:
            self.cm = configuration_manager(config_path, reload=True)
        except DatusException as e:
            if e.code == ErrorCode.COMMON_FILE_NOT_FOUND:
                console.print("[red]Configuration file not found.[/red]")
                console.print("Run 'datus-agent init' first, or pass --config <path>.")
            else:
                console.print(f"[red]{e.message}[/red]")
            self.cm = None
        except Exception as e:
            console.print(f"[red]Failed to load configuration: {e}[/red]")
            self.cm = None

    def run(self) -> int:
        if self.cm is None:
            return 1

        while True:
            action, payload = self._render_hub()
            if action == "quit":
                return 0
            if action == "add":
                self.add()
            elif action == "channel" and payload is not None:
                self._channel_submenu(payload)

    def _channels(self) -> Dict[str, Any]:
        channels = self.cm.get("channels", {}) or {}
        if not isinstance(channels, dict):
            return {}
        return channels

    # ------------------------------------------------------------------
    # Hub & submenus
    # ------------------------------------------------------------------
    def _render_hub(self) -> Tuple[str, Optional[str]]:
        channels = self._channels()
        rows: List[Tuple[str, str, bool]] = [
            ("__header__", f"Datus Gateway channels — {self.cm.config_path}", False),
            ("__add__", "+ Add a new channel", True),
            ("__sep__", _MENU_SEPARATOR, False),
        ]
        if not channels:
            rows.append(("__empty__", "(no channels configured)", False))
        else:
            name_width = max(len(n) for n in channels)
            for name, cfg in channels.items():
                if isinstance(cfg, dict):
                    adapter = cfg.get("adapter", "?")
                    enabled = cfg.get("enabled", True)
                else:
                    adapter = "?"
                    enabled = True
                status = "enabled" if enabled else "disabled"
                display = f"{name:<{name_width}}  ({adapter}, {status})"
                rows.append((f"channel:{name}", display, True))

        key = _select_menu(rows, default_key="__add__")
        if key is None:
            return ("quit", None)
        if key == "__add__":
            return ("add", None)
        if key.startswith("channel:"):
            return ("channel", key[len("channel:") :])
        return ("quit", None)

    def _channel_submenu(self, name: str) -> None:
        channels = self._channels()
        cfg = channels.get(name)
        if not isinstance(cfg, dict):
            console.print(f"[yellow]Channel '{name}' not found.[/yellow]")
            return

        adapter = cfg.get("adapter", "?")
        enabled = cfg.get("enabled", True)
        verbose = cfg.get("verbose", Verbose.ON.value)
        extra = cfg.get("extra", {})

        rows: List[Tuple[str, str, bool]] = [
            ("__header__", f"Channel: {name}", False),
            ("__info_adapter__", f"  adapter: {adapter}", False),
            ("__info_enabled__", f"  enabled: {enabled}", False),
            ("__info_verbose__", f"  verbose: {verbose}", False),
        ]
        if isinstance(extra, dict):
            for k, v in extra.items():
                rows.append((f"__info_extra_{k}__", f"  {k}: {_redact(k, v)}", False))
        rows.append(("__sep__", _MENU_SEPARATOR, False))

        toggle_label = "Disable channel" if enabled else "Enable channel"
        rows.append(("toggle_enabled", toggle_label, True))
        rows.append(("change_verbose", f"Change verbose (current: {verbose})", True))
        if adapter in ADAPTER_DEPS:
            deps = ADAPTER_DEPS[adapter]
            rows.append(("reinstall_deps", f"Reinstall pip deps ({' '.join(deps)})", True))
        rows.append(("delete", "Delete this channel", True))
        rows.append(("back", "Back", True))

        key = _select_menu(rows, default_key="back")
        if key is None or key == "back":
            return
        if key == "toggle_enabled":
            self._toggle_enabled(name)
        elif key == "change_verbose":
            self._change_verbose(name)
        elif key == "reinstall_deps":
            self._reinstall_deps(adapter)
        elif key == "delete":
            self.delete(name)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def add(self) -> int:
        channels = self._channels()

        adapters = list_adapters()
        if not adapters:
            console.print("[red]No channel adapters are registered.[/red]")
            return 1

        while True:
            name = Prompt.ask("Channel name (unique key under 'channels:')").strip()
            ok, err = _validate_channel_name(name, channels)
            if ok:
                break
            console.print(f"[red]{err}[/red]")

        adapter = select_choice(
            console,
            {a: a for a in adapters},
            default=adapters[0],
        )
        if adapter not in adapters:
            console.print("[red]No adapter selected.[/red]")
            return 1

        extra: Dict[str, Any] = {}
        for field, label, is_secret, required in ADAPTER_FIELDS.get(adapter, []):
            while True:
                if is_secret:
                    value = getpass(f"{label}: ").strip()
                else:
                    value = Prompt.ask(label).strip()
                if value or not required:
                    break
                console.print(f"[red]{field} is required.[/red]")
            if value:
                extra[field] = value

        enabled = Confirm.ask("Enable this channel now?", default=True)

        verbose_value = Verbose.ON.value
        if Confirm.ask("Override default verbose (brief)?", default=False):
            verbose_value = select_choice(
                console,
                {v.value: f"{v.value} ({v.name.lower()})" for v in Verbose},
                default=Verbose.ON.value,
            )

        try:
            cfg_model = ChannelConfig(
                adapter=adapter,
                enabled=enabled,
                verbose=Verbose(verbose_value),
                extra=extra,
            )
        except Exception as e:
            console.print(f"[red]Invalid channel configuration: {e}[/red]")
            return 1

        cfg_dict = cfg_model.model_dump(mode="json", exclude_none=True)

        if not self.cm.update_item("channels", {name: cfg_dict}, delete_old_key=False):
            console.print("[red]Failed to write channel to agent.yml.[/red]")
            return 1

        console.print(f"[green]Channel '{name}' saved to {self.cm.config_path}[/green]")

        deps = ADAPTER_DEPS.get(adapter, [])
        if deps:
            if Confirm.ask(f"Install required pip packages {deps}?", default=True):
                return self._pip_install(deps)
            console.print("[dim]Run the command below when ready:[/dim]")
            console.print(f"  {sys.executable} -m pip install {' '.join(deps)}")
        return 0

    def delete(self, name: Optional[str] = None) -> int:
        channels = self._channels()
        if not channels:
            console.print("[yellow]No channels configured — nothing to delete.[/yellow]")
            return 0

        if name is None:
            display = {
                n: f"{n} ({cfg.get('adapter', '?')})" if isinstance(cfg, dict) else n for n, cfg in channels.items()
            }
            name = select_choice(console, display, default=next(iter(display)))
            if name not in channels:
                console.print("[yellow]No channel selected.[/yellow]")
                return 0
        elif name not in channels:
            console.print(f"[yellow]Channel '{name}' not found.[/yellow]")
            return 0

        if not Confirm.ask(f"Delete channel '{name}'?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return 0

        try:
            self.cm.remove_item_recursively("channels", name)
        except DatusException as e:
            console.print(f"[red]{e.message}[/red]")
            return 1
        console.print(f"[green]Channel '{name}' removed from {self.cm.config_path}[/green]")
        return 0

    def _toggle_enabled(self, name: str) -> int:
        channels = self._channels()
        cfg = channels.get(name)
        if not isinstance(cfg, dict):
            console.print(f"[yellow]Channel '{name}' not found.[/yellow]")
            return 1
        new_cfg = dict(cfg)
        new_cfg["enabled"] = not cfg.get("enabled", True)
        if not self.cm.update_item("channels", {name: new_cfg}, delete_old_key=False):
            console.print("[red]Failed to update channel.[/red]")
            return 1
        state = "enabled" if new_cfg["enabled"] else "disabled"
        console.print(f"[green]Channel '{name}' {state}.[/green]")
        return 0

    def _change_verbose(self, name: str) -> int:
        channels = self._channels()
        cfg = channels.get(name)
        if not isinstance(cfg, dict):
            console.print(f"[yellow]Channel '{name}' not found.[/yellow]")
            return 1
        current = cfg.get("verbose", Verbose.ON.value)
        choice = select_choice(
            console,
            {v.value: f"{v.value} ({v.name.lower()})" for v in Verbose},
            default=current if current in {v.value for v in Verbose} else Verbose.ON.value,
        )
        if choice not in {v.value for v in Verbose}:
            console.print("[yellow]No change.[/yellow]")
            return 0
        if choice == current:
            console.print("[dim]Verbose unchanged.[/dim]")
            return 0
        new_cfg = dict(cfg)
        new_cfg["verbose"] = choice
        if not self.cm.update_item("channels", {name: new_cfg}, delete_old_key=False):
            console.print("[red]Failed to update channel.[/red]")
            return 1
        console.print(f"[green]Channel '{name}' verbose -> {choice}.[/green]")
        return 0

    def _reinstall_deps(self, adapter: str) -> int:
        deps = ADAPTER_DEPS.get(adapter, [])
        if not deps:
            console.print(f"[yellow]No pip deps registered for adapter '{adapter}'.[/yellow]")
            return 0
        if not Confirm.ask(f"Install required pip packages {deps}?", default=True):
            console.print("[dim]Cancelled.[/dim]")
            return 0
        return self._pip_install(deps)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pip_install(self, packages: List[str]) -> int:
        cmd = [sys.executable, "-m", "pip", "install", *packages]
        console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        try:
            completed = subprocess.run(cmd, check=False)
        except OSError as e:
            console.print(f"[red]Failed to invoke pip: {e}[/red]")
            return 1
        if completed.returncode != 0:
            console.print(f"[red]pip install exited with code {completed.returncode}[/red]")
            return completed.returncode
        console.print("[green]Dependencies installed.[/green]")
        return 0
