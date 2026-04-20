# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Interactive configuration wizard for Datus Claw IM channels.

Provides CRUD for the ``channels:`` section of ``agent.yml`` and an
optional ``pip install`` helper for adapter-specific SDK dependencies.
"""

from __future__ import annotations

import re
import subprocess
import sys
from getpass import getpass
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.prompt import Confirm, Prompt

from datus.claw.channel.registry import list_adapters, register_builtins
from datus.claw.models import ChannelConfig, Verbose
from datus.cli._cli_utils import select_choice
from datus.configuration.agent_config_loader import configuration_manager
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

_ACTIONS = ("list", "add", "delete", "install-deps")
_SECRET_KEY_PATTERN = re.compile(r"(secret|token|password)", re.IGNORECASE)
_ENV_PLACEHOLDER_PATTERN = re.compile(r"^\$\{[^}]+\}$")
_NAME_INVALID_CHARS = set(' \t\n/\\:*?"<>|')


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


class ChannelConfigurator:
    """Interactive CRUD for the ``channels:`` section of ``agent.yml``."""

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

    def run(self, action: Optional[str]) -> int:
        if self.cm is None:
            return 1

        if not action:
            action = self._prompt_action()
            if not action:
                return 0

        if action == "list":
            return self.list()
        if action == "add":
            return self.add()
        if action == "delete":
            return self.delete()
        if action == "install-deps":
            return self.install_deps_interactive()

        console.print(f"[red]Unknown action: {action}[/red]")
        return 1

    def _prompt_action(self) -> Optional[str]:
        choice = select_choice(
            console,
            {
                "list": "List configured channels",
                "add": "Add a new channel",
                "delete": "Delete a channel",
                "install-deps": "Install adapter pip dependencies",
            },
            default="list",
        )
        return choice if choice in _ACTIONS else None

    def _channels(self) -> Dict[str, Any]:
        channels = self.cm.get("channels", {}) or {}
        if not isinstance(channels, dict):
            return {}
        return channels

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def list(self) -> int:
        channels = self._channels()
        if not channels:
            console.print("[yellow]No channels configured.[/yellow]")
            console.print(f"Config file: {self.cm.config_path}")
            return 0

        console.print(f"[bold]Channels in {self.cm.config_path}:[/bold]")
        for name, cfg in channels.items():
            adapter = cfg.get("adapter", "?") if isinstance(cfg, dict) else "?"
            enabled = cfg.get("enabled", True) if isinstance(cfg, dict) else True
            verbose = cfg.get("verbose", Verbose.ON.value) if isinstance(cfg, dict) else "?"
            status = "[green]enabled[/green]" if enabled else "[dim]disabled[/dim]"
            console.print(f"\n  [cyan]{name}[/cyan]  ({adapter}, {status}, verbose={verbose})")
            extra = cfg.get("extra", {}) if isinstance(cfg, dict) else {}
            if isinstance(extra, dict) and extra:
                for k, v in extra.items():
                    console.print(f"    {k}: {_redact(k, v)}")
        return 0

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

    def delete(self) -> int:
        channels = self._channels()
        if not channels:
            console.print("[yellow]No channels configured — nothing to delete.[/yellow]")
            return 0

        display = {
            name: f"{name} ({cfg.get('adapter', '?')})" if isinstance(cfg, dict) else name
            for name, cfg in channels.items()
        }
        name = select_choice(console, display, default=next(iter(display)))
        if name not in channels:
            console.print("[yellow]No channel selected.[/yellow]")
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

    def install_deps_interactive(self) -> int:
        channels = self._channels()
        adapters_in_use = sorted(
            {
                cfg.get("adapter")
                for cfg in channels.values()
                if isinstance(cfg, dict) and cfg.get("adapter") in ADAPTER_DEPS
            }
        )
        if not adapters_in_use:
            console.print("[yellow]No channels with known pip deps are configured.[/yellow]")
            return 0

        adapter = select_choice(
            console,
            {a: f"{a} -> {ADAPTER_DEPS[a]}" for a in adapters_in_use},
            default=adapters_in_use[0],
        )
        deps = ADAPTER_DEPS.get(adapter, [])
        if not deps:
            console.print("[yellow]No pip deps registered for this adapter.[/yellow]")
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
