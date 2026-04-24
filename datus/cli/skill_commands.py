# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/skill`` slash command — unified CLI surface for the Skill Marketplace.

Entry point
-----------

- ``/skill``                          — open the interactive :class:`SkillApp`.
- ``/skill list``                     — open the app on the Installed tab.
- ``/skill search <query>``           — open the app on the Marketplace tab,
  seeded with ``<query>`` as a client-side filter.
- ``/skill login [url]``              — open the app directly on the login form
  (pre-filled with the marketplace URL if provided).
- ``/skill install <name> [version]`` — non-interactive install (scriptable).
- ``/skill publish <path> [--owner x]`` — non-interactive publish.
- ``/skill info <name>``              — non-interactive details table.
- ``/skill remove <name>``            — non-interactive remove with confirm.
- ``/skill update``                   — non-interactive bulk upgrade.
- ``/skill logout``                   — drop saved credentials.
- ``/skill help``                     — command reference.

The interactive path delegates to :class:`datus.cli.skill_app.SkillApp`,
a single prompt_toolkit Application that hosts tab switching, detail
drill-down, filter input, login capture, and two-press remove confirmation
in one event loop. All network I/O (marketplace search / install / login)
runs **after** ``SkillApp.run`` returns, driven from this module, so the
Application never blocks.
"""

from __future__ import annotations

import shlex
import shutil
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx
from rich.table import Table

from datus.cli._cli_utils import confirm_prompt
from datus.cli._render_utils import build_kv_table, build_row_table
from datus.cli.cli_styles import (
    TABLE_HEADER_STYLE,
    print_empty_set,
    print_error,
    print_info,
    print_success,
    print_usage,
    print_warning,
)
from datus.cli.skill_app import SkillApp, SkillSelection
from datus.tools.skill_tools.skill_config import SkillMetadata
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)

# Bounded re-entry when a SkillApp selection requires a refresh / login.
# Two refreshes are enough to cover: first open → login → reopen → refresh.
_MAX_REENTRY = 3


class SkillCommands:
    """Handlers for the ``/skill`` slash command."""

    def __init__(self, cli_instance: "DatusCLI"):
        self.cli = cli_instance
        self.console = cli_instance.console

    # ── Entry point ──────────────────────────────────────────────────────

    def cmd_skill(self, args: str) -> None:
        """Dispatch ``/skill`` based on its argument shape."""
        token = (args or "").strip()
        if not token:
            self._run_menu()
            return
        try:
            parts = shlex.split(token)
        except ValueError as exc:
            print_error(self.console, f"Invalid arguments: {exc}", prefix=False)
            return
        op, rest = parts[0], parts[1:]
        handlers: Dict[str, Any] = {
            "help": lambda r: self._show_usage(),
            "list": lambda r: self._run_menu(seed_tab="installed"),
            "search": lambda r: self._run_menu(seed_tab="marketplace", seed_search=" ".join(r)),
            "login": lambda r: self._run_menu(seed_tab="installed", open_login=True, seed_url=(r[0] if r else None)),
            "logout": lambda r: self._do_logout(),
            "install": self._cmd_install,
            "publish": self._cmd_publish,
            "info": self._cmd_info,
            "update": lambda r: self._do_update_all(),
            "remove": self._cmd_remove,
        }
        handler = handlers.get(op)
        if handler is None:
            print_error(self.console, f"Unknown skill command: {op}", prefix=False)
            self._show_usage()
            return
        handler(rest)

    # ── Interactive loop ─────────────────────────────────────────────────

    def _run_menu(
        self,
        *,
        seed_tab: Optional[str] = None,
        seed_search: Optional[str] = None,
        open_login: bool = False,
        seed_url: Optional[str] = None,
    ) -> None:
        """Drive the :class:`SkillApp` loop, re-entering on ``refresh`` / ``login``."""
        active_tab = seed_tab
        active_search = seed_search
        for _ in range(_MAX_REENTRY):
            manager = self._get_skill_manager()
            installed = self._safe_list_installed(manager)
            marketplace = self._safe_search_marketplace(manager, active_search or "")
            app = SkillApp(
                manager,
                self.console,
                installed=installed,
                marketplace=marketplace,
                seed_tab=active_tab,
                seed_search=active_search,
            )
            if open_login and seed_url:
                try:
                    app._login_url.text = seed_url
                except Exception:
                    pass
            if open_login:
                app._enter_login_form()
                open_login = False

            selection = self._run_app(app)
            if selection is None or selection.kind == "cancel":
                return
            if selection.kind == "install":
                self._do_install(selection.name or "", selection.version or "latest")
                return
            if selection.kind == "update":
                self._do_update_one(selection.name or "")
                return
            if selection.kind == "remove":
                self._do_remove(selection.name or "", confirmed=True)
                return
            if selection.kind == "logout":
                self._do_logout()
                return
            if selection.kind == "login":
                ok = self._do_login_with_credentials(
                    email=selection.email or "",
                    password=selection.password or "",
                    marketplace_url=selection.marketplace_url or getattr(manager.config, "marketplace_url", "") or "",
                )
                if not ok:
                    return
                active_tab = "installed"
                active_search = None
                continue
            if selection.kind == "refresh":
                active_tab = "marketplace"
                active_search = ""
                continue
            logger.debug("SkillApp returned unknown kind=%s", selection.kind)
            return

    def _run_app(self, app: SkillApp) -> Optional[SkillSelection]:
        """Run ``app`` with stdin released by the outer TUI (if any)."""
        tui_app = getattr(self.cli, "tui_app", None)
        ctx = tui_app.suspend_input() if tui_app is not None else nullcontext()
        with ctx:
            return app.run()

    # ── Non-interactive subcommands ──────────────────────────────────────

    def _cmd_install(self, rest: List[str]) -> None:
        if not rest:
            print_usage(self.console, "/skill install <name> [version]")
            return
        name = rest[0]
        version = rest[1] if len(rest) > 1 else "latest"
        self._do_install(name, version)

    def _cmd_publish(self, rest: List[str]) -> None:
        if not rest:
            print_usage(self.console, "/skill publish <path> [--owner <name>]")
            return
        skill_dir = rest[0]
        owner = ""
        if "--owner" in rest:
            idx = rest.index("--owner")
            if idx + 1 < len(rest):
                owner = rest[idx + 1]
        manager = self._get_skill_manager()
        print_info(self.console, f"Publishing skill from {skill_dir}...")
        ok, msg = manager.publish_to_marketplace(skill_dir, owner=owner)
        if ok:
            print_success(self.console, msg, symbol=True)
        else:
            print_error(self.console, msg, prefix=False)

    def _cmd_info(self, rest: List[str]) -> None:
        if not rest:
            print_usage(self.console, "/skill info <name>")
            return
        name = rest[0]
        manager = self._get_skill_manager()

        local = manager.get_skill(name)
        if local is not None:
            self._render_local_info(local)
        else:
            print_info(self.console, "Not installed locally. Checking marketplace...")

        try:
            client = manager._get_marketplace_client()
            remote = client.get_skill_info(name)
            self._render_marketplace_info(remote)
        except Exception as e:
            if local is None:
                print_warning(self.console, f"Skill '{name}' not found locally or in marketplace.")
            else:
                print_info(self.console, f"Marketplace lookup failed: {e}")

    def _cmd_remove(self, rest: List[str]) -> None:
        if not rest:
            print_usage(self.console, "/skill remove <name>")
            return
        name = rest[0]
        manager = self._get_skill_manager()
        skill = manager.get_skill(name)
        if skill is None:
            print_warning(self.console, f"Skill '{name}' not found locally.")
            return

        skill_path = skill.location
        if skill_path and skill_path.exists():
            if not confirm_prompt(self.console, f"Delete skill files at {skill_path}?"):
                print_warning(self.console, "Cancelled.")
                return
        self._do_remove(name, confirmed=True)

    # ── Business logic (shared by TUI and CLI paths) ─────────────────────

    def _do_install(self, name: str, version: str) -> None:
        manager = self._get_skill_manager()
        print_info(self.console, f"Installing {name}@{version} from marketplace...")
        ok, msg = manager.install_from_marketplace(name, version)
        if ok:
            print_success(self.console, msg, symbol=True)
        else:
            print_error(self.console, msg, prefix=False)

    def _do_remove(self, name: str, *, confirmed: bool) -> None:
        if not confirmed:
            # Interactive callers already confirmed; non-interactive ones go
            # through ``_cmd_remove`` which runs ``confirm_prompt`` first.
            return
        manager = self._get_skill_manager()
        skill = manager.get_skill(name)
        if skill is None:
            print_warning(self.console, f"Skill '{name}' not found locally.")
            return
        skill_path = skill.location
        removed = manager.registry.remove_skill(name)
        if not removed:
            print_error(self.console, f"Failed to remove skill '{name}'", prefix=False)
            return
        if skill_path and skill_path.exists():
            shutil.rmtree(str(skill_path), ignore_errors=True)
            print_info(self.console, f"Deleted files at {skill_path}")
        print_success(self.console, f"Removed skill '{name}'", symbol=True)

    def _do_update_one(self, name: str) -> None:
        manager = self._get_skill_manager()
        skill = manager.get_skill(name)
        if skill is None:
            print_warning(self.console, f"Skill '{name}' not found locally.")
            return
        try:
            client = manager._get_marketplace_client()
            remote = client.get_skill_info(name)
        except Exception as e:
            print_error(self.console, f"Marketplace lookup failed: {e}", prefix=False)
            return
        remote_version = remote.get("latest_version", "") or ""
        if remote_version and remote_version != skill.version:
            ok, msg = manager.install_from_marketplace(name, remote_version)
            if ok:
                print_success(self.console, f"Updated {name} to {remote_version}", symbol=True)
            else:
                print_error(self.console, f"Failed: {msg}", prefix=False)
        else:
            print_info(self.console, f"{name} is already up to date")

    def _do_update_all(self) -> None:
        manager = self._get_skill_manager()
        skills = manager.list_all_skills()
        marketplace_skills = [s for s in skills if s.source == "marketplace"]
        if not marketplace_skills:
            print_warning(self.console, "No marketplace-installed skills to update.")
            return
        updated = 0
        for skill in marketplace_skills:
            print_info(self.console, f"Checking {skill.name}...")
            try:
                client = manager._get_marketplace_client()
                remote = client.get_skill_info(skill.name)
                remote_version = remote.get("latest_version", "") or ""
                if remote_version and remote_version != skill.version:
                    ok, msg = manager.install_from_marketplace(skill.name, remote_version)
                    if ok:
                        print_success(self.console, f"  Updated {skill.name} to {remote_version}", symbol=True)
                        updated += 1
                    else:
                        print_error(self.console, f"  Failed: {msg}", prefix=False)
                else:
                    print_info(self.console, "  Already up to date")
            except Exception as e:
                print_error(self.console, f"  Error checking {skill.name}: {e}", prefix=False)
        self.console.print(f"[bold]{updated} skill(s) updated.[/]")

    def _do_login_with_credentials(self, *, email: str, password: str, marketplace_url: str) -> bool:
        """Exchange credentials for a JWT and persist it locally."""
        from datus.tools.skill_tools.marketplace_auth import save_token

        if not marketplace_url:
            print_error(self.console, "Marketplace URL is not configured", prefix=False)
            return False

        login_url = f"{marketplace_url.rstrip('/')}/api/auth/login"
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(login_url, json={"email": email, "password": password})
                if resp.status_code >= 400:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text
                    print_error(self.console, f"Login failed ({resp.status_code}): {detail}", prefix=False)
                    return False
                token = resp.cookies.get("town_token")
                if not token:
                    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    token = body.get("access_token") or body.get("token")
                if not token:
                    print_error(self.console, "Login succeeded but no token was returned.", prefix=False)
                    return False
                save_token(token, marketplace_url, email)
                print_success(self.console, f"Login successful! Token saved for {marketplace_url}", symbol=True)
                return True
        except httpx.ConnectError:
            print_error(self.console, f"Cannot connect to {login_url}", prefix=False)
            return False
        except Exception as exc:
            print_error(self.console, f"Login error: {exc}", prefix=False)
            return False

    def _do_logout(self) -> None:
        from datus.tools.skill_tools.marketplace_auth import clear_token

        manager = self._get_skill_manager()
        marketplace_url = getattr(manager.config, "marketplace_url", "") or ""
        if clear_token(marketplace_url):
            print_success(self.console, f"Logged out from {marketplace_url}")
        else:
            print_warning(self.console, f"No saved credentials for {marketplace_url}")

    # ── Rendering helpers (non-interactive table output) ─────────────────

    def _render_local_info(self, skill: SkillMetadata) -> None:
        payload: Dict[str, Any] = {
            "Name": skill.name,
            "Version": skill.version or "unversioned",
            "Source": skill.source or "local",
            "Location": str(skill.location) if skill.location else "-",
            "Tags": ", ".join(skill.tags) if skill.tags else "(none)",
            "License": skill.license or "-",
            "Description": skill.description or "-",
        }
        table = build_kv_table(payload, title=f"Local Skill: {skill.name}", max_cell_width=80)
        if table is not None:
            self.console.print(table)

    def _render_marketplace_info(self, remote: Dict[str, Any]) -> None:
        versions = remote.get("versions") or []
        version_labels = ", ".join(v.get("version", "?") for v in versions) if versions else "-"
        payload: Dict[str, Any] = {
            "Name": remote.get("name", "?"),
            "Latest Version": remote.get("latest_version", "-"),
            "Owner": remote.get("owner", "-"),
            "Promoted": "yes" if remote.get("promoted") else "no",
            "Usage Count": remote.get("usage_count", 0),
            "Versions": version_labels,
        }
        table = build_kv_table(payload, title="Marketplace Info", max_cell_width=80)
        if table is not None:
            self.console.print(table)

    # ── Data helpers ─────────────────────────────────────────────────────

    def _get_skill_manager(self):
        """Get the :class:`SkillManager` from the agent, or create a standalone one."""
        if self.cli.agent and hasattr(self.cli.agent, "skill_manager") and self.cli.agent.skill_manager:
            return self.cli.agent.skill_manager

        from datus.tools.skill_tools.skill_config import SkillConfig
        from datus.tools.skill_tools.skill_manager import SkillManager

        skills_conf: Dict[str, Any] = {}
        if hasattr(self.cli, "agent_config") and self.cli.agent_config:
            raw = getattr(self.cli.agent_config, "skills", {}) or {}
            if isinstance(raw, dict):
                skills_conf = raw
        config = SkillConfig.from_dict(skills_conf)
        return SkillManager(config=config)

    @staticmethod
    def _safe_list_installed(manager) -> List[SkillMetadata]:
        try:
            return list(manager.list_all_skills())
        except Exception as exc:
            logger.debug("list_all_skills failed: %s", exc)
            return []

    @staticmethod
    def _safe_search_marketplace(manager, query: str) -> List[Dict[str, Any]]:
        try:
            return list(manager.search_marketplace(query=query))
        except Exception as exc:
            logger.debug("search_marketplace failed: %s", exc)
            return []

    # ── Help ─────────────────────────────────────────────────────────────

    def _show_usage(self) -> None:
        table = Table(title="/skill commands", header_style=TABLE_HEADER_STYLE, show_lines=False)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description")
        rows = [
            ("/skill", "Open the interactive browser"),
            ("/skill list", "Open on the Installed tab"),
            ("/skill search <query>", "Open on Marketplace, seed a filter"),
            ("/skill info <name>", "Show local + marketplace details"),
            ("/skill install <name> [version]", "Install skill from marketplace"),
            ("/skill publish <path> [--owner <n>]", "Publish a local skill"),
            ("/skill update", "Upgrade all marketplace-installed skills"),
            ("/skill remove <name>", "Remove a locally-installed skill"),
            ("/skill login [url]", "Open the login form (pre-fill URL)"),
            ("/skill logout", "Clear saved marketplace credentials"),
            ("/skill help", "Show this help"),
        ]
        for cmd, desc in rows:
            table.add_row(cmd, desc)
        self.console.print(table)

    # ── Legacy non-interactive list path (kept for scripts / fallback) ──

    def cmd_skill_list(self) -> None:
        """Print installed skills as a Rich table (no TUI)."""
        manager = self._get_skill_manager()
        skills = manager.list_all_skills()
        if not skills:
            print_empty_set(self.console, "No skills installed locally.")
            return
        payload = [
            {
                "Name": s.name,
                "Version": s.version or "-",
                "Source": s.source or "local",
                "Path": str(s.location) if s.location else "-",
                "Tags": ", ".join(s.tags) if s.tags else "",
                "Description": s.description or "",
            }
            for s in skills
        ]
        table = build_row_table(
            payload,
            title="Installed Skills",
            columns=[
                ("Name", "Name"),
                ("Version", "Version"),
                ("Source", "Source"),
                ("Path", "Path"),
                ("Tags", "Tags"),
                ("Description", "Description"),
            ],
            max_cell_width=50,
        )
        if table is not None:
            self.console.print(table)
