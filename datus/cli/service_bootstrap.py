# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""REPL startup hook that pins service defaults and installs adapter packages.

Runs once per interactive REPL launch, **before** the prompt loop opens, to:

1. **Pin defaults to the project**: for each ``services.<section>`` (BI,
   Scheduler, Semantic), if no project pin is already set in
   ``./.datus/config.yml`` and the YAML resolves a deterministic default
   (single entry, or one entry flagged ``default: true``), write that
   choice back as the project pin so subsequent runs are explicit. When
   multiple entries are configured without a default, prompt the user to
   pick one synchronously.

2. **Install missing adapter packages in the background**: every
   configured service whose adapter Python package is not importable
   (``datus-bi-<x>``, ``datus-scheduler-<x>``, ``datus-semantic-<x>``)
   gets a ``pip install`` kicked off on a daemon thread, then a
   ``hot_reload_adapter`` so it becomes usable without a restart.

Hard guardrails — both must hold or the entire bootstrap is skipped:

- ``sys.stdin.isatty()`` AND ``sys.stdout.isatty()`` (no piped or
  CI-style invocations).
- ``DATUS_DISABLE_SERVICE_BOOTSTRAP`` env var unset (escape hatch for
  Docker / batch / regression suites).

Failures inside this module never abort startup; they print a one-line
message and move on. The user can always re-run ``/services`` to fix
anything by hand.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from datus.cli.cli_styles import print_error, print_info, print_success
from datus.cli.service_adapter_installer import (
    InstallResult,
    ensure_adapter,
    hot_reload_adapter,
    is_adapter_installed,
)
from datus.cli.service_client import service_type_label
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


# Section -> (services-dict accessor, active-getter, active-setter,
# default-flag resolver). One row per section keeps the bootstrap loop
# uniform and makes adding new service types a single-row change.
_SECTION_SPECS: Tuple[Tuple[str, str, str, str, str], ...] = (
    # (section,            services_attr,            active_getter,       active_setter,         default_resolver)
    ("bi_platforms", "dashboard_config", "active_dashboard", "set_active_dashboard", "default_dashboard_service"),
    ("schedulers", "scheduler_services", "active_scheduler", "set_active_scheduler", "default_scheduler_service"),
    ("semantic_layer", "semantic_layer_configs", "active_semantic", "set_active_semantic", "default_semantic_adapter"),
)


def _is_interactive() -> bool:
    """Return True only when both stdin and stdout are TTYs and the
    escape hatch env var is unset."""
    if os.environ.get("DATUS_DISABLE_SERVICE_BOOTSTRAP"):
        return False
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, OSError):
        return False


def run(cli: "DatusCLI") -> None:
    """Entry point — pin project defaults synchronously, then kick off
    the background install pass.

    The synchronous default-pinning step blocks startup briefly (only
    visible to the user when an interactive picker fires); the install
    pass returns immediately and runs to completion on a daemon thread.
    Either step is a no-op when ``_is_interactive()`` returns False.
    """
    if not _is_interactive():
        return

    agent_config = getattr(cli, "agent_config", None)
    if agent_config is None:
        return

    for section, services_attr, getter, setter, resolver in _SECTION_SPECS:
        try:
            _bootstrap_default(cli, section, services_attr, getter, setter, resolver)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("bootstrap_default(%s) failed: %s", section, exc)

    try:
        background_install_missing(cli)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("background_install_missing failed: %s", exc)


def _bootstrap_default(
    cli: "DatusCLI",
    section: str,
    services_attr: str,
    getter_name: str,
    setter_name: str,
    resolver_name: str,
) -> None:
    """Pin a project default for one section if not already pinned.

    Skips silently when the section is empty (the resolver itself raises
    on use), the project pin already names a configured service, or the
    user cancels the picker.
    """
    agent_config = cli.agent_config
    label = service_type_label(section)
    services = getattr(agent_config, services_attr, {}) or {}
    if not services:
        return

    getter = getattr(agent_config, getter_name, None)
    current_pin = getter() if callable(getter) else None
    if current_pin and current_pin in services:
        return

    setter = getattr(agent_config, setter_name, None)
    if not callable(setter):
        return

    resolver = getattr(agent_config, resolver_name, None)
    auto_pick: Optional[str] = None
    if callable(resolver):
        try:
            auto_pick = resolver()
        except Exception as exc:
            # ``default_*_service`` raises when ``default: true`` is on more
            # than one entry — surface it once and skip; the user can fix
            # the YAML or re-run the bootstrap by relaunching the REPL.
            print_error(cli.console, f"Cannot pick default {label}: {exc}", prefix=False)
            return

    if auto_pick is None and len(services) > 1:
        auto_pick = quick_pick_service(cli, section, list(services.keys()))

    if not auto_pick:
        return

    try:
        setter(auto_pick)
    except Exception as exc:
        print_error(cli.console, f"Failed to pin default {label}: {exc}", prefix=False)
        return
    print_success(cli.console, f"Pinned project default {label}: {auto_pick}", symbol=True)


def quick_pick_service(cli: "DatusCLI", section: str, names: List[str]) -> Optional[str]:
    """Tiny synchronous picker shown when a section has multiple entries
    and no default. Returns the chosen service name, or ``None`` when
    the user types ``q``/blank.

    Deliberately not a prompt_toolkit Application — we're called before
    the main TUI starts, but we want to stay portable to plain
    PromptSession launches too. ``input()`` works in both cases because
    ``_is_interactive()`` already verified stdin is a TTY.
    """
    label = service_type_label(section)
    cli.console.print(
        f"\n[bold]Multiple {label} services configured but no default set. Pick one to pin as the project default:[/]"
    )
    for i, name in enumerate(names, 1):
        cli.console.print(f"  [cyan]{i}[/]) {name}")
    cli.console.print("  [dim]q) skip — leave unpinned (will warn on first use)[/]")
    while True:
        try:
            choice = input(f"Choice [1-{len(names)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not choice or choice.lower() in ("q", "skip"):
            return None
        try:
            idx = int(choice) - 1
        except ValueError:
            print_error(cli.console, "Please enter a number or `q` to skip.", prefix=False)
            continue
        if 0 <= idx < len(names):
            return names[idx]
        print_error(cli.console, f"Out of range. Pick 1-{len(names)} or `q`.", prefix=False)


def background_install_missing(cli: "DatusCLI") -> None:
    """Spawn a daemon thread that pip-installs any configured service
    whose adapter package is not importable.

    Returns immediately. Failures inside the worker print a one-line
    message via ``cli.console`` (which is thread-safe for plain
    ``print``); they do not raise and they do not retry. The user can
    re-run ``/services <section>`` to retry a failed install
    interactively.
    """
    targets = _missing_install_targets(cli.agent_config)
    if not targets:
        return
    thread = threading.Thread(target=_install_worker, args=(cli, targets), daemon=True)
    thread.start()


def _missing_install_targets(agent_config: Any) -> List[Tuple[str, str]]:
    """Walk all three sections, return ``(section, adapter_type)`` for
    every configured service whose adapter package is not installed.

    The adapter type is read from the section-specific config object;
    BI uses ``DashboardConfig.adapter_type`` (alias may differ), while
    Scheduler / Semantic store the type as a dict key or ``type`` field.
    """
    targets: List[Tuple[str, str]] = []
    bi = getattr(agent_config, "dashboard_config", {}) or {}
    for name, cfg in bi.items():
        adapter_type = (getattr(cfg, "adapter_type", "") or name).strip().lower()
        if adapter_type and not is_adapter_installed("bi_platforms", adapter_type):
            targets.append(("bi_platforms", adapter_type))

    schedulers = getattr(agent_config, "scheduler_services", {}) or {}
    for name, cfg in schedulers.items():
        if not isinstance(cfg, dict):
            continue
        adapter_type = str(cfg.get("type") or name).strip().lower()
        if adapter_type and not is_adapter_installed("schedulers", adapter_type):
            targets.append(("schedulers", adapter_type))

    semantic = getattr(agent_config, "semantic_layer_configs", {}) or {}
    for name, cfg in semantic.items():
        # ``init_semantic_layer`` already enforces ``key == type``, so
        # using the name directly is safe.
        adapter_type = str(name).strip().lower()
        if adapter_type and not is_adapter_installed("semantic_layer", adapter_type):
            targets.append(("semantic_layer", adapter_type))

    # Deduplicate — multiple BI aliases of the same adapter type would
    # otherwise trigger ``pip install`` once per alias.
    seen: set = set()
    unique: List[Tuple[str, str]] = []
    for entry in targets:
        if entry in seen:
            continue
        seen.add(entry)
        unique.append(entry)
    return unique


def _install_worker(cli: "DatusCLI", targets: List[Tuple[str, str]]) -> None:
    for section, adapter_type in targets:
        result = _safe_ensure(section, adapter_type)
        if result is None:
            continue
        if result.ok:
            try:
                hot_reload_adapter(section, adapter_type)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("hot_reload_adapter(%s, %s) raised: %s", section, adapter_type, exc)
            print_info(
                cli.console,
                f"Installed {result.package} in background. Run /services {service_type_label(section)} to verify.",
            )
        else:
            print_error(
                cli.console,
                f"Background install of {result.package or adapter_type} failed: "
                f"{result.error or 'unknown error'}. Re-run /services {service_type_label(section)} to retry.",
                prefix=False,
            )


def _safe_ensure(section: str, adapter_type: str) -> Optional[InstallResult]:
    try:
        return ensure_adapter(section, adapter_type)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ensure_adapter(%s, %s) raised: %s", section, adapter_type, exc)
        return None


__all__ = [
    "run",
    "quick_pick_service",
    "background_install_missing",
]
