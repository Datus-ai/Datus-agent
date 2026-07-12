# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Install / uninstall / enumerate datus plugins.

Pure Python (no prompt_toolkit / Rich import) so it can be unit-tested by
monkey-patching ``subprocess.run`` / ``shutil.which`` and the registry
enumeration. Mirrors :mod:`datus.cli.upgrade_service`:

- **install** wraps ``uv pip install`` (falling back to ``pip``) over a source
  that may be a PyPI requirement, a ``.whl`` file, or a local directory — uv
  handles all three natively, so no per-source branching is needed. After the
  install it invalidates caches and re-scans the ``datus.plugins`` entry-point
  group to report which plugin(s) the package registered.
- **uninstall** maps a plugin *entry-point name* (the ``datus <name>``
  subcommand) back to its distribution and runs ``uv pip uninstall``.
- **list** joins the installed entry points with the configured profiles and
  the project's activation state.

Installation is otherwise registration-free: a freshly installed plugin is
discovered automatically on the next ``datus`` invocation.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class PluginInfo:
    """One installed ``datus.plugins`` entry point plus its config/activation."""

    name: str  # entry-point name == the ``datus <name>`` subcommand token
    package: str = ""  # distribution (pip) name, e.g. "datus-statsig-plugin"
    version: str = ""
    entry: str = ""  # "module:attr" target
    profiles: List[str] = field(default_factory=list)  # configured profile names
    active: Optional[bool] = None  # project activation state (None: unknown)
    active_profiles: Optional[List[str]] = None  # None: all profiles active


@dataclass
class InstallResult:
    ok: bool
    source: str = ""
    label: str = ""
    new_plugins: List[str] = field(default_factory=list)  # entry names newly registered
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


@dataclass
class UninstallResult:
    ok: bool
    plugin: str = ""
    package: str = ""
    label: str = ""
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


def _plugin_entry_names() -> Set[str]:
    """Return the set of currently-registered ``datus.plugins`` entry names."""
    from datus.plugins.registry import iter_plugin_entry_points

    return {getattr(ep, "name", None) for ep in iter_plugin_entry_points() if getattr(ep, "name", None)}


def _install_command(source: str, editable: bool = False) -> Tuple[List[str], str]:
    """Build the install command, preferring ``uv pip`` when available.

    Mirrors ``upgrade_service._upgrade_command``: ``uv pip install --python
    <sys.executable>`` reuses the active interpreter without requiring ``pip``
    to be seeded. ``editable`` adds ``-e`` (only meaningful for a local source
    tree). Returns ``(argv, label)``.
    """
    editable_flag = ["-e"] if editable else []
    uv_path = shutil.which("uv")
    if uv_path:
        return (
            [uv_path, "pip", "install", "--python", sys.executable, *editable_flag, source],
            "uv pip install",
        )
    return [sys.executable, "-m", "pip", "install", *editable_flag, source], "pip install"


def install(source: str, editable: bool = False) -> InstallResult:
    """Install a plugin from PyPI / a wheel / a local directory.

    After a successful install, invalidates the import + plugin caches and
    diffs the ``datus.plugins`` entry points so callers can report the plugin
    name(s) the package registered (empty when the package exposes none — a
    likely sign the user installed the wrong package).
    """
    source = (source or "").strip()
    if not source:
        return InstallResult(ok=False, source=source, error="no install source given")

    before = _plugin_entry_names()
    cmd, label = _install_command(source, editable)
    logger.info("Installing plugin: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:  # uv / pip missing, OSError, etc.
        return InstallResult(ok=False, source=source, label=label, error=str(exc))

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        return InstallResult(
            ok=False,
            source=source,
            label=label,
            stdout=stdout,
            stderr=stderr,
            error=f"{label} exited with code {proc.returncode}",
        )

    # Make the freshly-installed dist-info visible to a re-scan in this process.
    importlib.invalidate_caches()
    try:
        from datus.plugins.registry import invalidate_plugin_cache

        invalidate_plugin_cache()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("plugin cache invalidation failed after install: %s", exc)
    after = _plugin_entry_names()
    new_plugins = sorted(after - before)
    return InstallResult(ok=True, source=source, label=label, new_plugins=new_plugins, stdout=stdout, stderr=stderr)


def _distribution_for_plugin(name: str) -> Optional[str]:
    """Return the distribution (pip) name registering plugin ``name``, or None.

    Uses the entry point's ``.dist`` back-reference (Python 3.10+). Falls back
    to ``None`` when the metadata does not expose it so the caller can surface
    a clear error rather than uninstalling the wrong package.
    """
    from datus.plugins.registry import entry_points_for_group

    for ep in entry_points_for_group("datus.plugins", name=name):
        dist = getattr(ep, "dist", None)
        dist_name = getattr(dist, "name", None) if dist is not None else None
        if isinstance(dist_name, str) and dist_name:
            return dist_name
    return None


def _uninstall_command(package: str) -> Tuple[List[str], str]:
    """Build the uninstall command, preferring ``uv pip`` when available."""
    uv_path = shutil.which("uv")
    if uv_path:
        return [uv_path, "pip", "uninstall", "--python", sys.executable, package], "uv pip uninstall"
    return [sys.executable, "-m", "pip", "uninstall", "-y", package], "pip uninstall"


def uninstall(plugin_name: str) -> UninstallResult:
    """Uninstall the distribution that registers plugin ``plugin_name``."""
    plugin_name = (plugin_name or "").strip()
    if not plugin_name:
        return UninstallResult(ok=False, plugin=plugin_name, error="no plugin name given")

    package = _distribution_for_plugin(plugin_name)
    if not package:
        return UninstallResult(
            ok=False,
            plugin=plugin_name,
            error=f"no installed plugin named '{plugin_name}' (nothing to uninstall)",
        )

    cmd, label = _uninstall_command(package)
    logger.info("Uninstalling plugin '%s' (package %s): %s", plugin_name, package, " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:
        return UninstallResult(ok=False, plugin=plugin_name, package=package, label=label, error=str(exc))

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        return UninstallResult(
            ok=False,
            plugin=plugin_name,
            package=package,
            label=label,
            stdout=stdout,
            stderr=stderr,
            error=f"{label} exited with code {proc.returncode}",
        )
    importlib.invalidate_caches()
    try:
        from datus.plugins.registry import invalidate_plugin_cache

        invalidate_plugin_cache()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("plugin cache invalidation failed after uninstall: %s", exc)
    return UninstallResult(ok=True, plugin=plugin_name, package=package, label=label, stdout=stdout, stderr=stderr)


def list_plugins(agent_config=None) -> List[PluginInfo]:
    """Enumerate installed plugins with configured profiles and activation.

    ``agent_config`` (optional) supplies configured profiles
    (``plugin_services``) and the project's activation state
    (``active_plugin_names`` / ``active_plugin_profiles``). Without it, the
    ``profiles`` / ``active`` fields are left empty / ``None``.
    """
    from datus.plugins.registry import iter_plugin_entry_points

    active_names: Optional[Set[str]] = None
    plugin_services = {}
    if agent_config is not None:
        try:
            active_names = agent_config.active_plugin_names()
        except Exception as exc:  # noqa: BLE001 - listing must not crash on a bad config
            logger.debug("active_plugin_names() failed during list: %s", exc)
        plugin_services = getattr(agent_config, "plugin_services", {}) or {}

    infos: List[PluginInfo] = []
    for ep in iter_plugin_entry_points():
        name = getattr(ep, "name", None)
        if not isinstance(name, str) or not name:
            continue
        dist = getattr(ep, "dist", None)
        info = PluginInfo(
            name=name,
            package=str(getattr(dist, "name", "") or ""),
            version=str(getattr(dist, "version", "") or ""),
            entry=str(getattr(ep, "value", "") or ""),
            profiles=sorted((plugin_services.get(name) or {}).keys()),
        )
        if agent_config is not None:
            info.active = active_names is None or name in active_names
            try:
                info.active_profiles = agent_config.active_plugin_profiles(name)
            except Exception:  # noqa: BLE001 - best-effort activation detail
                info.active_profiles = None
        infos.append(info)
    return sorted(infos, key=lambda p: p.name)


__all__ = [
    "PluginInfo",
    "InstallResult",
    "UninstallResult",
    "install",
    "uninstall",
    "list_plugins",
]
