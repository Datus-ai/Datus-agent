# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Directory-backed plugin store under ``~/.datus/plugins/``.

Each installed plugin lives in ``~/.datus/plugins/{name}/`` — a ``pip install
--target`` tree with the plugin package and its dependencies vendored in, plus a
``datus-plugin.json`` metadata file describing how it was installed. This module
owns:

- **location** — :func:`plugins_root` / :func:`plugin_dir`
- **metadata** — :func:`read_meta` / :func:`write_meta` / :func:`iter_installed`
- **introspection** — :func:`introspect_target` reads a freshly-installed target
  tree's ``[datus.plugins]`` entry point and validates its bundled
  ``datus-plugin.yml`` manifest, without importing the package
- **activation** — :func:`activate` appends enabled plugin directories to
  ``sys.path`` so :mod:`datus.plugins.registry` (which relies on
  ``importlib.metadata.entry_points()`` scanning ``sys.path``) discovers them,
  then invalidates the import + registry caches

Besides the managed store, ``agent.plugin_paths`` (:class:`AgentConfig`) may
mount plugin directories living anywhere on disk. Each entry is already ONE
plugin's directory — the equivalent of a single ``{plugins_root}/{name}/``
subdirectory, not a root containing several — and is merged with the managed
store as a union; on a name clash the managed install wins.

Adding a directory to ``sys.path`` makes its ``.dist-info`` discoverable but does
**not** import the plugin package — manifest reading never executes plugin code,
and declared code refs are imported lazily by ``resolve_code_ref`` — so callers
may inject a directory before the ``plugins_enabled`` master switch is checked
without executing third-party module-level code.
"""

from __future__ import annotations

import configparser
import json
import os
import sys
from email.parser import Parser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from datus.plugins.base import MANIFEST_FILENAME, PluginManifest, read_manifest_file
from datus.plugins.registry import _SAFE_PLUGIN_NAME_RE, PLUGIN_ENTRY_POINT_GROUP, invalidate_plugin_cache
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

logger = get_logger(__name__)

# datus-owned per-directory metadata file (records install provenance).
MANIFEST_NAME = "datus-plugin.json"
# Retained original zip for ``zip:`` installs so ``export`` can return it verbatim.
ORIGIN_ZIP = ".datus-origin.zip"
MANIFEST_FORMAT = "datus-plugin"
MANIFEST_FORMAT_VERSION = 1

# Subcommand tokens owned by built-in handlers; a plugin must never claim one.
# Keep in sync with ``datus.cli.main._RESERVED_SUBCOMMANDS``.
RESERVED_PLUGIN_NAMES = frozenset({"upgrade", "skill", "plugin"})


class StoreError(DatusException):
    """A recoverable problem inspecting or writing the plugin store.

    A coded :class:`DatusException` (``PLUGIN_STORE_ERROR``) so CLI/API callers
    keep the repository's ``error_code=…`` contract, while remaining catchable
    as ``StoreError`` for the store's own recoverable control flow.
    """

    def __init__(self, message: str):
        super().__init__(ErrorCode.PLUGIN_STORE_ERROR, message=message)


def plugins_root() -> Path:
    """Return ``~/.datus/plugins`` (the installed-plugins root)."""
    return get_path_manager().plugins_dir


def plugin_dir(name: str) -> Path:
    """Return the directory a plugin named ``name`` is installed into."""
    return plugins_root() / name


def is_valid_name(name: str) -> bool:
    """True when ``name`` is a safe, non-reserved plugin/directory token."""
    return isinstance(name, str) and bool(_SAFE_PLUGIN_NAME_RE.match(name)) and name not in RESERVED_PLUGIN_NAMES


def ensure_valid_name(name: str) -> None:
    """Raise :class:`StoreError` when ``name`` is unsafe or reserved."""
    if not isinstance(name, str) or not name:
        raise StoreError("plugin name is empty")
    if name in RESERVED_PLUGIN_NAMES:
        raise StoreError(f"plugin name {name!r} is reserved (conflicts with a built-in `datus {name}` command)")
    if not _SAFE_PLUGIN_NAME_RE.match(name):
        raise StoreError(f"plugin name {name!r} is not a safe CLI token")


# ── Metadata (datus-plugin.json) ───────────────────────────────────────────


def meta_path(directory: Path) -> Path:
    """Return the ``datus-plugin.json`` path inside ``directory``."""
    return directory / MANIFEST_NAME


def read_meta(directory: Path) -> Optional[Dict[str, Any]]:
    """Read ``datus-plugin.json`` from ``directory``, or ``None`` if absent/bad.

    Never raises — a plugin directory without readable metadata is simply
    skipped by enumeration so one corrupt entry cannot break ``list``.
    """
    path = meta_path(directory)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        meta = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        logger.debug("unreadable %s: %s", path, exc)
        return None
    return meta if isinstance(meta, dict) else None


def write_meta(directory: Path, meta: Dict[str, Any]) -> None:
    """Write ``meta`` as ``datus-plugin.json`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    meta_path(directory).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def iter_installed() -> List[Dict[str, Any]]:
    """Enumerate installed plugins by scanning ``~/.datus/plugins/*/`` metadata.

    Returns each directory's metadata dict augmented with a ``_dir`` key (the
    absolute path). Directories without a readable ``datus-plugin.json`` are
    skipped. Filesystem errors resolve to an empty list — enumeration must never
    crash ``list``.
    """
    root = plugins_root()
    if not root.is_dir():
        return []
    installed: List[Dict[str, Any]] = []
    try:
        entries = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError as exc:
        logger.debug("plugins root not enumerable: %s", exc)
        return []
    for directory in entries:
        meta = read_meta(directory)
        if meta is None:
            continue
        meta = dict(meta)
        meta["_dir"] = str(directory)
        installed.append(meta)
    return installed


# ── Extra plugin paths (agent.plugin_paths) ────────────────────────────────


def plugin_name_for_dir(directory: Path) -> Optional[str]:
    """Best-effort plugin (entry-point) name for one plugin directory.

    Prefers the datus-owned ``datus-plugin.json`` (managed installs); falls
    back to the ``datus.plugins`` entry point declared by the tree's
    ``*.dist-info`` (externally built or path-mounted trees). Metadata-only —
    never imports plugin code and never raises; returns ``None`` when neither
    source yields a name.
    """
    meta = read_meta(directory)
    if meta is not None:
        name = meta.get("name")
        if isinstance(name, str) and name:
            return name
    dist_info = _find_plugin_dist_info(directory)
    if dist_info is None:
        return None
    parser = configparser.ConfigParser()
    parser.optionxform = str
    try:
        parser.read_string((dist_info / "entry_points.txt").read_text(encoding="utf-8", errors="replace"))
    except (OSError, configparser.Error):
        return None
    return next(iter(parser[PLUGIN_ENTRY_POINT_GROUP]), None)


def _module_ref_for_dir(directory: Path) -> Optional[str]:
    """Dotted package ref the plugin tree in ``directory`` registers, or ``None``."""
    meta = read_meta(directory)
    if meta is not None:
        entry_point = meta.get("entry_point")
        if isinstance(entry_point, str) and entry_point.strip():
            module_ref, _, attr = entry_point.partition(":")
            if not attr.strip() and module_ref.strip():
                return module_ref.strip()
    dist_info = _find_plugin_dist_info(directory)
    if dist_info is None:
        return None
    parser = configparser.ConfigParser()
    parser.optionxform = str
    try:
        parser.read_string((dist_info / "entry_points.txt").read_text(encoding="utf-8", errors="replace"))
    except (OSError, configparser.Error):
        return None
    section = parser[PLUGIN_ENTRY_POINT_GROUP]
    name = next(iter(section), None)
    if name is None:
        return None
    module_ref, _, attr = section[name].strip().partition(":")
    if attr.strip() or not module_ref.strip():
        return None
    return module_ref.strip()


def manifest_for_dir(directory: Path, name: str) -> Optional["PluginManifest"]:
    """Parse the manifest bundled in ONE plugin directory, without importing it.

    Mirrors :func:`plugin_name_for_dir`, and exists because
    :func:`datus.plugins.registry.load_plugin_manifest` resolves through
    ``sys.path`` entry points: a plugin mounted via ``agent.plugin_paths`` (or
    a managed store the host process never activated) is invisible to it even
    though the directory was selected successfully. Callers that already know
    which directory will execute must read the manifest from THAT directory,
    so the schema they act on always belongs to the code that will run.

    Returns ``None`` — never raises — when the tree declares no usable package
    ref or bundles no valid manifest.
    """
    module_ref = _module_ref_for_dir(directory)
    if module_ref is None:
        return None
    return read_manifest_file(directory.joinpath(*module_ref.split(".")), name)


def iter_extra_plugin_dirs(extra_paths: Optional[List[str]]) -> List[tuple]:
    """Resolve ``agent.plugin_paths`` entries to unique ``(name, dir)`` pairs.

    Each entry is expected to be ONE plugin's directory (the same layout as a
    single ``{plugins_root}/{name}/`` subdirectory). ``~`` and ``$ENV_VAR``
    are expanded. Entries that are missing, contain no recognizable datus
    plugin, carry an unsafe/reserved name, or repeat an earlier entry's name
    are warned about and skipped — a bad path must never block startup.
    """
    resolved: List[tuple] = []
    seen: Set[str] = set()
    for raw in extra_paths or []:
        if not isinstance(raw, str) or not raw.strip():
            continue
        directory = Path(os.path.expandvars(raw.strip())).expanduser()
        if not directory.is_dir():
            logger.warning("plugin_paths entry %r is not a directory; skipping.", raw)
            continue
        name = plugin_name_for_dir(directory)
        if name is None or not is_valid_name(name):
            logger.warning("plugin_paths entry %r contains no recognizable datus plugin; skipping.", raw)
            continue
        if name in seen:
            logger.warning("plugin_paths entry %r duplicates plugin %r; first entry wins.", raw, name)
            continue
        seen.add(name)
        resolved.append((name, directory))
    return resolved


# ── Introspection of a freshly-installed target tree ───────────────────────


def _find_plugin_dist_info(target_dir: Path) -> Optional[Path]:
    """Return the ``.dist-info`` dir declaring a ``[datus.plugins]`` entry point.

    Scans every ``*.dist-info/entry_points.txt`` in a ``pip install --target``
    tree and returns the sole one whose ``entry_points.txt`` contains a
    non-empty ``datus.plugins`` group. A target may bundle several
    distributions (through dependencies); returning the first sorted match
    would record an arbitrary plugin identity, so ``None`` is returned for
    both zero and multiple candidates — the caller then surfaces a clear
    "not a datus plugin" / ambiguity error.
    """
    candidates: List[Path] = []
    for entry_points in sorted(target_dir.glob("*.dist-info/entry_points.txt")):
        parser = configparser.ConfigParser()
        parser.optionxform = str  # entry-point names are case-sensitive
        try:
            parser.read_string(entry_points.read_text(encoding="utf-8", errors="replace"))
        except (OSError, configparser.Error):
            continue
        if PLUGIN_ENTRY_POINT_GROUP in parser and len(parser[PLUGIN_ENTRY_POINT_GROUP]) > 0:
            candidates.append(entry_points.parent)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logger.warning(
            "Target tree declares %d distributions with a `%s` entry point (%s); refusing to guess.",
            len(candidates),
            PLUGIN_ENTRY_POINT_GROUP,
            ", ".join(sorted(c.name for c in candidates)),
        )
    return None


def introspect_target(target_dir: Path) -> Dict[str, Any]:
    """Read plugin identity from a ``pip install --target`` tree, without import.

    Locates the ``.dist-info`` that declares a ``datus.plugins`` entry point,
    validates the package's bundled ``datus-plugin.yml`` manifest, and returns
    ``{name, distribution, version, entry_point, requires_python}``. Raises
    :class:`StoreError` when the tree contains no datus plugin, the entry
    point still targets an object (legacy class-based contract), or the
    manifest is missing/unparseable — so a broken plugin fails at install
    time, not at first use.
    """
    dist_info = _find_plugin_dist_info(target_dir)
    if dist_info is None:
        raise StoreError("installed package registers no `datus.plugins` entry point (not a datus plugin)")

    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read_string((dist_info / "entry_points.txt").read_text(encoding="utf-8", errors="replace"))
    section = parser[PLUGIN_ENTRY_POINT_GROUP]
    name = next(iter(section))
    entry_point = section[name].strip()

    module_ref, _, attr = entry_point.partition(":")
    if attr.strip():
        raise StoreError(
            f"entry point `{name} = {entry_point}` targets an object (legacy class-based contract); "
            f"rebuild the plugin against the {MANIFEST_FILENAME} manifest contract"
        )
    package_dir = target_dir.joinpath(*module_ref.strip().split("."))
    if read_manifest_file(package_dir, name) is None:
        raise StoreError(
            f"package bundles no valid {MANIFEST_FILENAME} under {module_ref.strip()}/ (see log for details)"
        )

    distribution = ""
    version = ""
    requires_python = ""
    metadata_file = dist_info / "METADATA"
    if metadata_file.is_file():
        headers = Parser().parsestr(metadata_file.read_text(encoding="utf-8", errors="replace"), headersonly=True)
        distribution = (headers.get("Name") or "").strip()
        version = (headers.get("Version") or "").strip()
        requires_python = (headers.get("Requires-Python") or "").strip()

    return {
        "name": name,
        "distribution": distribution,
        "version": version,
        "entry_point": entry_point,
        "requires_python": requires_python,
    }


# ── sys.path activation ────────────────────────────────────────────────────


def _append_to_syspath(directory: Path) -> bool:
    """Append ``directory`` to ``sys.path`` if not already present. Returns added?"""
    if not directory.is_dir():
        return False
    entry = str(directory)
    if entry in sys.path:
        return False
    sys.path.append(entry)
    return True


def activate(
    active_names: Optional[Set[str]],
    plugins_enabled: bool = True,
    extra_paths: Optional[List[str]] = None,
) -> List[str]:
    """Append enabled plugin directories to ``sys.path`` and refresh caches.

    ``active_names`` mirrors ``AgentConfig.active_plugin_names()``: ``None`` means
    "no filter" (activate every installed plugin) while a set is the project's
    activation whitelist. ``extra_paths`` (``agent.plugin_paths``) mounts
    additional plugin-level directories, merged with the managed store as a
    union under the same whitelist; a name claimed by the managed store wins
    over an extra path. Directories are **appended** (not prepended) so datus'
    own dependencies keep priority over a plugin's vendored copies. When
    ``plugins_enabled`` is false this is a no-op.

    Returns the plugin names newly added to ``sys.path`` this call (empty when
    nothing changed). Invalidates the import + plugin registry caches whenever a
    directory is added so an in-process re-scan discovers the entry points.
    """
    if not plugins_enabled:
        return []

    added: List[str] = []
    managed_names: Set[str] = set()
    for meta in iter_installed():
        name = meta.get("name")
        if not is_valid_name(name):
            continue
        managed_names.add(name)
        if active_names is not None and name not in active_names:
            continue
        if _append_to_syspath(Path(meta["_dir"])):
            added.append(name)

    for name, directory in iter_extra_plugin_dirs(extra_paths):
        if name in managed_names:
            logger.warning(
                "plugin_paths entry %s duplicates installed plugin %r; the managed install wins.", directory, name
            )
            continue
        if active_names is not None and name not in active_names:
            continue
        if _append_to_syspath(directory):
            added.append(name)

    if added:
        import importlib

        importlib.invalidate_caches()
        invalidate_plugin_cache()
    return added


def activate_paths(extra_paths: Optional[List[str]]) -> List[str]:
    """Append every ``agent.plugin_paths`` directory to ``sys.path``, unfiltered.

    Used by CLI dispatch/management paths that must make a path-mounted plugin's
    entry point discoverable BEFORE the activation gate runs — the exact mirror
    of :func:`activate_name` for managed plugins. Path-only (no import), so the
    ``plugins_enabled`` / per-project activation checks that follow still run
    before any plugin code executes. Returns the plugin names newly added;
    refreshes caches when anything was.
    """
    added: List[str] = []
    for name, directory in iter_extra_plugin_dirs(extra_paths):
        if _append_to_syspath(directory):
            added.append(name)
    if added:
        import importlib

        importlib.invalidate_caches()
        invalidate_plugin_cache()
    return added


def activate_name(name: str) -> bool:
    """Append a single installed plugin's directory to ``sys.path``.

    Used by the ``datus <plugin>`` dispatch path to make a managed plugin
    discoverable before the activation gate runs (path-only; no import). Returns
    whether the directory was newly added; refreshes caches when it was.
    """
    directory = plugin_dir(name)
    if not _append_to_syspath(directory):
        return False
    import importlib

    importlib.invalidate_caches()
    invalidate_plugin_cache()
    return True


__all__ = [
    "MANIFEST_NAME",
    "ORIGIN_ZIP",
    "MANIFEST_FORMAT",
    "MANIFEST_FORMAT_VERSION",
    "RESERVED_PLUGIN_NAMES",
    "StoreError",
    "plugins_root",
    "plugin_dir",
    "is_valid_name",
    "ensure_valid_name",
    "meta_path",
    "read_meta",
    "write_meta",
    "iter_installed",
    "plugin_name_for_dir",
    "manifest_for_dir",
    "iter_extra_plugin_dirs",
    "introspect_target",
    "activate",
    "activate_paths",
    "activate_name",
]
