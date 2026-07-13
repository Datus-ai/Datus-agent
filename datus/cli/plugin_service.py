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

import hashlib
import importlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# ── Plugin bundle (.dplug) format ──────────────────────────────────────────
# A ``.dplug`` is a zip of a ``datus-plugin.json`` manifest plus a ``wheels/``
# wheelhouse (the plugin wheel and every transitive dependency), built by
# ``datus plugin pack`` and installed fully offline by :func:`install_bundle`.
BUNDLE_EXT = ".dplug"
BUNDLE_FORMAT = "datus-plugin-bundle"
BUNDLE_FORMAT_VERSION = 1
BUNDLE_MANIFEST_NAME = "datus-plugin.json"
BUNDLE_WHEELS_DIR = "wheels"


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


def _rescan_plugins(before: Set[str]) -> List[str]:
    """Invalidate caches and return ``datus.plugins`` names newly registered.

    Shared by :func:`install` and :func:`install_bundle`: after the installer
    subprocess writes a new dist-info, the import machinery and the plugin
    registry cache must be dropped so an in-process re-scan sees it. Returns the
    sorted set difference against ``before`` (the names present pre-install).
    """
    importlib.invalidate_caches()
    try:
        from datus.plugins.registry import invalidate_plugin_cache

        invalidate_plugin_cache()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("plugin cache invalidation failed after install: %s", exc)
    return sorted(_plugin_entry_names() - before)


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


def install(source: str, editable: bool = False, force: bool = False) -> InstallResult:
    """Install a plugin from PyPI / a wheel / a local directory / a ``.dplug`` bundle.

    A ``.dplug`` source is a self-contained offline bundle and is dispatched to
    :func:`install_bundle` (``editable`` is ignored there; ``force`` skips its
    python/platform compatibility gate). Every other source wraps ``uv pip
    install`` (falling back to ``pip``) and resolves dependencies from the
    default index.

    After a successful install, invalidates the import + plugin caches and
    diffs the ``datus.plugins`` entry points so callers can report the plugin
    name(s) the package registered (empty when the package exposes none — a
    likely sign the user installed the wrong package).
    """
    source = (source or "").strip()
    if not source:
        return InstallResult(ok=False, source=source, error="no install source given")

    if source.lower().endswith(BUNDLE_EXT):
        return install_bundle(source, force=force)

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
    new_plugins = _rescan_plugins(before)
    return InstallResult(ok=True, source=source, label=label, new_plugins=new_plugins, stdout=stdout, stderr=stderr)


# ── Plugin bundle (.dplug) install ─────────────────────────────────────────


class BundleError(Exception):
    """A malformed, incompatible, or tampered ``.dplug`` plugin bundle."""


def _read_bundle_manifest(zf: zipfile.ZipFile) -> dict:
    """Read and shape-validate the ``datus-plugin.json`` manifest from a bundle.

    Raises :class:`BundleError` when the manifest is absent, unparseable, of an
    unknown ``format`` / ``format_version``, or missing the ``plugin.wheel`` /
    ``wheels`` fields the installer relies on.
    """
    try:
        raw = zf.read(BUNDLE_MANIFEST_NAME)
    except KeyError:
        raise BundleError(f"bundle has no {BUNDLE_MANIFEST_NAME} manifest")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise BundleError(f"unreadable {BUNDLE_MANIFEST_NAME}: {exc}")
    if not isinstance(manifest, dict):
        raise BundleError(f"{BUNDLE_MANIFEST_NAME} must be a JSON object")
    if manifest.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"not a datus plugin bundle (format={manifest.get('format')!r})")
    if manifest.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise BundleError(
            f"unsupported bundle format_version {manifest.get('format_version')!r} "
            f"(this datus supports {BUNDLE_FORMAT_VERSION})"
        )
    plugin = manifest.get("plugin")
    if not isinstance(plugin, dict) or not isinstance(plugin.get("wheel"), str) or not plugin["wheel"]:
        raise BundleError("manifest 'plugin.wheel' is missing")
    wheels = manifest.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        raise BundleError("manifest 'wheels' list is missing or empty")
    return manifest


def _python_satisfies(requires_python: str) -> Optional[bool]:
    """Whether the running interpreter satisfies a PEP 440 specifier.

    Returns ``None`` (unknown — do not block) when ``packaging`` is unavailable
    or the specifier is unparseable; pip re-checks ``Requires-Python`` at
    install time regardless, so this gate is only an early, friendlier error.
    """
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        return Version(platform.python_version()) in SpecifierSet(requires_python)
    except Exception:  # noqa: BLE001 - missing/unparseable → don't block
        return None


def _platform_matches(plat: str) -> bool:
    """Best-effort check that platform tag ``plat`` runs on this system.

    Uses ``packaging.tags`` when available; returns ``True`` (don't block) when
    it cannot be determined, deferring to pip's own tag check at install time.
    """
    try:
        from packaging.tags import sys_tags

        return any(plat == tag.platform for tag in sys_tags())
    except Exception:  # noqa: BLE001 - unknown → don't block
        return True


def _verify_bundle_compat(manifest: dict, force: bool = False) -> List[str]:
    """Return compatibility errors for a bundle against this interpreter.

    Checks ``compat.requires_python`` (best-effort) and a non-``any``
    ``compat.platform`` tag. ``force`` skips every check (checksums are still
    enforced separately). An empty list means "compatible / unknown".
    """
    if force:
        return []
    errors: List[str] = []
    compat = manifest.get("compat") or {}
    requires_python = compat.get("requires_python")
    if isinstance(requires_python, str) and requires_python.strip():
        if _python_satisfies(requires_python) is False:
            errors.append(f"bundle requires Python {requires_python}, running {platform.python_version()}")
    plat = compat.get("platform")
    if isinstance(plat, str) and plat.strip() and plat != "any" and not _platform_matches(plat):
        errors.append(f"bundle built for platform '{plat}', incompatible with this system (use --force to override)")
    return errors


def _guard_wheel_name(name: str) -> None:
    """Reject a manifest wheel ``file`` that is not a bare, safe filename.

    Wheel entries must be plain basenames; anything with a path separator,
    ``.``/``..`` components, or an absolute path is refused so a crafted
    manifest can never write outside the extraction directory.
    """
    if name in ("", ".", "..") or name != Path(name).name:
        raise BundleError(f"unsafe wheel filename in bundle: {name!r}")


def _sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_and_verify_wheels(zf: zipfile.ZipFile, manifest: dict, dest: Path) -> Path:
    """Extract every manifest-listed wheel into ``dest/wheels`` and checksum it.

    Only files named in the manifest are extracted, each by a path this code
    constructs (``wheels/<basename>``) rather than a name taken from the archive
    listing — so a crafted member name cannot escape ``dest`` (zip-slip safe).
    Each wheel's sha256 must match the manifest or :class:`BundleError` is
    raised before anything is installed.
    """
    wheels_dir = dest / BUNDLE_WHEELS_DIR
    wheels_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest["wheels"]:
        if not isinstance(entry, dict):
            raise BundleError("manifest 'wheels' entry is not an object")
        fname = entry.get("file")
        expected = entry.get("sha256")
        if not isinstance(fname, str) or not fname:
            raise BundleError("manifest wheel entry lacks a 'file' name")
        if not isinstance(expected, str) or not expected:
            raise BundleError(f"manifest has no sha256 for {fname}")
        _guard_wheel_name(fname)
        try:
            data = zf.read(f"{BUNDLE_WHEELS_DIR}/{fname}")
        except KeyError:
            raise BundleError(f"bundle is missing a wheel listed in the manifest: {fname}")
        target = wheels_dir / fname
        target.write_bytes(data)
        actual = _sha256_file(target)
        if actual.lower() != expected.lower():
            raise BundleError(f"checksum mismatch for {fname} (bundle may be corrupt or tampered)")
    return wheels_dir


def _bundle_install_command(main_wheel: Path, wheels_dir: Path) -> Tuple[List[str], str]:
    """Build the fully-offline install command for a bundle's main wheel.

    ``--no-index`` forbids any network access and ``--find-links <wheels_dir>``
    restricts dependency resolution to the extracted wheelhouse, so the bundle
    installs deterministically from its own contents.
    """
    uv_path = shutil.which("uv")
    if uv_path:
        return (
            [
                uv_path,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-index",
                "--find-links",
                str(wheels_dir),
                str(main_wheel),
            ],
            "uv pip install (offline)",
        )
    return (
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheels_dir),
            str(main_wheel),
        ],
        "pip install (offline)",
    )


def install_bundle(path: str, force: bool = False) -> InstallResult:
    """Install a plugin from a self-contained ``.dplug`` bundle, fully offline.

    A bundle is a zip of a ``datus-plugin.json`` manifest and a ``wheels/``
    wheelhouse (the plugin wheel plus every transitive dependency). Nothing is
    fetched from the network: uv/pip resolve solely from the extracted
    wheelhouse. Every wheel's sha256 is verified against the manifest before
    install; ``force`` skips only the python/platform compatibility gate. On
    success the caches are invalidated and the newly registered plugin name(s)
    reported, mirroring :func:`install`.
    """
    path = (path or "").strip()
    if not path:
        return InstallResult(ok=False, source=path, error="no bundle path given")
    bundle = Path(path).expanduser()
    if not bundle.is_file():
        return InstallResult(ok=False, source=path, error=f"bundle not found: {path}")

    label = "uv pip install (offline)"
    try:
        with zipfile.ZipFile(bundle) as zf:
            manifest = _read_bundle_manifest(zf)
            compat_errors = _verify_bundle_compat(manifest, force=force)
            if compat_errors:
                return InstallResult(ok=False, source=path, error="; ".join(compat_errors))
            with tempfile.TemporaryDirectory(prefix="datus-dplug-") as tmp:
                wheels_dir = _extract_and_verify_wheels(zf, manifest, Path(tmp))
                main_wheel = wheels_dir / manifest["plugin"]["wheel"]
                if not main_wheel.is_file():
                    return InstallResult(
                        ok=False,
                        source=path,
                        error=f"manifest 'plugin.wheel' {manifest['plugin']['wheel']!r} not present in bundle",
                    )
                cmd, label = _bundle_install_command(main_wheel, wheels_dir)
                before = _plugin_entry_names()
                logger.info("Installing plugin bundle: %s", " ".join(cmd))
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                returncode = proc.returncode
    except BundleError as exc:
        return InstallResult(ok=False, source=path, error=str(exc))
    except (zipfile.BadZipFile, OSError) as exc:
        return InstallResult(ok=False, source=path, error=f"invalid bundle: {exc}")
    except Exception as exc:  # noqa: BLE001 - subprocess/other; never crash the CLI
        return InstallResult(ok=False, source=path, label=label, error=str(exc))

    if returncode != 0:
        return InstallResult(
            ok=False,
            source=path,
            label=label,
            stdout=stdout,
            stderr=stderr,
            error=f"{label} exited with code {returncode}",
        )
    new_plugins = _rescan_plugins(before)
    return InstallResult(ok=True, source=path, label=label, new_plugins=new_plugins, stdout=stdout, stderr=stderr)


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
    "BundleError",
    "BUNDLE_EXT",
    "install",
    "install_bundle",
    "uninstall",
    "list_plugins",
]
