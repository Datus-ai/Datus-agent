# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Install / uninstall / enumerate datus plugins into ``~/.datus/plugins/``.

Pure Python (no prompt_toolkit / Rich import) so it can be unit-tested by
monkey-patching ``subprocess.run`` / ``shutil.which`` and the store.

Each plugin is installed into its own ``~/.datus/plugins/{name}/`` directory via
``pip install --target`` (dependencies vendored in), described by a
``datus-plugin.json`` metadata file (see :mod:`datus.plugins.store`). Enabled
directories are appended to ``sys.path`` at startup so the ``datus.plugins``
entry point is discovered.

Install sources use a ``{type}:{src}`` prefix; the type is optional and
defaults to ``pip`` (so a bare ``datus-foo`` == ``pip:datus-foo``):

- ``pip:<requirement>``    — a PyPI requirement (deps resolved from an index); default
- ``src:<directory>``      — a local plugin project directory
- ``whl:<file.whl>``       — a local wheel file
- ``git:<url>``            — a git repository (``git+`` auto-prepended)
- ``zip:<bundle.zip>``     — an offline wheelhouse bundle built by ``plugin pack``

For every install the provenance is recorded in the metadata so ``upgrade`` can
re-fetch with the original method and ``export`` can reproduce a distributable
``.zip`` (a ``zip:`` install additionally retains the original bundle verbatim).
"""

from __future__ import annotations

import hashlib
import importlib
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple

from datus.plugins import store
from datus.plugins.store import StoreError
from datus.utils.loggings import get_logger
from datus.utils.text_utils import redact_uri

logger = get_logger(__name__)


def _redacted_cmd(cmd: List[str]) -> str:
    """Join a subprocess command for logging, redacting credentials in URLs.

    Install sources may be credential-bearing (``git+https://user:token@…``) or
    PEP 508 direct references, so every token is passed through
    :func:`redact_uri` before it reaches the log.
    """
    return " ".join(redact_uri(token) for token in cmd)


# ── Offline wheelhouse bundle (.zip) format ────────────────────────────────
# A bundle ``.zip`` holds a ``datus-bundle.json`` manifest plus a ``wheels/``
# wheelhouse (the plugin wheel and — when ``bundle_deps`` — every transitive
# dependency), built by ``datus plugin pack`` and installed by ``install`` when
# the source is ``zip:``.
BUNDLE_EXT = ".zip"
BUNDLE_FORMAT = "datus-plugin-bundle"
BUNDLE_FORMAT_VERSION = 1
BUNDLE_MANIFEST_NAME = "datus-bundle.json"
BUNDLE_WHEELS_DIR = "wheels"

# Recognised ``{type}:{src}`` install-source prefixes.
INSTALL_TYPES = ("pip", "src", "whl", "git", "zip")


@dataclass
class PluginInfo:
    """One installed plugin plus its config/activation state."""

    name: str  # entry-point name == the ``datus <name>`` subcommand token
    package: str = ""  # distribution (pip) name, e.g. "datus-airflow-plugin"
    version: str = ""
    entry: str = ""  # "module:attr" target
    install_type: str = ""  # pip|src|whl|git|zip (managed) or "" (external)
    source: str = ""  # "managed" (~/.datus/plugins), "path" (agent.plugin_paths) or "external" (site-packages)
    profiles: List[str] = field(default_factory=list)  # configured profile names
    active: Optional[bool] = None  # project activation state (None: unknown)
    active_profiles: Optional[List[str]] = None  # None: all profiles active


@dataclass
class InstallResult:
    ok: bool
    source: str = ""
    label: str = ""
    name: str = ""  # installed plugin (entry-point) name
    version: str = ""
    plugin_dir: str = ""
    new_plugins: List[str] = field(default_factory=list)  # names newly available
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


@dataclass
class UninstallResult:
    ok: bool
    plugin: str = ""
    package: str = ""
    error: Optional[str] = None


@dataclass
class ExportResult:
    ok: bool
    name: str = ""
    out_path: str = ""
    error: Optional[str] = None


@dataclass
class UpgradeResult:
    ok: bool
    name: str = ""
    label: str = ""
    old_version: str = ""
    new_version: str = ""
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


# ── Bundle manifest / wheelhouse helpers (shared with plugin_pack) ──────────


class BundleError(Exception):
    """A malformed, incompatible, or tampered wheelhouse ``.zip`` bundle."""


def _sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bundle_manifest(zf: zipfile.ZipFile) -> dict:
    """Read and shape-validate the ``datus-bundle.json`` manifest from a bundle."""
    try:
        raw = zf.read(BUNDLE_MANIFEST_NAME)
    except KeyError:
        raise BundleError(f"bundle has no {BUNDLE_MANIFEST_NAME} manifest")
    import json

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


def _bundle_deps_flag(manifest: dict) -> bool:
    """Whether a bundle carries all its dependencies (offline-installable).

    Prefers the explicit ``bundle_deps`` field; when absent (older bundles),
    infers from the wheel count (more than the plugin wheel → deps bundled).
    """
    flag = manifest.get("bundle_deps")
    if isinstance(flag, bool):
        return flag
    return len(manifest.get("wheels", [])) > 1


def _python_satisfies(requires_python: str) -> Optional[bool]:
    """Whether the running interpreter satisfies a PEP 440 specifier (or None)."""
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        return Version(platform.python_version()) in SpecifierSet(requires_python)
    except Exception:  # noqa: BLE001 - missing/unparseable → don't block
        return None


def _platform_matches(plat: str) -> bool:
    """Best-effort check that platform tag ``plat`` runs on this system."""
    try:
        from packaging.tags import sys_tags

        return any(plat == tag.platform for tag in sys_tags())
    except Exception:  # noqa: BLE001 - unknown → don't block
        return True


def _verify_bundle_compat(manifest: dict, force: bool = False) -> List[str]:
    """Return compatibility errors for a bundle against this interpreter."""
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
    """Reject a manifest wheel ``file`` that is not a bare, safe filename."""
    if name in ("", ".", "..") or name != Path(name).name:
        raise BundleError(f"unsafe wheel filename in bundle: {name!r}")


def _extract_and_verify_wheels(zf: zipfile.ZipFile, manifest: dict, dest: Path) -> Path:
    """Extract every manifest-listed wheel into ``dest/wheels`` and checksum it.

    Only files named in the manifest are extracted, each by a path this code
    constructs (``wheels/<basename>``) rather than a name taken from the archive
    listing — so a crafted member name cannot escape ``dest`` (zip-slip safe).
    Each wheel's sha256 must match the manifest before anything is installed.
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


# ── Install-command builders ───────────────────────────────────────────────


def _target_install_command(spec: str, target: Path, upgrade: bool = False) -> Tuple[List[str], str]:
    """Build a ``pip install --target`` command, preferring ``uv`` when present."""
    upgrade_flag = ["--upgrade"] if upgrade else []
    uv_path = shutil.which("uv")
    if uv_path:
        return (
            [uv_path, "pip", "install", "--python", sys.executable, "--target", str(target), *upgrade_flag, spec],
            "uv pip install",
        )
    return [sys.executable, "-m", "pip", "install", "--target", str(target), *upgrade_flag, spec], "pip install"


def _bundle_install_command(
    main_wheel: Path, wheels_dir: Path, target: Path, bundle_deps: bool
) -> Tuple[List[str], str]:
    """Build the ``pip install --target`` command for a wheelhouse bundle.

    ``--find-links <wheels_dir>`` prefers the extracted wheelhouse. A with-deps
    bundle adds ``--no-index`` (fully offline); a no-deps bundle omits it so pip
    resolves the missing dependencies from an index.
    """
    flags: List[str] = ["--target", str(target), "--find-links", str(wheels_dir)]
    if bundle_deps:
        flags = ["--no-index", *flags]
    uv_path = shutil.which("uv")
    if uv_path:
        return (
            [uv_path, "pip", "install", "--python", sys.executable, *flags, str(main_wheel)],
            "uv pip install (offline)" if bundle_deps else "uv pip install",
        )
    return (
        [sys.executable, "-m", "pip", "install", *flags, str(main_wheel)],
        "pip install (offline)" if bundle_deps else "pip install",
    )


# ── Spec parsing ───────────────────────────────────────────────────────────


def parse_spec(spec: str) -> Tuple[str, str]:
    """Split a ``{type}:{src}`` install source. Raises ``ValueError`` if invalid.

    The type prefix is optional and defaults to ``pip``: a spec with no ``:`` —
    or whose token before the first ``:`` is not a recognised install type — is
    treated as a bare ``pip`` requirement (so ``datus-foo`` == ``pip:datus-foo``,
    and a PEP 508 direct reference like ``foo @ https://…`` passes through intact).

    Splits on the first ``:`` only, so git URLs (``git:https://…``) and Windows
    drive letters in ``src`` survive intact.
    """
    head, sep, rest = spec.partition(":")
    itype = head.strip().lower()
    if sep and itype in INSTALL_TYPES:
        src = rest.strip()
        if not src:
            raise ValueError(f"empty source after '{itype}:'")
        return itype, src
    # No recognised ``type:`` prefix → default to pip, whole spec is the requirement.
    src = spec.strip()
    if not src:
        raise ValueError("empty install source")
    return "pip", src


def _normalize_git(src: str) -> str:
    """Prepend ``git+`` to a bare git URL so pip treats it as a VCS source."""
    return src if src.startswith("git+") else f"git+{src}"


def _pip_spec_and_ref(itype: str, src: str) -> Tuple[str, str]:
    """Return ``(pip_spec, recorded_ref)`` for a non-zip install source.

    Raises :class:`StoreError` when a local path source does not exist.
    """
    if itype == "src":
        path = Path(src).expanduser()
        if not path.is_dir():
            raise StoreError(f"source directory not found: {src}")
        resolved = str(path.resolve())
        return resolved, resolved
    if itype == "whl":
        path = Path(src).expanduser()
        if not path.is_file() or not src.lower().endswith(".whl"):
            raise StoreError(f"wheel file not found: {src}")
        resolved = str(path.resolve())
        return resolved, resolved
    if itype == "git":
        normalized = _normalize_git(src)
        return normalized, normalized
    # pip
    return src, src


# ── Directory metadata + finalization ──────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _build_dir_meta(info: dict, itype: str, source: str, origin_artifact: Optional[str]) -> dict:
    """Assemble the ``datus-plugin.json`` metadata written into a plugin dir."""
    return {
        "format": store.MANIFEST_FORMAT,
        "format_version": store.MANIFEST_FORMAT_VERSION,
        "name": info["name"],
        "distribution": info.get("distribution", ""),
        "version": info.get("version", ""),
        "entry_point": info.get("entry_point", ""),
        "requires_python": info.get("requires_python", ""),
        "install": {
            "type": itype,
            "source": source,
            "ref": info.get("ref"),
            "origin_artifact": origin_artifact,
        },
        "installed_at": _now_iso(),
    }


def _replace_dir(dest: Path, src_dir: Path) -> None:
    """Replace ``dest`` with ``src_dir``, preserving the old ``dest`` on failure.

    The current ``dest`` (if any) is moved aside to a sibling backup before the
    freshly-staged ``src_dir`` is moved into place. The backup is restored on any
    failure and removed only after a clean swap, so an interrupted move can never
    destroy a working plugin or leave a half-written directory behind. Callers
    must stage the plugin's metadata into ``src_dir`` before calling this, so the
    committed directory is complete the instant it appears.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    backup = dest.parent / f"{dest.name}.datus-bak"
    if backup.exists():
        shutil.rmtree(backup)
    if dest.exists():
        dest.rename(backup)
    try:
        shutil.move(str(src_dir), str(dest))
    except Exception:
        # Roll back: drop any partial destination and restore the backup.
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        if backup.exists():
            backup.rename(dest)
        raise
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def _resolve_dest(name: str, dest_dir: Optional[str]) -> Path:
    """Destination directory for plugin ``name`` (``dest_dir`` overrides the store)."""
    return Path(dest_dir).expanduser() if dest_dir else store.plugin_dir(name)


def _refresh(name: str, directory: Path) -> None:
    """Make a freshly-installed plugin discoverable in this process.

    Ensures ``directory`` is on ``sys.path``, then unconditionally drops the
    import + plugin registry caches. The ``activate_*`` helpers only invalidate
    when they newly append a directory, so a same-name replace (``--force`` /
    ``upgrade``) — whose directory is already on ``sys.path`` — would otherwise
    keep serving the replaced plugin's stale manifest for the rest of the
    process.
    """
    try:
        if directory == store.plugin_dir(name):
            store.activate_name(name)
        else:
            store.activate_paths([str(directory)])
    except Exception as exc:  # noqa: BLE001 - defensive: never crash the install on refresh
        logger.debug("plugin path activation after install failed: %s", exc)
    importlib.invalidate_caches()
    try:
        from datus.plugins.registry import invalidate_plugin_cache

        invalidate_plugin_cache()
    except Exception as exc:  # noqa: BLE001 - defensive: never crash the install on refresh
        logger.debug("plugin cache invalidation failed after install: %s", exc)


# ── install ────────────────────────────────────────────────────────────────


def install(spec: str, force: bool = False, dest_dir: Optional[str] = None) -> InstallResult:
    """Install a plugin from a ``{type}:{src}`` source into ``~/.datus/plugins/``.

    Every non-zip source installs via ``pip install --target`` into the plugin's
    directory; ``zip:`` installs a self-contained wheelhouse bundle. Provenance
    is recorded in ``datus-plugin.json`` for later ``upgrade`` / ``export``. When
    the plugin is already installed, ``force`` replaces it.

    ``dest_dir`` (reserved for future use; not surfaced on the CLI yet) installs
    the plugin tree into that exact directory instead of
    ``~/.datus/plugins/{name}/`` — the one-directory-one-plugin layout that
    ``agent.plugin_paths`` mounts. Such an install is not enumerated by the
    managed store; mount it via ``agent.plugin_paths`` to use it.
    """
    spec = (spec or "").strip()
    if not spec:
        return InstallResult(ok=False, source=spec, error="no install source given")
    try:
        itype, src = parse_spec(spec)
    except ValueError as exc:
        return InstallResult(ok=False, source=spec, error=str(exc))

    dest_dir = (dest_dir or "").strip() or None
    if itype == "zip":
        return _install_zip(src, force=force, dest_dir=dest_dir)
    return _install_via_target(itype, src, force=force, dest_dir=dest_dir)


def _install_via_target(
    itype: str,
    src: str,
    force: bool,
    upgrade: bool = False,
    expected_name: Optional[str] = None,
    dest_dir: Optional[str] = None,
) -> InstallResult:
    """Install a ``pip``/``src``/``whl``/``git`` source via ``pip install --target``.

    ``expected_name`` pins the plugin identity: when set (an ``upgrade``), an
    installed distribution whose entry-point name differs is rejected so the
    upgrade cannot install a renamed plugin into a new directory while leaving
    the old one behind. ``dest_dir`` overrides the destination directory (see
    :func:`install`).
    """
    try:
        pip_spec, ref = _pip_spec_and_ref(itype, src)
    except StoreError as exc:
        return InstallResult(ok=False, source=src, error=str(exc))

    with tempfile.TemporaryDirectory(prefix="datus-install-") as tmp:
        target = Path(tmp) / "target"
        target.mkdir()
        cmd, label = _target_install_command(pip_spec, target, upgrade=upgrade)
        logger.info("Installing plugin (%s): %s", label, _redacted_cmd(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as exc:  # uv / pip missing, OSError, etc.
            return InstallResult(ok=False, source=src, label=label, error=str(exc))
        stdout, stderr = proc.stdout or "", proc.stderr or ""
        if proc.returncode != 0:
            return InstallResult(
                ok=False,
                source=src,
                label=label,
                stdout=stdout,
                stderr=stderr,
                error=f"{label} exited with code {proc.returncode}",
            )
        try:
            info = store.introspect_target(target)
            store.ensure_valid_name(info["name"])
        except StoreError as exc:
            return InstallResult(ok=False, source=src, label=label, stdout=stdout, stderr=stderr, error=str(exc))

        info["ref"] = ref if itype == "git" else None
        name = info["name"]
        if expected_name is not None and name != expected_name:
            return InstallResult(
                ok=False,
                source=src,
                label=label,
                name=name,
                stdout=stdout,
                stderr=stderr,
                error=(
                    f"upgrade would change plugin identity: installed distribution registers `{name}`, "
                    f"expected `{expected_name}`. Uninstall `{expected_name}` and install `{name}` explicitly instead."
                ),
            )
        dest = _resolve_dest(name, dest_dir)
        if dest.exists() and not force:
            error = (
                f"destination directory {dest} already exists (pass force=True to replace)"
                if dest_dir
                else f"plugin '{name}' is already installed (use --force to replace, or `datus plugin upgrade {name}`)"
            )
            return InstallResult(ok=False, source=src, name=name, error=error)
        # Stage metadata into the temp target BEFORE the swap so the committed
        # directory is complete the instant it appears and a metadata failure
        # never leaves the old plugin destroyed.
        store.write_meta(target, _build_dir_meta(info, itype, ref, origin_artifact=None))
        _replace_dir(dest, target)
        _refresh(name, dest)
        return InstallResult(
            ok=True,
            source=src,
            label=label,
            name=name,
            version=info.get("version", ""),
            plugin_dir=str(dest),
            new_plugins=[name],
            stdout=stdout,
            stderr=stderr,
        )


def _zip_dest_exists_error(name: str, dest: Path, dest_dir: Optional[str]) -> str:
    """Refusal message for a bundle install whose destination already exists."""
    if dest_dir:
        return f"destination directory {dest} already exists (pass force=True to replace)"
    return f"plugin '{name}' is already installed (use --force to replace)"


def _install_zip(src: str, force: bool, dest_dir: Optional[str] = None) -> InstallResult:
    """Install a plugin from a self-contained wheelhouse ``.zip`` bundle.

    ``dest_dir`` overrides the destination directory (see :func:`install`).
    """
    bundle = Path(src).expanduser()
    if not bundle.is_file():
        return InstallResult(ok=False, source=src, error=f"bundle not found: {src}")

    label = "pip install"
    try:
        with zipfile.ZipFile(bundle) as zf:
            manifest = _read_bundle_manifest(zf)
            compat_errors = _verify_bundle_compat(manifest, force=force)
            if compat_errors:
                return InstallResult(ok=False, source=src, error="; ".join(compat_errors))
            bundle_deps = _bundle_deps_flag(manifest)
            # Fast pre-check on the manifest's declared name (avoids a wasted
            # install when the plugin is already present and --force is absent).
            declared = manifest["plugin"].get("name")
            if isinstance(declared, str) and declared and not force:
                pre_dest = _resolve_dest(declared, dest_dir)
                if pre_dest.exists():
                    return InstallResult(
                        ok=False,
                        source=src,
                        name=declared,
                        error=_zip_dest_exists_error(declared, pre_dest, dest_dir),
                    )
            with tempfile.TemporaryDirectory(prefix="datus-zip-") as tmp:
                wheels_dir = _extract_and_verify_wheels(zf, manifest, Path(tmp))
                main_wheel = wheels_dir / manifest["plugin"]["wheel"]
                if not main_wheel.is_file():
                    return InstallResult(
                        ok=False,
                        source=src,
                        error=f"manifest 'plugin.wheel' {manifest['plugin']['wheel']!r} not present in bundle",
                    )
                target = Path(tmp) / "target"
                target.mkdir()
                cmd, label = _bundle_install_command(main_wheel, wheels_dir, target, bundle_deps)
                logger.info("Installing plugin bundle (%s): %s", label, _redacted_cmd(cmd))
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                stdout, stderr = proc.stdout or "", proc.stderr or ""
                if proc.returncode != 0:
                    return InstallResult(
                        ok=False,
                        source=src,
                        label=label,
                        stdout=stdout,
                        stderr=stderr,
                        error=f"{label} exited with code {proc.returncode}",
                    )
                try:
                    info = store.introspect_target(target)
                    store.ensure_valid_name(info["name"])
                except StoreError as exc:
                    return InstallResult(ok=False, source=src, label=label, error=str(exc))
                name = info["name"]
                dest = _resolve_dest(name, dest_dir)
                if dest.exists() and not force:
                    return InstallResult(
                        ok=False,
                        source=src,
                        name=name,
                        error=_zip_dest_exists_error(name, dest, dest_dir),
                    )
                # Stage the retained origin bundle + metadata into the temp
                # target before the swap so the committed directory is complete
                # and the old plugin survives any staging failure.
                shutil.copy2(bundle, target / store.ORIGIN_ZIP)
                store.write_meta(
                    target,
                    _build_dir_meta(info, "zip", str(bundle.resolve()), origin_artifact=store.ORIGIN_ZIP),
                )
                _replace_dir(dest, target)
                _refresh(name, dest)
                return InstallResult(
                    ok=True,
                    source=src,
                    label=label,
                    name=name,
                    version=info.get("version", ""),
                    plugin_dir=str(dest),
                    new_plugins=[name],
                    stdout=stdout,
                    stderr=stderr,
                )
    except BundleError as exc:
        return InstallResult(ok=False, source=src, error=str(exc))
    except (zipfile.BadZipFile, OSError) as exc:
        return InstallResult(ok=False, source=src, error=f"invalid bundle: {exc}")
    except Exception as exc:  # noqa: BLE001 - subprocess/other; never crash the CLI
        return InstallResult(ok=False, source=src, label=label, error=str(exc))


# ── pack / export ──────────────────────────────────────────────────────────


def pack(source: str = ".", out_dir: str = ".", with_deps: bool = False):
    """Build a distributable wheelhouse ``.zip`` from a plugin source directory.

    Thin delegate to :func:`datus.cli.plugin_pack.pack` (imported lazily to keep
    the install path import-light). Returns a ``PackResult``.
    """
    from datus.cli import plugin_pack

    return plugin_pack.pack(source, out_dir=out_dir, with_deps=with_deps)


def export(name: str, out_dir: str = ".") -> ExportResult:
    """Export an installed plugin as a distributable wheelhouse ``.zip``.

    A ``zip:`` install returns its retained original bundle verbatim (offline). A
    ``pip``/``src``/``whl``/``git`` install is re-materialised from its recorded
    source into a with-deps bundle (needs network).
    """
    name = (name or "").strip()
    if not name:
        return ExportResult(ok=False, error="no plugin name given")
    dest = store.plugin_dir(name)
    meta = store.read_meta(dest)
    if meta is None:
        return ExportResult(ok=False, name=name, error=f"no installed plugin named '{name}'")

    install_meta = meta.get("install") or {}
    itype = install_meta.get("type")
    source = install_meta.get("source") or ""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if itype == "zip":
        origin_name = install_meta.get("origin_artifact") or store.ORIGIN_ZIP
        origin_path = dest / origin_name
        if not origin_path.is_file():
            return ExportResult(ok=False, name=name, error="original bundle was not retained; cannot export")
        distribution = meta.get("distribution") or name
        version = meta.get("version") or "0"
        target = out / f"{distribution}-{version}{BUNDLE_EXT}"
        shutil.copy2(origin_path, target)
        return ExportResult(ok=True, name=name, out_path=str(target))

    if not source:
        return ExportResult(ok=False, name=name, error=f"plugin '{name}' has no recorded source to re-pack")
    result = pack(source, out_dir=str(out), with_deps=True)
    if not result.ok:
        return ExportResult(ok=False, name=name, error=result.error or "pack failed")
    return ExportResult(ok=True, name=name, out_path=result.bundle_path)


# ── upgrade ────────────────────────────────────────────────────────────────


def upgrade(name: str) -> UpgradeResult:
    """Re-install a plugin from its recorded source (the original method).

    ``pip`` adds ``--upgrade``; ``git`` re-fetches; ``src`` rebuilds from the
    recorded path. ``whl`` and ``zip`` are pinned artifacts and report that they
    cannot be upgraded in place.
    """
    name = (name or "").strip()
    if not name:
        return UpgradeResult(ok=False, error="no plugin name given")
    dest = store.plugin_dir(name)
    meta = store.read_meta(dest)
    if meta is None:
        return UpgradeResult(ok=False, name=name, error=f"no installed plugin named '{name}'")

    install_meta = meta.get("install") or {}
    itype = install_meta.get("type")
    source = install_meta.get("source") or ""
    old_version = meta.get("version", "")

    if itype in ("zip", "whl"):
        return UpgradeResult(
            ok=False,
            name=name,
            old_version=old_version,
            error=f"`{itype}` installs are pinned artifacts and cannot be upgraded in place; reinstall a newer one",
        )
    if not source:
        return UpgradeResult(ok=False, name=name, error=f"plugin '{name}' has no recorded source to upgrade from")

    result = _install_via_target(itype, source, force=True, upgrade=(itype == "pip"), expected_name=name)
    if not result.ok:
        return UpgradeResult(
            ok=False,
            name=name,
            label=result.label,
            old_version=old_version,
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.error,
        )
    return UpgradeResult(
        ok=True,
        name=result.name or name,
        label=result.label,
        old_version=old_version,
        new_version=result.version,
        stdout=result.stdout,
        stderr=result.stderr,
    )


# ── uninstall ──────────────────────────────────────────────────────────────


def _drop_from_syspath(directory: Path) -> None:
    """Remove a plugin directory from ``sys.path`` (best-effort)."""
    entry = str(directory)
    while entry in sys.path:
        sys.path.remove(entry)


def uninstall(plugin_name: str) -> UninstallResult:
    """Remove a managed plugin's ``~/.datus/plugins/{name}/`` directory."""
    plugin_name = (plugin_name or "").strip()
    if not plugin_name:
        return UninstallResult(ok=False, plugin=plugin_name, error="no plugin name given")

    dest = store.plugin_dir(plugin_name)
    meta = store.read_meta(dest)
    if meta is None or not dest.is_dir():
        return UninstallResult(
            ok=False,
            plugin=plugin_name,
            error=(
                f"no managed plugin named '{plugin_name}' under {store.plugins_root()} "
                "(externally pip-installed plugins are removed with pip)"
            ),
        )
    package = str(meta.get("distribution") or "")
    try:
        shutil.rmtree(dest)
    except OSError as exc:
        return UninstallResult(ok=False, plugin=plugin_name, package=package, error=str(exc))
    _drop_from_syspath(dest)
    importlib.invalidate_caches()
    try:
        from datus.plugins.registry import invalidate_plugin_cache

        invalidate_plugin_cache()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("plugin cache invalidation failed after uninstall: %s", exc)
    return UninstallResult(ok=True, plugin=plugin_name, package=package)


# ── list ───────────────────────────────────────────────────────────────────


def list_plugins(agent_config=None) -> List[PluginInfo]:
    """Enumerate installed plugins (managed + path-mounted + pip-installed).

    Managed plugins are read from ``~/.datus/plugins/*/datus-plugin.json``,
    then ``agent.plugin_paths`` mounts are introspected (one directory = one
    plugin), and externally installed ones are discovered via the
    ``datus.plugins`` entry-point group as a fallback (a name present in
    several sources prefers the earlier one). ``agent_config`` (optional)
    supplies configured profiles, the ``plugin_paths`` mounts and the
    project's activation state.
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

    by_name: dict[str, PluginInfo] = {}
    for meta in store.iter_installed():
        name = meta.get("name")
        if not store.is_valid_name(name):
            continue
        install_meta = meta.get("install") or {}
        by_name[name] = PluginInfo(
            name=name,
            package=str(meta.get("distribution") or ""),
            version=str(meta.get("version") or ""),
            entry=str(meta.get("entry_point") or ""),
            install_type=str(install_meta.get("type") or ""),
            source="managed",
        )

    # Path-mounted plugins (agent.plugin_paths) are listed even while inactive
    # (their entry points may be off sys.path), so they stay visible and
    # re-enable-able like a disabled managed plugin.
    extra_paths = getattr(agent_config, "plugin_paths", None) if agent_config is not None else None
    for name, directory in store.iter_extra_plugin_dirs(extra_paths):
        if name in by_name:
            continue
        try:
            ident = store.introspect_target(directory)
        except store.StoreError as exc:
            logger.debug("plugin_paths entry %s not listable: %s", directory, exc)
            continue
        by_name[name] = PluginInfo(
            name=name,
            package=str(ident.get("distribution") or ""),
            version=str(ident.get("version") or ""),
            entry=str(ident.get("entry_point") or ""),
            install_type="",
            source="path",
        )

    for ep in iter_plugin_entry_points():
        name = getattr(ep, "name", None)
        if not isinstance(name, str) or not name or name in by_name:
            continue
        dist = getattr(ep, "dist", None)
        by_name[name] = PluginInfo(
            name=name,
            package=str(getattr(dist, "name", "") or ""),
            version=str(getattr(dist, "version", "") or ""),
            entry=str(getattr(ep, "value", "") or ""),
            install_type="",
            source="external",
        )

    for name, info in by_name.items():
        info.profiles = sorted((plugin_services.get(name) or {}).keys())
        if agent_config is not None:
            info.active = active_names is None or name in active_names
            try:
                info.active_profiles = agent_config.active_plugin_profiles(name)
            except Exception:  # noqa: BLE001 - best-effort activation detail
                info.active_profiles = None

    return sorted(by_name.values(), key=lambda p: p.name)


__all__ = [
    "PluginInfo",
    "InstallResult",
    "UninstallResult",
    "ExportResult",
    "UpgradeResult",
    "BundleError",
    "BUNDLE_EXT",
    "BUNDLE_FORMAT",
    "BUNDLE_FORMAT_VERSION",
    "BUNDLE_MANIFEST_NAME",
    "BUNDLE_WHEELS_DIR",
    "INSTALL_TYPES",
    "parse_spec",
    "install",
    "pack",
    "export",
    "upgrade",
    "uninstall",
    "list_plugins",
]
