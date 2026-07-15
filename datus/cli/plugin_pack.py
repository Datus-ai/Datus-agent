# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Build a self-contained wheelhouse ``.zip`` plugin bundle.

A bundle is a zip of a ``datus-bundle.json`` manifest plus a ``wheels/``
wheelhouse. Two flavors:

* **without dependencies** (default) — only the plugin's own wheel. At install
  time pip resolves the dependencies from a package index (needs network).
* **with dependencies** (``--with-deps``) — the plugin wheel *and* every
  transitive dependency wheel, so it installs fully offline.

Packing is the *build-time* half and it **needs network** (to build/download the
plugin wheel, and its dependencies when ``--with-deps`` is set). Installing a
with-deps bundle does not. This module is kept separate from
:mod:`datus.cli.plugin_service` so the install path stays import-light and the
network-touching build path is independently testable (mock ``subprocess.run``).

The plugin wheel is resolved with ``pip wheel --no-deps`` which uniformly builds
from a source directory, downloads a PyPI requirement, or builds from a git URL
— so the same ``pack`` powers both ``datus plugin pack`` (source directory) and
``datus plugin export`` re-materialising a pip/git/src install.
"""

from __future__ import annotations

import configparser
import json
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import Parser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple

import yaml

from datus.cli.plugin_service import (
    BUNDLE_FORMAT,
    BUNDLE_FORMAT_VERSION,
    BUNDLE_MANIFEST_NAME,
    BUNDLE_WHEELS_DIR,
    _sha256_file,
)
from datus.plugins.base import MANIFEST_FILENAME, parse_manifest
from datus.plugins.registry import PLUGIN_ENTRY_POINT_GROUP
from datus.utils.loggings import get_logger
from datus.utils.text_utils import redact_uri

logger = get_logger(__name__)

# Bundle files are always plain ``.zip`` (the offline wheelhouse format).
BUNDLE_ZIP_EXT = ".zip"


@dataclass
class PackResult:
    ok: bool
    bundle_path: str = ""
    plugin_name: str = ""
    wheel_count: int = 0
    with_deps: bool = False
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


class PackError(Exception):
    """A recoverable failure while building a bundle (surfaced as PackResult)."""

    def __init__(self, message: str, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


def _run(cmd: List[str], what: str) -> subprocess.CompletedProcess:
    """Run a build/download subprocess, raising :class:`PackError` on failure."""
    # Sources may carry credentials (``git+https://user:token@…`` / PEP 508
    # direct refs), so redact each token before logging the command.
    logger.info("pack: %s", " ".join(redact_uri(token) for token in cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:  # noqa: BLE001 - tool missing, OSError, etc.
        raise PackError(f"{what} failed to start: {exc}")
    if proc.returncode != 0:
        raise PackError(f"{what} failed (exit {proc.returncode})", stdout=proc.stdout or "", stderr=proc.stderr or "")
    return proc


def _parse_wheel_filename(filename: str) -> Tuple[str, str, str]:
    """Return ``(distribution, version, platform_tag)`` for a wheel filename.

    Prefers ``packaging.utils.parse_wheel_filename`` (canonical name + real
    tags); falls back to a positional split of the ``name-version-…`` stem when
    ``packaging`` is unavailable. ``platform_tag`` is ``"any"`` for a pure wheel.
    """
    try:
        from packaging.utils import parse_wheel_filename

        name, version, _build, tags = parse_wheel_filename(filename)
        platforms = {tag.platform for tag in tags}
        plat = "any" if platforms == {"any"} else sorted(platforms)[0]
        return str(name), str(version), plat
    except Exception:  # noqa: BLE001 - packaging missing / non-standard name → best effort
        stem = filename[:-4] if filename.lower().endswith(".whl") else filename
        parts = stem.split("-")
        name = parts[0].replace("_", "-") if parts else filename
        version = parts[1] if len(parts) > 1 else "0"
        plat = parts[-1] if len(parts) >= 5 else "any"
        return name, version, plat


def _pip_wheel_no_deps(source: str, out_dir: Path) -> Path:
    """Build/download just the plugin's own wheel (no deps) from any source.

    ``pip wheel --no-deps`` accepts the same specifiers as ``pip install`` — a
    local project directory, a PyPI requirement, or a ``git+`` URL — so one call
    covers ``src`` / ``pip`` / ``git`` sources uniformly.
    """
    cmd = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(out_dir), source]
    _run(cmd, "build plugin wheel")
    wheels = sorted(out_dir.glob("*.whl"))
    if not wheels:
        raise PackError(f"no wheel produced from source {source!r}")
    return wheels[-1]


def _resolve_main_wheel(source: str, work_dir: Path) -> Path:
    """Produce the plugin's own wheel in ``work_dir`` from any supported source.

    An existing ``.whl`` is copied verbatim; everything else (project directory,
    PyPI requirement, git URL) is resolved via ``pip wheel --no-deps``.
    """
    candidate = Path(source).expanduser()
    if source.lower().endswith(".whl") and candidate.is_file():
        dest = work_dir / candidate.name
        shutil.copy2(candidate, dest)
        return dest
    return _pip_wheel_no_deps(source, work_dir)


def _download_dependencies(main_wheel: Path, wheels_dir: Path) -> None:
    """Download the plugin wheel's transitive dependencies into ``wheels_dir``.

    ``--only-binary=:all:`` keeps the wheelhouse wheel-only (no sdists that would
    need building at install time), so a with-deps bundle installs fully offline.
    """
    cmd = [sys.executable, "-m", "pip", "download", "--only-binary=:all:", "-d", str(wheels_dir), str(main_wheel)]
    _run(cmd, "download dependencies")


def _read_plugin_entry(wheel_path: Path) -> Tuple[str, str]:
    """Read the ``datus.plugins`` entry point ``(name, target)`` from a wheel.

    Raises :class:`PackError` when the wheel declares no entry points or none
    in the ``datus.plugins`` group, when the entry point still targets an
    object (legacy class-based contract), or when the wheel does not bundle
    the package's ``datus-plugin.yml`` — i.e. it is not a valid manifest
    plugin, caught before any dependency download happens.
    """
    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()
        candidates = [n for n in names if n.endswith(".dist-info/entry_points.txt")]
        if not candidates:
            raise PackError(f"{wheel_path.name} declares no entry points (not a datus plugin)")
        text = zf.read(candidates[0]).decode("utf-8", errors="replace")
    parser = configparser.ConfigParser()
    parser.optionxform = str  # entry-point names are case-sensitive
    try:
        parser.read_string(text)
    except configparser.Error as exc:
        raise PackError(f"{wheel_path.name} has an unparseable entry_points.txt: {exc}")
    if PLUGIN_ENTRY_POINT_GROUP not in parser:
        raise PackError(f"{wheel_path.name} has no [{PLUGIN_ENTRY_POINT_GROUP}] entry point (not a datus plugin)")
    section = parser[PLUGIN_ENTRY_POINT_GROUP]
    for name in section:
        target = section[name].strip()
        module_ref, _, attr = target.partition(":")
        if attr.strip():
            raise PackError(
                f"{wheel_path.name} entry point `{name} = {target}` targets an object (legacy class-based "
                f"contract); rebuild the plugin against the {MANIFEST_FILENAME} manifest contract"
            )
        manifest_arcname = "/".join(module_ref.strip().split(".")) + f"/{MANIFEST_FILENAME}"
        if manifest_arcname not in names:
            raise PackError(
                f"{wheel_path.name} does not bundle {manifest_arcname} — include {MANIFEST_FILENAME} "
                "in the package data"
            )
        # Validate the manifest CONTENTS (not just its presence) so an invalid
        # manifest is rejected at build time rather than surfacing only when a
        # consumer tries to install the bundle.
        with zipfile.ZipFile(wheel_path) as zf:
            manifest_text = zf.read(manifest_arcname).decode("utf-8", errors="replace")
        try:
            manifest_data = yaml.safe_load(manifest_text)
        except yaml.YAMLError as exc:
            raise PackError(f"{wheel_path.name} bundles an unparseable {manifest_arcname}: {exc}")
        if parse_manifest(manifest_data, name, Path(module_ref.strip())) is None:
            raise PackError(
                f"{wheel_path.name} bundles an invalid {manifest_arcname} (see log for details); "
                f"fix the {MANIFEST_FILENAME} manifest contract before packing"
            )
        return name, target
    raise PackError(f"{wheel_path.name} has an empty [{PLUGIN_ENTRY_POINT_GROUP}] entry-point group")


def _read_requires_python(wheel_path: Path) -> Optional[str]:
    """Return the wheel's ``Requires-Python`` metadata value, or ``None``."""
    with zipfile.ZipFile(wheel_path) as zf:
        candidates = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        if not candidates:
            return None
        text = zf.read(candidates[0]).decode("utf-8", errors="replace")
    value = Parser().parsestr(text, headersonly=True).get("Requires-Python")
    return value.strip() if value else None


def _builder_tag() -> str:
    """Return a ``datus/<version>`` provenance stamp for the manifest."""
    try:
        import importlib.metadata as importlib_metadata

        return f"datus/{importlib_metadata.version('datus-agent')}"
    except Exception:  # noqa: BLE001 - version metadata absent in odd installs
        return "datus"


def _build_manifest(
    main_wheel: Path,
    entry_name: str,
    entry_target: str,
    requires_python: Optional[str],
    wheels_dir: Path,
    with_deps: bool,
) -> dict:
    """Assemble the ``datus-bundle.json`` manifest from the gathered wheelhouse."""
    distribution, version, _ = _parse_wheel_filename(main_wheel.name)
    wheel_files = sorted(p for p in wheels_dir.iterdir() if p.suffix == ".whl")
    platforms = {_parse_wheel_filename(p.name)[2] for p in wheel_files}
    non_any = sorted(p for p in platforms if p != "any")
    bundle_platform = non_any[0] if non_any else "any"
    wheels = [
        {
            "file": wheel.name,
            "sha256": _sha256_file(wheel),
            "role": "plugin" if wheel.name == main_wheel.name else "dependency",
        }
        for wheel in wheel_files
    ]
    return {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_FORMAT_VERSION,
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "builder": _builder_tag(),
        "bundle_deps": bool(with_deps),
        "plugin": {
            "name": entry_name,
            "distribution": distribution,
            "version": version,
            "wheel": main_wheel.name,
            "entry_point": entry_target,
        },
        "compat": {
            "requires_python": requires_python or "",
            "platform": bundle_platform,
        },
        "wheels": wheels,
    }


def _write_bundle(manifest: dict, wheels_dir: Path, out_dir: Path) -> Path:
    """Zip the manifest + wheelhouse into ``<dist>-<version>.zip``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plugin = manifest["plugin"]
    out_path = out_dir / f"{plugin['distribution']}-{plugin['version']}{BUNDLE_ZIP_EXT}"
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(BUNDLE_MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
        for entry in manifest["wheels"]:
            zf.write(wheels_dir / entry["file"], f"{BUNDLE_WHEELS_DIR}/{entry['file']}")
    return out_path


def pack(source: str = ".", out_dir: str = ".", with_deps: bool = False) -> PackResult:
    """Build a wheelhouse ``.zip`` bundle from a plugin source. Requires network.

    ``source`` defaults to the current directory (a plugin project); it may also
    be an existing ``.whl``, a PyPI requirement, or a ``git+`` URL. With
    ``with_deps`` the bundle includes every transitive dependency wheel (offline
    install); without it only the plugin wheel is packed and dependencies are
    resolved from an index at install time. ``out_dir`` defaults to the current
    directory. Returns a :class:`PackResult`; never raises — build failures are
    captured with their subprocess output.
    """
    source = (source or ".").strip() or "."
    try:
        with TemporaryDirectory(prefix="datus-pack-") as tmp:
            work = Path(tmp)
            build_dir = work / "build"
            wheels_dir = work / BUNDLE_WHEELS_DIR
            build_dir.mkdir()
            wheels_dir.mkdir()

            main_wheel_src = _resolve_main_wheel(source, build_dir)
            entry_name, entry_target = _read_plugin_entry(main_wheel_src)

            main_wheel = wheels_dir / main_wheel_src.name
            shutil.copy2(main_wheel_src, main_wheel)
            if with_deps:
                _download_dependencies(main_wheel, wheels_dir)

            requires_python = _read_requires_python(main_wheel)
            manifest = _build_manifest(main_wheel, entry_name, entry_target, requires_python, wheels_dir, with_deps)
            out_path = _write_bundle(manifest, wheels_dir, Path(out_dir).expanduser())
    except PackError as exc:
        return PackResult(ok=False, error=str(exc), stdout=exc.stdout, stderr=exc.stderr)
    except (zipfile.BadZipFile, OSError) as exc:
        return PackResult(ok=False, error=f"pack failed: {exc}")

    return PackResult(
        ok=True,
        bundle_path=str(out_path),
        plugin_name=entry_name,
        wheel_count=len(manifest["wheels"]),
        with_deps=with_deps,
    )


__all__ = ["PackResult", "PackError", "pack"]
