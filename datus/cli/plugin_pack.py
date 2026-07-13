# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Build a self-contained ``.dplug`` plugin bundle (plugin wheel + all deps).

A ``.dplug`` is a zip of a ``datus-plugin.json`` manifest plus a ``wheels/``
wheelhouse — the plugin's own wheel and every transitive dependency wheel — so
it can be installed on an air-gapped machine with no PyPI access (see
:func:`datus.cli.plugin_service.install_bundle`).

Packing is the *build-time* half and it **needs network** (to resolve and
download dependencies from an index). Installing the resulting bundle does not.
This module is kept separate from :mod:`datus.cli.plugin_service` so the
offline install path stays import-light and the network-touching build path is
independently testable (mock ``subprocess.run``).

The build has three steps:

1. **Resolve the main wheel** from the source — build it from a project
   directory (``uv build`` / ``python -m build``), copy an existing ``.whl``,
   or download it from PyPI (``pip download --no-deps``).
2. **Introspect + gather** — read the wheel's ``datus.plugins`` entry point and
   ``Requires-Python`` (failing early if it is not a datus plugin), then
   ``pip download`` its dependencies into the wheelhouse.
3. **Write the bundle** — checksum every wheel, emit the manifest, and zip.
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

from datus.cli.plugin_service import (
    BUNDLE_EXT,
    BUNDLE_FORMAT,
    BUNDLE_FORMAT_VERSION,
    BUNDLE_MANIFEST_NAME,
    BUNDLE_WHEELS_DIR,
    _sha256_file,
)
from datus.plugins.registry import PLUGIN_ENTRY_POINT_GROUP
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class PackResult:
    ok: bool
    bundle_path: str = ""
    plugin_name: str = ""
    wheel_count: int = 0
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
    logger.info("pack: %s", " ".join(cmd))
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


def _build_wheel_from_dir(src: Path, out_dir: Path) -> Path:
    """Build a wheel from a project directory, preferring ``uv build``."""
    uv_path = shutil.which("uv")
    if uv_path:
        cmd = [uv_path, "build", "--wheel", "--out-dir", str(out_dir), str(src)]
    else:
        cmd = [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir), str(src)]
    _run(cmd, "build wheel")
    wheels = sorted(out_dir.glob("*.whl"))
    if not wheels:
        raise PackError(f"no wheel produced from {src}")
    return wheels[-1]


def _download_named_wheel(requirement: str, out_dir: Path) -> Path:
    """Download just the plugin's own wheel (no deps) for a PyPI requirement."""
    cmd = [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "-d", str(out_dir), requirement]
    _run(cmd, "download plugin wheel")
    wheels = sorted(out_dir.glob("*.whl"))
    if not wheels:
        raise PackError(f"no wheel downloaded for requirement {requirement!r}")
    return wheels[-1]


def _resolve_main_wheel(source: str, work_dir: Path) -> Path:
    """Produce the plugin's own wheel in ``work_dir`` from any supported source."""
    candidate = Path(source).expanduser()
    if source.lower().endswith(".whl") and candidate.is_file():
        dest = work_dir / candidate.name
        shutil.copy2(candidate, dest)
        return dest
    if candidate.is_dir():
        return _build_wheel_from_dir(candidate, work_dir)
    return _download_named_wheel(source, work_dir)


def _download_dependencies(
    main_wheel: Path,
    wheels_dir: Path,
    python_version: Optional[str] = None,
    platform_tag: Optional[str] = None,
) -> None:
    """Download the plugin wheel's transitive dependencies into ``wheels_dir``.

    ``--only-binary=:all:`` keeps the wheelhouse wheel-only (no sdists that
    would need building at install time). ``python_version`` / ``platform_tag``
    cross-target another interpreter/OS when packing on a different machine.
    """
    cmd = [sys.executable, "-m", "pip", "download", "--only-binary=:all:", "-d", str(wheels_dir)]
    if python_version:
        cmd += ["--python-version", python_version]
    if platform_tag:
        cmd += ["--platform", platform_tag]
    cmd.append(str(main_wheel))
    _run(cmd, "download dependencies")


def _read_plugin_entry(wheel_path: Path) -> Tuple[str, str]:
    """Read the ``datus.plugins`` entry point ``(name, target)`` from a wheel.

    Raises :class:`PackError` when the wheel declares no entry points or none in
    the ``datus.plugins`` group — i.e. it is not a datus plugin, caught before
    any dependency download happens.
    """
    with zipfile.ZipFile(wheel_path) as zf:
        candidates = [n for n in zf.namelist() if n.endswith(".dist-info/entry_points.txt")]
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
        return name, section[name].strip()
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
    platform_tag: Optional[str],
) -> dict:
    """Assemble the ``datus-plugin.json`` manifest from the gathered wheelhouse."""
    distribution, version, _ = _parse_wheel_filename(main_wheel.name)
    wheel_files = sorted(p for p in wheels_dir.iterdir() if p.suffix == ".whl")
    platforms = {_parse_wheel_filename(p.name)[2] for p in wheel_files}
    non_any = sorted(p for p in platforms if p != "any")
    bundle_platform = platform_tag or (non_any[0] if non_any else "any")
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
    """Zip the manifest + wheelhouse into ``<dist>-<version>-<platform>.dplug``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plugin = manifest["plugin"]
    platform_tag = manifest["compat"]["platform"] or "any"
    out_path = out_dir / f"{plugin['distribution']}-{plugin['version']}-{platform_tag}{BUNDLE_EXT}"
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(BUNDLE_MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
        for entry in manifest["wheels"]:
            zf.write(wheels_dir / entry["file"], f"{BUNDLE_WHEELS_DIR}/{entry['file']}")
    return out_path


def pack(
    source: str,
    out_dir: str = ".",
    python_version: Optional[str] = None,
    platform_tag: Optional[str] = None,
) -> PackResult:
    """Build a ``.dplug`` bundle from a plugin source. Requires network access.

    ``source`` may be a project directory, an existing ``.whl``, or a PyPI
    requirement. ``python_version`` / ``platform_tag`` are forwarded to ``pip
    download`` to cross-target another environment. Returns a :class:`PackResult`;
    never raises — build failures are captured with their subprocess output.
    """
    source = (source or "").strip()
    if not source:
        return PackResult(ok=False, error="no pack source given")
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
            _download_dependencies(main_wheel, wheels_dir, python_version, platform_tag)

            requires_python = _read_requires_python(main_wheel)
            manifest = _build_manifest(main_wheel, entry_name, entry_target, requires_python, wheels_dir, platform_tag)
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
    )


__all__ = ["PackResult", "PackError", "pack"]
