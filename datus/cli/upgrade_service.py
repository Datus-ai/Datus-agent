# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Self-upgrade helpers for the ``datus upgrade`` subcommand.

Two concerns live here, both pure Python (no prompt_toolkit / Rich import)
so they can be unit-tested by monkey-patching ``subprocess.run``,
``shutil.which``, ``httpx.Client`` and ``importlib.metadata.distributions``:

1. **Package discovery + upgrade.** Enumerate every ``datus-*``
   distribution installed in the active interpreter
   (:func:`enumerate_datus_packages`) and run a single ``uv pip install
   --upgrade`` (or ``pip install --upgrade``) over the non-editable ones
   (:func:`upgrade_packages`). Editable / source-tree installs are
   detected via ``direct_url.json`` (mirrors
   ``ci/nightly_manifest.package_info``) and skipped — pip-upgrading over
   a checkout would be wrong.

2. **Latest-version check.** Query the public PyPI JSON API for the
   latest ``datus-agent`` release (:func:`get_latest_version`), cached on
   disk for 24h under ``{datus_home}/cache/version_check.json``. The REPL
   banner uses :func:`newer_version_available` to decide whether to show
   a one-line upgrade hint. Every network / disk failure degrades to
   ``None`` so a check never blocks or breaks startup.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

DISTRIBUTION_NAME = "datus-agent"
_PYPI_URL = "https://pypi.org/pypi/{name}/json"
_CACHE_TTL_SECONDS = 24 * 60 * 60
_HTTP_TIMEOUT = 3.0  # short so the background check never lingers


@dataclass
class DatusPackage:
    """One installed ``datus-*`` distribution."""

    name: str  # canonical (PEP 503) dist name, e.g. "datus-agent"
    version: str  # currently installed version
    editable: bool = False  # editable / source-tree install → skip pip upgrade


@dataclass
class UpgradeResult:
    """Outcome of :func:`upgrade_packages`.

    ``ok`` is the field callers branch on; ``stdout`` / ``stderr`` are
    captured so the CLI can render a tail of the failed pip/uv run.
    """

    ok: bool
    packages: List[str] = field(default_factory=list)
    skipped_editable: List[str] = field(default_factory=list)
    # (name, old_version, new_version) for packages whose version actually
    # changed after the upgrade. Empty when everything was already current.
    changes: List[Tuple[str, str, str]] = field(default_factory=list)
    label: str = ""
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


def _normalize(name: str) -> str:
    """PEP 503 normalization for comparison (``datus_foo`` → ``datus-foo``)."""
    return name.lower().replace("_", "-")


def _is_editable(dist: importlib_metadata.Distribution) -> bool:
    """True for editable / local source (directory) installs.

    Per PEP 610, a ``direct_url.json`` describes a directory install via
    ``dir_info`` and a local wheel/archive install via ``archive_info``.
    A distribution is treated as editable/source when it was installed from
    a directory (``dir_info`` present), either explicitly editable
    (``dir_info.editable`` true) or from a local ``file://`` path. Local
    wheels/archives (``archive_info``) and missing / malformed metadata are
    treated as normal (non-editable) installs that pip/uv can upgrade.
    """
    try:
        text = dist.read_text("direct_url.json")
    except Exception:  # pragma: no cover - defensive, metadata read can raise
        return False
    if not text:
        return False
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    dir_info = data.get("dir_info")
    if not isinstance(dir_info, dict):
        # Local wheels/archives carry ``archive_info`` (no ``dir_info``); pip/uv
        # can upgrade those, so they are not editable/source installs.
        return False
    if dir_info.get("editable"):
        return True
    return str(data.get("url", "")).startswith("file://")


def enumerate_datus_packages() -> List[DatusPackage]:
    """Return every installed ``datus-*`` distribution, sorted by name.

    ``datus-agent`` is always present in the result: when it is not an
    installed distribution (running straight from a source checkout) we
    synthesize an entry from :data:`datus.__version__` and flag it
    ``editable`` so the upgrade path never pip-installs over the tree.
    """
    found: dict[str, DatusPackage] = {}
    for dist in importlib_metadata.distributions():
        try:
            raw = dist.metadata.get("Name") or ""
        except Exception:  # pragma: no cover - corrupt metadata
            continue
        if not raw:
            continue
        norm = _normalize(raw)
        if not norm.startswith("datus-"):
            continue
        # First writer wins; duplicate dist dirs for the same name are rare
        # but possible, and either entry is acceptable for our purposes.
        if norm not in found:
            found[norm] = DatusPackage(name=norm, version=dist.version, editable=_is_editable(dist))

    if DISTRIBUTION_NAME not in found:
        from datus import __version__

        found[DISTRIBUTION_NAME] = DatusPackage(name=DISTRIBUTION_NAME, version=__version__, editable=True)

    return sorted(found.values(), key=lambda p: p.name)


def _upgrade_command(pkgs: List[str]) -> Tuple[List[str], str]:
    """Build the upgrade command, preferring ``uv pip`` when available.

    Mirrors ``service_adapter_installer._install_command`` but adds
    ``--upgrade`` and accepts multiple packages. ``uv tool install`` /
    bare ``uv venv`` environments do not seed ``pip``, so routing through
    ``uv pip install --python <sys.executable>`` reuses the active
    interpreter without requiring ``pip`` to be present. Returns
    ``(argv, label)`` where ``label`` names the underlying tool for error
    messages.
    """
    uv_path = shutil.which("uv")
    if uv_path:
        return [uv_path, "pip", "install", "--upgrade", "--python", sys.executable, *pkgs], "uv pip install --upgrade"
    return [sys.executable, "-m", "pip", "install", "--upgrade", *pkgs], "pip install --upgrade"


def upgrade_packages(packages: List[DatusPackage]) -> UpgradeResult:
    """Upgrade every non-editable package in one ``uv``/``pip`` invocation.

    Editable / source-tree installs are skipped (recorded in
    ``skipped_editable``). When nothing is upgradable the function returns
    ``ok=True`` with an explanatory ``error`` and never shells out.
    """
    upgradable = [p.name for p in packages if not p.editable]
    skipped = [p.name for p in packages if p.editable]
    if not upgradable:
        return UpgradeResult(
            ok=True,
            packages=[],
            skipped_editable=skipped,
            error="No upgradable packages (all editable / source installs).",
        )

    old_versions = {p.name: p.version for p in packages if not p.editable}
    cmd, label = _upgrade_command(upgradable)
    logger.info("Upgrading datus packages: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:  # uv / pip missing, OSError, etc.
        return UpgradeResult(ok=False, packages=upgradable, skipped_editable=skipped, label=label, error=str(exc))

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        return UpgradeResult(
            ok=False,
            packages=upgradable,
            skipped_editable=skipped,
            label=label,
            stdout=stdout,
            stderr=stderr,
            error=f"{label} exited with code {proc.returncode}",
        )
    return UpgradeResult(
        ok=True,
        packages=upgradable,
        skipped_editable=skipped,
        changes=_collect_changes(old_versions, upgradable),
        label=label,
        stdout=stdout,
        stderr=stderr,
    )


def _collect_changes(old_versions: dict, names: List[str]) -> List[Tuple[str, str, str]]:
    """Re-read installed versions after an upgrade and return the diffs.

    ``importlib.invalidate_caches()`` makes the freshly-installed
    ``dist-info`` directories visible to a re-scan in the same process
    (mirrors ``service_adapter_installer.ensure_adapter``). Only packages
    whose version actually changed are returned; a package that was
    already current is omitted.
    """
    importlib.invalidate_caches()
    after = {p.name: p.version for p in enumerate_datus_packages()}
    changes: List[Tuple[str, str, str]] = []
    for name in names:
        old = old_versions.get(name, "")
        new = after.get(name, old)
        if new != old:
            changes.append((name, old, new))
    return changes


# ── latest-version check (public PyPI JSON + 24h on-disk cache) ────────


def _cache_path(agent_config=None) -> Path:
    """Resolve ``{datus_home}/cache/version_check.json``."""
    from datus.utils.path_manager import get_path_manager

    return get_path_manager(agent_config=agent_config).datus_home / "cache" / "version_check.json"


def _read_cache(path: Path) -> Optional[dict]:
    """Return the cached payload if present and younger than the TTL."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    checked_at = data.get("checked_at", 0)
    if not isinstance(checked_at, (int, float)):
        return None
    if (time.time() - checked_at) > _CACHE_TTL_SECONDS:
        return None
    return data


def _write_cache(path: Path, latest: Optional[str]) -> None:
    """Persist the check result (best-effort; failures are swallowed).

    Negative results (``latest=None``) are cached too so an offline run
    does not re-hit the network on every launch within the TTL window.
    """
    payload = {"checked_at": time.time(), "latest": latest, "name": DISTRIBUTION_NAME}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def _fetch_latest_from_pypi() -> Optional[str]:
    """Fetch the latest ``datus-agent`` version from PyPI, or ``None``."""
    import httpx

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.get(_PYPI_URL.format(name=DISTRIBUTION_NAME))
            resp.raise_for_status()
            version = resp.json().get("info", {}).get("version")
    except Exception as exc:  # offline, timeout, HTTP error, bad JSON
        logger.debug("PyPI version check failed: %s", exc)
        return None
    return version if isinstance(version, str) and version else None


def get_latest_version(*, agent_config=None, force: bool = False) -> Optional[str]:
    """Latest ``datus-agent`` version, using a fresh cache when available.

    Returns ``None`` on any failure / offline state. Always refreshes the
    on-disk cache when it fetches (including negative results).
    """
    path = _cache_path(agent_config)
    if not force:
        cached = _read_cache(path)
        if cached is not None:
            return cached.get("latest")
    latest = _fetch_latest_from_pypi()
    _write_cache(path, latest)
    return latest


def newer_version_available(current: str, *, agent_config=None, cached_only: bool = False) -> Optional[str]:
    """Return the latest version if it is strictly newer than ``current``.

    ``cached_only=True`` never touches the network — it consults only a
    fresh on-disk cache (used at REPL startup so a cold cache shows no
    hint and the check never blocks). Returns ``None`` when there is no
    newer version, no cached/fetched answer, or either version string is
    unparseable.
    """
    from packaging.version import InvalidVersion, Version

    if cached_only:
        cached = _read_cache(_cache_path(agent_config))
        latest = cached.get("latest") if cached else None
    else:
        latest = get_latest_version(agent_config=agent_config)

    if not latest:
        return None
    try:
        return latest if Version(latest) > Version(current) else None
    except InvalidVersion:
        return None


__all__ = [
    "DatusPackage",
    "UpgradeResult",
    "enumerate_datus_packages",
    "upgrade_packages",
    "get_latest_version",
    "newer_version_available",
]
