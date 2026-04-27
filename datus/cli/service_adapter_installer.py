# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Adapter package auto-installer + hot-reloader for the ``/services`` TUI.

When the user picks a ``type`` for a new ``services.bi_platforms`` or
``services.schedulers`` entry, the underlying adapter package
(``datus-bi-<type>`` / ``datus-scheduler-<type>``) might not be
installed yet. The TUI calls into this module to:

1. Detect whether the adapter is already importable
   (:func:`is_adapter_installed`).
2. Run ``pip install`` for the missing package
   (:func:`ensure_adapter`) — the install runs through
   ``sys.executable -m pip`` so it targets the same interpreter that
   loaded ``datus-cli`` (works for venv / pipx / uv-installed CLIs).
3. Import the adapter module and refresh the BI / scheduler registries
   without restarting the process (:func:`hot_reload_adapter`).

Pure Python on purpose — the module has no prompt_toolkit dependency so
it can be unit-tested by monkey-patching ``subprocess.run`` and
``importlib.util.find_spec``.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# Section → (pip-prefix, import-prefix). Append the lower-cased ``type``
# field to either prefix to obtain the actual pip distribution name and
# the importable module name. The mapping mirrors the package naming
# convention documented in the data-engineering quickstart guide
# (``datus-bi-superset`` ships ``datus_bi_superset``;
# ``datus-scheduler-airflow`` ships ``datus_scheduler_airflow``).
_PKG_PREFIXES: dict[str, Tuple[str, str]] = {
    "bi_platforms": ("datus-bi-", "datus_bi_"),
    "schedulers": ("datus-scheduler-", "datus_scheduler_"),
    "semantic_layer": ("datus-semantic-", "datus_semantic_"),
}


# Entry-point group each section's adapter packages register under.
# ``datus-bi-<x>`` exposes ``datus.bi_adapters`` → ``<x> = datus_bi_<x>:register``;
# ``datus-scheduler-<x>`` exposes ``datus.schedulers`` similarly. Importing
# the module alone does NOT register the adapter — registration runs only
# inside the ``register`` callable that the entry-point points at, so
# :func:`hot_reload_adapter` resolves and calls it directly. Bypasses the
# caching that ``adapter_registry.discover_adapters()`` would otherwise
# apply on second-and-later calls.
_EP_GROUPS: dict[str, str] = {
    "bi_platforms": "datus.bi_adapters",
    "schedulers": "datus.schedulers",
    "semantic_layer": "datus.semantic_adapters",
}


@dataclass
class InstallResult:
    """Outcome of :func:`ensure_adapter`.

    ``ok`` is the only field callers must inspect; the rest is captured
    so the TUI can render a tail of pip's output and surface a precise
    failure message in the error bar.
    """

    ok: bool
    package: str
    import_name: str
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


def package_for(section: str, adapter_type: str) -> Tuple[str, str]:
    """Return ``(pip_pkg_name, import_module_name)`` for ``(section, type)``.

    Raises :class:`ValueError` for unsupported sections / blank types so
    the caller stops early instead of issuing a malformed pip command
    like ``pip install datus-bi-`` (which pip happily accepts and
    spends 30 seconds resolving before failing).
    """
    prefixes = _PKG_PREFIXES.get(section)
    if prefixes is None:
        raise ValueError(f"Unsupported service section: {section!r}")
    type_token = (adapter_type or "").strip().lower()
    if not type_token:
        raise ValueError("adapter_type must be a non-empty string")
    pkg_prefix, import_prefix = prefixes
    return f"{pkg_prefix}{type_token}", f"{import_prefix}{type_token}"


def is_adapter_installed(section: str, adapter_type: str) -> bool:
    """True when the adapter module is already importable."""
    try:
        _, import_name = package_for(section, adapter_type)
    except ValueError:
        return False
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        # ``find_spec`` raises ``ValueError`` on partially-initialized
        # parent packages and ``ImportError`` when an ancestor is
        # missing. Both cases mean "not importable".
        return False


def ensure_adapter(
    section: str,
    adapter_type: str,
    *,
    line_callback: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """Install the adapter package via ``sys.executable -m pip`` if missing.

    ``line_callback`` (when provided) receives one line of pip
    stdout/stderr at a time so the TUI can display a live tail. The
    function blocks until pip exits — the caller is expected to run it
    on a worker thread when the UI must stay responsive.
    """
    try:
        pkg, import_name = package_for(section, adapter_type)
    except ValueError as exc:
        return InstallResult(ok=False, package="", import_name="", error=str(exc))

    if is_adapter_installed(section, adapter_type):
        return InstallResult(ok=True, package=pkg, import_name=import_name)

    cmd = [sys.executable, "-m", "pip", "install", pkg]
    logger.info("Installing adapter package: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - exercised via mocked subprocess
        return InstallResult(ok=False, package=pkg, import_name=import_name, error=str(exc))

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if line_callback is not None:
        for line in stdout.splitlines():
            line_callback(line)
        for line in stderr.splitlines():
            line_callback(line)
    if proc.returncode != 0:
        return InstallResult(
            ok=False,
            package=pkg,
            import_name=import_name,
            stdout=stdout,
            stderr=stderr,
            error=f"pip install exited with code {proc.returncode}",
        )

    importlib.invalidate_caches()
    return InstallResult(
        ok=True,
        package=pkg,
        import_name=import_name,
        stdout=stdout,
        stderr=stderr,
    )


def hot_reload_adapter(section: str, adapter_type: str) -> bool:
    """Import the freshly-installed adapter module and call its registrar.

    Adapter packages declare their registration callable through an
    entry-point (``datus.bi_adapters`` / ``datus.schedulers``); importing
    the package itself does **not** register the adapter — the registrar
    is only invoked when something walks those entry points. We do that
    walk explicitly here so a pip install + this function brings the
    adapter online without a process restart, even when the registry's
    own ``discover_adapters()`` has already cached an earlier (empty)
    scan.

    Returns ``True`` when at least one ``register`` call succeeded for
    the requested ``adapter_type``. Returns ``False`` when the package
    is missing or its entry point fails to load.
    """
    try:
        _, import_name = package_for(section, adapter_type)
    except ValueError:
        return False
    # Refresh importlib's loader cache so a freshly-installed
    # distribution's metadata is visible to ``entry_points()`` below.
    importlib.invalidate_caches()
    try:
        if import_name in sys.modules:
            importlib.reload(sys.modules[import_name])
        else:
            importlib.import_module(import_name)
    except ImportError as exc:
        logger.debug("Hot reload of %s failed: %s", import_name, exc)
        return False
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Hot reload of %s raised: %s", import_name, exc)
        return False

    registered = False
    ep_group = _EP_GROUPS.get(section)
    if ep_group:
        try:
            import importlib.metadata as metadata

            entries = metadata.entry_points()
            # Python 3.10+ exposes ``select(group=..., name=...)``; pre-3.10
            # returns a ``dict``-shaped object. Fall back gracefully.
            if hasattr(entries, "select"):
                candidates = list(entries.select(group=ep_group, name=adapter_type))
            else:  # pragma: no cover - legacy Python
                candidates = [ep for ep in entries.get(ep_group, []) if ep.name == adapter_type]
            for ep in candidates:
                try:
                    register_fn = ep.load()
                except Exception as exc:
                    logger.warning("Failed to load entry point %s: %s", ep, exc)
                    continue
                if callable(register_fn):
                    try:
                        register_fn()
                        registered = True
                    except Exception as exc:
                        logger.warning("`%s` register() raised: %s", import_name, exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Entry-point lookup for %s failed: %s", import_name, exc)

    # Best-effort refresh of the BI registry's own discovery state so
    # callers that go through ``discover_adapters()`` (e.g. the listing
    # path in ``service_client``) see the new entry too.
    if section == "bi_platforms":
        try:
            from datus_bi_core import adapter_registry  # type: ignore[import-not-found]
        except ImportError:
            return registered
        discover = getattr(adapter_registry, "discover_adapters", None)
        if callable(discover):
            try:
                discover()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("adapter_registry.discover_adapters() raised: %s", exc)
    elif section == "semantic_layer":
        try:
            from datus.tools.semantic_tools.registry import semantic_adapter_registry
        except ImportError:
            return registered
        discover = getattr(semantic_adapter_registry, "discover_adapters", None)
        if callable(discover):
            try:
                discover()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("semantic_adapter_registry.discover_adapters() raised: %s", exc)

    return registered


__all__ = [
    "InstallResult",
    "ensure_adapter",
    "hot_reload_adapter",
    "is_adapter_installed",
    "package_for",
]
