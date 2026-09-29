# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Engine binding lifecycle: lazy import, in-memory connections, file reload."""

from __future__ import annotations

import copy
import glob
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

import yaml

from datus.tools.semantic_tools.dosi.config import DosiConfig
from datus.tools.semantic_tools.dosi.dialects import normalize_dialect
from datus.tools.semantic_tools.dosi.model import (
    legacy_window_error,
    legacy_window_findings,
    load_document,
)
from datus.tools.semantic_tools.exceptions import SemanticCoreException

_INSTALL_HINT = "dosi-engine is missing; reinstall datus-agent from its lockfile"


# Per-metric fragments, not whole models. Datus excludes this directory from
# both of its own model walks.
_METRIC_FRAGMENT_DIR = "metrics"


def resolve_model_files(config: DosiConfig) -> list[str]:
    """Resolve every model file available to a Dosi runtime configuration.

    ``semantic_model_path`` remains an explicit single-file pin. Directory
    configurations return every OSI model beneath the directory, in
    deterministic order, so the runtime can build a datasource-wide metric
    catalog over one native engine per file.

    The walk is recursive because Datus discovers authored models that way:
    both its authoring inventory and its YAML-to-knowledge-base sync use
    ``rglob``. Matching only the top level would leave a model that Datus
    accepts and indexes invisible to every query.

    ``metrics`` is skipped for the same reason those two skip it: it holds
    per-metric fragments rather than whole models, and loading a fragment as a
    document fails.
    """
    if config.semantic_model_path:
        return [config.semantic_model_path]
    models_dir = config.semantic_models_path
    if models_dir:
        candidates = sorted(
            path
            for ext in ("*.yaml", "*.yml", "*.json")
            for path in glob.glob(os.path.join(models_dir, "**", ext), recursive=True)
            if _METRIC_FRAGMENT_DIR not in os.path.relpath(path, models_dir).split(os.sep)
        )
        if candidates:
            return candidates
        raise SemanticCoreException(f"no OSI model file (*.yaml/*.yml/*.json) in {models_dir!r}")
    raise SemanticCoreException(
        "Dosi requires semantic_model_path (an OSI model file) or semantic_models_path (a directory containing one)"
    )


def resolve_model_file(config: DosiConfig) -> str:
    """The OSI model file to load: explicit semantic_model_path, else the sole
    model file in semantic_models_path (the Datus directory convention).

    Raises SemanticCoreException when nothing resolves, or when a directory
    holds several models (the engine loads exactly one).
    """
    candidates = resolve_model_files(config)
    if len(candidates) == 1:
        return candidates[0]
    models_dir = config.semantic_models_path
    if models_dir:
        raise SemanticCoreException(
            f"{len(candidates)} model files in {models_dir!r}; set semantic_model_path to select one"
        )
    raise AssertionError("multiple models cannot resolve from semantic_model_path")


def load_binding() -> Any:
    """Import the mandatory dosi-engine binding with a repair hint."""
    try:
        import dosi_engine
    except ImportError as exc:  # pragma: no cover - exercised via fake absence
        raise SemanticCoreException(_INSTALL_HINT) from exc
    return dosi_engine


def datus_extension_version() -> str:
    """Return the DATUS extension version implemented by the active engine."""
    binding = load_binding()
    capabilities = getattr(binding, "DATUS_EXT", None)
    version = capabilities.get("version") if isinstance(capabilities, dict) else None
    if version is None or not str(version).strip():
        raise SemanticCoreException(
            "the installed dosi-engine does not expose DATUS_EXT.version; "
            "install a current dosi-engine build before authoring versioned "
            "DATUS extensions"
        )
    return str(version).strip()


def datus_authoring_contract() -> dict[str, Any]:
    """Return a defensive copy of the active engine's authoring contract."""

    contract = getattr(load_binding(), "DATUS_AUTHORING_CONTRACT", None)
    if not isinstance(contract, dict):
        raise SemanticCoreException(
            "the installed dosi-engine does not expose DATUS_AUTHORING_CONTRACT; install a current dosi-engine build"
        )

    contract_version = str(contract.get("extension_version") or "").strip()
    engine_version = datus_extension_version()
    if contract_version != engine_version:
        raise SemanticCoreException(
            "the installed dosi-engine exposes inconsistent DATUS authoring "
            f"metadata: contract version {contract_version!r}, engine version "
            f"{engine_version!r}"
        )
    contract_osi_version = str(contract.get("osi_spec_version") or "").strip()
    engine_osi_version = str(getattr(load_binding(), "SPEC_VERSION", "") or "").strip()
    if contract_osi_version != engine_osi_version:
        raise SemanticCoreException(
            "the installed dosi-engine exposes inconsistent OSI authoring "
            f"metadata: contract version {contract_osi_version!r}, engine version "
            f"{engine_osi_version!r}"
        )
    return copy.deepcopy(contract)


def datus_authoring_contract_digest() -> str:
    """Return the stable digest for the active engine's authoring contract."""

    digest = str(getattr(load_binding(), "DATUS_AUTHORING_CONTRACT_DIGEST", "") or "").strip()
    if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
        raise SemanticCoreException(
            "the installed dosi-engine does not expose a valid "
            "DATUS_AUTHORING_CONTRACT_DIGEST; install a current dosi-engine build"
        )
    return digest


class EngineHandle:
    """One engine per model file, rebuilt when its file inputs change.

    Every access re-stats ``semantic_model_path`` (one os.stat, negligible
    next to the call it guards) so edits to the OSI YAML are picked up
    without restarting the process. A SQLite datasource adds the source file
    to the same signature: a schema change there must reopen the native
    storage connection (or rebuild the companion for an older engine).
    """

    def __init__(self, config: DosiConfig):
        self._config = config
        self._lock = threading.Lock()
        self._engine: Optional[Any] = None
        self._model_signature: Optional[tuple] = None

    def _sqlite_source(self) -> Optional[str]:
        """The SQLite file backing this handle's db_config, if any."""
        db_config = self._config.db_config or {}
        if str(db_config.get("type") or "").strip().lower() != "sqlite":
            return None
        source = db_config.get("uri") or db_config.get("path")
        if not source:
            return None
        from datus.tools.semantic_tools.dosi.sqlite_bridge import normalize_sqlite_source

        return normalize_sqlite_source(str(source))

    def _sqlite_signature(self) -> Optional[float]:
        source = self._sqlite_source()
        if source is None:
            return None
        from datus.tools.semantic_tools.dosi.sqlite_bridge import sqlite_source_mtime

        try:
            return sqlite_source_mtime(source)
        except OSError:
            # Let _runtime_connections raise the actionable error.
            return None

    @property
    def profile_name(self) -> Optional[str]:
        """The connection profile to execute on, when one is configured.

        ``connection`` is an explicit profile selection. Otherwise the inline
        datasource name is used, falling back to ``default``.
        """
        config = self._config
        if config.connection:
            return config.connection
        if config.db_config:
            return config.datasource or "default"
        return None

    def model_file(self) -> str:
        """The resolved OSI model file path (raises if unresolvable)."""
        return resolve_model_file(self._config)

    def get(self) -> Any:
        config = self._config
        model_file = resolve_model_file(config)
        try:
            stat = os.stat(model_file)
        except OSError as exc:
            raise SemanticCoreException(f"cannot read semantic model {model_file!r}: {exc}") from exc
        with self._lock:
            signature = (stat.st_mtime_ns, stat.st_size, self._sqlite_signature())
            if self._engine is None or signature != self._model_signature:
                # A changed SQLite source must reopen the engine connection.
                self._engine = self._build(config, model_file)
                self._model_signature = signature
            return self._engine

    def _build(self, config: DosiConfig, model_file: str) -> Any:
        binding = load_binding()
        try:
            findings = legacy_window_findings(load_document(model_file))
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            # The native loader owns syntax/schema diagnostics. This preflight
            # exists only to catch syntactically valid legacy hints that Dosi
            # would otherwise treat as inert metadata.
            findings = []
        if findings:
            raise SemanticCoreException(legacy_window_error(findings))
        try:
            return binding.Engine(
                model_path=model_file,
                connections=self._runtime_connections(config),
                pool_size=config.pool_size,
            )
        except binding.OsiError as exc:
            raise SemanticCoreException(f"Dosi failed to load model {model_file!r}: {exc}") from exc

    def _runtime_connections(self, config: DosiConfig) -> Optional[dict[str, dict[str, Any]]]:
        """Normalize ``db_config`` into the engine's in-memory mapping.

        The engine's connections vocabulary IS the agent.yml datasource
        vocabulary, so fields pass through verbatim. Datasource aliases
        are normalized where the native executor uses a narrower vocabulary;
        the `type` alias is normalized (the engine derives the dialect from it)
        and the entry is marked default so connection-less execution lands on
        it.
        """
        if not config.db_config:
            return None
        entry = dict(config.db_config or {})
        db_type = str(entry.get("type") or "").strip().lower()
        if db_type == "oracle":
            # datus-oracle calls the PDB/service target `service_name` and
            # accepts `database` as a compatibility alias. The native Dosi
            # executor uses the shared Connection.database field exclusively,
            # so preserve the Datus spelling and materialize its native alias.
            # Match datus-oracle precedence when both aliases are present:
            # the recommended service_name wins over database.
            service_name = entry.get("service_name")
            if service_name:
                entry["database"] = service_name
        if db_type == "sqlite":
            source = entry.get("uri") or entry.get("path")
            if not source:
                raise SemanticCoreException(
                    "SQLite datasource config has no `uri`/`path`; cannot "
                    "configure the engine's SQLite storage connector"
                )
            # `path` is a Datus compatibility alias; the native connections
            # vocabulary uses `uri`. URI spelling itself passes through so the
            # engine owns the file-path contract.
            entry["uri"] = source
            entry.pop("path", None)
        dialect = normalize_dialect(entry.get("type"))
        if dialect:
            entry["type"] = dialect
        entry.setdefault("default", True)
        return {self.profile_name or "default": entry}


class EngineRegistry:
    """Directory-aware lifecycle for one native Dosi engine per model file."""

    def __init__(self, config: DosiConfig):
        self._config = config
        self._lock = threading.Lock()
        self._signature: tuple[tuple[str, int, int], ...] = ()
        self._handles: dict[str, EngineHandle] = {}

    @staticmethod
    def _file_signature(path: str) -> Optional[tuple[str, int, int]]:
        try:
            stat = Path(path).stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SemanticCoreException(f"cannot read semantic model {path!r}: {exc}") from exc
        return path, stat.st_mtime_ns, stat.st_size

    def snapshot(
        self,
    ) -> tuple[tuple[tuple[str, int, int], ...], tuple[tuple[str, EngineHandle], ...]]:
        """Return a stable file signature and matching engine handles."""
        with self._lock:
            entries = [(path, self._file_signature(path)) for path in resolve_model_files(self._config)]
            files = [path for path, entry_signature in entries if entry_signature is not None]
            if not files:
                raise SemanticCoreException("no readable OSI model file resolved for this datasource")
            signature = tuple(entry_signature for _, entry_signature in entries if entry_signature is not None)
            if signature != self._signature:
                previous = self._handles
                self._handles = {
                    path: previous.get(path)
                    or EngineHandle(self._config.model_copy(update={"semantic_model_path": path}))
                    for path in files
                }
                self._signature = signature
            return self._signature, tuple((path, self._handles[path]) for path in files)
