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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from datus.tools.semantic_tools.dosi.config import DosiConfig
from datus.tools.semantic_tools.dosi.dialects import normalize_dialect
from datus.tools.semantic_tools.dosi.model import (
    document_model_names,
    legacy_window_error,
    legacy_window_findings,
    load_document,
)
from datus.tools.semantic_tools.exceptions import SemanticCoreException
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_INSTALL_HINT = "dosi-engine is missing; reinstall datus-agent from its lockfile"


# Per-metric fragments, not whole models. Datus excludes this directory from
# both of its own model walks.
_METRIC_FRAGMENT_DIR = "metrics"

# ``subject/semantic_models/ontology/<datasource>/`` holds Apache Ossie
# ontologies, so no datasource may own the ``ontology`` model directory.
ONTOLOGY_DIR_NAME = "ontology"

_MODEL_SUFFIXES = ("*.yaml", "*.yml", "*.json")


def _normalize_sqlite_source(source: str) -> str:
    """Resolve Datus SQLite URI spellings to the underlying file."""
    text = str(source).strip()
    if text.lower().startswith("sqlite:"):
        text = text[len("sqlite:") :]
        while text.startswith("//"):
            text = text[1:]
    return os.path.abspath(os.path.expanduser(text))


def _sqlite_source_mtime(sqlite_path: str) -> float:
    """Include SQLite WAL and SHM writes when checking connection freshness."""
    newest = os.path.getmtime(sqlite_path)
    for suffix in ("-wal", "-shm"):
        sidecar = sqlite_path + suffix
        if os.path.exists(sidecar):
            newest = max(newest, os.path.getmtime(sidecar))
    return newest


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
        if Path(models_dir).name == ONTOLOGY_DIR_NAME:
            raise SemanticCoreException(
                f"{models_dir!r} is reserved for ontology files "
                "(subject/semantic_models/ontology/<datasource>/); rename the datasource"
            )
        candidates = sorted(
            path
            for ext in _MODEL_SUFFIXES
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


def resolve_ontology_files(config: DosiConfig) -> list[str]:
    """Every ontology file under ``config.ontology_dir``, in a stable order.

    A missing directory means the datasource has no ontology, which is not an
    error here: only the ontology-aware agent turns that into one.
    """
    ontology_dir = config.ontology_dir
    if not ontology_dir or not os.path.isdir(ontology_dir):
        return []
    return sorted(
        path for ext in _MODEL_SUFFIXES for path in glob.glob(os.path.join(ontology_dir, "**", ext), recursive=True)
    )


def _load_quietly(path: str) -> Optional[dict[str, Any]]:
    """A YAML/JSON document as a mapping, or None when it is not one."""
    try:
        document = load_document(path)
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return None
    return document


def document_kind(document: Optional[dict[str, Any]]) -> str:
    """``mapping``, ``ontology``, ``model`` or ``other``, by structure.

    The same rule the engine applies (apache/ossie#458): a mapping has all of
    ``concept_mappings``, ``ontology_ref`` and ``semantic_model_ref``; an
    ontology has an ``ontology:`` list; a model is flat (``name`` +
    ``datasets``) or the ``semantic_model:`` wrapper. Never the file name.
    """
    if not isinstance(document, dict):
        return "other"
    if all(key in document for key in ("concept_mappings", "ontology_ref", "semantic_model_ref")):
        return "mapping"
    if isinstance(document.get("ontology"), list):
        return "ontology"
    if "semantic_model" in document or ("name" in document and "datasets" in document):
        return "model"
    return "other"


def _ref(document: dict[str, Any], key: str) -> dict[str, Any]:
    ref = document.get(key)
    return ref if isinstance(ref, dict) else {}


@dataclass(frozen=True)
class OntologyBinding:
    """How one engine loads its ontology.

    ``mapping_path`` is a mapping document (apache/ossie#458). Without one,
    ``ontology_path`` is the deprecated layout: an ontology embedding its
    mapping and model. ``model_file`` is None when the mapping carries its
    semantic model inline, so the mapping alone is the engine's input.
    ``references`` are the local files the mapping reaches by relative
    ``iri``; the engine reads them itself, so they belong to its inputs.
    """

    model_file: Optional[str]
    ontology_path: Optional[str] = None
    mapping_path: Optional[str] = None
    references: tuple[str, ...] = ()


def plan_ontologies(model_files: list[str], ontology_files: list[str]) -> tuple[dict[str, OntologyBinding], list[str]]:
    """Bind every model file, and every self-contained mapping, to its ontology.

    Returns ``(engine key -> binding, unpaired files)``. The key is the model
    file, or the mapping file for a mapping whose model is inline.

    - A mapping binds the model file whose semantic model carries the name
      its ``semantic_model_ref`` names; an inline model makes it its own
      engine. Its ontology is resolved by the engine (inline or ``iri``),
      else here by name among the ontology files.
    - Files a mapping uses (its ontology, and a model it reaches by ``iri``)
      are consumed. The remaining ontology documents pair the old way, by file
      stem (:func:`pair_ontologies`), as the deprecated layout; any other
      leftover file is reported, never paired as an ontology.
    - Anything ambiguous — two mappings onto one model, a name nothing
      carries — is reported unpaired rather than guessed.
    """
    documents = {path: _load_quietly(path) for path in ontology_files}
    kinds = {path: document_kind(doc) for path, doc in documents.items()}
    ontologies_by_name: dict[str, list[str]] = {}
    for path, kind in kinds.items():
        if kind == "ontology":
            ontologies_by_name.setdefault(str(documents[path].get("name") or ""), []).append(path)
    models_by_name: dict[str, list[str]] = {}
    for model in model_files:
        for name in document_model_names(_load_quietly(model) or {}):
            models_by_name.setdefault(name, []).append(model)

    bindings: dict[str, OntologyBinding] = {}
    unpaired: list[str] = []
    consumed: set[str] = set()
    claims: dict[str, list[str]] = {}
    for path in sorted(p for p, k in kinds.items() if k == "mapping"):
        document = documents[path]
        ontology_ref = _ref(document, "ontology_ref")
        model_ref = _ref(document, "semantic_model_ref")
        ontology_path = None
        references = tuple(
            os.path.normpath(os.path.join(os.path.dirname(path), iri))
            for iri in (ontology_ref.get("iri"), model_ref.get("iri"))
            if isinstance(iri, str) and "://" not in iri
        )
        # The engine follows the iri itself; the file it lands on is this
        # mapping's, not one to pair again by stem.
        consumed.update(references)
        if "document" not in ontology_ref and "iri" not in ontology_ref:
            named = ontologies_by_name.get(str(ontology_ref.get("name") or ""), [])
            if len(named) != 1:
                unpaired.append(path)
                continue
            ontology_path = named[0]
        if "document" in model_ref:
            model_file = None
        else:
            matches = models_by_name.get(str(model_ref.get("name") or ""), [])
            if len(matches) == 1:
                model_file = matches[0]
            elif not matches and "iri" in model_ref:
                model_file = None  # the engine follows the iri itself
            else:
                unpaired.append(path)
                continue
        key = model_file or path
        claims.setdefault(key, []).append(path)
        bindings[key] = OntologyBinding(
            model_file=model_file, ontology_path=ontology_path, mapping_path=path, references=references
        )
        if ontology_path:
            consumed.add(os.path.normpath(ontology_path))
    for key, mappings in claims.items():
        if len(mappings) > 1:
            # One engine loads one mapping document; never pick one silently.
            del bindings[key]
            unpaired.extend(mappings)

    leftover = [p for p, k in kinds.items() if k != "mapping" and os.path.normpath(p) not in consumed]
    legacy = [p for p in leftover if kinds[p] == "ontology"]
    unpaired.extend(p for p in leftover if kinds[p] != "ontology")
    pairs, legacy_unpaired = pair_ontologies([m for m in model_files if m not in bindings], legacy)
    for model, ontology in pairs.items():
        bindings[model] = OntologyBinding(model_file=model, ontology_path=ontology)
    return bindings, sorted(unpaired + legacy_unpaired)


def pair_ontologies(model_files: list[str], ontology_files: list[str]) -> tuple[dict[str, str], list[str]]:
    """Pair each model file with the ontology that shares its file stem.

    One model and one ontology pair regardless of names, which covers the
    common single-model datasource. Returns ``(model -> ontology, unpaired
    ontologies)``; two ontologies claiming one model are both left unpaired
    rather than picking one silently.
    """
    if len(model_files) == 1 and len(ontology_files) == 1:
        return {model_files[0]: ontology_files[0]}, []
    by_stem: dict[str, list[str]] = {}
    for ontology in ontology_files:
        by_stem.setdefault(Path(ontology).stem, []).append(ontology)
    pairs: dict[str, str] = {}
    paired: set[str] = set()
    for model in model_files:
        matches = by_stem.get(Path(model).stem, [])
        if len(matches) == 1:
            pairs[model] = matches[0]
            paired.add(matches[0])
    return pairs, [ontology for ontology in ontology_files if ontology not in paired]


def binding_supports_ontology(binding: Any = None) -> bool:
    """Whether the installed binding answers ontology questions natively.

    Loading an ontology predates this; ``describe`` and ``paths`` on the
    engine are what the ontology agent needs, and arrive together.
    """
    try:
        engine_cls = getattr(binding or load_binding(), "Engine", None)
    except SemanticCoreException:
        return False
    return callable(getattr(engine_cls, "describe", None)) and callable(getattr(engine_cls, "paths", None))


def binding_supports_mapping_documents(binding: Any = None) -> bool:
    """Whether the installed binding loads Ossie mapping documents
    (apache/ossie#458): its ``Engine`` takes ``mapping_path``."""
    try:
        engine_cls = getattr(binding or load_binding(), "Engine", None)
    except SemanticCoreException:
        return False
    signature = getattr(engine_cls, "__text_signature__", None) or ""
    return "mapping_path" in signature


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
        return _normalize_sqlite_source(str(source))

    def _ontology_signature(self) -> Optional[tuple[int, ...]]:
        """An edited ontology, mapping or file it references rebuilds the engine, like an edited model."""
        signature: list[int] = []
        for path in (self._config.ontology_path, self._config.mapping_path):
            if not path:
                continue
            try:
                stat = os.stat(path)
            except OSError as exc:
                raise SemanticCoreException(f"cannot read ontology {path!r}: {exc}") from exc
            signature += [stat.st_mtime_ns, stat.st_size]
        for path in self._config.mapping_references:
            # An iri the engine never follows (the model was found by name)
            # may name no file; its appearing still rebuilds the engine.
            try:
                stat = os.stat(path)
            except OSError:
                signature += [-1, -1]
            else:
                signature += [stat.st_mtime_ns, stat.st_size]
        return tuple(signature) or None

    @property
    def ontology_path(self) -> Optional[str]:
        """The ontology file this engine loads, if any (embedded layout, or one a mapping names)."""
        return self._config.ontology_path

    @property
    def mapping_path(self) -> Optional[str]:
        """The Ossie mapping document this engine loads, if any."""
        return self._config.mapping_path

    @property
    def inputs(self) -> tuple[Optional[str], Optional[str], list[str]]:
        """The ontology-side files this engine was planned with; a change re-creates the handle."""
        return self._config.ontology_path, self._config.mapping_path, self._config.mapping_references

    @property
    def has_ontology(self) -> bool:
        """Whether this engine loads an ontology, by mapping or the old way."""
        return bool(self._config.ontology_path or self._config.mapping_path)

    @property
    def ontology_source(self) -> Optional[str]:
        """The file an ontology error should name: the mapping, else the ontology."""
        return self._config.mapping_path or self._config.ontology_path

    def _sqlite_signature(self) -> Optional[float]:
        source = self._sqlite_source()
        if source is None:
            return None
        try:
            return _sqlite_source_mtime(source)
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
            signature = (stat.st_mtime_ns, stat.st_size, self._sqlite_signature(), self._ontology_signature())
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
            ontology_kwargs: dict[str, Any] = {}
            if config.ontology_path:
                ontology_kwargs["ontology_path"] = config.ontology_path
            if config.mapping_path:
                ontology_kwargs["mapping_path"] = config.mapping_path
            # A mapping that carries its model inline is the engine's only
            # input; naming it as the model too would give the model twice.
            model_kwargs = {} if model_file == config.mapping_path else {"model_path": model_file}
            if not ontology_kwargs:
                return binding.Engine(
                    model_path=model_file,
                    connections=self._runtime_connections(config),
                    pool_size=config.pool_size,
                )
            # Ontology lowering reports constructs it keeps as text (``note:``)
            # as RuntimeWarnings, which reach a CLI user's terminal on every
            # load. The same notes are on ``Engine.warnings``; log them instead.
            with warnings.catch_warnings(record=True) as lowering_notes:
                warnings.simplefilter("always", RuntimeWarning)
                engine = binding.Engine(
                    connections=self._runtime_connections(config),
                    pool_size=config.pool_size,
                    **model_kwargs,
                    **ontology_kwargs,
                )
            for note in lowering_notes:
                logger.debug("Ontology %s: %s", self.ontology_source, note.message)
            return engine
        except TypeError as exc:
            if config.mapping_path and "mapping_path" in str(exc):
                raise SemanticCoreException(
                    f"{config.mapping_path!r} is an Ossie mapping document, and the installed "
                    "dosi-engine cannot load one; upgrade dosi-engine"
                ) from exc
            raise
        except binding.OsiError as exc:
            if self.has_ontology:
                raise SemanticCoreException(
                    f"Dosi failed to load model {model_file!r} with ontology {self.ontology_source!r}: "
                    f"[{getattr(exc, 'code', '')}] {exc}"
                ) from exc
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
        self._keys: list[str] = []
        self._unpaired_ontologies: list[str] = []

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
            ontology_files = resolve_ontology_files(self._config)
            try:
                model_paths = resolve_model_files(self._config)
            except SemanticCoreException:
                # A datasource whose only model lives inline in a mapping
                # document has no model files at all.
                if not ontology_files:
                    raise
                model_paths = []
            entries = [(path, self._file_signature(path)) for path in model_paths]
            files = [path for path, entry_signature in entries if entry_signature is not None]
            # Adding, removing or renaming an ontology re-pairs the handles;
            # content edits are caught per handle by EngineHandle.get. A
            # mapping's references live in its content, so ontology files
            # are re-planned on every content change too.
            signature = tuple(entry_signature for _, entry_signature in entries if entry_signature is not None) + tuple(
                entry for entry in (self._file_signature(path) for path in ontology_files) if entry is not None
            )
            if signature != self._signature:
                bindings, unpaired = plan_ontologies(files, ontology_files)
                self._unpaired_ontologies = unpaired
                # Model files first, in catalog order; then the engines whose
                # only input is a mapping document.
                keys = files + sorted(key for key, b in bindings.items() if b.model_file is None)
                previous = self._handles
                handles: dict[str, EngineHandle] = {}
                for key in keys:
                    bound = bindings.get(key)
                    update = {
                        "semantic_model_path": key,
                        "ontology_path": bound.ontology_path if bound else None,
                        "mapping_path": bound.mapping_path if bound else None,
                        "mapping_references": list(bound.references) if bound else [],
                    }
                    kept = previous.get(key)
                    if kept is not None and kept.inputs == (
                        update["ontology_path"],
                        update["mapping_path"],
                        update["mapping_references"],
                    ):
                        handles[key] = kept
                    else:
                        handles[key] = EngineHandle(self._config.model_copy(update=update))
                self._handles = handles
                self._keys = keys
                self._signature = signature
            if not self._keys:
                raise SemanticCoreException("no readable OSI model file resolved for this datasource")
            return self._signature, tuple((key, self._handles[key]) for key in self._keys)

    def unpaired_ontologies(self) -> list[str]:
        """Ontology files the last snapshot could not pair with a model file."""
        with self._lock:
            return list(self._unpaired_ontologies)
