# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Isolated semantic drafts and controlled publication for private authoring runtimes."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from datus.storage.semantic_model.artifact_file import (
    atomic_write_bytes,
    ontology_bundle_lock,
    semantic_artifact_lock,
)
from datus.tools.semantic_tools.dosi.config import DosiConfig
from datus.tools.semantic_tools.dosi.engine import EngineHandle, document_kind, plan_ontologies


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class SemanticDraftScope:
    """Project shadow file paths onto the caller's unchanged resource grants."""

    def __init__(self, draft: SemanticAuthoringDraft):
        self.draft = draft
        self.root = draft.root

    def allows_file(self, path: Path) -> bool:
        return self.draft.scope.allows_file(self.draft.live_path(path))

    def allows_host_read(self, path: Path) -> bool:
        return self.draft.scope.allows_host_read(path)


class SemanticAuthoringDraft:
    """Copy authorized inputs, retain exact baselines and publish one validated bundle.

    Business editing stays in the plugin. The host owns real project I/O, scope,
    version checks and publication/rollback using the same locks as Agent readers.
    """

    def __init__(self, scope: Any):
        self.scope = scope
        self.live_root = scope.root
        self._temp = tempfile.TemporaryDirectory(prefix="datus-semantic-draft-")
        self.root = Path(self._temp.name).resolve()
        self.config = copy.copy(scope.config)
        self.config.path_manager = copy.copy(scope.config.path_manager)
        self.config._project_root = self.root
        self.config.path_manager._project_root = self.root
        self.baseline: dict[str, bytes] = {}
        self.published: dict[str, bytes] = {}
        self.validation: dict[str, Any] | None = None
        self.kb_sync_pending = False
        with ontology_bundle_lock(self.live_root / "subject/ontology"):
            for path in self._live_files():
                with semantic_artifact_lock(path, load_before_lock=False):
                    name = path.relative_to(self.live_root).as_posix()
                    self.path(name)
                    self.baseline[name] = path.read_bytes()
        for name, content in self.baseline.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        self.filesystem_scope = SemanticDraftScope(self)

    def _live_files(self) -> list[Path]:
        subject = self.live_root / "subject"
        return (
            sorted(
                path
                for path in subject.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".yaml", ".yml", ".json"}
                and (
                    path.parent == subject / "ontology"
                    or (
                        path.is_relative_to(subject / "semantic_models")
                        and "metrics" not in path.relative_to(subject / "semantic_models").parts[1:-1]
                    )
                )
                and self.scope.allows_file(path)
            )
            if subject.is_dir()
            else []
        )

    def live_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute() and path.is_relative_to(self.root):
            return self.live_root / path.relative_to(self.root)
        return path if path.is_absolute() else self.live_root / path

    def path(self, name: str) -> Path:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or relative.as_posix() != name:
            raise ValueError("Artifact paths must be canonical project-relative paths")
        path = self.live_root / relative
        if path.resolve() != path or not self.scope.allows_file(path):
            raise PermissionError(f"Artifact is outside scoped_context.project_files: {name}")
        return path

    def snapshot(self) -> dict[str, Any]:
        return {
            "files": {name: content.decode("utf-8") for name, content in self.baseline.items()},
            "revisions": {name: digest(content) for name, content in self.baseline.items()},
            "datasource": self.scope.config.current_datasource,
        }

    def _contents(self, files: dict[str, str]) -> dict[str, bytes]:
        contents = {name: (self.root / name).read_bytes() for name in self.baseline}
        # Newly planned semantic models are not present in the original snapshot.
        models = self.root / "subject/semantic_models"
        for path in models.rglob("*") if models.is_dir() else ():
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".json"}:
                name = path.relative_to(self.root).as_posix()
                self.path(name)
                contents[name] = path.read_bytes()
        for name, text in files.items():
            self.path(name)
            if PurePosixPath(name).parent != PurePosixPath("subject/ontology") or Path(name).suffix not in {
                ".yaml",
                ".yml",
                ".json",
            }:
                raise ValueError("Private artifact contributions must be directly under subject/ontology")
            if not isinstance(text, str):
                raise TypeError("Artifact contents must be UTF-8 text")
            contents[name] = text.encode("utf-8")
        return contents

    def _require_ontology_inventory(self) -> None:
        # An unreadable mapping may refer to this model or shared ontology.
        # Do not inspect its contents or certify a partial dependency inventory.
        directory = self.live_root / "subject/ontology"
        for path in directory.iterdir() if directory.is_dir() else ():
            if (
                path.is_file()
                and path.suffix.lower() in {".yaml", ".yml", ".json"}
                and not self.scope.allows_file(path)
            ):
                raise PermissionError(
                    "Ontology artifacts exist outside scoped_context.project_files; the full mapping inventory is required for publication"
                )

    def inspect(
        self,
        files: dict[str, str],
        mapping_file: str,
        operation: str = "validate",
        arguments: dict | None = None,
        *,
        follow_shared_ontology: bool = True,
    ) -> dict:
        """Use production binding discovery and native lowering on exact draft bytes."""
        self._require_ontology_inventory()
        contents = self._contents(files)
        if mapping_file not in contents:
            raise ValueError("Selected mapping is unavailable in the authorized draft")
        documents = {name: self._document(name, body) for name, body in contents.items()}

        selected = documents[mapping_file]
        if document_kind(selected) != "mapping":
            raise ValueError("Selected artifact is not an ontology mapping")
        if any(not isinstance(selected[key], dict) for key in ("ontology_ref", "semantic_model_ref")):
            raise ValueError("Selected mapping references must be objects")
        ontology_name = selected["ontology_ref"].get("name")
        ontology_paths = (
            {self._reference_path(mapping_file, selected["ontology_ref"])} if follow_shared_ontology else set()
        )
        ontology_names = (
            {ontology_name} if follow_shared_ontology and isinstance(ontology_name, str) and ontology_name else set()
        )
        contributed_ontologies = set()
        for name in files:
            kind = document_kind(documents[name])
            if kind == "ontology":
                contributed_ontologies.add(name)
                ontology_paths.add(self.root / name)
                ontology_names.add(str(documents[name].get("name") or ""))
                if name in self.baseline:
                    previous = self._document(name, self.baseline[name])
                    if document_kind(previous) == "ontology":
                        ontology_names.add(str(previous.get("name") or ""))
            elif kind != "mapping":
                raise ValueError(f"Contributed artifact is not a valid ontology or mapping document: {name}")
        changed_models = {
            self.root / name
            for name, content in contents.items()
            if document_kind(documents[name]) == "model" and self.baseline.get(name) != content
        }
        ontology_paths.discard(None)
        affected = sorted(
            name
            for name, doc in documents.items()
            if document_kind(doc) == "mapping"
            and (
                name == mapping_file
                or name in files
                or (
                    isinstance(doc["ontology_ref"], dict)
                    and str(doc["ontology_ref"].get("name") or "") in ontology_names
                )
                or self._reference_path(name, doc["ontology_ref"]) in ontology_paths
                or self._reference_path(name, doc["semantic_model_ref"]) in changed_models
            )
        )
        with tempfile.TemporaryDirectory(prefix="datus-bundle-validation-") as directory:
            root = Path(directory).resolve()
            for name, body in contents.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
            ontology_files = [
                str(root / name) for name, doc in documents.items() if document_kind(doc) in {"mapping", "ontology"}
            ]
            selected_engine = None
            notes = []
            bound_ontologies = set()
            for name in affected:
                doc = documents[name]
                for key in ("ontology_ref", "semantic_model_ref"):
                    ref = doc[key]
                    if not isinstance(ref, dict):
                        raise ValueError(f"Mapping reference must be an object: {name}: {key}")
                    if name not in files and isinstance(ref.get("document"), dict):
                        continue
                    if key == "ontology_ref" and set(ref) == {"name"}:
                        continue
                    iri = ref.get("iri")
                    if (
                        not isinstance(iri, str)
                        or not iri
                        or ":" in iri
                        or Path(iri).is_absolute()
                        or "document" in ref
                    ):
                        raise ValueError(f"Unsupported local reference: {name}: {key}")
                    target = (root / name).parent.joinpath(iri).resolve()
                    if not target.is_relative_to(root.resolve()) or target.relative_to(root).as_posix() not in contents:
                        raise PermissionError(
                            f"Affected mapping dependency is outside the authorized draft: {name}: {key}"
                        )
                inline = isinstance(doc["semantic_model_ref"].get("document"), dict)
                models_dir = root / "subject/semantic_models"
                if inline:
                    model = root / name
                    models_dir /= self.config.current_datasource
                else:
                    model = (root / name).parent.joinpath(doc["semantic_model_ref"]["iri"]).resolve()
                    models_dir /= model.relative_to(models_dir).parts[0]
                models = [
                    str(root / path)
                    for path, value in documents.items()
                    if document_kind(value) == "model" and (root / path).is_relative_to(models_dir)
                ]
                bindings, _ = plan_ontologies(models, ontology_files, str(models_dir))
                binding = bindings.get(str(model))
                if binding is None or binding.mapping_path != str(root / name):
                    raise ValueError(f"Production Agent discovery cannot bind mapping {name}")
                ontology_source = binding.ontology_path or self._reference_path(name, doc["ontology_ref"], root=root)
                if ontology_source is not None:
                    bound_ontologies.add(Path(ontology_source).relative_to(root).as_posix())
                native = EngineHandle(
                    DosiConfig(
                        semantic_model_path=str(model),
                        ontology_dir=str(root / "subject/ontology"),
                        ontology_path=binding.ontology_path,
                        mapping_path=binding.mapping_path,
                        mapping_references=list(binding.references),
                    )
                ).get()
                notes.extend(str(note) for note in native.warnings)
                if name == mapping_file:
                    selected_engine = native
            if selected_engine is None:
                raise ValueError("The selected mapping was not validated")
            if contributed_ontologies - bound_ontologies:
                raise ValueError("Each contributed ontology must be validated through a bound mapping")
            args = arguments or {}
            if operation == "paths":
                return {"paths": selected_engine.paths(args["source"], args["target"])}
            if operation == "describe":
                return selected_engine.describe(args.get("name") or None)
            if operation != "validate":
                raise ValueError("Unsupported bundle inspection operation")
            for query in args.get("record_queries") or []:
                selected_engine.compile_select(query)
            for query in args.get("metric_queries") or []:
                selected_engine.compile(query)
            result = {
                "valid": True,
                "checks": {
                    "engine": "passed",
                    "agent_binding": "passed",
                    "query_compile": "passed"
                    if args.get("record_queries") or args.get("metric_queries")
                    else "not_run",
                    "execution": "not_run",
                },
                "warnings": sorted(set(notes)),
                "catalog": selected_engine.describe(),
                "affected_mappings": affected,
                "validated_digests": {name: digest(body) for name, body in contents.items()},
            }
            self.validation = copy.deepcopy(result)
            return result

    @staticmethod
    def _document(name: str, body: bytes) -> Any:
        try:
            return json.loads(body) if name.endswith(".json") else yaml.safe_load(body)
        except (ValueError, yaml.YAMLError):
            return {}

    def _reference_path(self, name: str, ref: Any, *, root: Path | None = None) -> Path | None:
        iri = ref.get("iri") if isinstance(ref, dict) else None
        if not isinstance(iri, str) or not iri or ":" in iri or Path(iri).is_absolute():
            return None
        return ((root or self.root) / name).parent.joinpath(iri).resolve()

    def publish(self, files: dict[str, str], semantic_file: str, expected_revision: str) -> dict[str, Any]:
        if self.published:
            raise RuntimeError("This draft has already been published")
        contents = self._contents(files)
        if semantic_file not in contents or digest(contents[semantic_file]) != expected_revision:
            raise ValueError("The semantic model changed after joint validation")
        if files and (
            self.validation is None
            or self.validation["validated_digests"] != {name: digest(body) for name, body in contents.items()}
        ):
            raise ValueError("Joint validation does not match the current bundle")
        model_mapping_validation = None
        if not files and self.baseline.get(semantic_file) != contents[semantic_file]:
            for name, body in contents.items():
                if PurePosixPath(name).parent != PurePosixPath("subject/ontology"):
                    continue
                document = self._document(name, body)
                if (
                    document_kind(document) == "mapping"
                    and self._reference_path(name, document["semantic_model_ref"]) == self.root / semantic_file
                ):
                    model_mapping_validation = self.inspect({}, name, follow_shared_ontology=False)
                    break
        outputs = {name: contents[name] for name in [semantic_file, *files]}
        for name in outputs:
            self.path(name)
        with ExitStack() as stack:
            from datus.agent.node.semantic_authoring import require_unique_model_target

            stack.enter_context(ontology_bundle_lock(self.live_root / "subject/ontology", create=True))
            for name in sorted(set(self.baseline) | set(outputs)):
                stack.enter_context(semantic_artifact_lock(self.path(name), load_before_lock=False))
            self._require_ontology_inventory()
            model_path = self.path(semantic_file)
            document = self._document(semantic_file, contents[semantic_file])
            if document_kind(document) == "model":
                models_dir = self.live_root / "subject/semantic_models"
                model_directory = models_dir / model_path.relative_to(models_dir).parts[0]
                require_unique_model_target(
                    self.scope.config,
                    model_path,
                    str(document["name"]),
                    creating=not model_path.is_file(),
                    model_directory=model_directory,
                )
            current = {path.relative_to(self.live_root).as_posix(): path.read_bytes() for path in self._live_files()}
            if current != self.baseline:
                raise ValueError("Artifact files changed since draft creation; start a new turn")
            try:
                for name, body in outputs.items():
                    path = self.path(name)
                    atomic_write_bytes(path, body)
                    self.published[name] = body
            except BaseException:
                self._restore_pending()
                raise
        return {
            "semantic_models": [semantic_file],
            "files": list(outputs),
            "validation": model_mapping_validation,
        }

    def rollback(self) -> None:
        if not self.published:
            return
        with ontology_bundle_lock(self.live_root / "subject/ontology", create=True):
            with ExitStack() as stack:
                for name in sorted(self.published):
                    stack.enter_context(semantic_artifact_lock(self.path(name), load_before_lock=False))
                for name, content in self.published.items():
                    path = self.path(name)
                    if not path.is_file() or path.read_bytes() != content:
                        raise RuntimeError("Published artifacts changed before recovery; refusing to overwrite")
                self._restore_pending()

    def _restore_pending(self) -> None:
        failures = []
        for name in reversed(tuple(self.published)):
            try:
                path = self.path(name)
                if name in self.baseline:
                    atomic_write_bytes(path, self.baseline[name])
                else:
                    path.unlink(missing_ok=True)
            except OSError as exc:
                failures.append(exc)
            else:
                del self.published[name]
        if failures:
            raise OSError("Some semantic bundle files could not be restored. Recovery remains pending.") from failures[
                0
            ]

    def close(self) -> None:
        self._temp.cleanup()
