# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Exact-byte draft isolation, publication conflict and recovery contracts."""

import json
from pathlib import Path

import pytest
import yaml

from datus.agent.node.configurable_scope import ConfigurableScope
from datus.agent.semantic_authoring_draft import SemanticAuthoringDraft, digest
from datus.schemas.agent_spec import AgentSpec


@pytest.fixture
def draft(real_agent_config):
    scope = ConfigurableScope(
        AgentSpec(
            name="author", scoped_context={"project_files": ["subject/semantic_models/**", "subject/ontology/**"]}
        ),
        real_agent_config,
    )
    model = scope.root / "subject/semantic_models/warehouse/sales.yaml"
    model.parent.mkdir(parents=True)
    model.write_text(
        yaml.safe_dump(
            {
                "version": "0.2.0.dev0",
                "name": "sales",
                "datasets": [
                    {
                        "name": "orders",
                        "source": "orders",
                        "fields": [
                            {"name": "id", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "id"}]}}
                        ],
                    }
                ],
            }
        )
    )
    workspace = SemanticAuthoringDraft(scope)
    try:
        yield workspace
    finally:
        workspace.close()


def _files():
    return {
        "subject/ontology/Sales.json": json.dumps({"version": "0.2.0.dev0", "name": "Sales", "ontology": []}),
        "subject/ontology/Sales.mapping.json": json.dumps(
            {
                "version": "0.2.0.dev0",
                "name": "SalesMapping",
                "ontology_ref": {"name": "Sales", "iri": "Sales.json"},
                "semantic_model_ref": {"name": "sales", "iri": "../semantic_models/warehouse/sales.yaml"},
                "concept_mappings": [],
            }
        ),
    }


def _change_model(draft):
    name = "subject/semantic_models/warehouse/sales.yaml"
    content = (draft.root / name).read_bytes() + b"description: Draft only\n"
    (draft.root / name).write_bytes(content)
    return name, content


def test_semantic_draft_stays_private_then_publishes_validated_json_bundle(draft):
    name, content = _change_model(draft)
    assert (draft.live_root / name).read_bytes() != content
    files = _files()
    checked = draft.inspect(files, "subject/ontology/Sales.mapping.json")
    assert checked["checks"]["agent_binding"] == "passed"
    assert checked["checks"]["execution"] == "not_run"
    draft.publish(files, name, digest(content))
    assert (draft.live_root / name).read_bytes() == content
    assert json.loads((draft.live_root / "subject/ontology/Sales.json").read_text())["name"] == "Sales"
    draft.rollback()
    assert (draft.live_root / name).read_bytes() == draft.baseline[name]
    assert not (draft.live_root / "subject/ontology/Sales.json").exists()


def test_joint_validation_cannot_certify_an_unrelated_invalid_contributed_mapping(draft):
    files = _files()
    mapping = json.loads(files["subject/ontology/Sales.mapping.json"])
    mapping["name"] = "UnrelatedMapping"
    mapping["ontology_ref"] = {"name": "AnotherOntology"}
    mapping["semantic_model_ref"]["iri"] = "../semantic_models/warehouse/missing.yaml"
    files["subject/ontology/unrelated.mapping.json"] = json.dumps(mapping)

    with pytest.raises(PermissionError, match="Affected mapping dependency"):
        draft.inspect(files, "subject/ontology/Sales.mapping.json")
    assert draft.validation is None


def test_model_only_publication_still_validates_a_dependent_mapping(draft):
    files = _files()
    mapping = json.loads(files["subject/ontology/Sales.mapping.json"])
    mapping["ontology_ref"]["iri"] = "missing.json"
    files["subject/ontology/Sales.mapping.json"] = json.dumps(mapping)
    for name, body in files.items():
        live = draft.live_root / name
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text(body)

    coordinated = SemanticAuthoringDraft(draft.scope)
    try:
        name, content = _change_model(coordinated)
        with pytest.raises(PermissionError, match="Affected mapping dependency"):
            coordinated.publish({}, name, digest(content))
        assert (coordinated.live_root / name).read_bytes() == coordinated.baseline[name]
        assert not coordinated.published
    finally:
        coordinated.close()


def test_model_only_publication_leaves_unmodified_inline_mapping_alone(draft):
    files = _files()
    mapping = json.loads(files["subject/ontology/Sales.mapping.json"])
    inline = {
        **mapping,
        "name": "InlineMapping",
        "semantic_model_ref": {
            "name": "sales",
            "document": yaml.safe_load((draft.root / "subject/semantic_models/warehouse/sales.yaml").read_bytes()),
        },
    }
    files["subject/ontology/inline.mapping.json"] = json.dumps(inline)
    for name, body in files.items():
        live = draft.live_root / name
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text(body)

    coordinated = SemanticAuthoringDraft(draft.scope)
    try:
        name, content = _change_model(coordinated)
        receipt = coordinated.publish({}, name, digest(content))
        assert receipt["files"] == [name]
        assert receipt["validation"]["affected_mappings"] == ["subject/ontology/Sales.mapping.json"]
        assert (draft.live_root / "subject/ontology/inline.mapping.json").read_text() == files[
            "subject/ontology/inline.mapping.json"
        ]
    finally:
        coordinated.close()


def test_publication_rechecks_new_out_of_scope_json_model_names(draft):
    from datus.schemas.agent_models import ScopedContext

    name = "subject/semantic_models/warehouse/sales.yaml"
    draft.scope.prepare(ScopedContext(project_files=[name]))
    coordinated = SemanticAuthoringDraft(draft.scope)
    try:
        name, content = _change_model(coordinated)
        conflicting = draft.live_root / "subject/semantic_models/warehouse/hidden.json"
        conflicting.write_text(json.dumps(yaml.safe_load(coordinated.baseline[name])))
        with pytest.raises(ValueError, match="model name conflicts") as error:
            coordinated.publish({}, name, digest(content))
        assert "hidden.json" not in str(error.value)
        assert (draft.live_root / name).read_bytes() == coordinated.baseline[name]
    finally:
        coordinated.close()


def test_renamed_ontology_rechecks_mappings_using_the_prior_name(draft):
    files = _files()
    legacy = json.loads(files["subject/ontology/Sales.mapping.json"])
    legacy["name"] = "LegacyMapping"
    legacy["ontology_ref"] = {"name": "Sales"}
    legacy["semantic_model_ref"] = {"name": "legacy", "iri": "../semantic_models/warehouse/legacy.yaml"}
    files["subject/ontology/legacy.mapping.json"] = json.dumps(legacy)
    files["subject/semantic_models/warehouse/legacy.yaml"] = (
        next(iter(draft.baseline.values())).decode().replace("name: sales", "name: legacy", 1)
    )
    for name, body in files.items():
        live = draft.live_root / name
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text(body)

    coordinated = SemanticAuthoringDraft(draft.scope)
    try:
        renamed = {
            "subject/ontology/Sales.json": files["subject/ontology/Sales.json"].replace('"Sales"', '"NewSales"'),
            "subject/ontology/Sales.mapping.json": files["subject/ontology/Sales.mapping.json"].replace(
                '"Sales"', '"NewSales"'
            ),
        }
        with pytest.raises(ValueError, match="legacy.mapping.json"):
            coordinated.inspect(renamed, "subject/ontology/Sales.mapping.json")
        assert coordinated.validation is None
    finally:
        coordinated.close()


def test_rollback_retries_only_files_that_still_need_restoring(draft, monkeypatch):
    from datus.agent import semantic_authoring_draft

    name, content = _change_model(draft)
    model = draft.live_root / name
    files = _files()
    draft.inspect(files, "subject/ontology/Sales.mapping.json")
    draft.publish(files, name, digest(content))
    original_write = semantic_authoring_draft.atomic_write_bytes

    def fail_model_write(path, body):
        if path == model:
            raise OSError("Model is temporarily unwritable")
        return original_write(path, body)

    with monkeypatch.context() as blocked:
        blocked.setattr(semantic_authoring_draft, "atomic_write_bytes", fail_model_write)
        with pytest.raises(OSError, match="Recovery remains pending"):
            draft.rollback()
    assert model.read_bytes() == content
    assert draft.published == {name: content}
    assert all(not (draft.live_root / key).exists() for key in files)

    draft.rollback()
    assert not draft.published
    assert model.read_bytes() == draft.baseline[name]


def test_publish_keeps_failed_recovery_available_for_retry(draft, monkeypatch):
    from datus.agent import semantic_authoring_draft

    name, content = _change_model(draft)
    files = _files()
    ontology = draft.live_root / "subject/ontology/Sales.json"
    mapping = draft.live_root / "subject/ontology/Sales.mapping.json"
    draft.inspect(files, "subject/ontology/Sales.mapping.json")
    original_write = semantic_authoring_draft.atomic_write_bytes
    original_unlink = Path.unlink

    def fail_mapping_write(path, body):
        if path == mapping:
            raise OSError("Mapping write failed")
        return original_write(path, body)

    def fail_ontology_removal(path, *args, **kwargs):
        if path == ontology:
            raise OSError("Ontology cannot be removed yet")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as blocked:
        blocked.setattr(semantic_authoring_draft, "atomic_write_bytes", fail_mapping_write)
        blocked.setattr(Path, "unlink", fail_ontology_removal)
        with pytest.raises(OSError, match="Recovery remains pending"):
            draft.publish(files, name, digest(content))
    assert list(draft.published) == ["subject/ontology/Sales.json"]
    assert (draft.live_root / name).read_bytes() == draft.baseline[name]

    draft.rollback()
    assert not ontology.exists()
    assert not mapping.exists()


@pytest.mark.parametrize("change", ["model", "new_mapping", "validation"])
def test_publication_rejects_concurrent_or_unvalidated_changes_without_overwriting(draft, change):
    name, content = _change_model(draft)
    files = _files()
    draft.inspect(files, "subject/ontology/Sales.mapping.json")
    if change == "model":
        path = draft.live_root / name
        path.write_bytes(b"External edit\n")
    elif change == "new_mapping":
        path = draft.live_root / "subject/ontology/concurrent.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"External mapping\n")
    else:
        path = draft.live_root / name
        files["subject/ontology/Sales.json"] += " "
    before = path.read_bytes()
    with pytest.raises(ValueError, match="changed since|validation does not match"):
        draft.publish(files, name, digest(content))
    assert path.read_bytes() == before
    assert not (draft.live_root / "subject/ontology/Sales.json").exists()


def test_write_failure_restores_model_and_removes_partial_bundle(draft, monkeypatch):
    from datus.agent import semantic_authoring_draft as module

    name, content = _change_model(draft)
    files = _files()
    draft.inspect(files, "subject/ontology/Sales.mapping.json")
    write = module.atomic_write_bytes

    def fail_mapping(path, body):
        if Path(path).name == "Sales.mapping.json":
            raise OSError("injected disk failure")
        return write(path, body)

    monkeypatch.setattr(module, "atomic_write_bytes", fail_mapping)
    with pytest.raises(OSError, match="injected disk failure"):
        draft.publish(files, name, digest(content))
    assert (draft.live_root / name).read_bytes() == draft.baseline[name]
    assert not (draft.live_root / "subject/ontology/Sales.json").exists()


@pytest.mark.parametrize("reference", ["../../../../outside.yaml", "https://example.org/model.yaml", "/tmp/model.yaml"])
def test_bundle_validation_refuses_escaping_or_remote_references(draft, reference):
    files = _files()
    mapping = json.loads(files["subject/ontology/Sales.mapping.json"])
    mapping["semantic_model_ref"]["iri"] = reference
    files["subject/ontology/Sales.mapping.json"] = json.dumps(mapping)
    with pytest.raises((ValueError, PermissionError), match="reference|dependency"):
        draft.inspect(files, "subject/ontology/Sales.mapping.json")
    assert not (draft.live_root / "subject/ontology/Sales.json").exists()


@pytest.mark.parametrize("body", [b"invalid: [", b"ontology_ref: broken\nsemantic_model_ref: {}\nconcept_mappings: []"])
def test_unrelated_invalid_document_does_not_block_selected_bundle(draft, body):
    name = "subject/ontology/unrelated.yaml"
    draft.baseline[name] = body
    path = draft.root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(draft.baseline[name])
    assert draft.inspect(_files(), "subject/ontology/Sales.mapping.json")["valid"] is True


def _shared_draft(draft, scope_case):
    from datus.agent.node.configurable_scope import ConfigurableScope

    files = _files()
    for name, text in files.items():
        path = draft.live_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    foreign = draft.live_root / "subject/semantic_models/crm/sales.yaml"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(next(iter(draft.baseline.values())))
    original = foreign.read_bytes()
    mapping = json.loads(files["subject/ontology/Sales.mapping.json"])
    mapping["name"] = "crm_mapping"
    mapping["ontology_ref"] = {"name": "Sales"}
    mapping["semantic_model_ref"]["iri"] = "../semantic_models/crm/sales.yaml"
    other = draft.live_root / "subject/ontology/crm.mapping.json"
    other.write_text(json.dumps(mapping))
    scope = ConfigurableScope(
        AgentSpec(
            name="shared",
            scoped_context={
                "project_files": [
                    *(
                        ["subject/ontology/Sales.json", "subject/ontology/Sales.mapping.json"]
                        if scope_case == "mapping_missing"
                        else ["subject/ontology/**"]
                    ),
                    "subject/semantic_models/warehouse/**"
                    if scope_case == "model_missing"
                    else "subject/semantic_models/**",
                ]
            },
        ),
        draft.scope.config,
    )
    coordinated = SemanticAuthoringDraft(scope)
    return coordinated, foreign, original


@pytest.mark.parametrize("scope_case", ["complete", "model_missing", "mapping_missing"])
def test_shared_ontology_checks_other_datasource_and_does_not_expand_file_scope(draft, scope_case):
    coordinated, foreign, original = _shared_draft(draft, scope_case)
    try:
        if scope_case == "complete":
            result = coordinated.inspect({}, "subject/ontology/Sales.mapping.json")
            assert result["affected_mappings"] == [
                "subject/ontology/Sales.mapping.json",
                "subject/ontology/crm.mapping.json",
            ]
        else:
            with pytest.raises(PermissionError, match="outside"):
                coordinated.inspect({}, "subject/ontology/Sales.mapping.json")
        assert foreign.read_bytes() == original
    finally:
        coordinated.close()


def test_datasource_named_metrics_is_included_but_legacy_metric_folder_is_not(draft):
    content = next(iter(draft.baseline.values()))
    model = draft.live_root / "subject/semantic_models/metrics/sales.yaml"
    model.parent.mkdir(parents=True)
    model.write_bytes(content)
    legacy = draft.live_root / "subject/semantic_models/warehouse/metrics/legacy.yaml"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(content)
    snapshot = SemanticAuthoringDraft(draft.scope)
    try:
        assert "subject/semantic_models/metrics/sales.yaml" in snapshot.snapshot()["files"]
        assert "subject/semantic_models/warehouse/metrics/legacy.yaml" not in snapshot.snapshot()["files"]
    finally:
        snapshot.close()


def test_model_only_publication_cannot_skip_an_unavailable_mapping(draft):
    coordinated, foreign, original = _shared_draft(draft, "mapping_missing")
    try:
        name, changed = _change_model(coordinated)
        with pytest.raises(PermissionError, match="full mapping inventory"):
            coordinated.publish({}, name, digest(changed))
        assert (coordinated.live_root / name).read_bytes() == coordinated.baseline[name]
        assert foreign.read_bytes() == original
    finally:
        coordinated.close()
