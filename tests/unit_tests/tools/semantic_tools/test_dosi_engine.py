# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
import sys

import pytest

from datus.tools.semantic_tools.dosi.config import DosiConfig
from datus.tools.semantic_tools.dosi.engine import (
    EngineHandle,
    EngineRegistry,
    _normalize_sqlite_source,
    _sqlite_source_mtime,
    datus_authoring_contract,
    datus_authoring_contract_digest,
    datus_extension_version,
)
from datus.tools.semantic_tools.exceptions import SemanticCoreException
from tests.unit_tests.tools.semantic_tools.dosi_fakes import build_fake_module


@pytest.fixture
def fake_binding(monkeypatch):
    binding = build_fake_module()
    monkeypatch.setitem(sys.modules, "dosi_engine", binding)
    return binding


def test_datus_extension_version_reads_active_engine_capabilities(fake_binding, monkeypatch):
    monkeypatch.setattr(fake_binding, "DATUS_EXT", {"version": "7.3"})

    assert datus_extension_version() == "7.3"


def test_datus_extension_version_requires_capability_metadata(fake_binding, monkeypatch):
    monkeypatch.delattr(fake_binding, "DATUS_EXT")

    with pytest.raises(SemanticCoreException, match=r"DATUS_EXT\.version"):
        datus_extension_version()


def test_datus_authoring_contract_is_copied_and_version_checked(fake_binding, monkeypatch):
    source = {
        "extension_version": "7.3",
        "osi_spec_version": fake_binding.SPEC_VERSION,
        "capabilities": {"window": {}},
    }
    monkeypatch.setattr(fake_binding, "DATUS_EXT", {"version": "7.3"})
    monkeypatch.setattr(fake_binding, "DATUS_AUTHORING_CONTRACT", source, raising=False)

    contract = datus_authoring_contract()
    contract["capabilities"]["window"]["changed"] = True

    assert source == {
        "extension_version": "7.3",
        "osi_spec_version": fake_binding.SPEC_VERSION,
        "capabilities": {"window": {}},
    }


def test_datus_authoring_contract_rejects_version_mismatch(fake_binding, monkeypatch):
    monkeypatch.setattr(fake_binding, "DATUS_EXT", {"version": "1.5"})
    monkeypatch.setattr(
        fake_binding,
        "DATUS_AUTHORING_CONTRACT",
        {"extension_version": "1.4", "osi_spec_version": fake_binding.SPEC_VERSION},
        raising=False,
    )

    with pytest.raises(SemanticCoreException, match="inconsistent"):
        datus_authoring_contract()


def test_datus_authoring_contract_rejects_osi_version_mismatch(fake_binding, monkeypatch):
    monkeypatch.setattr(
        fake_binding,
        "DATUS_AUTHORING_CONTRACT",
        {"extension_version": "1.2", "osi_spec_version": "future"},
        raising=False,
    )

    with pytest.raises(SemanticCoreException, match="inconsistent OSI"):
        datus_authoring_contract()


@pytest.mark.parametrize(
    "digest",
    [
        "md5:nope",
        "sha256:" + "g" * 64,
    ],
)
def test_datus_authoring_contract_digest_requires_sha256(digest, fake_binding, monkeypatch):
    monkeypatch.setattr(fake_binding, "DATUS_AUTHORING_CONTRACT_DIGEST", digest, raising=False)

    with pytest.raises(SemanticCoreException, match="valid"):
        datus_authoring_contract_digest()


def test_non_object_model_root_is_mapped_by_native_loader(tmp_path, fake_binding, monkeypatch):
    model_path = tmp_path / "model.yaml"
    model_path.write_text("- not\n- an\n- object\n")

    class RejectingEngine:
        def __init__(self, **kwargs):
            raise fake_binding.ModelError("invalid document root", code="invalid_model")

    monkeypatch.setattr(fake_binding, "Engine", RejectingEngine)

    with pytest.raises(SemanticCoreException, match="Dosi failed to load model"):
        EngineHandle(DosiConfig(semantic_model_path=str(model_path))).get()


def test_registry_skips_model_removed_between_listing_and_stat(tmp_path, monkeypatch):
    stable = tmp_path / "stable.yaml"
    vanished = tmp_path / "vanished.yaml"
    stable.write_text("version: '0.2.0.dev0'\nsemantic_model: []\n")
    registry = EngineRegistry(DosiConfig(semantic_models_path=str(tmp_path)))
    monkeypatch.setattr(
        "datus.tools.semantic_tools.dosi.engine.resolve_model_files",
        lambda _config: [str(stable), str(vanished)],
    )

    signature, handles = registry.snapshot()

    assert [path for path, *_ in signature] == [str(stable)]
    assert [path for path, _ in handles] == [str(stable)]


def test_registry_rejects_snapshot_when_every_listed_model_disappears(tmp_path, monkeypatch):
    vanished = tmp_path / "vanished.yaml"
    registry = EngineRegistry(DosiConfig(semantic_models_path=str(tmp_path)))
    monkeypatch.setattr(
        "datus.tools.semantic_tools.dosi.engine.resolve_model_files",
        lambda _config: [str(vanished)],
    )

    with pytest.raises(SemanticCoreException, match="no readable OSI model file"):
        registry.snapshot()


def test_resolve_model_files_finds_models_in_subdirectories(tmp_path):
    """Datus discovers authored models recursively -- its authoring inventory
    and its YAML-to-KB sync both walk the tree -- so a model it accepts under a
    subdirectory has to be visible here too, or the same file exists for
    authoring and for the knowledge base while being absent from queries."""
    from datus.tools.semantic_tools.dosi.engine import resolve_model_files

    document = "version: '0.2.0.dev0'\nsemantic_model: []\n"
    (tmp_path / "flat.yaml").write_text(document)
    nested = tmp_path / "orders"
    nested.mkdir()
    (nested / "orders.yml").write_text(document)

    resolved = resolve_model_files(DosiConfig(semantic_models_path=str(tmp_path)))

    assert [os.path.relpath(path, tmp_path) for path in resolved] == [
        "flat.yaml",
        os.path.join("orders", "orders.yml"),
    ]


def test_resolve_model_files_skips_the_metric_fragment_directory(tmp_path):
    """``metrics`` holds per-metric fragments, not whole documents. Datus omits
    it from both of its own model walks, and loading a fragment as a document
    fails."""
    from datus.tools.semantic_tools.dosi.engine import resolve_model_files

    (tmp_path / "flat.yaml").write_text("version: '0.2.0.dev0'\nsemantic_model: []\n")
    fragments = tmp_path / "metrics"
    fragments.mkdir()
    (fragments / "revenue.yml").write_text("name: revenue\n")

    resolved = resolve_model_files(DosiConfig(semantic_models_path=str(tmp_path)))

    assert [os.path.relpath(path, tmp_path) for path in resolved] == ["flat.yaml"]


def test_resolve_model_files_still_reports_an_empty_directory(tmp_path):
    from datus.tools.semantic_tools.dosi.engine import resolve_model_files

    (tmp_path / "notes").mkdir()

    with pytest.raises(SemanticCoreException, match="no OSI model file"):
        resolve_model_files(DosiConfig(semantic_models_path=str(tmp_path)))


@pytest.mark.parametrize("prefix", ["sqlite:///", "sqlite://", "sqlite:", ""])
def test_sqlite_source_uri_resolves_to_one_file(tmp_path, prefix):
    source = tmp_path / "orders.sqlite"
    assert _normalize_sqlite_source(f"{prefix}{source}") == str(source)


def test_sqlite_wal_changes_engine_file_signature(tmp_path):
    source = tmp_path / "orders.sqlite"
    source.touch()
    base = _sqlite_source_mtime(str(source))
    wal = tmp_path / "orders.sqlite-wal"
    wal.touch()
    os.utime(wal, (base + 10, base + 10))
    assert _sqlite_source_mtime(str(source)) == base + 10


def test_sqlite_connection_preserves_native_uri(tmp_path):
    model = tmp_path / "model.yaml"
    model.write_text("version: '0.2.0.dev0'\nsemantic_model: []\n")
    source = tmp_path / "orders.sqlite"
    config = DosiConfig(
        semantic_model_path=str(model),
        datasource="orders",
        db_config={"type": "sqlite", "path": str(source)},
    )
    connection = EngineHandle(config)._runtime_connections(config)["orders"]
    assert connection == {"type": "sqlite", "uri": str(source), "default": True}
