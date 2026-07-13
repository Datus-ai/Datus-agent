from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "verify_nightly_adapter_sources.py"
MODULE_SPEC = importlib.util.spec_from_file_location("verify_nightly_adapter_sources", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load verify_nightly_adapter_sources module from {MODULE_PATH}")
verify_sources = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(verify_sources)


class _FakeDistribution:
    def __init__(self, direct_url: str | None):
        self.direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return self.direct_url


def _local_distribution(path: Path) -> _FakeDistribution:
    return _FakeDistribution(json.dumps({"url": path.resolve().as_uri(), "dir_info": {}}))


def test_verify_local_sources_accepts_every_expected_checkout(monkeypatch, tmp_path):
    external_root = tmp_path / "external"
    distributions = {
        name: _local_distribution(external_root / relative_path)
        for name, relative_path in verify_sources.EXPECTED_LOCAL_PACKAGES.items()
    }
    monkeypatch.setattr(verify_sources.metadata, "distribution", distributions.__getitem__)

    assert verify_sources.verify_local_sources(external_root) == []


def test_expected_sources_include_storage_packages():
    assert verify_sources.EXPECTED_LOCAL_PACKAGES["datus-storage-base"] == ("datus-storage-adapters/datus-storage-base")
    assert verify_sources.EXPECTED_LOCAL_PACKAGES["datus-storage-postgresql"] == (
        "datus-storage-adapters/datus-storage-postgresql"
    )


def test_verify_local_sources_rejects_registry_package(monkeypatch, tmp_path):
    external_root = tmp_path / "external"
    distributions = {
        name: _local_distribution(external_root / relative_path)
        for name, relative_path in verify_sources.EXPECTED_LOCAL_PACKAGES.items()
    }
    distributions["datus-semantic-core"] = _FakeDistribution(None)
    monkeypatch.setattr(verify_sources.metadata, "distribution", distributions.__getitem__)

    errors = verify_sources.verify_local_sources(external_root)

    assert errors == ["datus-semantic-core: package has no direct_url.json and was likely installed from a registry"]


def test_verify_local_sources_rejects_wrong_checkout(monkeypatch, tmp_path):
    external_root = tmp_path / "external"
    distributions = {
        name: _local_distribution(external_root / relative_path)
        for name, relative_path in verify_sources.EXPECTED_LOCAL_PACKAGES.items()
    }
    distributions["datus-db-core"] = _local_distribution(tmp_path / "somewhere-else")
    monkeypatch.setattr(verify_sources.metadata, "distribution", distributions.__getitem__)

    errors = verify_sources.verify_local_sources(external_root)

    assert len(errors) == 1
    assert errors[0].startswith("datus-db-core: expected checkout source ")
    assert errors[0].endswith(f", got {(tmp_path / 'somewhere-else').resolve()}")


def test_verify_semantic_adapter_imports_requires_shared_contract(monkeypatch):
    modules = {
        "datus_semantic_core.models": SimpleNamespace(),
        "datus_semantic_metricflow": SimpleNamespace(),
    }
    monkeypatch.setattr(verify_sources.importlib, "import_module", modules.__getitem__)

    assert verify_sources.verify_semantic_adapter_imports() == [
        "datus-semantic-core is missing SemanticValidationError"
    ]


def test_verify_storage_adapter_imports_requires_shared_contract(monkeypatch):
    shared_fts = SimpleNamespace(FtsSpec=object())
    modules = {
        "datus_storage_base.vector.fts": shared_fts,
        "datus.storage.fts": SimpleNamespace(FtsSpec=shared_fts.FtsSpec),
        "datus_storage_postgresql.vector": SimpleNamespace(),
    }
    monkeypatch.setattr(verify_sources.importlib, "import_module", modules.__getitem__)

    assert verify_sources.verify_storage_adapter_imports() == [
        "datus-storage-base FTS contract is missing: FtsField, FtsIndexStatus, normalize_fts_spec",
        "datus-storage-postgresql vector adapter is missing PgvectorBackend",
    ]


def test_verify_storage_adapter_imports_requires_agent_to_reexport_shared_contract(monkeypatch):
    shared_fts = SimpleNamespace(
        FtsField=object(),
        FtsIndexStatus=object(),
        FtsSpec=object(),
        normalize_fts_spec=object(),
    )
    agent_fts = SimpleNamespace(
        FtsField=shared_fts.FtsField,
        FtsIndexStatus=object(),
        FtsSpec=shared_fts.FtsSpec,
        normalize_fts_spec=shared_fts.normalize_fts_spec,
    )
    modules = {
        "datus_storage_base.vector.fts": shared_fts,
        "datus.storage.fts": agent_fts,
        "datus_storage_postgresql.vector": SimpleNamespace(PgvectorBackend=object()),
    }
    monkeypatch.setattr(verify_sources.importlib, "import_module", modules.__getitem__)

    assert verify_sources.verify_storage_adapter_imports() == [
        "Datus Agent FTS contract does not re-export datus-storage-base: FtsIndexStatus"
    ]


def test_verify_storage_adapter_imports_accepts_agent_shared_contract(monkeypatch):
    shared_fts = SimpleNamespace(
        FtsField=object(),
        FtsIndexStatus=object(),
        FtsSpec=object(),
        normalize_fts_spec=object(),
    )
    modules = {
        "datus_storage_base.vector.fts": shared_fts,
        "datus.storage.fts": shared_fts,
        "datus_storage_postgresql.vector": SimpleNamespace(PgvectorBackend=object()),
    }
    monkeypatch.setattr(verify_sources.importlib, "import_module", modules.__getitem__)

    assert verify_sources.verify_storage_adapter_imports() == []
