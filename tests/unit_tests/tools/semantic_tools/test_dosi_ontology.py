# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Ontology support in the embedded Dosi runtime: discovery, pairing, loading and the ontology calls."""

import os
import sys

import pytest

from datus.tools.semantic_tools.dosi.config import DosiConfig
from datus.tools.semantic_tools.dosi.engine import (
    EngineRegistry,
    binding_supports_ontology,
    pair_ontologies,
    resolve_model_files,
    resolve_ontology_files,
)
from datus.tools.semantic_tools.dosi.errors import SemanticValidationException
from datus.tools.semantic_tools.dosi.runtime import DosiRuntime
from datus.tools.semantic_tools.exceptions import SemanticCoreException
from tests.unit_tests.tools.semantic_tools.dosi_fakes import (
    ONTOLOGY_CATALOG,
    ONTOLOGY_PATHS,
    FakeEngine,
    FakeOntologyEngine,
    QueryError,
    build_fake_module,
)

MODEL_YAML = "version: '0.2.0.dev0'\nsemantic_model: []\n"
ONTOLOGY_YAML = "version: 0.2.0.dev0\nname: Aviation\nontology: []\n"


@pytest.fixture
def ontology_binding(monkeypatch):
    FakeEngine.instances.clear()
    binding = build_fake_module(ontology=True)
    monkeypatch.setitem(sys.modules, "dosi_engine", binding)
    yield binding
    FakeEngine.instances.clear()


@pytest.fixture
def legacy_binding(monkeypatch):
    FakeEngine.instances.clear()
    binding = build_fake_module()
    monkeypatch.setitem(sys.modules, "dosi_engine", binding)
    yield binding
    FakeEngine.instances.clear()


def _layout(tmp_path, models=("aviation",), ontologies=("aviation",)):
    models_dir = tmp_path / "semantic_models" / "flights_db"
    ontology_dir = tmp_path / "semantic_models" / "ontology" / "flights_db"
    models_dir.mkdir(parents=True)
    for stem in models:
        (models_dir / f"{stem}.yaml").write_text(MODEL_YAML)
    if ontologies:
        ontology_dir.mkdir(parents=True)
        for stem in ontologies:
            (ontology_dir / f"{stem}.yaml").write_text(ONTOLOGY_YAML)
    return models_dir, ontology_dir


def _distinct_metrics_per_model(monkeypatch):
    """Several model files in one datasource must not declare the same metric."""

    def metrics(engine):
        stem = os.path.splitext(os.path.basename(engine.model_path))[0]
        return [{"name": f"{stem}_count", "kind": "aggregate", "datasets": [stem], "measures": [f"{stem}_count"]}]

    monkeypatch.setattr(FakeEngine, "metrics", metrics)


def _runtime(models_dir, ontology_dir, **overrides):
    return DosiRuntime(
        DosiConfig(
            datasource="flights_db",
            semantic_models_path=str(models_dir),
            ontology_dir=str(ontology_dir),
            **overrides,
        )
    )


# ---------------------------------------------------------------- discovery


class TestOntologyDiscovery:
    def test_missing_dir_means_no_ontology(self, tmp_path):
        assert resolve_ontology_files(DosiConfig(ontology_dir=str(tmp_path / "absent"))) == []

    def test_unset_dir_means_no_ontology(self):
        assert resolve_ontology_files(DosiConfig()) == []

    def test_finds_every_ontology_suffix_recursively(self, tmp_path):
        for rel in ("a.yaml", "b.yml", "nested/c.json"):
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        (tmp_path / "notes.txt").write_text("not an ontology")
        found = resolve_ontology_files(DosiConfig(ontology_dir=str(tmp_path)))
        assert [os.path.relpath(path, tmp_path) for path in found] == [
            "a.yaml",
            "b.yml",
            os.path.join("nested", "c.json"),
        ]

    def test_datasource_named_ontology_is_rejected(self, tmp_path):
        reserved = tmp_path / "semantic_models" / "ontology"
        reserved.mkdir(parents=True)
        (reserved / "m.yaml").write_text(MODEL_YAML)
        with pytest.raises(SemanticCoreException, match="reserved for ontology files"):
            resolve_model_files(DosiConfig(semantic_models_path=str(reserved)))


class TestPairOntologies:
    def test_single_model_and_single_ontology_pair_regardless_of_names(self):
        assert pair_ontologies(["/m/sales.yaml"], ["/o/whatever.yaml"]) == ({"/m/sales.yaml": "/o/whatever.yaml"}, [])

    def test_pairs_by_file_stem_across_suffixes(self):
        pairs, unpaired = pair_ontologies(
            ["/m/aviation.yaml", "/m/retail.yml"],
            ["/o/retail.json", "/o/aviation.yml"],
        )
        assert pairs == {"/m/aviation.yaml": "/o/aviation.yml", "/m/retail.yml": "/o/retail.json"}
        assert unpaired == []

    def test_unmatched_ontology_is_reported_not_guessed(self):
        pairs, unpaired = pair_ontologies(["/m/aviation.yaml", "/m/retail.yaml"], ["/o/hr.yaml"])
        assert pairs == {}
        assert unpaired == ["/o/hr.yaml"]

    def test_two_ontologies_claiming_one_model_leave_it_unpaired(self):
        pairs, unpaired = pair_ontologies(
            ["/m/aviation.yaml", "/m/retail.yaml"],
            ["/o/a/aviation.yaml", "/o/b/aviation.yaml"],
        )
        assert pairs == {}
        assert sorted(unpaired) == ["/o/a/aviation.yaml", "/o/b/aviation.yaml"]

    def test_no_ontologies(self):
        assert pair_ontologies(["/m/aviation.yaml"], []) == ({}, [])


class TestBindingCapability:
    def test_legacy_binding_lacks_ontology_calls(self, legacy_binding):
        assert binding_supports_ontology() is False

    def test_ontology_binding_is_detected(self, ontology_binding):
        assert binding_supports_ontology() is True

    def test_missing_binding_is_not_supported(self, monkeypatch):
        def fail():
            raise SemanticCoreException("missing")

        monkeypatch.setattr("datus.tools.semantic_tools.dosi.engine.load_binding", fail)
        assert binding_supports_ontology() is False


# ---------------------------------------------------------------- loading


class TestEngineLoading:
    def test_paired_model_loads_with_its_ontology_and_mapping(self, tmp_path, ontology_binding):
        models_dir, ontology_dir = _layout(tmp_path)
        registry = EngineRegistry(
            DosiConfig(semantic_models_path=str(models_dir), ontology_dir=str(ontology_dir), ontology_mapping="m1")
        )
        _, handles = registry.snapshot()
        engine = handles[0][1].get()
        assert engine.ontology_path == str(ontology_dir / "aviation.yaml")
        assert engine.mapping == "m1"

    def test_ontologies_are_not_loaded_unless_requested(self, tmp_path, ontology_binding):
        models_dir, _ = _layout(tmp_path)
        _, handles = EngineRegistry(DosiConfig(semantic_models_path=str(models_dir))).snapshot()
        engine = handles[0][1].get()
        assert engine.ontology_path is None
        assert engine.mapping is None

    def test_legacy_binding_without_ontology_kwargs_still_loads_plain_models(self, tmp_path, legacy_binding):
        models_dir, _ = _layout(tmp_path, ontologies=())
        _, handles = EngineRegistry(
            DosiConfig(semantic_models_path=str(models_dir), ontology_dir=str(tmp_path / "none"))
        ).snapshot()
        assert handles[0][1].get().model_path.endswith("aviation.yaml")

    def test_editing_the_ontology_rebuilds_the_engine(self, tmp_path, ontology_binding):
        models_dir, ontology_dir = _layout(tmp_path)
        registry = EngineRegistry(DosiConfig(semantic_models_path=str(models_dir), ontology_dir=str(ontology_dir)))
        handle = registry.snapshot()[1][0][1]
        first = handle.get()
        ontology_file = ontology_dir / "aviation.yaml"
        ontology_file.write_text(ONTOLOGY_YAML + "# edited\n")
        stat = ontology_file.stat()
        os.utime(ontology_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
        assert handle.get() is not first

    def test_adding_an_ontology_repairs_handles(self, tmp_path, ontology_binding):
        models_dir, ontology_dir = _layout(tmp_path, ontologies=())
        registry = EngineRegistry(DosiConfig(semantic_models_path=str(models_dir), ontology_dir=str(ontology_dir)))
        assert registry.snapshot()[1][0][1].ontology_path is None
        ontology_dir.mkdir(parents=True)
        (ontology_dir / "aviation.yaml").write_text(ONTOLOGY_YAML)
        assert registry.snapshot()[1][0][1].ontology_path == str(ontology_dir / "aviation.yaml")

    def test_ontology_load_failure_names_the_ontology_and_code(self, tmp_path, ontology_binding, monkeypatch):
        models_dir, ontology_dir = _layout(tmp_path)

        class RejectingEngine(FakeOntologyEngine):
            def __init__(self, **kwargs):
                raise ontology_binding.ModelError("pick one with --mapping", code="ontology_mapping_unresolved")

        monkeypatch.setattr(ontology_binding, "Engine", RejectingEngine)
        registry = EngineRegistry(DosiConfig(semantic_models_path=str(models_dir), ontology_dir=str(ontology_dir)))
        with pytest.raises(SemanticCoreException, match=r"with ontology .*aviation\.yaml.*ontology_mapping_unresolved"):
            registry.snapshot()[1][0][1].get()


# ---------------------------------------------------------------- runtime calls


class TestRuntimeOntology:
    def test_has_ontology_and_info(self, tmp_path, ontology_binding):
        runtime = _runtime(*_layout(tmp_path))
        assert runtime.has_ontology() is True
        [info] = runtime.ontology_info()
        assert info["name"] == "Aviation"
        assert info["model_file"].endswith("aviation.yaml")

    def test_no_ontology(self, tmp_path, ontology_binding):
        runtime = _runtime(*_layout(tmp_path, ontologies=()))
        assert runtime.has_ontology() is False
        assert runtime.unpaired_ontologies() == []

    def test_unpaired_ontology_is_visible(self, tmp_path, ontology_binding, monkeypatch):
        _distinct_metrics_per_model(monkeypatch)
        runtime = _runtime(*_layout(tmp_path, models=("aviation", "retail"), ontologies=("hr",)))
        assert runtime.has_ontology() is False
        assert [os.path.basename(path) for path in runtime.unpaired_ontologies()] == ["hr.yaml"]

    @pytest.mark.asyncio
    async def test_describe_catalog_is_the_engine_payload(self, tmp_path, ontology_binding):
        runtime = _runtime(*_layout(tmp_path))
        catalog = await runtime.describe()
        assert catalog["concepts"] == ONTOLOGY_CATALOG["concepts"]
        assert catalog["model_file"] == "aviation.yaml"

    @pytest.mark.asyncio
    async def test_describe_catalog_across_several_ontologies(self, tmp_path, ontology_binding, monkeypatch):
        _distinct_metrics_per_model(monkeypatch)
        runtime = _runtime(*_layout(tmp_path, models=("aviation", "retail"), ontologies=("aviation", "retail")))
        catalog = await runtime.describe()
        assert [entry["model_file"] for entry in catalog["ontologies"]] == ["aviation.yaml", "retail.yaml"]

    @pytest.mark.asyncio
    async def test_describe_concept_detail(self, tmp_path, ontology_binding):
        runtime = _runtime(*_layout(tmp_path))
        detail = await runtime.describe("FlightAlert")
        status = next(p for p in detail["properties"] if p["path"] == "FlightAlert.status")
        assert status["domain"] == ["OPEN", "RESOLVED"]

    @pytest.mark.asyncio
    async def test_describe_routes_to_the_ontology_that_knows_the_concept(
        self, tmp_path, ontology_binding, monkeypatch
    ):
        _distinct_metrics_per_model(monkeypatch)
        runtime = _runtime(*_layout(tmp_path, models=("aviation", "retail"), ontologies=("aviation", "retail")))
        original = FakeOntologyEngine.describe

        def describe(engine, name=None):
            if name == "FlightAlert" and not engine.model_path.endswith("retail.yaml"):
                raise QueryError("unknown concept", code="unknown_concept")
            return original(engine, name)

        monkeypatch.setattr(FakeOntologyEngine, "describe", describe)
        detail = await runtime.describe("FlightAlert")
        assert detail["concept"] == "FlightAlert"

    @pytest.mark.asyncio
    async def test_unknown_concept_is_a_structured_rejection(self, tmp_path, ontology_binding):
        runtime = _runtime(*_layout(tmp_path))
        with pytest.raises(SemanticValidationException) as excinfo:
            await runtime.describe("Spaceship")
        assert excinfo.value.payload.code == "unknown_concept"
        assert "FlightAlert" in excinfo.value.payload.candidates

    @pytest.mark.asyncio
    async def test_concept_paths(self, tmp_path, ontology_binding):
        runtime = _runtime(*_layout(tmp_path))
        result = await runtime.concept_paths("Flight", "Airport")
        assert result == ONTOLOGY_PATHS
        assert FakeEngine.instances[-1].paths_calls == [("Flight", "Airport")]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call", ["catalog", "concept", "paths"])
    async def test_ontology_calls_require_an_ontology(self, tmp_path, ontology_binding, call):
        runtime = _runtime(*_layout(tmp_path, ontologies=()))
        with pytest.raises(SemanticCoreException, match="no ontology is loaded"):
            if call == "catalog":
                await runtime.describe()
            elif call == "concept":
                await runtime.describe("Flight")
            else:
                await runtime.concept_paths("Flight", "Airport")

    @pytest.mark.asyncio
    async def test_ontology_calls_require_a_capable_binding(self, tmp_path, legacy_binding):
        runtime = _runtime(*_layout(tmp_path))
        with pytest.raises(SemanticCoreException, match="cannot describe ontologies"):
            await runtime.describe()

    @pytest.mark.asyncio
    async def test_select_builds_a_detail_query(self, tmp_path, ontology_binding):
        runtime = _runtime(*_layout(tmp_path))
        result = await runtime.select(
            from_concept="FlightAlert",
            fields=["FlightAlert.id", "FlightAlert.priority"],
            where="FlightAlert.status = 'OPEN'",
            order_by=["-FlightAlert.id"],
            limit=10,
        )
        [call] = FakeEngine.instances[-1].select_calls
        assert call["query"] == {
            "from": "FlightAlert",
            "fields": ["FlightAlert.id", "FlightAlert.priority"],
            "where_sql": "FlightAlert.status = 'OPEN'",
            "order_by": [{"key": "FlightAlert.id", "desc": True}],
            "limit": 10,
        }
        assert result.columns == ["alert_id", "priority"]
        assert result.metadata["row_count"] == 2
        assert result.metadata["requested_columns"][0] == {"requested": "FlightAlert.id", "column": "alert_id"}

    @pytest.mark.asyncio
    async def test_select_value_not_in_domain_keeps_violations(self, tmp_path, ontology_binding, monkeypatch):
        runtime = _runtime(*_layout(tmp_path))
        error = QueryError(
            '"unresolved" is not a value of FlightAlert.status',
            code="value_not_in_domain",
            candidates=["OPEN", "RESOLVED"],
        )
        error.violations = [
            {"reference": "FlightAlert.status", "literal": "unresolved", "allowed": ["OPEN", "RESOLVED"]}
        ]

        def failing_select(engine, query, **kwargs):
            raise error

        monkeypatch.setattr(FakeOntologyEngine, "select", failing_select)
        with pytest.raises(SemanticValidationException) as excinfo:
            await runtime.select(
                from_concept="FlightAlert",
                fields=["FlightAlert.id"],
                where="FlightAlert.status = 'unresolved'",
            )
        payload = excinfo.value.payload
        assert payload.code == "value_not_in_domain"
        assert payload.detail["violations"][0]["allowed"] == ["OPEN", "RESOLVED"]
        assert payload.candidates == ["OPEN", "RESOLVED"]

    @pytest.mark.asyncio
    async def test_concept_time_dimension_receives_the_grain(self, tmp_path, ontology_binding):
        runtime = _runtime(*_layout(tmp_path))
        await runtime.query_metrics(
            metrics=["revenue"],
            dimensions=["Delay.flight.scheduled_departure"],
            time_granularity="day",
        )
        query = FakeEngine.instances[-1].execute_calls[-1]["query"]
        assert query["group_by"] == [{"field": "Delay.flight.scheduled_departure", "grain": "day"}]
        assert "dimension" not in query.get("time_range", {})

    @pytest.mark.asyncio
    async def test_wrong_root_rejection_keeps_candidates_and_hint(self, tmp_path, ontology_binding):
        runtime = _runtime(*_layout(tmp_path))
        engine_error = QueryError(
            'a relationship path pins this query to dataset "flights"',
            code="invalid_relationship_path",
            candidates=["Delay.flight.arrives_at.code"],
            hint="root the group-by on the concept the metric lives on: --group-by Delay.flight.arrives_at.code",
        )
        runtime.has_ontology()  # builds the engine
        engine = FakeEngine.instances[-1]
        engine.fail_with = engine_error
        with pytest.raises(SemanticValidationException) as excinfo:
            await runtime.query_metrics(metrics=["revenue"], dimensions=["Flight.arrives_at.code"])
        payload = excinfo.value.payload
        assert payload.code == "invalid_relationship_path"
        assert payload.candidates == ["Delay.flight.arrives_at.code"]
        assert "Delay.flight.arrives_at.code" in payload.hint
