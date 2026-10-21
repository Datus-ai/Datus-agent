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
    OntologyBinding,
    binding_supports_ontology,
    document_kind,
    pair_ontologies,
    plan_ontologies,
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
    def test_paired_model_loads_with_its_ontology(self, tmp_path, ontology_binding):
        models_dir, ontology_dir = _layout(tmp_path)
        registry = EngineRegistry(DosiConfig(semantic_models_path=str(models_dir), ontology_dir=str(ontology_dir)))
        _, handles = registry.snapshot()
        engine = handles[0][1].get()
        assert engine.ontology_path == str(ontology_dir / "aviation.yaml")
        assert engine.mapping_path is None

    def test_ontologies_are_not_loaded_unless_requested(self, tmp_path, ontology_binding):
        models_dir, _ = _layout(tmp_path)
        _, handles = EngineRegistry(DosiConfig(semantic_models_path=str(models_dir))).snapshot()
        engine = handles[0][1].get()
        assert engine.ontology_path is None
        assert engine.mapping_path is None

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

    def test_ontology_lowering_notes_do_not_reach_the_terminal(self, tmp_path, ontology_binding, monkeypatch):
        import warnings

        models_dir, ontology_dir = _layout(tmp_path)

        class NotingEngine(FakeOntologyEngine):
            def __init__(self, **kwargs):
                warnings.warn("Flight.arrival_delay: `derived_by` rule kept as text", RuntimeWarning, stacklevel=2)
                super().__init__(**kwargs)

        monkeypatch.setattr(ontology_binding, "Engine", NotingEngine)
        registry = EngineRegistry(DosiConfig(semantic_models_path=str(models_dir), ontology_dir=str(ontology_dir)))
        with warnings.catch_warnings(record=True) as escaped:
            warnings.simplefilter("always")
            engine = registry.snapshot()[1][0][1].get()
        assert engine.ontology_path == str(ontology_dir / "aviation.yaml")
        assert [str(w.message) for w in escaped] == []

    def test_ontology_load_failure_names_the_ontology_and_code(self, tmp_path, ontology_binding, monkeypatch):
        models_dir, ontology_dir = _layout(tmp_path)

        class RejectingEngine(FakeOntologyEngine):
            def __init__(self, **kwargs):
                raise ontology_binding.ModelError("pick one with --mapping", code="ontology_mapping_unresolved")

        monkeypatch.setattr(ontology_binding, "Engine", RejectingEngine)
        registry = EngineRegistry(DosiConfig(semantic_models_path=str(models_dir), ontology_dir=str(ontology_dir)))
        with pytest.raises(SemanticCoreException, match=r"with ontology .*aviation\.yaml.*ontology_mapping_unresolved"):
            registry.snapshot()[1][0][1].get()


# ---------------------------------------------------------------- mapping documents


MAPPING_YAML = """version: 0.2.0.dev0
name: aviation_mapping
ontology_ref: {name: Aviation, iri: ./ontology.yaml}
semantic_model_ref: {name: aviation}
concept_mappings: []
"""
NAMED_MODEL_YAML = "version: '0.2.0.dev0'\nsemantic_model:\n  - name: aviation\n    datasets: []\n"
FLAT_MODEL_YAML = "version: 0.2.0.dev0\nname: aviation\ndatasets: []\n"
INLINE_MAPPING_YAML = """version: 0.2.0.dev0
name: mini_mapping
ontology_ref: {name: Mini, document: {version: 0.2.0.dev0, name: Mini, ontology: []}}
semantic_model_ref: {name: mini, document: {version: 0.2.0.dev0, name: mini, datasets: []}}
concept_mappings: []
"""


def _mapping_layout(tmp_path, model_yaml=NAMED_MODEL_YAML, mapping_yaml=MAPPING_YAML):
    """apache/ossie#458: the ontology and a mapping document in the ontology dir."""
    models_dir = tmp_path / "semantic_models" / "flights_db"
    ontology_dir = tmp_path / "semantic_models" / "ontology" / "flights_db"
    models_dir.mkdir(parents=True)
    ontology_dir.mkdir(parents=True)
    (models_dir / "flights.yaml").write_text(model_yaml)
    (ontology_dir / "ontology.yaml").write_text(ONTOLOGY_YAML)
    (ontology_dir / "mapping.yaml").write_text(mapping_yaml)
    return models_dir, ontology_dir


class TestMappingDocuments:
    def test_kinds_are_read_from_structure(self):
        import yaml

        assert document_kind(yaml.safe_load(MAPPING_YAML)) == "mapping"
        assert document_kind(yaml.safe_load(ONTOLOGY_YAML)) == "ontology"
        assert document_kind(yaml.safe_load(NAMED_MODEL_YAML)) == "model"
        assert document_kind(yaml.safe_load(FLAT_MODEL_YAML)) == "model"
        assert document_kind({"cases": []}) == "other"

    @pytest.mark.parametrize("model_yaml", [NAMED_MODEL_YAML, FLAT_MODEL_YAML])
    def test_a_mapping_binds_the_model_its_ref_names(self, tmp_path, model_yaml):
        models_dir, ontology_dir = _mapping_layout(tmp_path, model_yaml=model_yaml)
        model = str(models_dir / "flights.yaml")
        bindings, unpaired = plan_ontologies(
            [model], [str(ontology_dir / f) for f in ("mapping.yaml", "ontology.yaml")]
        )
        # The mapping finds its ontology by iri; the ontology file is consumed,
        # not paired a second time by stem.
        assert bindings == {
            model: OntologyBinding(
                model_file=model,
                mapping_path=str(ontology_dir / "mapping.yaml"),
                references=(str(ontology_dir / "ontology.yaml"),),
            )
        }
        assert unpaired == []

    def test_a_mapping_without_iri_finds_its_ontology_by_name(self, tmp_path):
        mapping = MAPPING_YAML.replace("{name: Aviation, iri: ./ontology.yaml}", "{name: Aviation}")
        models_dir, ontology_dir = _mapping_layout(tmp_path, mapping_yaml=mapping)
        model = str(models_dir / "flights.yaml")
        bindings, _ = plan_ontologies([model], [str(ontology_dir / f) for f in ("mapping.yaml", "ontology.yaml")])
        assert bindings[model].ontology_path == str(ontology_dir / "ontology.yaml")

    def test_a_mapping_naming_no_model_is_reported(self, tmp_path):
        models_dir, ontology_dir = _mapping_layout(
            tmp_path, mapping_yaml=MAPPING_YAML.replace("{name: aviation}", "{name: retail}")
        )
        bindings, unpaired = plan_ontologies([str(models_dir / "flights.yaml")], [str(ontology_dir / "mapping.yaml")])
        assert bindings == {}
        assert unpaired == [str(ontology_dir / "mapping.yaml")]

    def test_two_mappings_onto_one_model_are_both_reported(self, tmp_path):
        models_dir, ontology_dir = _mapping_layout(tmp_path)
        (ontology_dir / "second.yaml").write_text(MAPPING_YAML.replace("aviation_mapping", "second"))
        files = [str(ontology_dir / f) for f in ("mapping.yaml", "ontology.yaml", "second.yaml")]
        bindings, unpaired = plan_ontologies([str(models_dir / "flights.yaml")], files)
        assert bindings == {}
        assert unpaired == sorted([files[0], files[2]])

    @pytest.mark.parametrize(
        "stray_name, stray_text",
        [
            pytest.param("extra_model.yaml", NAMED_MODEL_YAML, id="semantic_model"),
            pytest.param("notes.yaml", "cases: []\n", id="unrelated_document"),
            pytest.param("broken.yaml", "ontology: [unclosed\n", id="unparsable"),
        ],
    )
    def test_a_stray_file_never_becomes_an_ontology(self, tmp_path, stray_name, stray_text):
        # One unbound model and one leftover file would pair regardless of
        # names; only an ontology document may be paired that way.
        models_dir, ontology_dir = _layout(tmp_path, ontologies=())
        ontology_dir.mkdir(parents=True)
        stray = ontology_dir / stray_name
        stray.write_text(stray_text)
        bindings, unpaired = plan_ontologies([str(models_dir / "aviation.yaml")], [str(stray)])
        assert bindings == {}
        assert unpaired == [str(stray)]

    def test_a_model_a_mapping_reaches_by_iri_is_consumed(self, tmp_path):
        # The three-file layout with the model beside the mapping: the engine
        # follows both iris, and neither target is reported or paired again.
        models_dir, ontology_dir = _layout(tmp_path, ontologies=())
        ontology_dir.mkdir(parents=True)
        (ontology_dir / "ontology.yaml").write_text(ONTOLOGY_YAML)
        (ontology_dir / "model.yaml").write_text(NAMED_MODEL_YAML.replace("aviation", "beside"))
        mapping = MAPPING_YAML.replace("{name: aviation}", "{name: beside, iri: ./model.yaml}")
        (ontology_dir / "mapping.yaml").write_text(mapping)
        files = [str(ontology_dir / f) for f in ("mapping.yaml", "model.yaml", "ontology.yaml")]
        bindings, unpaired = plan_ontologies([str(models_dir / "aviation.yaml")], files)
        assert bindings == {
            files[0]: OntologyBinding(model_file=None, mapping_path=files[0], references=(files[2], files[1]))
        }
        assert unpaired == []

    def test_the_engine_gets_the_model_and_the_mapping(self, tmp_path, ontology_binding):
        models_dir, ontology_dir = _mapping_layout(tmp_path)
        registry = EngineRegistry(DosiConfig(semantic_models_path=str(models_dir), ontology_dir=str(ontology_dir)))
        [(key, handle)] = registry.snapshot()[1]
        engine = handle.get()
        assert key == str(models_dir / "flights.yaml")
        assert engine.given_model_path == key
        assert engine.mapping_path == str(ontology_dir / "mapping.yaml")
        assert engine.ontology_path is None
        assert handle.has_ontology

    def test_an_inline_mapping_is_an_engine_of_its_own(self, tmp_path, ontology_binding):
        models_dir, ontology_dir = _layout(tmp_path, ontologies=())
        ontology_dir.mkdir(parents=True)
        (ontology_dir / "mini.yaml").write_text(INLINE_MAPPING_YAML)
        registry = EngineRegistry(DosiConfig(semantic_models_path=str(models_dir), ontology_dir=str(ontology_dir)))
        handles = dict(registry.snapshot()[1])
        mini = str(ontology_dir / "mini.yaml")
        assert list(handles) == [str(models_dir / "aviation.yaml"), mini]
        engine = handles[mini].get()
        # The model is inside the mapping, so the mapping is the only input.
        assert engine.given_model_path is None
        assert engine.mapping_path == mini

    @pytest.mark.parametrize(
        "mapping_yaml, edited",
        [
            pytest.param(MAPPING_YAML, "ontology.yaml", id="ontology_ref"),
            pytest.param(
                MAPPING_YAML.replace("{name: aviation}", "{name: beside, iri: ./model.yaml}"),
                "model.yaml",
                id="semantic_model_ref",
            ),
        ],
    )
    def test_editing_a_file_a_mapping_reaches_by_iri_rebuilds_the_engine(
        self, tmp_path, ontology_binding, mapping_yaml, edited
    ):
        # The engine reads the iri target itself, so its edits must invalidate
        # the engine exactly like an edit to the mapping does.
        models_dir, ontology_dir = _mapping_layout(tmp_path, mapping_yaml=mapping_yaml)
        (ontology_dir / "model.yaml").write_text(NAMED_MODEL_YAML.replace("aviation", "beside"))
        registry = EngineRegistry(DosiConfig(semantic_models_path=str(models_dir), ontology_dir=str(ontology_dir)))
        handle = dict(registry.snapshot()[1])[
            str(models_dir / "flights.yaml") if edited == "ontology.yaml" else str(ontology_dir / "mapping.yaml")
        ]
        first = handle.get()
        target = ontology_dir / edited
        target.write_text(target.read_text() + "# edited\n")
        stat = target.stat()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
        registry.snapshot()
        assert handle.get() is not first

    def test_a_datasource_whose_only_model_is_inline_loads(self, tmp_path, ontology_binding):
        ontology_dir = tmp_path / "semantic_models" / "ontology" / "flights_db"
        ontology_dir.mkdir(parents=True)
        (ontology_dir / "mini.yaml").write_text(INLINE_MAPPING_YAML)
        runtime = _runtime(tmp_path / "semantic_models" / "flights_db", ontology_dir)
        assert runtime.has_ontology() is True
        [info] = runtime.ontology_info()
        assert info["mappings"] == ["aviation_mapping"]
        assert info["model_file"] == str(ontology_dir / "mini.yaml")

    def test_an_older_binding_names_the_upgrade(self, tmp_path, ontology_binding, monkeypatch):
        models_dir, ontology_dir = _mapping_layout(tmp_path)

        class OldEngine(FakeOntologyEngine):
            def __init__(self, *, mapping_path=None, **kwargs):
                if mapping_path is not None:
                    raise TypeError("Engine() got an unexpected keyword argument 'mapping_path'")
                super().__init__(**kwargs)

        monkeypatch.setattr(ontology_binding, "Engine", OldEngine)
        registry = EngineRegistry(DosiConfig(semantic_models_path=str(models_dir), ontology_dir=str(ontology_dir)))
        with pytest.raises(SemanticCoreException, match="mapping document.*upgrade dosi-engine"):
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
            # Ordered by the requested path, which the engine names ``alert_id``.
            "order_by": [{"key": "alert_id", "desc": True}],
            "limit": 10,
        }
        [probe] = FakeEngine.instances[-1].compile_select_calls
        assert "order_by" not in probe["query"] and "limit" not in probe["query"]
        # Output columns carry the requested spelling; the SQL keeps the engine's aliases.
        assert result.columns == ["FlightAlert.id", "FlightAlert.priority"]
        assert result.data == [
            {"FlightAlert.id": "A007", "FlightAlert.priority": "P1"},
            {"FlightAlert.id": "A011", "FlightAlert.priority": "P1"},
        ]
        assert result.metadata["row_count"] == 2
        assert result.metadata["requested_columns"][0] == {"requested": "FlightAlert.id", "column": "alert_id"}

    @pytest.mark.parametrize(
        "result, expected_columns, expected_rows",
        [
            pytest.param(
                {
                    "columns": ["alert_id", "flight_alerts_flight__flight_number", "row_rank"],
                    "rows": [{"alert_id": "A007", "flight_alerts_flight__flight_number": "SK202", "row_rank": 1}],
                    "requested_columns": [
                        {"requested": "FlightAlert.id", "column": "alert_id"},
                        {"requested": "FlightAlert.flight.number", "column": "flight_alerts_flight__flight_number"},
                    ],
                },
                ["FlightAlert.id", "FlightAlert.flight.number", "row_rank"],
                [{"FlightAlert.id": "A007", "FlightAlert.flight.number": "SK202", "row_rank": 1}],
                id="join_path_names_become_requested_paths",
            ),
            pytest.param(
                {
                    "columns": ["alert_id", "FlightAlert.id"],
                    "rows": [{"alert_id": "A007", "FlightAlert.id": "x"}],
                    "requested_columns": [{"requested": "FlightAlert.id", "column": "alert_id"}],
                },
                ["alert_id", "FlightAlert.id"],
                [{"alert_id": "A007", "FlightAlert.id": "x"}],
                id="collision_keeps_engine_names",
            ),
            pytest.param(
                {"columns": ["status", "order_count"], "rows": [{"status": "paid", "order_count": 2}]},
                ["status", "order_count"],
                [{"status": "paid", "order_count": 2}],
                id="no_mapping_is_untouched",
            ),
            pytest.param(
                {
                    "columns": ["alert_id"],
                    "rows": [],
                    "requested_columns": [{"requested": "", "column": "alert_id"}, "malformed"],
                },
                ["alert_id"],
                [],
                id="malformed_entries_are_ignored",
            ),
            pytest.param(
                {
                    "columns": ["iata_code", "airport_code"],
                    "rows": [{"iata_code": "SK", "airport_code": "JFK"}],
                    "requested_columns": [
                        {"requested": "carriers.iata_code", "column": "iata_code"},
                        {"requested": "Delay.flight.arrives_at.code", "column": "airport_code"},
                    ],
                },
                ["iata_code", "Delay.flight.arrives_at.code"],
                [{"iata_code": "SK", "Delay.flight.arrives_at.code": "JFK"}],
                id="dataset_spelling_keeps_engine_name",
            ),
        ],
    )
    def test_output_columns_are_named_by_requested_paths(self, result, expected_columns, expected_rows):
        columns, rows = DosiRuntime._label_requested_columns(result, ["carriers", "flights"])
        assert columns == expected_columns
        assert rows == expected_rows

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "order_by, expected",
        [
            (["priority"], [{"key": "priority", "desc": False}]),
            (["-FlightAlert.created_at"], [{"key": "FlightAlert.created_at", "desc": True}]),
        ],
    )
    async def test_select_order_keys_that_are_not_requested_paths_are_untouched(
        self, tmp_path, ontology_binding, order_by, expected
    ):
        runtime = _runtime(*_layout(tmp_path))
        await runtime.select(from_concept="FlightAlert", fields=["FlightAlert.id"], order_by=order_by)
        engine = FakeEngine.instances[-1]
        assert engine.select_calls[0]["query"]["order_by"] == expected
        assert engine.compile_select_calls == []

    @pytest.mark.asyncio
    async def test_select_order_translation_failure_defers_to_the_real_call(
        self, tmp_path, ontology_binding, monkeypatch
    ):
        runtime = _runtime(*_layout(tmp_path))

        def failing_compile(engine, query, **kwargs):
            raise QueryError("boom", code="unknown_dataset")

        monkeypatch.setattr(FakeOntologyEngine, "compile_select", failing_compile)
        await runtime.select(from_concept="FlightAlert", fields=["FlightAlert.id"], order_by=["FlightAlert.id"])
        query = FakeEngine.instances[-1].select_calls[0]["query"]
        assert query["order_by"] == [{"key": "FlightAlert.id", "desc": False}]

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
