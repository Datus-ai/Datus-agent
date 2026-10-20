# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""The ontology tools against the real Dosi binding on the aviation fixture.

The fixture is the Dosi aviation scenario: flights reach airports through two
roles (origin / destination) and again through their route, alert status and
priority are closed value sets, and delays and alerts are two children of a
flight. Each case below is a question the ontology answers and the model alone
does not. Cases that need ``Engine.describe`` / ``paths`` skip on a binding that
predates them; loading and dataset-spelled queries run on every binding.
"""

from __future__ import annotations

import csv
import io
import shutil
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from datus.configuration.agent_config import AgentConfig, NodeConfig
from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.tools.semantic_tools.dosi.engine import (
    binding_supports_mapping_documents,
    binding_supports_ontology,
)

pytestmark = [pytest.mark.acceptance, pytest.mark.nightly]

FIXTURE_ROOT = Path(__file__).parents[2] / "data" / "semantic_models"
MODEL_PATH = FIXTURE_ROOT / "aviation" / "aviation.yaml"
SEED_PATH = FIXTURE_ROOT / "aviation" / "seed.sql"
ONTOLOGY_PATH = FIXTURE_ROOT / "ontology" / "aviation" / "aviation.yaml"
# The same ontology as an apache/ossie#458 mapping document beside a bare
# ontology; the mapping names the model by `semantic_model_ref.name`.
MAPPING_DOCUMENT_DIR = FIXTURE_ROOT / "ontology" / "aviation-mapping-document"

requires_ontology_binding = pytest.mark.skipif(
    not binding_supports_ontology(),
    reason="installed dosi-engine binding has no Engine.describe/paths",
)


@pytest.fixture(
    params=[
        "embedded",
        pytest.param(
            "mapping_document",
            marks=pytest.mark.skipif(
                not binding_supports_mapping_documents(),
                reason="installed dosi-engine binding cannot load mapping documents",
            ),
        ),
    ]
)
def agent_config(tmp_path, request) -> AgentConfig:
    """Every case runs on both layouts: the deprecated ontology embedding its
    mapping, and a mapping document beside a bare ontology."""
    database = tmp_path / "aviation.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(SEED_PATH.read_text(encoding="utf-8"))

    config = AgentConfig(
        nodes={"semantic": NodeConfig(model="mock", input=None)},
        home=str(tmp_path / "home"),
        project_name="dosi_ontology_acceptance",
        project_root=str(tmp_path / "workspace"),
        target="mock",
        models={
            "mock": {
                "type": "openai",
                "api_key": "unused",
                "model": "unused",
                "base_url": "http://127.0.0.1:1",
            }
        },
        services={"datasources": {"aviation": {"type": "duckdb", "uri": str(database), "default": True}}},
    )
    config.current_datasource = "aviation"
    shutil.copy(MODEL_PATH, config.path_manager.semantic_model_path("aviation") / "aviation.yaml")
    ontology_dir = config.path_manager.ontology_path("aviation")
    ontology_dir.mkdir(parents=True)
    if request.param == "embedded":
        shutil.copy(ONTOLOGY_PATH, ontology_dir / "aviation.yaml")
    else:
        for name in ("ontology.yaml", "mapping.yaml"):
            shutil.copy(MAPPING_DOCUMENT_DIR / name, ontology_dir / name)
    return config


def _tools(config: AgentConfig, **kwargs) -> SemanticTools:
    # Ontology and metric queries do not use MetricRAG; replacing that storage
    # dependency keeps the suite on the real binding and SQL execution.
    with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
        return SemanticTools(agent_config=config, **kwargs)


@pytest.fixture
def ontology_tools(agent_config) -> SemanticTools:
    tool = _tools(agent_config, ontology_enabled=True)
    assert type(tool.runtime).__module__ == "datus.tools.semantic_tools.dosi.runtime"
    return tool


def _rows(tool: SemanticTools, result) -> list[dict[str, str]]:
    assert result.success == 1, result.error
    cached = tool.get_cached_query_metrics_result(result.result["result_id"])
    return list(csv.DictReader(io.StringIO(cached["csv"])))


class TestLoading:
    def test_ontology_loads_beside_the_model(self, ontology_tools):
        runtime = ontology_tools.runtime
        assert runtime.has_ontology()
        [graph] = runtime_lineage(runtime)
        assert graph["ontology"]["name"] == "Aviation"

    def test_other_agents_do_not_load_the_ontology(self, agent_config):
        runtime = _tools(agent_config).runtime
        assert not runtime.has_ontology()
        [graph] = runtime_lineage(runtime)
        assert "ontology" not in graph or not graph["ontology"]

    def test_dataset_spelled_metric_query_is_unchanged(self, ontology_tools):
        result = ontology_tools.query_metrics(metrics=["total_delay_minutes"], dimensions=["carriers.iata_code"])
        rows = {row["iata_code"]: int(row["total_delay_minutes"]) for row in _rows(ontology_tools, result)}
        assert rows == {"SK": 264, "UA": 84}


def runtime_lineage(runtime):
    from datus.utils.async_utils import run_async

    return run_async(runtime.lineage_graph())


@requires_ontology_binding
class TestOntologyAnswers:
    def test_catalog_names_the_concepts(self, ontology_tools):
        result = ontology_tools.describe_concept()
        assert result.success == 1, result.error
        names = {row["name"] for row in result.result["concepts"]}
        assert {"Airport", "Flight", "FlightAlert", "Delay"} <= names

    def test_concept_detail_carries_closed_value_sets(self, ontology_tools):
        result = ontology_tools.describe_concept("FlightAlert")
        assert result.success == 1, result.error
        domains = {prop["path"]: prop.get("domain") for prop in result.result["properties"]}
        assert domains["FlightAlert.status"] == ["OPEN", "RESOLVED"]
        assert domains["FlightAlert.priority"] == ["P0", "P1", "P2", "P3"]

    def test_unknown_concept_is_a_structured_rejection(self, ontology_tools):
        result = ontology_tools.describe_concept("Spaceship")
        assert result.success == 0
        assert result.result["code"] == "unknown_concept"

    def test_four_paths_to_an_airport_are_two_meanings(self, ontology_tools):
        result = ontology_tools.find_concept_paths("Flight", "Airport")
        assert result.success == 1, result.error
        assert {path["path"] for path in result.result["paths"]} == {
            "Flight.arrives_at",
            "Flight.departs_from",
            "Flight.follows_route.destination",
            "Flight.follows_route.origin",
        }
        assert len(result.result["equivalences"]) == 2

    def test_a_guessed_literal_is_refused_with_the_allowed_values(self, ontology_tools):
        result = ontology_tools.select_records(
            from_concept="FlightAlert",
            fields=["FlightAlert.id"],
            where="FlightAlert.status = 'unresolved'",
        )
        assert result.success == 0
        assert result.result["code"] == "value_not_in_domain"
        [violation] = result.result["detail"]["violations"]
        assert violation["allowed"] == ["OPEN", "RESOLVED"]

    def test_records_two_hops_out_by_role(self, ontology_tools):
        result = ontology_tools.select_records(
            from_concept="FlightAlert",
            fields=["FlightAlert.id", "FlightAlert.flight.number", "FlightAlert.flight.arrives_at.name"],
            where=(
                "FlightAlert.status = 'OPEN' AND FlightAlert.priority = 'P1' "
                "AND FlightAlert.flight.arrives_at.code = 'JFK'"
            ),
        )
        assert result.result["columns"] == [
            "FlightAlert.id",
            "FlightAlert.flight.number",
            "FlightAlert.flight.arrives_at.name",
        ]
        rows = _rows(ontology_tools, result)
        assert {row["FlightAlert.id"] for row in rows} == {"A007", "A011"}

    def test_records_order_by_a_requested_concept_path(self, ontology_tools):
        result = ontology_tools.select_records(
            from_concept="FlightAlert",
            fields=["FlightAlert.id", "FlightAlert.priority"],
            where="FlightAlert.status = 'OPEN'",
            order_by=["-FlightAlert.id"],
        )
        rows = _rows(ontology_tools, result)
        ids = [row["FlightAlert.id"] for row in rows]
        assert ids == sorted(ids, reverse=True)
        assert len(ids) == 8

    def test_metrics_order_by_a_requested_concept_path(self, ontology_tools):
        result = ontology_tools.query_metrics(
            metrics=["delayed_flights"],
            dimensions=["Delay.flight.arrives_at.code"],
            order_by=["-Delay.flight.arrives_at.code"],
        )
        codes = [row["Delay.flight.arrives_at.code"] for row in _rows(ontology_tools, result)]
        assert codes == ["SFO", "SEA", "LAX", "JFK"]

    def test_a_metric_grouped_from_the_wrong_root_names_the_fix(self, ontology_tools):
        result = ontology_tools.query_metrics(metrics=["delayed_flights"], dimensions=["Flight.arrives_at.code"])
        assert result.success == 0
        assert result.result["code"] == "invalid_relationship_path"
        assert "Delay.flight.arrives_at.code" in result.result["candidates"]

    def test_the_role_changes_the_answer(self, ontology_tools):
        def by_airport(path):
            result = ontology_tools.query_metrics(metrics=["delayed_flights"], dimensions=[path])
            rows = _rows(ontology_tools, result)
            return {row[path]: int(row["delayed_flights"]) for row in rows}

        assert by_airport("Delay.flight.arrives_at.code") == {"LAX": 3, "JFK": 2, "SFO": 2, "SEA": 1}
        assert by_airport("Delay.flight.departs_from.code") == {"SFO": 6, "LAS": 1, "JFK": 1}


# ---------------------------------------------------------------- one-file layout

MINI_INLINE_DIR = FIXTURE_ROOT / "ontology" / "mini-flights-inline"


@pytest.fixture
def inline_tools(tmp_path) -> SemanticTools:
    """A datasource whose only model is the one inline in a mapping document:
    `semantic_models/mini/` holds nothing, the ontology dir one file."""
    database = tmp_path / "mini.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute((MINI_INLINE_DIR / "seed.sql").read_text(encoding="utf-8"))
    config = AgentConfig(
        nodes={"semantic": NodeConfig(model="mock", input=None)},
        home=str(tmp_path / "home"),
        project_name="dosi_inline_mapping",
        project_root=str(tmp_path / "workspace"),
        target="mock",
        models={"mock": {"type": "openai", "api_key": "unused", "model": "unused", "base_url": "http://127.0.0.1:1"}},
        services={"datasources": {"mini": {"type": "duckdb", "uri": str(database), "default": True}}},
    )
    config.current_datasource = "mini"
    ontology_dir = config.path_manager.ontology_path("mini")
    ontology_dir.mkdir(parents=True)
    shutil.copy(MINI_INLINE_DIR / "mapping.yaml", ontology_dir / "mapping.yaml")
    return _tools(config, ontology_enabled=True)


@requires_ontology_binding
@pytest.mark.skipif(
    not binding_supports_mapping_documents(),
    reason="installed dosi-engine binding cannot load mapping documents",
)
class TestInlineMapping:
    def test_the_mapping_alone_is_the_ontology_and_the_model(self, inline_tools):
        runtime = inline_tools.runtime
        assert runtime.has_ontology()
        [info] = runtime.ontology_info()
        assert info["name"] == "MiniFlights"
        assert info["mappings"] == ["mini_flights_mapping"]

    def test_its_metrics_answer_by_concept_path(self, inline_tools):
        result = inline_tools.query_metrics(
            metrics=["airport_average_departure_delay"], dimensions=["Flight.departs_from.code"]
        )
        rows = _rows(inline_tools, result)
        by_origin = {row["Flight.departs_from.code"]: row["airport_average_departure_delay"] for row in rows}
        assert {code: float(value) for code, value in by_origin.items()} == {
            "ATL": 16,
            "DCA": 20,
            "ORD": 30,
            "SFO": 32,
        }
