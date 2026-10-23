# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``query_rows`` against the real Dosi binding on the aviation fixture.

Flights reach carriers many-to-one, alerts reach carriers through flights,
airports are reached from a flight two ways (origin and destination), and a
carrier has many flights. Each expected row set is read straight from the
seeded DuckDB file, so the tool is checked against the data, not against
itself.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from datus.configuration.agent_config import AgentConfig, NodeConfig
from datus.tools.func_tool.semantic_tools import SemanticTools

pytestmark = [pytest.mark.acceptance, pytest.mark.nightly]

FIXTURE_DIR = Path(__file__).parents[2] / "data" / "semantic_models" / "aviation"
MODEL_PATH = FIXTURE_DIR / "aviation.yaml"
SEED_PATH = FIXTURE_DIR / "seed.sql"


@pytest.fixture
def database(tmp_path) -> Path:
    path = tmp_path / "aviation.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute(SEED_PATH.read_text(encoding="utf-8"))
    return path


@pytest.fixture
def rows_tool(tmp_path, database, monkeypatch) -> SemanticTools:
    config = AgentConfig(
        nodes={"semantic": NodeConfig(model="mock", input=None)},
        home=str(tmp_path / "home"),
        project_name="dosi_query_rows_acceptance",
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
    model_dir = Path(config.project_root) / "subject" / "semantic_models" / "aviation"
    model_dir.mkdir(parents=True)
    (model_dir / "aviation.yaml").write_text(MODEL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(config.path_manager, "semantic_model_path", lambda datasource: model_dir)

    # Row queries do not use MetricRAG; replacing it keeps the suite on the
    # real engine binding, SQL execution and the tool contract.
    with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
        tool = SemanticTools(agent_config=config)
    assert type(tool.runtime).__module__ == "datus.tools.semantic_tools.dosi.runtime"
    return tool


def _rows(tool: SemanticTools, result) -> list[dict[str, str]]:
    """The full rows behind a successful result, read the way an agent pages them."""
    assert result.success == 1, result.error
    page = tool.get_query_metrics_result(result.result["result_id"])
    assert page.success == 1, page.error
    return list(csv.DictReader(io.StringIO(page.result["csv"])))


def _sql(database: Path, sql: str) -> list[tuple]:
    with duckdb.connect(str(database), read_only=True) as connection:
        return connection.execute(sql).fetchall()


def test_a_two_hop_lookup_adds_columns_not_rows(rows_tool, database):
    result = rows_tool.query_rows(
        dataset="flight_alerts",
        fields=["flight_alerts.alert_id", "carriers.iata_code"],
        where="flight_alerts.status = 'OPEN'",
    )
    rows = _rows(rows_tool, result)
    assert result.result["columns"] == ["alert_id", "carriers__iata_code"]
    expected = _sql(
        database,
        "SELECT a.alert_id, c.iata_code FROM flight_alerts a JOIN flights f USING (flight_id) "
        "JOIN carriers c USING (carrier_id) WHERE a.status = 'OPEN'",
    )
    assert len(expected) == 8
    assert sorted((row["alert_id"], row["carriers__iata_code"]) for row in rows) == sorted(expected)


def test_distinct_lists_each_value_once(rows_tool, database):
    result = rows_tool.query_rows(
        dataset="flight_alerts",
        fields=["flights.flight_id"],
        where="flight_alerts.status = 'OPEN'",
        distinct=True,
    )
    flights = sorted(row["flights__flight_id"] for row in _rows(rows_tool, result))
    expected = sorted(
        flight_id
        for (flight_id,) in _sql(database, "SELECT DISTINCT flight_id FROM flight_alerts WHERE status = 'OPEN'")
    )
    assert flights == expected
    assert len(flights) == 6


def test_order_by_a_requested_field_with_a_limit(rows_tool, database):
    result = rows_tool.query_rows(
        dataset="flights",
        fields=["flights.flight_id", "flights.passenger_count"],
        order_by=["-flights.passenger_count", "flights.flight_id"],
        limit=3,
    )
    rows = _rows(rows_tool, result)
    expected = _sql(
        database, "SELECT flight_id, passenger_count FROM flights ORDER BY passenger_count DESC, flight_id LIMIT 3"
    )
    assert [(row["flight_id"], int(row["passenger_count"])) for row in rows] == expected
    assert "truncated" not in result.result["metadata"]


def test_an_ambiguous_path_is_refused_and_a_relationship_prefix_picks_one(rows_tool, database):
    refused = rows_tool.query_rows(dataset="flights", fields=["flights.flight_id", "airports.city"])
    assert refused.success == 0
    assert refused.result["code"] == "ambiguous_join_path"
    assert any("flights_arrives_at" in candidate for candidate in refused.result["candidates"])

    result = rows_tool.query_rows(dataset="flights", fields=["flights.flight_id", "flights_arrives_at.city"])
    rows = _rows(rows_tool, result)
    expected = _sql(
        database,
        "SELECT f.flight_id, a.city FROM flights f JOIN airports a ON f.destination_airport_code = a.airport_code",
    )
    assert sorted((row["flight_id"], row["flights_arrives_at__city"]) for row in rows) == sorted(expected)


def test_a_one_to_many_field_is_refused_with_the_finer_dataset(rows_tool):
    result = rows_tool.query_rows(dataset="carriers", fields=["carriers.iata_code", "flights.flight_number"])
    assert result.success == 0
    assert result.result["code"] == "detail_fanout"
    assert "flights" in result.result["hint"]


def test_an_unknown_dataset_lists_the_datasets(rows_tool):
    result = rows_tool.query_rows(dataset="flight", fields=["flight.flight_id"])
    assert result.success == 0
    assert result.result["code"] == "unknown_dataset"
    assert "flights" in result.result["candidates"]
