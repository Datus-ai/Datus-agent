"""Real query_metrics with owned ORC fixtures and an explicit HS2 profile."""

import csv
import io
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from datus_db_core.testing.hs2 import Environment, compare, render, selected_cases

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.skipif(
        not os.getenv("HS2_TEST_PROFILE"), reason="Select HS2_TEST_PROFILE to run dedicated HS2 contracts"
    ),
]
ROOT = Path(__file__).resolve().parents[3] / "fixtures/hs2"
CASES = selected_cases(ROOT, "metrics")


@pytest.fixture(scope="module")
def hs2_tool(tmp_path_factory):
    from datus_hive import register

    from datus.configuration.agent_config import AgentConfig, NodeConfig
    from datus.tools.func_tool.semantic_tools import SemanticTools

    register()
    environment = Environment(ROOT, "agent-query-metrics")
    try:
        database = environment.ensure("relational")
        connection = {**environment.connection, "database": database, "default": True}
        root = tmp_path_factory.mktemp("hs2_metrics")
        config = AgentConfig(
            nodes={"semantic": NodeConfig(model="mock", input=None)},
            home=str(root / "home"),
            project_name="hs2_contract",
            project_root=str(root / "workspace"),
            target="mock",
            models={
                "mock": {"type": "openai", "api_key": "unused", "model": "unused", "base_url": "http://127.0.0.1:1"}
            },
            services={"datasources": {"hs2_contract": connection}},
        )
        config.current_datasource = "hs2_contract"
        model_dir = Path(config.path_manager.semantic_model_path("hs2_contract"))
        model_dir.mkdir(parents=True, exist_ok=True)
        text = (ROOT / "scenarios/relational/model.yaml").read_text()
        (model_dir / "model.yaml").write_text(render(text, database))
        with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
            tool = SemanticTools(agent_config=config)
        assert type(tool.runtime).__module__ == "datus.tools.semantic_tools.dosi.runtime"
        yield tool, environment
    finally:
        environment.close()


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_query_metrics_contract(hs2_tool, case):
    tool, environment = hs2_tool
    query = case["query"]
    dimensions = [item["field"] for item in query.get("group_by", [])]
    grains = {item["grain"] for item in query.get("group_by", []) if item.get("grain")}
    assert len(grains) <= 1, "Agent API applies one granularity to time dimensions"
    record = {"case": case["id"], "check": "query_metrics_csv", "status": "failed"}
    environment.checks.append(record)
    result = tool.query_metrics(
        metrics=query["metrics"],
        dimensions=dimensions,
        where=query.get("where_sql"),
        time_granularity=next(iter(grains), None),
    )
    assert result.success == 1, result.error
    page = tool.get_query_metrics_result(result.result["result_id"])
    assert page.success == 1, page.error
    rows = list(csv.reader(io.StringIO(page.result["csv"])))
    compare(case["expected"], rows[0], rows[1:], csv_values=True)
    record["status"] = "passed"
