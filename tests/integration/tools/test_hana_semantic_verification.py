"""Read-only HANA regressions through native Dosi and Agent verification tools."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from datus.configuration.agent_config import AgentConfig, NodeConfig
from datus.tools.func_tool import DBFuncTool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.tools.func_tool.semantic_verification_tools import SemanticVerificationTools

pytestmark = [pytest.mark.integration, pytest.mark.nightly]

MODEL = """
version: '0.2.0.dev0'
semantic_model:
- name: hana_native_regression
  datasets:
  - name: sales
    source: DUMMY
    primary_key: [id]
    fields:
    - {name: id, expression: {dialects: [{dialect: HANA, expression: '1'}]}}
    - {name: region, expression: {dialects: [{dialect: HANA, expression: "TO_VARCHAR(DATE '2026-10-01', 'YYYY-MM')"}]}}
    - {name: amount, expression: {dialects: [{dialect: HANA, expression: '100'}]}}
  metrics:
  - name: revenue
    expression: {dialects: [{dialect: HANA, expression: 'SUM(sales.amount)'}]}
  - name: orders
    expression: {dialects: [{dialect: HANA, expression: 'COUNT(sales.id)'}]}
  - name: average_order
    expression: {dialects: [{dialect: HANA, expression: 'SUM(sales.amount) / NULLIF(COUNT(sales.id), 0)'}]}
    custom_extensions:
    - vendor_name: DATUS
      data: '{"v":"1.11","derive":{"type":"compose","expr":"revenue / NULLIF(orders, 0)"}}'
"""


def test_hana_derived_metric_bindings_and_business_verification(tmp_path, monkeypatch):
    if not os.environ.get("HANA_HOST") or not os.environ.get("HANA_USER") or not os.environ.get("HANA_PASSWORD"):
        pytest.skip("HANA_HOST, HANA_USER and HANA_PASSWORD are required")
    testing = pytest.importorskip("datus_hana.testing")
    datasource = testing.config_from_env().model_dump(by_alias=True)
    datasource.update(type="hana", default=True)
    config = AgentConfig(
        nodes={"semantic": NodeConfig(model="mock", input=None)},
        home=str(tmp_path / "home"),
        project_name="hana_native_regression",
        project_root=str(tmp_path / "workspace"),
        target="mock",
        models={"mock": {"type": "openai", "api_key": "unused", "model": "unused", "base_url": "http://127.0.0.1:1"}},
        services={"datasources": {"hana_regression": datasource}},
    )
    config.current_datasource = "hana_regression"
    model_dir = Path(config.project_root) / "subject/semantic_models/hana_regression"
    model_dir.mkdir(parents=True)
    artifact = model_dir / "model.yaml"
    artifact.write_text(MODEL, encoding="utf-8")
    monkeypatch.setattr(config.path_manager, "semantic_model_path", lambda datasource: model_dir)
    with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
        semantic = SemanticTools(agent_config=config)

    queried = semantic.query_metrics(metrics=["average_order"], dimensions=["sales.region"], order_by=["region"])
    assert queried.success, queried.error
    cached = semantic.get_cached_query_metrics_result(queried.result["result_id"])
    assert cached["columns"] == ["region", "average_order"]
    assert queried.result["metadata"]["outputs"][0]["column"] == "average_order"

    db = DBFuncTool(agent_config=config, read_only=True)
    verifier = SemanticVerificationTools(db, semantic, GenerationEvidence(), lambda: artifact)
    result = verifier.verify_semantic_outputs(
        [
            {
                "name": "Native NULLIF average",
                "metric": "average_order",
                "query": {"dimensions": ["sales.region"]},
                "reference_sql": "SELECT TO_VARCHAR(DATE '2026-10-01', 'YYYY-MM') AS REGION, 100 AS TOTAL FROM DUMMY",
                "reference_column": "TOTAL",
                "dimension_mapping": {"REGION": "sales.region"},
            }
        ]
    )
    assert result.success, result.error
    check = result.result["checks"][0]
    assert check["status"] == "passed", check
    assert check["reference_groups"] == check["metric_groups"] == 1
