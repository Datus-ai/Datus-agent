"""Progressive references preserve the native contract and core structure."""

import copy
import re
from importlib import resources

import pytest
import yaml

from datus.agent.node.semantic_authoring import prepare_authoring_references
from datus.tools.semantic_tools.dosi import authoring_spec, engine


def _advanced_model(dialect):
    skill = resources.files("datus.resources.skills").joinpath("dosi-semantic-authoring/SKILL.md").read_text()
    example = re.findall(r"```yaml\n(.*?)```", skill, re.S)[0]
    return example.replace("<osi_dialect>", dialect).replace(
        "<datus_extension_version>", engine.datus_extension_version()
    )


@pytest.mark.parametrize("dialect", ["STARROCKS", "DUCKDB"])
def test_complete_skill_patterns_compile_with_native_engine(dialect):
    binding = engine.load_binding()
    model = _advanced_model(dialect)
    validated = binding.validate(model)
    assert validated["valid"], validated
    compiled = binding.Engine(model_text=model).compile(
        {
            "metrics": ["average_order_value", "revenue_rank", "market_count", "revenue_score"],
            "group_by": [{"field": "metric_time", "grain": "month"}, {"field": "orders.market"}],
        },
        dialect=dialect,
    )
    assert "sql" in compiled
    if dialect == "DUCKDB":
        import duckdb

        with duckdb.connect() as db:
            db.execute("CREATE SCHEMA analytics")
            db.execute(
                "CREATE TABLE analytics.orders AS SELECT '2024-01' AS month, "
                "'market' || CAST(i AS VARCHAR) AS market, i AS order_id, (9-i)*10 AS amount "
                "FROM range(1,9) t(i)"
            )
            rows = [dict(zip([c[0] for c in db.description], row)) for row in db.execute(compiled["sql"]).fetchall()]
        by_market = {r["market"]: r for r in rows}
        for i in range(1, 9):
            row = by_market[f"market{i}"]
            assert row["average_order_value"] == (9 - i) * 10
            assert row["revenue_rank"] == i
            assert row["market_count"] == 8
            assert row["revenue_score"] == (90 if i <= 3 else 0 if i >= 6 else 40)


def test_native_authoring_errors_explain_complete_object_repairs():
    from datus.tools.semantic_tools.dosi.authoring import dosi_validation_text_payload

    document = yaml.safe_load(_advanced_model("STARROCKS"))
    metrics = document["semantic_model"][0]["metrics"]
    metrics[2]["expression"]["dialects"][0]["expression"] = "SUM(orders.amount)"
    result = dosi_validation_text_payload(yaml.safe_dump(document))
    mismatch = next(e for e in result["compile_errors"] if e["code"] == "derive_expression_mismatch")
    assert "does not generate" in mismatch["hint"]
    assert "rejected batch" in mismatch["hint"]
    from datus.tools.semantic_tools.dosi.model import validation_messages

    assert "metric 'average_order_value'" in "; ".join(validation_messages(result))

    metrics[2] = yaml.safe_load(_advanced_model("STARROCKS"))["semantic_model"][0]["metrics"][2]
    metrics[3]["expression"]["dialects"][0]["expression"] = "RANK() OVER (ORDER BY SUM(orders.amount))"
    result = dosi_validation_text_payload(yaml.safe_dump(document))
    window = next(e for e in result["compile_errors"] if e["code"] == "window_in_metric")
    assert "capability-window.yaml" in window["hint"]


@pytest.fixture
def contract(monkeypatch):
    value = {
        "extension_version": "7.3",
        "envelope": {"requires": {"current_silent_keys": ["params"]}},
        "capabilities": [
            {"key": key, "extension": extension, "carriers": ["metric"], "since": "7.3"}
            for key, extension in [("window", "D-WINDOW"), ("params", "D-PARAM")]
        ],
        "carriers": {
            "metric": {"window": {"schema_ref": "definitions.window"}, "params": {"type": "array"}},
        },
        "definitions": {
            "window": {"shape": {"partition": {"schema_ref": "definitions.partition"}}},
            "partition": {"mode": {"enum": ["query_dimensions"]}, "rules": ["Keep the complete population."]},
        },
        "authoring_guidance": [
            {"capability": "D-WINDOW", "use_when": "Rank a population."},
            {"capability": "D-PARAM", "use_when": "Vary a bounded input."},
        ],
        "examples": {"rolling": {"v": "7.3", "window": {"count": 3}, "params": [{"name": "n"}]}},
    }
    monkeypatch.setattr(engine, "datus_authoring_contract", lambda: copy.deepcopy(value))
    monkeypatch.setattr(engine, "datus_extension_version", lambda: "7.3")
    monkeypatch.setattr(authoring_spec, "datus_extension_authoring_spec_digest", lambda: "sha256:contract")
    monkeypatch.setattr(
        authoring_spec, "datus_extension_authoring_spec_text", lambda: yaml.safe_dump(value, sort_keys=False)
    )
    return value


def test_compact_core_preserves_every_shape_and_enum():
    assert list(yaml.safe_load_all(authoring_spec.compact_authoring_spec_text("STARROCKS"))) == list(
        document
        for document in yaml.safe_load_all(authoring_spec.authoring_spec_text("STARROCKS"))
        if document is not None
    )


def test_directory_covers_every_capability_and_points_to_native_rules(contract):
    overview = yaml.safe_load(authoring_spec.authoring_overview_text())
    documents = authoring_spec.authoring_reference_documents()
    assert {row["key"] for row in overview["capabilities"]} == {row["key"] for row in contract["capabilities"]}
    assert overview["carriers"] == contract["carriers"]
    assert overview["envelope"] == contract["envelope"]
    assert overview["authoring_guidance"] == contract["authoring_guidance"]
    for row in overview["capabilities"]:
        details = yaml.safe_load(documents[row["reference"]])
        assert details["capabilities"] == [c for c in contract["capabilities"] if c["key"] == row["key"]]
        assert details["carriers"]["metric"][row["key"]] == contract["carriers"]["metric"][row["key"]]
        assert details["examples"] == contract["examples"]
        assert details["contract_digest"] == "sha256:contract"
        assert all(name in documents for name in details["related_references"].values())
    window = yaml.safe_load(documents["capability-window.yaml"])
    assert window["definitions"] == contract["definitions"]
    assert yaml.safe_load(documents["datus.yaml"]) == contract
    assert documents["osi-core.yaml"] == authoring_spec.authoring_spec_text("<osi_dialect>")


@pytest.mark.parametrize("reference", ["window", 7])
def test_invalid_schema_reference_returns_contract_error(contract, reference):
    from datus.tools.semantic_tools.exceptions import SemanticCoreException

    contract["carriers"]["metric"]["window"]["schema_ref"] = reference
    with pytest.raises(SemanticCoreException, match="Unresolved authoring schema reference"):
        authoring_spec.authoring_reference_documents()


def test_native_validation_retains_non_object_compile_diagnostics(monkeypatch):
    from types import SimpleNamespace

    from datus.tools.semantic_tools.dosi import authoring

    errors = ["Native diagnostic", {"code": "not_an_aggregate", "message": "Aggregate required"}]
    monkeypatch.setattr(
        authoring, "load_binding", lambda: SimpleNamespace(validate=lambda _: {"compile_errors": errors})
    )
    result = authoring.dosi_validation_text_payload("semantic_model: []")
    assert result["compile_errors"][0] == "Native diagnostic"
    assert "aggregate SQL" in result["compile_errors"][1]["hint"]


def test_reference_cache_reuses_content_and_refreshes_same_version_changes(contract, tmp_path):
    first = prepare_authoring_references(tmp_path)
    before = {p.name: p.stat().st_mtime_ns for p in first.iterdir()}
    assert prepare_authoring_references(tmp_path) == first
    assert {p.name: p.stat().st_mtime_ns for p in first.iterdir()} == before

    contract["definitions"]["partition"]["rules"].append("Preserve tied values.")
    second = prepare_authoring_references(tmp_path)
    assert second != first
    assert yaml.safe_load((second / "capability-window.yaml").read_text())["definitions"] == contract["definitions"]
    assert {p.name: p.stat().st_mtime_ns for p in first.iterdir()} == before


@pytest.mark.parametrize("authoring_scope", ["full", "datasets"])
def test_modeling_reads_generated_references_in_strict_mode_and_resumes(
    contract, real_agent_config, mock_llm_create, authoring_scope
):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

    def new_node(session_id):
        node = SemanticModelingAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
            session_id=session_id,
            authoring_scope=authoring_scope,
        )
        node.input = SemanticNodeInput(user_message="Model order revenue", authoring_scope=authoring_scope)
        return node

    node = new_node("authoring_references")
    filesystem = node._make_filesystem_tool(strict=True)
    context = node._prepare_template_context(node.input)
    prompt = node._get_session_system_prompt(template_context=context)
    data_dir = real_agent_config.path_manager.session_data_dir(node.session_id)
    reference = next(data_dir.rglob("capability-window.yaml"))
    assert str(reference.parent) in prompt
    assert yaml.safe_load(filesystem.read_file(str(reference)).result)["definitions"] == contract["definitions"]
    assert filesystem._classify(str(reference)).read_only
    assert not filesystem.write_file(str(reference), "changed").success
    assert reference.is_relative_to(node._make_filesystem_policy().session_data_dir)
    other = new_node("another_session")
    assert not other._make_filesystem_tool(strict=True).read_file(str(reference)).success

    # A resumed prompt must retain readable references even after cache cleanup.
    reference.unlink()
    resumed = new_node(node.session_id)
    assert resumed._get_session_system_prompt(template_context=context) == prompt
    assert (
        yaml.safe_load(resumed.filesystem_func_tool.read_file(str(reference)).result)["definitions"]
        == contract["definitions"]
    )

    contract["definitions"]["partition"]["rules"].append("Preserve tied values.")
    updated = new_node(node.session_id)
    new_prompt = updated._get_session_system_prompt(template_context=context)
    assert new_prompt != prompt
    current = prepare_authoring_references(data_dir / "semantic_authoring") / "capability-window.yaml"
    assert str(current.parent) in new_prompt
    assert (
        yaml.safe_load(updated.filesystem_func_tool.read_file(str(current)).result)["definitions"]
        == contract["definitions"]
    )
