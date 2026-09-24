# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the optional planning skill's narrow plan-file writer."""

import json
import re

import pytest

from datus.tools.func_tool.semantic_plan_file_tools import SemanticPlanFileTools


def test_writes_reviewable_plan_without_semantic_model_mutation(tmp_path):
    tool = SemanticPlanFileTools(tmp_path)
    plan = {
        "summary": "orders -> revenue",
        "datasets": [{"name": "orders", "action": "create", "source_kind": "physical"}],
        "field_decisions": [],
        "relationships": [],
        "metrics": [{"name": "revenue", "action": "create", "kind": "atomic", "role": "business_output"}],
    }

    result = tool.write_semantic_model_plan_file("sales_model", json.dumps(plan))

    assert result.success == 1
    assert re.fullmatch(
        r"\.datus/semantic-model-plans/[0-9a-f]{32}/semantic-model-plan\.json",
        result.result["path"],
    )
    saved = tmp_path / result.result["path"]
    assert json.loads(saved.read_text()) == plan
    assert not (tmp_path / "subject").exists()


def test_rejects_unsafe_name_and_invalid_json_without_overwriting(tmp_path):
    tool = SemanticPlanFileTools(tmp_path)
    good = tool.write_semantic_model_plan_file("sales_model", '{"summary":"first"}')
    saved = tmp_path / good.result["path"]

    assert tool.write_semantic_model_plan_file("../subject", "{}").success == 0
    assert tool.write_semantic_model_plan_file("sales_model", "not json").success == 0
    assert tool.write_semantic_model_plan_file("sales_model", "[]").success == 0
    assert json.loads(saved.read_text()) == {"summary": "first"}


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_rejects_non_finite_numbers_without_overwriting(tmp_path, number):
    tool = SemanticPlanFileTools(tmp_path)
    good = tool.write_semantic_model_plan_file("sales_model", '{"summary":"first"}')
    saved = tmp_path / good.result["path"]

    result = tool.write_semantic_model_plan_file("sales_model", f'{{"score":{number}}}')

    assert result.success == 0
    assert "valid JSON" in result.error
    assert json.loads(saved.read_text()) == {"summary": "first"}


def test_revisions_reuse_plan_id_but_different_models_do_not(tmp_path):
    tool = SemanticPlanFileTools(tmp_path)
    first = tool.write_semantic_model_plan_file("sales_model", '{"summary":"first"}')
    revision = tool.write_semantic_model_plan_file("sales_model", '{"summary":"revised"}')
    other = tool.write_semantic_model_plan_file("inventory_model", '{"summary":"other"}')

    assert revision.result["path"] == first.result["path"]
    assert other.result["path"] != first.result["path"]
    assert json.loads((tmp_path / first.result["path"]).read_text()) == {"summary": "revised"}
    assert json.loads((tmp_path / other.result["path"]).read_text()) == {"summary": "other"}


def test_rejects_symlinked_plan_directory_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    plan_parent = workspace / ".datus"
    plan_parent.mkdir(parents=True)
    outside.mkdir()
    (plan_parent / "semantic-model-plans").symlink_to(outside, target_is_directory=True)

    result = SemanticPlanFileTools(workspace).write_semantic_model_plan_file("sales_model", "{}")

    assert result.success == 0
    assert not (outside / "sales_model").exists()
