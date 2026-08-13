# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.tools.func_tool.generation_evidence."""

from datus.tools.func_tool.generation_evidence import (
    GenerationEvidence,
    _metadata_from_result,
    _result_payload,
    _result_success,
)


class TestResultSuccess:
    def test_dict_success_1(self):
        assert _result_success({"success": 1}) is True

    def test_dict_success_true(self):
        assert _result_success({"success": True}) is True

    def test_dict_success_0(self):
        assert _result_success({"success": 0}) is False

    def test_dict_no_success_key(self):
        assert _result_success({}) is False

    def test_object_with_success_attr(self):
        class Obj:
            success = 1

        assert _result_success(Obj()) is True

    def test_object_with_success_false(self):
        class Obj:
            success = False

        assert _result_success(Obj()) is False

    def test_plain_value_returns_false(self):
        assert _result_success(None) is False
        assert _result_success("str") is False
        assert _result_success(42) is False


class TestResultPayload:
    def test_dict_returns_result_key(self):
        assert _result_payload({"result": "payload"}) == "payload"

    def test_dict_missing_result_key_returns_none(self):
        assert _result_payload({}) is None

    def test_object_with_result_attr(self):
        class Obj:
            result = "attr_payload"

        assert _result_payload(Obj()) == "attr_payload"

    def test_plain_returns_none(self):
        assert _result_payload(None) is None


class TestMetadataFromResult:
    def test_extracts_metadata_from_dict_result(self):
        result = {"result": {"metadata": {"sql": "SELECT 1"}}}
        assert _metadata_from_result(result) == {"sql": "SELECT 1"}

    def test_non_dict_metadata_returns_empty(self):
        result = {"result": {"metadata": "not a dict"}}
        assert _metadata_from_result(result) == {}

    def test_no_metadata_key_returns_empty(self):
        result = {"result": {}}
        assert _metadata_from_result(result) == {}

    def test_object_payload_with_metadata_attr(self):
        class Payload:
            metadata = {"key": "value"}

        class Obj:
            result = Payload()

        assert _metadata_from_result(Obj()) == {"key": "value"}

    def test_non_dict_object_metadata_returns_empty(self):
        class Payload:
            metadata = "not a dict"

        class Obj:
            result = Payload()

        assert _metadata_from_result(Obj()) == {}


class TestGenerationEvidence:
    def test_initial_state(self):
        ev = GenerationEvidence()
        assert ev.validation_passed is False
        assert ev.metric_sqls == {}
        assert ev.kb_sync_passed is False

    def test_kb_sync_passed_when_any_kind_set(self):
        ev = GenerationEvidence()
        ev.mark_kb_sync("metric", ["revenue"])
        assert ev.kb_sync_passed is True
        assert ev.metric_kb_sync_passed is True
        assert ev.has_metric_kb_sync(["revenue"])
        assert not ev.has_metric_kb_sync(["revenue", "orders"])
        assert not ev.has_metric_kb_sync()

        ev.mark_kb_sync("metric", ["orders"])
        assert ev.has_metric_kb_sync(["revenue", "orders"])

    def test_kb_sync_semantic(self):
        ev = GenerationEvidence()
        ev.mark_kb_sync("semantic")
        assert ev.semantic_kb_sync_passed is True

    def test_kb_sync_generic(self):
        ev = GenerationEvidence()
        ev.mark_kb_sync()
        assert ev.generic_kb_sync_passed is True

    def test_reset_clears_all_request_evidence_without_replacing_object(self, tmp_path):
        artifact = tmp_path / "sales.yml"
        artifact.write_text("semantic_model: sales\n", encoding="utf-8")
        ev = GenerationEvidence(
            validation_passed=True,
            metric_sqls={"revenue": "select 1"},
            semantic_kb_sync_passed=True,
            metric_kb_sync_passed=True,
            metric_kb_sync_metrics={"revenue"},
            generic_kb_sync_passed=True,
        )
        ev.record_semantic_artifact_validation("sales", artifact)

        ev.reset()

        assert ev == GenerationEvidence()

    def test_artifact_mutation_invalidates_publish_evidence(self, tmp_path):
        artifact = tmp_path / "sales.yml"
        artifact.write_text("semantic_model: sales\n", encoding="utf-8")
        ev = GenerationEvidence(
            validation_passed=True,
            metric_sqls={"revenue": "select 1"},
            semantic_kb_sync_passed=True,
            metric_kb_sync_passed=True,
            metric_kb_sync_metrics={"revenue"},
        )
        ev.record_semantic_artifact_validation("sales", artifact)

        ev.invalidate_artifact_evidence()

        assert ev.validation_passed is False
        assert ev.metric_sqls == {}
        assert ev.validated_semantic_artifacts == {}
        assert ev.kb_sync_passed is False
        assert ev.metric_kb_sync_metrics == set()

    def test_record_validation_result_success(self):
        ev = GenerationEvidence()
        ev.record_validation_result({"success": 1, "result": {"valid": True}})
        assert ev.validation_passed is True

    def test_record_validation_result_not_valid(self):
        ev = GenerationEvidence()
        ev.record_validation_result({"success": 1, "result": {"valid": False}})
        assert ev.validation_passed is False

    def test_record_validation_result_failure(self):
        ev = GenerationEvidence()
        ev.record_validation_result({"success": 0, "result": {"valid": True}})
        assert ev.validation_passed is False

    def test_semantic_validation_is_bound_to_model_and_file_content(self, tmp_path):
        artifact = tmp_path / "sales.yml"
        artifact.write_text("semantic_model: sales\n", encoding="utf-8")
        ev = GenerationEvidence()
        ev.record_validation_result(
            {
                "success": 1,
                "result": {
                    "valid": True,
                    "semantic_model_name": "sales",
                    "semantic_model_file": str(artifact),
                },
            }
        )

        assert ev.semantic_artifact_validation_passed("sales", artifact)
        assert not ev.semantic_artifact_validation_passed("finance", artifact)

        artifact.write_text("semantic_model: changed\n", encoding="utf-8")

        assert not ev.semantic_artifact_validation_passed("sales", artifact)

    def test_full_validation_satisfies_both_final_scopes(self, tmp_path):
        artifact = tmp_path / "sales.yml"
        artifact.write_text("semantic_model: sales\n", encoding="utf-8")
        ev = GenerationEvidence()
        ev.record_semantic_artifact_validation("sales", artifact, validation_scope="all")

        assert ev.semantic_artifact_validation_passed("sales", artifact, required_scope="all")
        assert ev.semantic_artifact_validation_passed("sales", artifact, required_scope="semantic_model")

    def test_explicit_validation_checks_do_not_satisfy_publish_gate(self, tmp_path):
        artifact = tmp_path / "sales.yml"
        artifact.write_text("semantic_model: sales\n", encoding="utf-8")
        ev = GenerationEvidence()

        ev.record_validation_result(
            {
                "success": 1,
                "result": {
                    "valid": True,
                    "checks": ["authoring_quality"],
                    "semantic_model_name": "sales",
                    "semantic_model_file": str(artifact),
                },
            }
        )

        assert ev.validation_passed is False
        assert not ev.semantic_artifact_validation_passed("sales", artifact)

    def test_record_metric_dry_run_failure_ignored(self):
        ev = GenerationEvidence()
        ev.record_metric_dry_run(["revenue_total"], {"success": 0})
        assert ev.metric_sqls == {}

    def test_record_metric_dry_run_stores_sql_from_metadata(self):
        ev = GenerationEvidence()
        result = {"success": 1, "result": {"metadata": {"sql": "SELECT SUM(revenue)"}}}
        ev.record_metric_dry_run(["revenue_total"], result)
        assert ev.metric_sqls["revenue_total"] == "SELECT SUM(revenue)"

    def test_record_metric_dry_run_stores_metric_sqls_dict(self):
        ev = GenerationEvidence()
        metric_sqls = {
            "__query_metrics_dry_run__": "SELECT combined",
            "revenue_total": "SELECT revenue",
        }
        result = {"success": 1, "result": {"metadata": {"metric_sqls": metric_sqls}}}
        ev.record_metric_dry_run(["revenue_total"], result)
        assert ev.metric_sqls == metric_sqls

    def test_record_metric_dry_run_multi_metric_combined_sql_key(self):
        ev = GenerationEvidence()
        result = {"success": 1, "result": {"metadata": {"sql": "SELECT 1"}}}
        ev.record_metric_dry_run(["m1", "m2"], result)
        # more than one metric -> stored under combined key
        assert "__query_metrics_dry_run__" in ev.metric_sqls

    def test_record_metric_dry_run_single_metric_uses_name_as_key(self):
        ev = GenerationEvidence()
        result = {"success": 1, "result": {"metadata": {"sql": "SELECT 1"}}}
        ev.record_metric_dry_run(["revenue_total"], result)
        assert "revenue_total" in ev.metric_sqls
