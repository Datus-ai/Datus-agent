# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.tools.func_tool.generation_evidence."""

from datus.tools.func_tool.generation_evidence import (
    GenerationEvidence,
    _deduplicate_preserve_order,
    _metadata_from_result,
    _normalized_metric_alias_map,
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


class TestDeduplicatePreserveOrder:
    def test_removes_duplicates(self):
        assert _deduplicate_preserve_order(["a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_preserves_order(self):
        assert _deduplicate_preserve_order(["c", "b", "a"]) == ["c", "b", "a"]

    def test_empty(self):
        assert _deduplicate_preserve_order([]) == []


class TestNormalizedMetricAliasMap:
    def test_maps_alias_to_canonical(self):
        result = _normalized_metric_alias_map({"rev_total": "revenue_total"})
        assert result["rev_total"] == "revenue_total"

    def test_also_maps_normalized_alias(self):
        result = _normalized_metric_alias_map({"Rev Total": "revenue_total"})
        assert result.get("rev_total") == "revenue_total"

    def test_skips_non_string_entries(self):
        result = _normalized_metric_alias_map({1: "canonical", "alias": 2})
        assert result == {}

    def test_skips_empty_entries(self):
        result = _normalized_metric_alias_map({"": "canonical", "alias": ""})
        assert result == {}


class TestGenerationEvidence:
    def test_initial_state(self):
        ev = GenerationEvidence()
        assert ev.validation_passed is False
        assert ev.metric_dry_run_passed is False
        assert ev.kb_sync_passed is False
        assert ev.storage_revision == 0

    def test_kb_sync_passed_when_any_kind_set(self):
        ev = GenerationEvidence()
        ev.mark_kb_sync("metric")
        assert ev.kb_sync_passed is True
        assert ev.metric_kb_sync_passed is True
        assert ev.storage_revision == 1

    def test_kb_sync_semantic(self):
        ev = GenerationEvidence()
        ev.mark_kb_sync("semantic")
        assert ev.semantic_kb_sync_passed is True

    def test_kb_sync_generic(self):
        ev = GenerationEvidence()
        ev.mark_kb_sync()
        assert ev.generic_kb_sync_passed is True

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

    def test_record_metric_dry_run_success(self):
        ev = GenerationEvidence()
        result = {"success": 1, "result": {"metadata": {}}}
        ev.record_metric_dry_run(["revenue_total"], result)
        assert ev.metric_dry_run_passed is True
        assert "revenue_total" in ev.metric_dry_run_metrics

    def test_record_metric_dry_run_failure_ignored(self):
        ev = GenerationEvidence()
        ev.record_metric_dry_run(["revenue_total"], {"success": 0})
        assert ev.metric_dry_run_passed is False
        assert "revenue_total" not in ev.metric_dry_run_metrics

    def test_record_metric_dry_run_string_metrics(self):
        ev = GenerationEvidence()
        ev.record_metric_dry_run("revenue_total", {"success": 1, "result": {"metadata": {}}})
        assert "revenue_total" in ev.metric_dry_run_metrics

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
        assert ev.metric_sqls["revenue_total"] == "SELECT revenue"
        assert ev.metric_dry_run_queries[0].get("sql") == "SELECT combined"

    def test_has_metric_dry_run_no_names(self):
        ev = GenerationEvidence()
        ev.metric_dry_run_passed = True
        assert ev.has_metric_dry_run() is True

    def test_has_metric_dry_run_with_names_subset(self):
        ev = GenerationEvidence()
        ev.metric_dry_run_passed = True
        ev.metric_dry_run_metrics = {"a", "b"}
        assert ev.has_metric_dry_run(["a"]) is True
        assert ev.has_metric_dry_run(["a", "c"]) is False

    def test_has_metric_dry_run_not_passed(self):
        ev = GenerationEvidence()
        ev.metric_dry_run_metrics = {"a"}
        assert ev.has_metric_dry_run(["a"]) is False

    def test_set_metric_queryability_contracts_filters_invalid(self):
        ev = GenerationEvidence()
        contracts = [
            {"source": "s1", "dimension_hints": ["col_a"], "metric_hints": ["m1"]},
            {"source": "s2"},  # no dimension_hints or time_group_hints — filtered
        ]
        ev.set_metric_queryability_contracts(contracts)
        assert len(ev.metric_queryability_contracts) == 1
        assert ev.metric_queryability_contracts[0]["source"] == "s1"

    def test_set_metric_queryability_contracts_deduplicates_metric_hints(self):
        ev = GenerationEvidence()
        ev.set_metric_queryability_contracts([{"dimension_hints": ["col_a"], "metric_hints": ["m1", "m1", "m2"]}])
        assert ev.metric_queryability_contracts[0]["metric_hints"] == ["m1", "m2"]

    def test_set_metric_queryability_contracts_applies_alias_rewrites(self):
        ev = GenerationEvidence()
        ev.set_metric_queryability_contracts(
            [{"dimension_hints": ["col_a"], "metric_hints": ["rev_alias"]}],
            metric_aliases={"rev_alias": "revenue_total"},
        )
        hints = ev.metric_queryability_contracts[0]["metric_hints"]
        assert "revenue_total" in hints
        alias_rewrites = ev.metric_queryability_contracts[0].get("metric_alias_rewrites", {})
        assert alias_rewrites.get("rev_alias") == "revenue_total"

    def test_has_required_queryability_dry_runs_no_contracts(self):
        ev = GenerationEvidence()
        assert ev.has_required_queryability_dry_runs() is True

    def test_missing_queryability_contracts_empty_when_satisfied(self):
        ev = GenerationEvidence()
        ev.set_metric_queryability_contracts([{"dimension_hints": ["col_a"], "metric_hints": ["revenue_total"]}])
        ev.record_metric_dry_run(
            ["revenue_total"],
            {"success": 1, "result": {"metadata": {}}},
            dimensions=["col_a"],
        )
        missing = ev.missing_queryability_contracts(["revenue_total"])
        assert missing == []

    def test_missing_queryability_contracts_nonempty_when_unsatisfied(self):
        ev = GenerationEvidence()
        ev.set_metric_queryability_contracts([{"dimension_hints": ["col_a"], "metric_hints": ["revenue_total"]}])
        ev.metric_dry_run_passed = True
        ev.metric_dry_run_metrics.add("revenue_total")
        # No dry_run_queries at all -> contract not matched
        missing = ev.missing_queryability_contracts(["revenue_total"])
        assert len(missing) == 1

    def test_record_metric_dry_run_time_granularity_explicit(self):
        ev = GenerationEvidence()
        result = {"success": 1, "result": {"metadata": {}}}
        ev.record_metric_dry_run(["rev"], result, time_granularity="month")
        query = ev.metric_dry_run_queries[0]
        assert query["time_granularity"] == "month"
        assert query["time_granularity_explicit"] is True

    def test_record_metric_dry_run_time_granularity_from_dimensions(self):
        ev = GenerationEvidence()
        result = {"success": 1, "result": {"metadata": {}}}
        ev.record_metric_dry_run(["rev"], result, dimensions=["metric_time__month"])
        query = ev.metric_dry_run_queries[0]
        assert query["time_granularity"] == "month"
        assert query["time_granularity_explicit"] is False

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
