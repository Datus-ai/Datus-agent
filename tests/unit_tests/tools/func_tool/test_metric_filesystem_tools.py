# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import pytest
import yaml

from datus.tools.func_tool.metric_filesystem_tools import MetricFilesystemFuncTool


class TestMetricFilesystemFuncTool:
    def test_write_file_merges_existing_semantic_model(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ac_manage" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """
data_source:
  name: orders
  sql_table: ac_manage.orders
  measures:
    - name: order_count
      agg: COUNT
      expr: "1"
  dimensions:
    - name: ds
      type: TIME
""".lstrip(),
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ac_manage/orders.yml",
            """
data_source:
  name: orders
  sql_table: ac_manage.orders
  measures:
    - name: paid_order_count
      agg: SUM
      expr: "CASE WHEN status = 'paid' THEN 1 ELSE 0 END"
""".lstrip(),
        )

        assert result.success == 1
        docs = list(yaml.safe_load_all(target.read_text(encoding="utf-8")))
        data_source = docs[0]["data_source"]
        assert [measure["name"] for measure in data_source["measures"]] == [
            "order_count",
            "paid_order_count",
        ]
        assert data_source["dimensions"][0]["name"] == "ds"

    def test_write_file_rejects_conflicting_measure_overwrite(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ac_manage" / "orders.yml"
        target.parent.mkdir(parents=True)
        original = """
data_source:
  name: orders
  sql_table: ac_manage.orders
  measures:
    - name: order_count
      agg: COUNT
      expr: "1"
""".lstrip()
        target.write_text(original, encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ac_manage/orders.yml",
            """
data_source:
  name: orders
  sql_table: ac_manage.orders
  measures:
    - name: order_count
      agg: SUM
      expr: amount
""".lstrip(),
        )

        assert result.success == 0
        assert "Refusing to overwrite measure 'order_count'" in result.error
        assert target.read_text(encoding="utf-8") == original

    def test_write_file_merges_existing_metric_file(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ac_manage" / "metrics" / "orders_metrics.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """
metric:
  name: order_count
  type: measure_proxy
  type_params:
    measure: order_count
""".lstrip(),
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ac_manage/metrics/orders_metrics.yml",
            """
metric:
  name: paid_order_count
  type: measure_proxy
  type_params:
    measure: paid_order_count
""".lstrip(),
        )

        assert result.success == 1
        docs = list(yaml.safe_load_all(target.read_text(encoding="utf-8")))
        assert [doc["metric"]["name"] for doc in docs] == ["order_count", "paid_order_count"]

    def test_write_file_rejects_conflicting_metric_overwrite(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ac_manage" / "metrics" / "orders_metrics.yml"
        target.parent.mkdir(parents=True)
        original = """
metric:
  name: order_count
  type: measure_proxy
  type_params:
    measure: order_count
""".lstrip()
        target.write_text(original, encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ac_manage/metrics/orders_metrics.yml",
            """
metric:
  name: order_count
  type: ratio
  type_params:
    numerator: paid_order_count
    denominator: order_count
""".lstrip(),
        )

        assert result.success == 0
        assert "Refusing to overwrite metric 'order_count'" in result.error
        assert target.read_text(encoding="utf-8") == original

    def test_osi_authoring_skips_metricflow_merge(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ac_manage" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """
semantic_model:
  - name: ac_manage
    datasets:
      - name: orders
        source:
          table: orders
""".lstrip(),
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(
            root_path=str(project),
            current_node="gen_metrics",
            authoring_format="osi",
        )
        incoming = """
semantic_model:
  - name: ac_manage
    datasets:
      - name: orders
        source:
          table: orders
        fields:
          - name: amount
""".lstrip()

        result = tool.write_file("subject/semantic_models/ac_manage/orders.yml", incoming)

        assert result.success == 1
        assert target.read_text(encoding="utf-8") == incoming

    def test_write_file_strict_external_path_is_rejected_before_merge(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        project.mkdir()
        target = tmp_path / "outside" / "subject" / "semantic_models" / "ac_manage" / "metrics" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """
metric:
  name: order_count
  type: measure_proxy
  type_params:
    measure: order_count
""".lstrip(),
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics", strict=True)

        def fail_if_called(*_args, **_kwargs):
            pytest.fail("external file was read before strict path rejection")

        monkeypatch.setattr(tool, "_merge_metric_content", fail_if_called)

        result = tool.write_file(
            str(target),
            """
metric:
  name: paid_order_count
  type: measure_proxy
  type_params:
    measure: paid_order_count
""".lstrip(),
        )

        assert result.success == 0
        assert "outside workspace" in result.error

    def test_quote_escape_repair_only_touches_yaml_error_location(self):
        content = """
data_source:
  name: orders
  sql_query: |
    SELECT '\\'' AS literal_value
  measures:
    - name: paid_order_count
      agg: SUM
      expr: "CASE WHEN status = \\'paid\\' THEN 1 END"
""".lstrip()

        repaired = MetricFilesystemFuncTool._repair_invalid_yaml_single_quote_escapes(content)

        docs = list(yaml.safe_load_all(repaired))
        data_source = docs[0]["data_source"]
        assert data_source["measures"][0]["expr"] == "CASE WHEN status = 'paid' THEN 1 END"
        assert "SELECT '\\'' AS literal_value" in data_source["sql_query"]


class TestEditFile:
    """Tests for MetricFilesystemFuncTool.edit_file — covers lines 65-97."""

    def test_edit_file_in_semantic_yaml(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text("data_source:\n  name: orders\n  description: old\n", encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")
        result = tool.edit_file(
            "subject/semantic_models/orders.yml",
            "description: old",
            "description: new",
        )
        assert result.success == 1
        assert "description: new" in target.read_text(encoding="utf-8")

    def test_edit_file_old_string_not_found(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text("data_source:\n  name: orders\n", encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")
        result = tool.edit_file(
            "subject/semantic_models/orders.yml",
            "nonexistent string",
            "replacement",
        )
        assert result.success == 0

    def test_edit_file_outside_semantic_yaml_no_postprocess(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir(parents=True)
        target = project / "notes.txt"
        target.write_text("hello world\n", encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")
        result = tool.edit_file("notes.txt", "hello", "goodbye")
        assert result.success == 1

    def test_edit_file_postprocess_restores_on_invalid_yaml(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "orders.yml"
        target.parent.mkdir(parents=True)
        original = "data_source:\n  name: orders\n"
        target.write_text(original, encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")
        # Replace valid YAML with something that becomes invalid after edit
        result = tool.edit_file(
            "subject/semantic_models/orders.yml",
            "name: orders",
            "name: orders\n  bad: : :",
        )
        # Should fail due to invalid YAML and restore original
        assert result.success == 0
        # Original content should be restored
        assert target.read_text(encoding="utf-8") == original


class TestMergeSemanticModelContentErrors:
    """Tests for _merge_semantic_model_content error paths — lines 213-216, 220-228."""

    def test_rejects_missing_existing_data_source(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text("semantic_model:\n  name: orders\n", encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/orders.yml",
            "data_source:\n  name: orders\n  measures:\n    - name: revenue\n      agg: SUM\n      expr: revenue\n",
        )
        assert result.success == 0
        assert "data_source" in result.error.lower()

    def test_rejects_missing_incoming_data_source(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "data_source:\n  name: orders\n  measures:\n    - name: revenue\n      agg: SUM\n      expr: revenue\n",
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/orders.yml",
            "semantic_model:\n  name: orders\n",
        )
        assert result.success == 0
        assert "data_source" in result.error.lower()


class TestMergeMetricContentErrors:
    """Tests for _merge_metric_content error paths — lines 245-248, 256-257, 290-291."""

    def test_rejects_when_existing_has_no_metrics(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ds" / "metrics" / "orders_metrics.yml"
        target.parent.mkdir(parents=True)
        target.write_text("some_key:\n  name: something\n", encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ds/metrics/orders_metrics.yml",
            "metric:\n  name: new_metric\n  type: measure_proxy\n",
        )
        assert result.success == 0
        assert "metric" in result.error.lower()

    def test_rejects_when_incoming_has_no_metrics(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ds" / "metrics" / "orders_metrics.yml"
        target.parent.mkdir(parents=True)
        target.write_text("metric:\n  name: order_count\n  type: measure_proxy\n", encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ds/metrics/orders_metrics.yml",
            "some_key:\n  name: something\n",
        )
        assert result.success == 0
        assert "metric" in result.error.lower()


class TestNormalizeMetricSubjectTreeTags:
    """Tests for _normalize_metric_subject_tree_tags — lines 296-327."""

    def test_normalizes_subject_tree_tag_in_metric(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "myds" / "metrics" / "orders_metrics.yml"
        target.parent.mkdir(parents=True)
        # Write a metric with a subject_tree tag that will be normalized
        target.write_text(
            "metric:\n"
            "  name: order_count\n"
            "  type: measure_proxy\n"
            "  type_params:\n"
            "    measure: order_count\n"
            "  locked_metadata:\n"
            "    tags:\n"
            "      - 'subject_tree: metrics/OrderCount'\n",
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")
        result = tool.write_file(
            "subject/semantic_models/myds/metrics/orders_metrics.yml",
            "metric:\n  name: new_metric\n  type: measure_proxy\n  type_params:\n    measure: new_metric\n",
        )
        assert result.success == 1


class TestStaticHelpers:
    """Tests for static methods: _metric_scope_from_path, _merge_metric_fields, etc."""

    def test_metric_scope_from_path_with_metrics_suffix(self, tmp_path):
        path = tmp_path / "subject" / "semantic_models" / "myds" / "metrics" / "orders_metrics.yml"
        datasource, table_name = MetricFilesystemFuncTool._metric_scope_from_path(path)
        assert datasource == "myds"
        assert table_name == "orders"

    def test_metric_scope_from_path_without_suffix(self, tmp_path):
        path = tmp_path / "subject" / "semantic_models" / "myds" / "metrics" / "revenue.yml"
        _, table_name = MetricFilesystemFuncTool._metric_scope_from_path(path)
        assert table_name == "revenue"

    def test_metric_from_doc_non_dict_returns_none(self):
        assert MetricFilesystemFuncTool._metric_from_doc("not_a_dict") is None
        assert MetricFilesystemFuncTool._metric_from_doc(None) is None

    def test_metric_from_doc_no_metric_key_returns_none(self):
        assert MetricFilesystemFuncTool._metric_from_doc({"other": "value"}) is None

    def test_metric_from_doc_valid_returns_dict(self):
        result = MetricFilesystemFuncTool._metric_from_doc({"metric": {"name": "m1"}})
        assert result == {"name": "m1"}

    def test_merge_metric_fields_fills_empty(self):
        existing = {"name": "m1", "type": "measure_proxy", "description": ""}
        incoming = {"name": "m1", "description": "new description", "extra": "value"}
        merged = MetricFilesystemFuncTool._merge_metric_fields(existing, incoming)
        assert merged["description"] == "new description"
        assert merged["extra"] == "value"
        assert merged["type"] == "measure_proxy"

    def test_merge_metric_fields_preserves_existing_non_empty(self):
        existing = {"name": "m1", "description": "existing"}
        incoming = {"name": "m1", "description": "new"}
        merged = MetricFilesystemFuncTool._merge_metric_fields(existing, incoming)
        assert merged["description"] == "existing"

    def test_metric_definition_conflict_detects_type_change(self):
        existing = {"name": "m1", "type": "measure_proxy"}
        incoming = {"name": "m1", "type": "ratio"}
        assert MetricFilesystemFuncTool._metric_definition_conflict(existing, incoming) == "type"

    def test_metric_definition_conflict_no_conflict(self):
        existing = {"name": "m1", "type": "measure_proxy"}
        incoming = {"name": "m1", "type": "measure_proxy"}
        assert MetricFilesystemFuncTool._metric_definition_conflict(existing, incoming) == ""

    def test_metric_definition_conflict_missing_value_skipped(self):
        existing = {"name": "m1", "type": None}
        incoming = {"name": "m1", "type": "measure_proxy"}
        assert MetricFilesystemFuncTool._metric_definition_conflict(existing, incoming) == ""

    def test_find_data_source_doc_found(self):
        docs = [{"other": "value"}, {"data_source": {"name": "orders"}}]
        idx, doc, ds = MetricFilesystemFuncTool._find_data_source_doc(docs)
        assert idx == 1
        assert ds == {"name": "orders"}

    def test_find_data_source_doc_not_found(self):
        docs = [{"other": "value"}]
        idx, doc, ds = MetricFilesystemFuncTool._find_data_source_doc(docs)
        assert idx == -1
        assert ds is None

    def test_named_item_conflict_detects_field_diff(self):
        existing = {"name": "m1", "agg": "SUM"}
        incoming = {"name": "m1", "agg": "COUNT"}
        assert MetricFilesystemFuncTool._named_item_conflict(existing, incoming, ("agg",)) == "agg"

    def test_named_item_conflict_no_conflict(self):
        existing = {"name": "m1", "agg": "SUM"}
        incoming = {"name": "m1", "agg": "SUM"}
        assert MetricFilesystemFuncTool._named_item_conflict(existing, incoming, ("agg",)) == ""

    def test_named_item_conflict_empty_values_skipped(self):
        existing = {"name": "m1", "agg": None}
        incoming = {"name": "m1", "agg": "SUM"}
        assert MetricFilesystemFuncTool._named_item_conflict(existing, incoming, ("agg",)) == ""

    def test_stable_yaml_value_is_deterministic(self):
        val = {"b": 2, "a": 1}
        s1 = MetricFilesystemFuncTool._stable_yaml_value(val)
        s2 = MetricFilesystemFuncTool._stable_yaml_value(val)
        assert s1 == s2

    def test_merge_stable_scalar_fills_empty(self):
        merged = {"name": "orders"}
        incoming = {"sql_table": "orders_table"}
        err = MetricFilesystemFuncTool._merge_stable_scalar(merged, incoming, "sql_table", "orders")
        assert err == ""
        assert merged["sql_table"] == "orders_table"

    def test_merge_stable_scalar_conflict_returns_error(self):
        merged = {"sql_table": "original_table"}
        incoming = {"sql_table": "different_table"}
        err = MetricFilesystemFuncTool._merge_stable_scalar(merged, incoming, "sql_table", "orders")
        assert err != ""
        assert "original_table" in err
        assert "different_table" in err

    def test_merge_data_sources_name_conflict_returns_error(self):
        tool = MetricFilesystemFuncTool.__new__(MetricFilesystemFuncTool)
        existing_ds = {"name": "orders"}
        incoming_ds = {"name": "customers"}
        _, error = tool._merge_data_sources(existing_ds, incoming_ds)
        assert error != ""
        assert "orders" in error

    def test_merge_named_items_conflict_returns_error(self):
        tool = MetricFilesystemFuncTool.__new__(MetricFilesystemFuncTool)
        existing = [{"name": "order_count", "agg": "COUNT", "expr": "1"}]
        incoming = [{"name": "order_count", "agg": "SUM", "expr": "amount"}]
        _, error = tool._merge_named_items("measures", existing, incoming, ("agg", "expr"), "orders")
        assert error != ""
        assert "order_count" in error
