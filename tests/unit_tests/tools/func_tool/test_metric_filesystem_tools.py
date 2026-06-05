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

    def test_write_file_rejects_semantic_merge_without_incoming_data_source(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ac_manage" / "orders.yml"
        target.parent.mkdir(parents=True)
        original = """
data_source:
  name: orders
  sql_table: ac_manage.orders
""".lstrip()
        target.write_text(original, encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ac_manage/orders.yml",
            """
metric:
  name: order_count
  type: measure_proxy
""".lstrip(),
        )

        assert result.success == 0
        assert "without a data_source document" in result.error
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

    def test_write_file_normalizes_subject_tree_tags_after_metric_merge(self, tmp_path):
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
  locked_metadata:
    tags:
      - Metrics
      - "subject_tree: Metrics/ac_manage/orders"
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
  locked_metadata:
    tags:
      - ac_manage
      - "subject_tree: ac_manage/ac_manage/orders"
""".lstrip(),
        )

        assert result.success == 1
        docs = list(yaml.safe_load_all(target.read_text(encoding="utf-8")))
        assert docs[0]["metric"]["locked_metadata"]["tags"][1] == "subject_tree: ac_manage/orders"
        assert docs[1]["metric"]["locked_metadata"]["tags"][1] == "subject_tree: ac_manage/orders"

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

    def test_write_file_rejects_metric_merge_without_incoming_metric(self, tmp_path):
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
data_source:
  name: orders
""".lstrip(),
        )

        assert result.success == 0
        assert "without metric documents" in result.error
        assert target.read_text(encoding="utf-8") == original

    def test_write_file_strict_external_path_is_rejected_before_merge(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        project.mkdir()
        target = tmp_path / "outside" / "subject" / "semantic_models" / "ac_manage" / "metrics" / "orders_metrics.yml"
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

    def test_write_file_normalizes_metric_subject_tree_tags(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ac_manage" / "metrics" / "orders_metrics.yml"
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ac_manage/metrics/orders_metrics.yml",
            """
metric:
  name: order_count
  type: measure_proxy
  type_params:
    measure: order_count
  locked_metadata:
    tags:
      - Metrics
      - "subject_tree: Metrics/ac_manage/orders"
---
metric:
  name: paid_order_count
  type: measure_proxy
  type_params:
    measure: paid_order_count
  locked_metadata:
    tags:
      - ac_manage
      - "subject_tree: ac_manage/ac_manage/orders"
""".lstrip(),
        )

        assert result.success == 1
        docs = list(yaml.safe_load_all(target.read_text(encoding="utf-8")))
        assert docs[0]["metric"]["locked_metadata"]["tags"][1] == "subject_tree: ac_manage/orders"

    def test_edit_file_repairs_overescaped_sql_quotes_in_semantic_model(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ac_manage" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """
data_source:
  name: orders
  sql_table: ac_manage.orders
  measures:
    - name: paid_order_count
      agg: SUM
      expr: "CASE WHEN status = 'paid' THEN 1 END"
""".lstrip(),
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.edit_file(
            "subject/semantic_models/ac_manage/orders.yml",
            "status = 'paid'",
            "status = \\'paid\\'",
        )

        assert result.success == 1
        docs = list(yaml.safe_load_all(target.read_text(encoding="utf-8")))
        measure = docs[0]["data_source"]["measures"][0]
        assert measure["expr"] == "CASE WHEN status = 'paid' THEN 1 END"

    def test_write_file_repairs_overescaped_sql_quotes_before_merge(self, tmp_path):
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
      expr: "CASE WHEN status = \\'paid\\' THEN 1 END"
""".lstrip(),
        )

        assert result.success == 1
        docs = list(yaml.safe_load_all(target.read_text(encoding="utf-8")))
        measures = {measure["name"]: measure for measure in docs[0]["data_source"]["measures"]}
        assert measures["paid_order_count"]["expr"] == "CASE WHEN status = 'paid' THEN 1 END"

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

    def test_edit_file_normalizes_metric_subject_tree_tags(self, tmp_path):
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
  locked_metadata:
    tags:
      - ac_manage
      - "subject_tree: ac_manage/orders"
""".lstrip(),
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.edit_file(
            "subject/semantic_models/ac_manage/metrics/orders_metrics.yml",
            "subject_tree: ac_manage/orders",
            "subject_tree: ac_manage/ac_manage/orders",
        )

        assert result.success == 1
        docs = list(yaml.safe_load_all(target.read_text(encoding="utf-8")))
        assert docs[0]["metric"]["locked_metadata"]["tags"][1] == "subject_tree: ac_manage/orders"

    def test_edit_file_restores_original_when_yaml_postprocess_fails(self, tmp_path):
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

        result = tool.edit_file(
            "subject/semantic_models/ac_manage/metrics/orders_metrics.yml",
            "measure: order_count",
            "measure: [",
        )

        assert result.success == 0
        assert "invalid edited YAML" in result.error
        assert target.read_text(encoding="utf-8") == original
