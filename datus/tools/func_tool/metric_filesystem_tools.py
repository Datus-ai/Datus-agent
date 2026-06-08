# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from datus.storage.metric.store import normalize_metric_name
from datus.storage.metric.subject_path import normalize_metric_subject_tree_tag
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.fs_path_policy import PathZone, ResolvedPath


class MetricFilesystemFuncTool(FilesystemFuncTool):
    """Filesystem tool variant for MetricFlow YAML generation.

    Batch metric generation often updates the same semantic model and metrics
    files across several batches. Plain ``write_file`` replacement can drop
    measures, dimensions, or metrics created by earlier batches, so existing
    MetricFlow YAML files are merged structurally.
    """

    def write_file(self, path: str, content: str, file_type: str = "") -> FuncToolResult:  # type: ignore[override]
        resolved = self._classify(path)
        policy_error = self._reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error

        target_path = resolved.resolved
        preprocess_result = self._preprocess_yaml_content(target_path, content)
        if not preprocess_result.success:
            return preprocess_result
        content = str(preprocess_result.result or "")

        if self._is_metric_file_path(target_path):
            if not self._should_merge_metric_file(target_path):
                normalize_result = self._normalize_metric_subject_tree_tags(target_path, content)
                if not normalize_result.success:
                    return normalize_result
                content = str(normalize_result.result or "")
                return super().write_file(path, content, file_type)

            merge_result = self._merge_metric_content(target_path, content)
            if not merge_result.success:
                return merge_result
            result = super().write_file(path, str(merge_result.result or ""), file_type)
            if result.success:
                result.result = f"Metric file merged successfully: {resolved.display}"
            return result

        if not self._should_merge_semantic_model(target_path):
            return super().write_file(path, content, file_type)

        merge_result = self._merge_semantic_model_content(target_path, content)
        if not merge_result.success:
            return merge_result
        result = super().write_file(path, str(merge_result.result or ""), file_type)
        if result.success:
            result.result = f"Semantic model merged successfully: {resolved.display}"
        return result

    def edit_file(self, path: str, old_string: str, new_string: str) -> FuncToolResult:  # type: ignore[override]
        resolved = self._classify(path)
        policy_error = self._reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error

        target_path = resolved.resolved
        original_content: Optional[str] = None
        should_restore = self._is_semantic_yaml_path(target_path) and target_path.exists() and target_path.is_file()
        if should_restore:
            try:
                original_content = target_path.read_text(encoding="utf-8")
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Cannot read YAML file before edit: {exc}")

        result = super().edit_file(path, old_string, new_string)
        if not result.success:
            return result
        if not self._is_semantic_yaml_path(target_path) or not target_path.exists():
            return result

        postprocess_result = self._postprocess_yaml_file(target_path)
        if not postprocess_result.success:
            if original_content is not None:
                try:
                    target_path.write_text(original_content, encoding="utf-8")
                except OSError as exc:
                    return FuncToolResult(
                        success=0,
                        error=f"{postprocess_result.error}; additionally failed to restore original file: {exc}",
                    )
            return postprocess_result
        return result

    def _reject_write_policy(self, resolved: ResolvedPath) -> Optional[FuncToolResult]:
        if resolved.zone == PathZone.HIDDEN:
            return self._not_found(resolved)
        if self.strict and resolved.zone == PathZone.EXTERNAL:
            return self._strict_reject(resolved)
        if resolved.read_only:
            return self._read_only_reject(resolved)
        return None

    def _postprocess_yaml_file(self, target_path: Path) -> FuncToolResult:
        try:
            content = target_path.read_text(encoding="utf-8")
        except OSError as exc:
            return FuncToolResult(success=0, error=f"Cannot read edited YAML file: {exc}")

        preprocess_result = self._preprocess_yaml_content(target_path, content)
        if not preprocess_result.success:
            return preprocess_result
        normalized_content = str(preprocess_result.result or "")

        try:
            list(yaml.safe_load_all(normalized_content))
        except yaml.YAMLError as exc:
            return FuncToolResult(success=0, error=f"Cannot normalize invalid edited YAML file: {exc}")

        if self._is_metric_file_path(target_path):
            normalize_result = self._normalize_metric_subject_tree_tags(target_path, normalized_content)
            if not normalize_result.success:
                return normalize_result
            normalized_content = str(normalize_result.result or "")

        if normalized_content != content:
            try:
                target_path.write_text(normalized_content, encoding="utf-8")
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Cannot normalize edited YAML file: {exc}")
        return FuncToolResult(result=normalized_content)

    def _should_merge_semantic_model(self, target_path: Path) -> bool:
        if not target_path.exists() or not target_path.is_file():
            return False
        if target_path.suffix.lower() not in {".yml", ".yaml"}:
            return False
        parts = target_path.parts
        if "subject" not in parts or "semantic_models" not in parts:
            return False
        subject_idx = parts.index("subject")
        if len(parts) <= subject_idx + 1 or parts[subject_idx + 1] != "semantic_models":
            return False
        return "metrics" not in parts[subject_idx + 2 : -1]

    def _should_merge_metric_file(self, target_path: Path) -> bool:
        return target_path.exists() and target_path.is_file() and self._is_metric_file_path(target_path)

    def _is_metric_file_path(self, target_path: Path) -> bool:
        if target_path.suffix.lower() not in {".yml", ".yaml"}:
            return False
        parts = target_path.parts
        if "subject" not in parts or "semantic_models" not in parts:
            return False
        subject_idx = parts.index("subject")
        if len(parts) <= subject_idx + 1 or parts[subject_idx + 1] != "semantic_models":
            return False
        return "metrics" in parts[subject_idx + 2 : -1]

    def _is_semantic_yaml_path(self, target_path: Path) -> bool:
        if target_path.suffix.lower() not in {".yml", ".yaml"}:
            return False
        parts = target_path.parts
        if "subject" not in parts or "semantic_models" not in parts:
            return False
        subject_idx = parts.index("subject")
        return len(parts) > subject_idx + 1 and parts[subject_idx + 1] == "semantic_models"

    def _preprocess_yaml_content(self, target_path: Path, content: str) -> FuncToolResult:
        if not self._is_semantic_yaml_path(target_path):
            return FuncToolResult(result=content)
        return FuncToolResult(result=self._repair_invalid_yaml_single_quote_escapes(content))

    @staticmethod
    def _repair_invalid_yaml_single_quote_escapes(content: str) -> str:
        repaired = content
        changed = False
        for _ in range(20):
            try:
                list(yaml.safe_load_all(repaired))
                return repaired if changed else content
            except yaml.YAMLError as exc:
                if "unknown escape character" not in str(exc) or "'" not in str(exc):
                    return content
                offset = MetricFilesystemFuncTool._yaml_error_offset(repaired, getattr(exc, "problem_mark", None))
                if offset is None or offset <= 0:
                    return content
                if repaired[offset] != "'" or repaired[offset - 1] != "\\":
                    return content
                repaired = repaired[: offset - 1] + repaired[offset:]
                changed = True
        return content

    @staticmethod
    def _yaml_error_offset(content: str, mark: object) -> Optional[int]:
        line = getattr(mark, "line", None)
        column = getattr(mark, "column", None)
        if not isinstance(line, int) or not isinstance(column, int) or line < 0 or column < 0:
            return None
        lines = content.splitlines(keepends=True)
        if line >= len(lines) or column >= len(lines[line]):
            return None
        return sum(len(item) for item in lines[:line]) + column

    def _merge_semantic_model_content(self, target_path: Path, incoming_content: str) -> FuncToolResult:
        try:
            existing_docs = list(yaml.safe_load_all(target_path.read_text(encoding="utf-8")))
            incoming_docs = list(yaml.safe_load_all(incoming_content))
        except yaml.YAMLError as exc:
            return FuncToolResult(success=0, error=f"Cannot merge invalid semantic model YAML: {exc}")
        except OSError as exc:
            return FuncToolResult(success=0, error=f"Cannot read existing semantic model file: {exc}")

        existing_idx, existing_doc, existing_ds = self._find_data_source_doc(existing_docs)
        _, _, incoming_ds = self._find_data_source_doc(incoming_docs)
        if existing_ds is None:
            return FuncToolResult(
                success=0,
                error="Cannot merge existing semantic model YAML without a data_source document.",
            )
        if incoming_ds is None:
            return FuncToolResult(
                success=0,
                error="Cannot merge semantic model YAML update without a data_source document.",
            )

        merged_ds, error = self._merge_data_sources(existing_ds, incoming_ds)
        if error:
            return FuncToolResult(success=0, error=error)

        merged_doc = dict(existing_doc or {})
        merged_doc["data_source"] = merged_ds
        existing_docs[existing_idx] = merged_doc
        merged_content = yaml.safe_dump_all(existing_docs, allow_unicode=True, sort_keys=False)
        return self._normalize_metric_subject_tree_tags(target_path, merged_content)

    def _merge_metric_content(self, target_path: Path, incoming_content: str) -> FuncToolResult:
        try:
            existing_docs = list(yaml.safe_load_all(target_path.read_text(encoding="utf-8")))
            incoming_docs = list(yaml.safe_load_all(incoming_content))
        except yaml.YAMLError as exc:
            return FuncToolResult(success=0, error=f"Cannot merge invalid metric YAML: {exc}")
        except OSError as exc:
            return FuncToolResult(success=0, error=f"Cannot read existing metric file: {exc}")

        existing_by_name: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        for idx, doc in enumerate(existing_docs):
            metric = self._metric_from_doc(doc)
            name = normalize_metric_name(metric.get("name") if metric else "")
            if name and metric is not None:
                existing_by_name[name] = (idx, metric)
        if not existing_by_name:
            return FuncToolResult(success=0, error="Cannot merge existing metric YAML without metric documents.")

        saw_incoming_metric = False
        for incoming_doc in incoming_docs:
            incoming_metric = self._metric_from_doc(incoming_doc)
            incoming_name = normalize_metric_name(incoming_metric.get("name") if incoming_metric else "")
            if not incoming_name or incoming_metric is None:
                continue
            saw_incoming_metric = True
            existing_entry = existing_by_name.get(incoming_name)
            if existing_entry is None:
                existing_by_name[incoming_name] = (len(existing_docs), incoming_metric)
                existing_docs.append(incoming_doc)
                continue

            existing_idx, existing_metric = existing_entry
            conflict_field = self._metric_definition_conflict(existing_metric, incoming_metric)
            if conflict_field:
                metric_name = incoming_metric.get("name") or existing_metric.get("name") or incoming_name
                return FuncToolResult(
                    success=0,
                    error=(
                        f"Refusing to overwrite metric '{metric_name}': field '{conflict_field}' differs. "
                        "Metric names must be unique within a datasource; preserve the existing definition "
                        "or choose a new metric name."
                    ),
                )
            merged_metric = self._merge_metric_fields(existing_metric, incoming_metric)
            merged_doc = dict(existing_docs[existing_idx] or {})
            merged_doc["metric"] = merged_metric
            existing_docs[existing_idx] = merged_doc
            existing_by_name[incoming_name] = (existing_idx, merged_metric)

        if not saw_incoming_metric:
            return FuncToolResult(success=0, error="Cannot merge metric YAML update without metric documents.")

        merged_content = yaml.safe_dump_all(existing_docs, allow_unicode=True, sort_keys=False)
        return self._normalize_metric_subject_tree_tags(target_path, merged_content)

    def _normalize_metric_subject_tree_tags(self, target_path: Path, content: str) -> FuncToolResult:
        try:
            docs = list(yaml.safe_load_all(content))
        except yaml.YAMLError as exc:
            return FuncToolResult(success=0, error=f"Cannot normalize invalid metric YAML: {exc}")

        datasource, table_name = self._metric_scope_from_path(target_path)
        changed = False
        for doc in docs:
            metric = self._metric_from_doc(doc)
            if metric is None:
                continue
            locked_metadata = metric.get("locked_metadata")
            if not isinstance(locked_metadata, dict):
                continue
            tags = locked_metadata.get("tags")
            if not isinstance(tags, list):
                continue
            normalized_tags = []
            for tag in tags:
                normalized = (
                    normalize_metric_subject_tree_tag(tag, datasource=datasource, table_name=table_name)
                    if isinstance(tag, str)
                    else tag
                )
                normalized_tags.append(normalized)
                changed = changed or normalized != tag
            locked_metadata["tags"] = normalized_tags

        if not changed:
            return FuncToolResult(result=content)
        return FuncToolResult(result=yaml.safe_dump_all(docs, allow_unicode=True, sort_keys=False))

    @staticmethod
    def _metric_scope_from_path(target_path: Path) -> Tuple[str, str]:
        parts = list(target_path.parts)
        datasource = ""
        if "semantic_models" in parts:
            idx = parts.index("semantic_models")
            if len(parts) > idx + 1:
                datasource = parts[idx + 1]
        stem = target_path.stem
        table_name = stem[: -len("_metrics")] if stem.endswith("_metrics") else stem
        return datasource, table_name or "Unknown"

    @staticmethod
    def _metric_from_doc(doc: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(doc, dict):
            return None
        metric = doc.get("metric")
        return metric if isinstance(metric, dict) else None

    @staticmethod
    def _merge_metric_fields(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(existing)
        for field, value in incoming.items():
            if field not in merged or merged.get(field) in (None, "", []):
                merged[field] = value
        return merged

    @classmethod
    def _metric_definition_conflict(cls, existing: Dict[str, Any], incoming: Dict[str, Any]) -> str:
        for field in ("type", "type_params"):
            existing_value = existing.get(field)
            incoming_value = incoming.get(field)
            if existing_value in (None, "", []) or incoming_value in (None, "", []):
                continue
            if cls._stable_yaml_value(existing_value) != cls._stable_yaml_value(incoming_value):
                return field
        return ""

    @staticmethod
    def _stable_yaml_value(value: Any) -> str:
        return yaml.safe_dump(value, allow_unicode=True, sort_keys=True)

    @staticmethod
    def _find_data_source_doc(docs: List[Any]) -> Tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        for idx, doc in enumerate(docs):
            if isinstance(doc, dict) and isinstance(doc.get("data_source"), dict):
                return idx, doc, doc["data_source"]
        return -1, None, None

    def _merge_data_sources(
        self,
        existing_ds: Dict[str, Any],
        incoming_ds: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        existing_name = str(existing_ds.get("name") or "").strip()
        incoming_name = str(incoming_ds.get("name") or "").strip()
        if existing_name and incoming_name and existing_name != incoming_name:
            return {}, (
                f"Refusing to overwrite semantic model '{existing_name}' with data_source '{incoming_name}'. "
                "Use a separate file for a different data_source."
            )

        merged = dict(existing_ds)
        for field in ("name", "description"):
            if not merged.get(field) and incoming_ds.get(field):
                merged[field] = incoming_ds[field]

        for field in ("sql_table", "sql_query"):
            error = self._merge_stable_scalar(merged, incoming_ds, field, existing_name or incoming_name)
            if error:
                return {}, error

        for field, conflict_fields in (
            ("identifiers", ("type", "expr")),
            ("measures", ("agg", "expr", "filter", "agg_params", "non_additive_dimension")),
            ("dimensions", ("type", "expr", "type_params")),
        ):
            merged_items, error = self._merge_named_items(
                field,
                merged.get(field) or [],
                incoming_ds.get(field) or [],
                conflict_fields,
                existing_name or incoming_name,
            )
            if error:
                return {}, error
            if merged_items:
                merged[field] = merged_items

        for field, value in incoming_ds.items():
            if field in {"name", "description", "sql_table", "sql_query", "identifiers", "measures", "dimensions"}:
                continue
            if field not in merged:
                merged[field] = value

        return merged, ""

    @staticmethod
    def _merge_stable_scalar(
        merged: Dict[str, Any],
        incoming: Dict[str, Any],
        field: str,
        data_source_name: str,
    ) -> str:
        existing_value = merged.get(field)
        incoming_value = incoming.get(field)
        if not existing_value and incoming_value:
            merged[field] = incoming_value
            return ""
        if existing_value and incoming_value and existing_value != incoming_value:
            return (
                f"Refusing to change data_source '{data_source_name}' field '{field}' from "
                f"{existing_value!r} to {incoming_value!r}. Edit intentionally or write a new data_source file."
            )
        return ""

    def _merge_named_items(
        self,
        section: str,
        existing_items: List[Any],
        incoming_items: List[Any],
        conflict_fields: Tuple[str, ...],
        data_source_name: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        merged: List[Dict[str, Any]] = [dict(item) for item in existing_items if isinstance(item, dict)]
        index_by_name = {
            str(item.get("name") or "").strip(): idx
            for idx, item in enumerate(merged)
            if str(item.get("name") or "").strip()
        }

        for raw_item in incoming_items:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            existing_idx = index_by_name.get(name)
            if existing_idx is None:
                index_by_name[name] = len(merged)
                merged.append(item)
                continue

            existing = merged[existing_idx]
            conflict_field = self._named_item_conflict(existing, item, conflict_fields)
            if conflict_field:
                return [], (
                    f"Refusing to overwrite {section[:-1]} '{name}' in data_source '{data_source_name}': "
                    f"field '{conflict_field}' differs. Preserve the existing definition or choose a new name."
                )
            for field, value in item.items():
                if field not in existing or existing.get(field) in (None, "", []):
                    existing[field] = value

        return merged, ""

    @staticmethod
    def _named_item_conflict(existing: Dict[str, Any], incoming: Dict[str, Any], fields: Tuple[str, ...]) -> str:
        for field in fields:
            existing_value = existing.get(field)
            incoming_value = incoming.get(field)
            if existing_value in (None, "", []) or incoming_value in (None, "", []):
                continue
            if existing_value != incoming_value:
                return field
        return ""
