# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Set

from datus.storage.metric.store import MetricRAG
from datus.storage.semantic_model.store import SemanticModelRAG


def existing_semantic_metrics(semantic_rag: SemanticModelRAG, metric_rag: MetricRAG) -> tuple[Set[str], Set[str]]:
    """
    Get all existing semantic models and metrics from storage.
    """
    all_semantic_models, all_metrics = set(), set()
    for semantic_model in semantic_rag.search_all("", select_fields=["id"]):
        all_semantic_models.add(str(semantic_model["id"]))
    for metric in metric_rag.search_all_metrics(select_fields=["id"]):
        all_metrics.add(str(metric["id"]))
    return all_semantic_models, all_metrics


def gen_semantic_model_id(
    catalog_name: str,
    database_name: str,
    schema_name: str,
    table_name: str,
):
    # todo use f"{catalog_name}.{database_name}.{schema_name}.{table_name}"
    return f"{catalog_name}_{database_name}_{schema_name}_{table_name}"


def gen_metric_id(
    subject_path: list,
    semantic_model_name: str,
    metric_name: str,
):
    """Generate unique ID for metric.

    Args:
        subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
        semantic_model_name: Semantic model name
        metric_name: Metric name

    Returns:
        Unique metric ID
    """
    path_str = "/".join(subject_path) if subject_path else ""
    return f"{path_str}/{semantic_model_name}_{metric_name}"
