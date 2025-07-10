from typing import Set

from datus.storage.metric.store import SemanticMetricsRAG


def exists_semantic_metrics(storage: SemanticMetricsRAG, build_mode: str = "overwrite") -> tuple[Set[str], Set[str]]:
    if build_mode == "overwrite":
        return set([]), set([])
    if build_mode == "incremental":
        all_semantic_models = set()
        for semantic_model in storage.search_all_semantic_models(""):
            all_semantic_models.add(semantic_model["id"])
        all_metrics = set()
        for metric in storage.search_all_metrics(""):
            all_metrics.add(metric["id"])
        return all_semantic_models, all_metrics


def gen_semantic_model_id(
    catalog_name: str,
    database_name: str,
    schema_name: str,
    table_name: str,
):
    return f"{catalog_name}_{database_name}_{schema_name}_{table_name}"


def gen_metric_id(
    domain: str,
    layer1: str,
    layer2: str,
    semantic_model_name: str,
    metric_name: str,
):
    return f"{domain}_{layer1}_{layer2}_{semantic_model_name}_{metric_name}"
