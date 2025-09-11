from typing import Set

from datus.storage.metric.store import SemanticMetricsRAG


def exists_semantic_metrics(storage: SemanticMetricsRAG, build_mode: str = "overwrite") -> tuple[Set[str], Set[str]]:
    all_semantic_models, all_metrics = set(), set()
    if build_mode == "overwrite":
        return all_semantic_models, all_metrics
    if build_mode == "incremental":
        for semantic_model in storage.search_all_semantic_models(
            "", selected_fields=["id", "catalog_database_schema", "semantic_model_name"]
        ):
            all_semantic_models.add(str(semantic_model["id"]))
        for metric in storage.search_all_metrics("", select_fields=["id"]):
            all_metrics.add(str(metric["id"]))
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
