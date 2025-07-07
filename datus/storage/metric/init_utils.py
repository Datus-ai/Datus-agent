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
