# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""OntologyAnalystAgenticNode: metric and record questions over an Apache Ossie ontology."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode
from datus.schemas.ontology_analyst_agentic_node_models import OntologyAnalystNodeResult
from datus.tools.semantic_tools.dosi.engine import binding_supports_ontology
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class OntologyAnalystAgenticNode(AskMetricsAgenticNode):
    """AskMetrics plus the datasource's ontology: concepts, roles, value domains and record queries.

    The ontology lives at ``subject/semantic_models/ontology/<datasource>/`` and
    is loaded beside the model it shares a file stem with. Without one the node
    refuses to run and points to ``ask_metrics``; it never degrades silently.
    """

    NODE_NAME = "ontology_analyst"
    result_class = OntologyAnalystNodeResult
    RESULT_ACTION_TYPES = ("query_metrics", "select_records")
    FALLBACK_PROMPT_TEMPLATE = "ontology_analyst_system"
    CONCEPT_CATALOG_PROMPT_LIMIT = 60
    DEFAULT_TOOLS = (
        "semantic_tools.describe_concept",
        "semantic_tools.find_concept_paths",
        "semantic_tools.select_records",
        "semantic_tools.list_metrics",
        "semantic_tools.get_metric",
        "semantic_tools.query_metrics",
        "semantic_tools.attribution_analyze",
        "context_search_tools.search_metrics",
    )

    def setup_tools(self) -> None:
        self.concept_catalog: Dict[str, Any] = {}
        self.concept_catalog_mode: str = "none"
        self.concept_catalog_count: int = 0
        self.concept_catalog_prompt: str = ""
        super().setup_tools()
        if not self.agent_config or self.startup_error or not self.semantic_tools:
            return
        self.startup_error = self._prepare_ontology()
        if self.startup_error:
            logger.warning("OntologyAnalyst unavailable: %s", self.startup_error)

    def _semantic_tools_kwargs(self) -> Dict[str, Any]:
        mapping = self.node_config.get("ontology_mapping")
        return {
            "ontology_enabled": True,
            "ontology_mapping": str(mapping).strip() if mapping else None,
        }

    def _prepare_ontology(self) -> Optional[str]:
        """Load the ontology catalog; return why the node cannot run, or None."""
        if not binding_supports_ontology():
            return (
                "the installed dosi-engine binding cannot describe ontologies; "
                "upgrade dosi-engine to a build whose Engine has describe() and paths()"
            )
        runtime = self.semantic_tools.runtime
        if runtime is None:
            return self.semantic_tools._runtime_unavailable_message()

        datasource = getattr(runtime, "datasource", None) or ""
        ontology_dir = self._ontology_dir(datasource)
        try:
            has_ontology = runtime.has_ontology()
        except Exception as exc:  # noqa: BLE001 - reported as the startup error
            return f"failed to load the ontology for datasource {datasource!r}: {exc}"
        if not has_ontology:
            unpaired = runtime.unpaired_ontologies()
            if unpaired:
                names = ", ".join(Path(path).name for path in unpaired)
                return (
                    f"ontology file(s) {names} under {ontology_dir} pair with no semantic model file; "
                    "name each ontology after the model file it describes"
                )
            return (
                f"no ontology for datasource {datasource!r}. Add one under {ontology_dir}/ "
                "or use ask_metrics for metric questions."
            )

        from datus.utils.async_utils import run_async

        try:
            catalog = run_async(runtime.describe())
        except Exception as exc:  # noqa: BLE001 - reported as the startup error
            return f"failed to describe the ontology for datasource {datasource!r}: {exc}"
        self._set_concept_catalog(catalog if isinstance(catalog, dict) else {})
        return None

    def _ontology_dir(self, datasource: str) -> str:
        path_manager = getattr(self.agent_config, "path_manager", None)
        if path_manager is None or not datasource:
            return "subject/semantic_models/ontology/<datasource>"
        return str(path_manager.ontology_path(datasource))

    @staticmethod
    def _catalog_concepts(catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Concept rows of one catalog, or of every catalog in a multi-ontology payload."""
        if isinstance(catalog.get("ontologies"), list):
            rows: List[Dict[str, Any]] = []
            for entry in catalog["ontologies"]:
                if isinstance(entry, dict):
                    rows.extend(OntologyAnalystAgenticNode._catalog_concepts(entry))
            return rows
        concepts = catalog.get("concepts")
        if isinstance(concepts, list):
            return [row for row in concepts if isinstance(row, dict)]
        for value in catalog.values():
            if isinstance(value, list) and value and all(isinstance(row, dict) and "name" in row for row in value):
                return list(value)
        return []

    def _set_concept_catalog(self, catalog: Dict[str, Any]) -> None:
        concepts = self._catalog_concepts(catalog)
        self.concept_catalog = catalog
        self.concept_catalog_count = len(concepts)
        limit = self._resolve_positive_int("concept_catalog_prompt_limit", self.CONCEPT_CATALOG_PROMPT_LIMIT)
        if not catalog:
            self.concept_catalog_mode = "none"
            self.concept_catalog_prompt = ""
        elif len(concepts) <= limit:
            self.concept_catalog_mode = "full"
            self.concept_catalog_prompt = json.dumps(catalog, ensure_ascii=False, indent=2, default=str)
        else:
            self.concept_catalog_mode = "partial"
            excerpt = {
                "concept_names": [str(row.get("name")) for row in concepts],
                "total_concepts": len(concepts),
            }
            self.concept_catalog_prompt = json.dumps(excerpt, ensure_ascii=False, indent=2, default=str)

    def _resolve_positive_int(self, key: str, default: int) -> int:
        raw = self.node_config.get(key, default)
        if isinstance(raw, bool):
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid ontology_analyst %s=%r; using default %s", key, raw, default)
            return default
        return value if value > 0 else default

    def _extra_prompt_context(self) -> Dict[str, Any]:
        return {
            "concept_catalog_mode": self.concept_catalog_mode,
            "concept_catalog_count": self.concept_catalog_count,
            "concept_catalog_prompt": self.concept_catalog_prompt,
        }
