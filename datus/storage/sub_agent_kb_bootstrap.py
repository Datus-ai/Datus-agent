# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import ScopedContextLists, SubAgentConfig
from datus.storage.conditions import Node, or_
from datus.storage.ext_knowledge.store import ExtKnowledgeRAG
from datus.storage.metric.store import MetricRAG
from datus.storage.reference_sql.store import ReferenceSqlRAG
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.storage.scoped_filter import _table_condition_for_token
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.reference_paths import split_reference_path

logger = get_logger(__name__)

SUPPORTED_COMPONENTS = ("metadata", "semantic_model", "metrics", "reference_sql", "ext_knowledge")

# "overwrite" now behaves like "plan" (validation only) since sub-agents
# use the shared global storage with WHERE filters instead of separate copies.
SubAgentBootstrapStrategy = Literal["overwrite", "plan"]


@dataclass(slots=True)
class ComponentResult:
    component: str
    status: Literal["success", "skipped", "error", "plan"]
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class BootstrapResult:
    should_bootstrap: bool
    reason: Optional[str]
    storage_path: str
    strategy: SubAgentBootstrapStrategy
    results: List[ComponentResult]


class SubAgentBootstrapper:
    """Validate scoped-context configuration for a sub-agent.

    Since sub-agents now query the shared global storage via WHERE filters,
    the bootstrapper no longer copies data.  ``run()`` always performs a
    *plan*-style validation: it checks which rows in the global storage
    match the scoped context and reports missing/invalid tokens.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        sub_agent: Optional[SubAgentConfig] = None,
        sub_agent_name: Optional[str] = None,
        check_exists: bool = True,
    ):
        self.agent_config = agent_config
        self._valid_sub_agent(sub_agent_name, sub_agent, check_exists)
        self.dialect = getattr(self.agent_config, "db_type", "")
        self.storage_path = self.agent_config.rag_storage_path()

    def _valid_sub_agent(
        self,
        sub_agent_name: Optional[str] = None,
        sub_agent: Optional[SubAgentConfig] = None,
        check_exists: bool = True,
    ):
        if sub_agent:
            self.sub_agent_name = sub_agent.system_prompt
            if check_exists:
                self._valid_sub_agent_in_main(self.sub_agent_name)
            self.sub_agent = sub_agent
        elif sub_agent_name:
            self.sub_agent_name = sub_agent_name
            self.sub_agent = SubAgentConfig.model_validate(self._valid_sub_agent_in_main(sub_agent_name))
        else:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message="Subagent name and configuration cannot be empty at the same time",
            )

    def _valid_sub_agent_in_main(self, sub_agent_name: str) -> Dict[str, Any]:
        sub_in_main_config = self.agent_config.sub_agent_config(sub_agent_name)
        if not sub_in_main_config:
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message=f"Subagent configuration named `{sub_agent_name}` not found in agent configuration",
            )
        return sub_in_main_config

    def run(
        self,
        selected_components: Optional[List[str]] = None,
        strategy: SubAgentBootstrapStrategy = "plan",
    ) -> BootstrapResult:
        if not self.sub_agent.has_scoped_context():
            return BootstrapResult(
                should_bootstrap=False,
                reason="Scope context is empty, no need to execute",
                storage_path=self.storage_path,
                strategy=strategy,
                results=[],
            )
        if strategy not in ("overwrite", "plan"):
            raise DatusException(
                code=ErrorCode.COMMON_VALIDATION_FAILED,
                message=f"Unsupported strategy '{strategy}'. Expected 'overwrite' or 'plan'.",
            )

        if not selected_components:
            selected_components = SUPPORTED_COMPONENTS

        normalized_components = list(dict.fromkeys([c.lower() for c in selected_components]))
        results: List[ComponentResult] = []
        context_lists = self._context_lists()

        handlers = {
            "metadata": ("tables", self._handle_metadata),
            "semantic_model": ("tables", self._handle_semantic_model),
            "metrics": ("metrics", self._handle_metrics),
            "reference_sql": ("sqls", self._handle_reference_sql),
            "ext_knowledge": ("ext_knowledge", self._handle_ext_knowledge),
        }
        for component in normalized_components:
            handler_entry = handlers.get(component)
            if handler_entry is None:
                results.append(
                    ComponentResult(
                        component=component,
                        status="error",
                        message=f"Unsupported component '{component}'. Supported: {', '.join(SUPPORTED_COMPONENTS)}",
                    )
                )
                continue
            attr_name, handler = handler_entry
            try:
                result = handler(getattr(context_lists, attr_name))
            except Exception as exc:  # pragma: no cover - safety net
                logger.exception("Failed to validate %s component", component)
                result = ComponentResult(component=component, status="error", message=str(exc))
            results.append(result)

        return BootstrapResult(
            storage_path=self.storage_path, strategy="plan", results=results, should_bootstrap=True, reason=None
        )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _context_lists(self) -> ScopedContextLists:
        if self.sub_agent.scoped_context:
            return self.sub_agent.scoped_context.as_lists()
        return ScopedContextLists()

    def _ensure_source_ready(self, db_path: str, component: str) -> bool:
        import os

        if os.path.isdir(db_path):
            return True
        logger.warning("Global storage path '%s' not found for component '%s'.", db_path, component)
        return False

    def _count_rows(self, storage, condition: Optional[Node]) -> int:
        try:
            return storage._count_rows(where=condition)
        except Exception:
            return 0

    # --------------------------------------------------------------------- #
    # Metadata (plan / validation only)
    # --------------------------------------------------------------------- #
    def _handle_metadata(self, tables: List[str]) -> ComponentResult:
        if not tables:
            return ComponentResult(
                component="metadata", status="skipped", message="No tables defined in scoped context."
            )

        global_path = self.agent_config.rag_storage_path()
        if not self._ensure_source_ready(global_path, "metadata"):
            return ComponentResult(
                component="metadata", status="error", message="Global metadata store is not initialized."
            )

        source = SchemaWithValueRAG(self.agent_config)
        condition_map, invalid_tokens = self._metadata_conditions(tables)

        if not condition_map:
            details = {"invalid": invalid_tokens} if invalid_tokens else None
            return ComponentResult(
                component="metadata", status="skipped", message="No valid table filters resolved.", details=details
            )

        aggregate_condition = self._combine_conditions(condition_map)
        schema_table = source.schema_store._search_all(
            where=aggregate_condition,
            select_fields=[
                "identifier",
                "catalog_name",
                "database_name",
                "schema_name",
                "table_name",
                "table_type",
                "definition",
            ],
        )
        schema_rows = schema_table.to_pylist()
        missing = self._missing_tokens(source.schema_store, condition_map)

        details = {
            "match_count": len(schema_rows),
            "tables": [self._format_table_identifier(item) for item in schema_rows[:20]],
            "missing": missing,
            "invalid": invalid_tokens,
        }
        return ComponentResult(component="metadata", status="plan", message="Metadata plan generated.", details=details)

    # --------------------------------------------------------------------- #
    # Semantic Model (plan / validation only)
    # --------------------------------------------------------------------- #
    def _handle_semantic_model(self, semantic_models: List[str]) -> ComponentResult:
        if not semantic_models:
            return ComponentResult(
                component="semantic_model", status="skipped", message="No semantic models defined in scoped context."
            )

        global_path = self.agent_config.rag_storage_path()
        if not self._ensure_source_ready(global_path, "semantic_model"):
            return ComponentResult(
                component="semantic_model", status="error", message="Global semantic model store is not initialized."
            )

        condition_map, invalid_tokens = self._metadata_conditions(semantic_models)
        if not condition_map:
            details = {"invalid": invalid_tokens} if invalid_tokens else None
            return ComponentResult(
                component="semantic_model",
                status="skipped",
                message="No valid semantic model filters resolved.",
                details=details,
            )

        source = SemanticModelRAG(self.agent_config)
        aggregate_condition = self._combine_conditions(condition_map)
        semantic_table = source.storage._search_all(where=aggregate_condition)
        semantic_rows = semantic_table.to_pylist()
        missing = self._missing_tokens(source.storage, condition_map)

        details = {
            "match_count": len(semantic_rows),
            "semantic_objects": [row.get("id") for row in semantic_rows[:20]],
            "missing": missing,
            "invalid": invalid_tokens,
        }
        return ComponentResult(
            component="semantic_model", status="plan", message="Semantic model plan generated.", details=details
        )

    # --------------------------------------------------------------------- #
    # Metrics (plan / validation only)
    # --------------------------------------------------------------------- #
    def _handle_metrics(self, metrics: List[str]) -> ComponentResult:
        if not metrics:
            return ComponentResult(
                component="metrics", status="skipped", message="No metrics defined in scoped context."
            )

        global_path = self.agent_config.rag_storage_path()
        if not self._ensure_source_ready(global_path, "metrics"):
            return ComponentResult(
                component="metrics", status="error", message="Global metrics store is not initialized."
            )

        source = MetricRAG(self.agent_config)

        metric_rows = []
        invalid_tokens = []
        missing = []
        for metric in metrics:
            parts = split_reference_path(metric.replace("/", "."))
            if not parts:
                invalid_tokens.append(metric)
                continue
            metric_table = source.search_all_metrics(subject_path=parts)
            if len(metric_table) > 0:
                metric_rows.extend(metric_table)
            else:
                missing.append(metric)

        details = {
            "match_count": len(metric_rows),
            "metrics": [self._format_subject_identifier(row) for row in metric_rows[:20]],
            "missing": missing,
            "invalid": invalid_tokens,
        }
        return ComponentResult(component="metrics", status="plan", message="Metrics plan generated.", details=details)

    # --------------------------------------------------------------------- #
    # Reference SQL (plan / validation only)
    # --------------------------------------------------------------------- #
    def _handle_reference_sql(self, historical_sql: List[str]) -> ComponentResult:
        if not historical_sql:
            return ComponentResult(
                component="reference_sql",
                status="skipped",
                message="No reference SQL identifiers defined in scoped context.",
            )

        global_path = self.agent_config.rag_storage_path()
        if not self._ensure_source_ready(global_path, "reference_sql"):
            return ComponentResult(
                component="reference_sql", status="error", message="Global reference SQL store is not initialized."
            )

        source = ReferenceSqlRAG(self.agent_config)
        invalid_tokens = []
        missing = []
        sql_rows = []
        for sql in historical_sql:
            parts = split_reference_path(sql.replace("/", "."))
            if not parts:
                invalid_tokens.append(sql)
                continue
            sql_table = source.search_all_reference_sql(subject_path=parts)
            if len(sql_table) > 0:
                sql_rows.extend(sql_table)
            else:
                missing.append(sql)

        details = {
            "match_count": len(sql_rows),
            "entries": [self._format_subject_identifier(row) for row in sql_rows[:20]],
            "missing": missing,
            "invalid": invalid_tokens,
        }
        return ComponentResult(
            component="reference_sql", status="plan", message="Reference SQL plan generated.", details=details
        )

    # --------------------------------------------------------------------- #
    # Ext Knowledge (plan / validation only)
    # --------------------------------------------------------------------- #
    def _handle_ext_knowledge(self, ext_knowledge_tokens: List[str]) -> ComponentResult:
        if not ext_knowledge_tokens:
            return ComponentResult(
                component="ext_knowledge",
                status="skipped",
                message="No ext knowledge identifiers defined in scoped context.",
            )

        global_path = self.agent_config.rag_storage_path()
        if not self._ensure_source_ready(global_path, "ext_knowledge"):
            return ComponentResult(
                component="ext_knowledge", status="error", message="Global ext knowledge store is not initialized."
            )

        source = ExtKnowledgeRAG(self.agent_config)
        invalid_tokens = []
        missing = []
        knowledge_rows = []
        for token in ext_knowledge_tokens:
            parts = split_reference_path(token.replace("/", "."))
            if not parts:
                invalid_tokens.append(token)
                continue
            rows = source.store.search_all_knowledge(subject_path=parts)
            if len(rows) > 0:
                knowledge_rows.extend(rows)
            else:
                missing.append(token)

        details = {
            "match_count": len(knowledge_rows),
            "entries": [self._format_subject_identifier(row) for row in knowledge_rows[:20]],
            "missing": missing,
            "invalid": invalid_tokens,
        }
        return ComponentResult(
            component="ext_knowledge", status="plan", message="Ext knowledge plan generated.", details=details
        )

    # --------------------------------------------------------------------- #
    # Condition helpers (still used by ScopedFilterBuilder & plan validation)
    # --------------------------------------------------------------------- #
    def _metadata_conditions(self, tokens: Iterable[str]) -> tuple[List[tuple[str, Node]], List[str]]:
        mapped: List[tuple[str, Node]] = []
        invalid: List[str] = []
        for raw in tokens:
            token = raw.strip()
            if not token:
                continue
            condition = self._metadata_condition_for_token(token)
            if condition is None:
                invalid.append(token)
            else:
                mapped.append((token, condition))
        return mapped, invalid

    def _metadata_condition_for_token(self, token: str) -> Optional[Node]:
        return _table_condition_for_token(token, self.dialect or "")

    def _combine_conditions(self, condition_map: List[tuple[str, Node]]) -> Optional[Node]:
        if not condition_map:
            return None
        nodes = [node for _, node in condition_map]
        if len(nodes) == 1:
            return nodes[0]
        return or_(*nodes)

    def _missing_tokens(self, storage, condition_map: List[tuple[str, Node]]) -> List[str]:
        missing: List[str] = []
        for token, node in condition_map:
            if self._count_rows(storage, node) == 0:
                missing.append(token)
        return missing

    @staticmethod
    def _format_table_identifier(row: Dict[str, Any]) -> str:
        return ".".join(
            filter(
                None,
                [
                    row.get("catalog_name"),
                    row.get("database_name"),
                    row.get("schema_name"),
                    row.get("table_name"),
                ],
            )
        )

    @staticmethod
    def _format_subject_identifier(row: Dict[str, Any]) -> str:
        return f"{'/'.join(row.get('subject_path'))}/{row.get('name')}"
