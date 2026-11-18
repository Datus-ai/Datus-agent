# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Agent-owned semantic authoring shared by built-in nodes and plugin hosts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, List, Optional

import yaml

from datus.configuration.agent_config import AgentConfig
from datus.storage.semantic_model.artifact_file import semantic_artifact_lock
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState, OsiSemanticModelTargetTools
from datus.tools.func_tool.semantic_model_editing_tools import SemanticModelEditingTools
from datus.tools.func_tool.semantic_plan_file_tools import SemanticPlanFileTools
from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.tools.registry.contracts import tool_contract
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

SEMANTIC_READ_ONLY_DB_TOOLS = frozenset(
    {"list_databases", "list_schemas", "list_tables", "search_table", "describe_table", "execute_sql"}
)
AUTHORING_TOOL_NAMES = frozenset(
    {
        "list_existing_osi_semantic_models",
        "plan_osi_semantic_model_target",
        "bind_osi_semantic_model_target",
        "upsert_osi_datasets",
        "delete_osi_datasets",
        "upsert_osi_metrics",
        "delete_osi_metrics",
        "verify_semantic_outputs",
    }
)
AUTHORING_MUTATION_TOOL_IDS = frozenset(
    {
        "semantic_tools.upsert_osi_datasets",
        "semantic_tools.delete_osi_datasets",
        "semantic_tools.upsert_osi_metrics",
        "semantic_tools.delete_osi_metrics",
        "filesystem_tools.edit_file",
    }
)


def requests_semantic_authoring(selections: list[str]) -> bool:
    """Recognize explicit authoring tools, including unambiguous short names."""
    return any(name.rsplit(".", 1)[-1] in AUTHORING_TOOL_NAMES for name in selections)


class SemanticAuthoringSession:
    """One execution owner's target, workspace, tools and completion evidence.

    This class has no model loop. Built-in nodes and configurable plugin hosts
    supply their own filesystem policy and runtime database context.
    """

    permission_category = "semantic_tools"

    def __init__(
        self,
        agent_config: AgentConfig,
        *,
        node_name: str = "semantic_modeling",
        authoring_scope: str = "full",
        target_state: OsiSemanticModelTargetState | None = None,
        generation_evidence: GenerationEvidence | None = None,
        path_allowed: Callable[[Path], bool] | None = None,
    ) -> None:
        if authoring_scope not in {"datasets", "full"}:
            raise ValueError("authoring_scope must be datasets or full")
        self.agent_config = agent_config
        self.node_name = node_name
        self.authoring_scope = authoring_scope
        self.osi_target_state = target_state or OsiSemanticModelTargetState()
        self.generation_evidence = generation_evidence or GenerationEvidence()
        self.path_allowed = path_allowed
        self.osi_target_tools = OsiSemanticModelTargetTools(
            agent_config,
            target_state=self.osi_target_state,
            generation_evidence=self.generation_evidence,
            path_allowed=path_allowed,
        )
        self._existing_models_checked = False
        self.semantic_tools = None
        self.db_func_tool = None
        self.generation_tools = None
        self.filesystem_func_tool = None
        self.semantic_model_workspace = None
        self.semantic_model_editing_tools = None
        self.semantic_verification_tools = None
        self.semantic_plan_file_tool = SemanticPlanFileTools(agent_config.project_root, path_allowed=path_allowed)
        self._runtime_db_context: Callable[[], dict[str, Any]] = lambda: {}
        self._reconciled_sha256 = ""
        self._scope_metric_baseline: bytes | None = None
        self.kb_sync_attempted = False

    def setup(
        self,
        *,
        filesystem_factory: Callable[..., Any],
        runtime_db_context: Callable[[], dict[str, Any]],
        filesystem_root: str,
        database_config: AgentConfig | None = None,
    ) -> None:
        """Compose shared authoring state, optionally keeping DB reads on the live project.

        Isolated model drafts still resolve SQL reference files against the
        caller's real project, matching the permission hook's file resolution.
        """
        from datus.tools.func_tool.database import DBFuncTool
        from datus.tools.func_tool.semantic_verification_tools import SemanticVerificationTools

        self._runtime_db_context = runtime_db_context
        self.osi_target_tools.integrity_config = database_config if database_config is not None else self.agent_config
        try:
            self.semantic_tools = SemanticTools(
                self.agent_config,
                sub_agent_name=self.node_name,
                generation_evidence=self.generation_evidence,
                runtime_db_context_provider=runtime_db_context,
                warehouse_dry_run_provider=self.warehouse_dry_run,
                semantic_model_path_provider=lambda: self.osi_target_state.selected_path,
                semantic_metric_names_provider=self.validation_metric_names,
                initialize_runtime=self.path_allowed is None,
                enforce_metric_scope=self.path_allowed is not None,
            )
        except Exception as exc:
            logger.error("Failed to setup semantic tools: %s", exc)
        try:
            self.db_func_tool = DBFuncTool(
                filesystem_root=filesystem_root,
                agent_config=database_config if database_config is not None else self.agent_config,
                sub_agent_name=self.node_name,
                read_only=True,
            )
        except Exception as exc:
            logger.error("Failed to setup database tools: %s", exc)
        try:
            self.generation_tools = GenerationTools(
                self.agent_config,
                generation_evidence=self.generation_evidence,
                osi_target_state=self.osi_target_state,
                require_bound_osi_target=True,
            )
        except Exception as exc:
            logger.error("Failed to setup semantic-modeling generation tools: %s", exc)
        try:
            self.filesystem_func_tool = filesystem_factory(
                mutation_guard=self.mutation_guard,
                mutation_callback=self.record_mutation,
            )
            self.semantic_model_workspace = self.filesystem_func_tool.workspace
            self.semantic_model_editing_tools = SemanticModelEditingTools(self.semantic_model_workspace)
        except Exception as exc:
            logger.error("Failed to setup semantic-modeling filesystem tools: %s", exc)
        if self.authoring_scope == "full":
            self.semantic_verification_tools = SemanticVerificationTools(
                self.db_func_tool,
                self.semantic_tools,
                self.generation_evidence,
                lambda: self.osi_target_state.selected_path,
            )

    def available_tools(self) -> list[Any]:
        tools = []
        if self.semantic_tools is not None:
            tools.extend(
                t
                for t in self.semantic_tools.available_tools()
                if t.name in {"get_metric", "query_metrics", "validate_semantic"}
            )
        tools.extend(
            trans_to_function_tool(getattr(self, name))
            for name in (
                "list_existing_osi_semantic_models",
                "plan_osi_semantic_model_target",
                "bind_osi_semantic_model_target",
            )
        )
        if self.db_func_tool is not None:
            tools.extend(t for t in self.db_func_tool.available_tools() if t.name in SEMANTIC_READ_ONLY_DB_TOOLS)
        if self.filesystem_func_tool is not None:
            tools.extend(self.filesystem_func_tool.available_tools())
        if self.semantic_model_editing_tools is not None:
            tools.extend(
                t
                for t in self.semantic_model_editing_tools.available_tools()
                if self.authoring_scope != "datasets" or t.name not in {"upsert_osi_metrics", "delete_osi_metrics"}
            )
        tools.extend(self.semantic_plan_file_tool.available_tools())
        if self.semantic_verification_tools is not None:
            tools.extend(self.semantic_verification_tools.available_tools())
        return tools

    def begin_turn(self) -> None:
        self.generation_evidence.reset()
        self.osi_target_state.reset()
        self.osi_target_tools.invalidate_inventory()
        self.reset_completion()

    def reset_completion(self) -> None:
        self._existing_models_checked = False
        self._reconciled_sha256 = ""
        self._scope_metric_baseline = None
        self.kb_sync_attempted = False

    def mutation_guard(self, path: Path) -> None:
        if self._reconciled_sha256:
            raise ValueError("Semantic authoring has reached KB reconciliation; start a new turn to edit")
        self.osi_target_state.require_selected_path(path)

    def record_mutation(self, path: str | Path | None = None) -> None:
        self.generation_evidence.record_artifact_mutation(path)
        self.osi_target_state.record_selected_write()
        self._reconciled_sha256 = ""

    def rollback(self) -> bool:
        # After successful KB reconciliation, retain the matching artifact if a
        # later plugin step fails. Restoring only YAML would diverge from the KB.
        if self._reconciled_sha256:
            return False
        self._reconciled_sha256 = ""
        return bool(self.semantic_model_workspace and self.semantic_model_workspace.rollback_failed_authoring())

    def validation_metric_names(self) -> list[str]:
        path = self.osi_target_state.selected_path
        if not path or not self.osi_target_state.touched_metric_names or not self.generation_tools:
            return []
        names = self.generation_tools.extract_osi_metric_names(path)
        present, _ = self.osi_target_state.partition_touched_metrics(names)
        return present

    def warehouse_dry_run(self, sql: str) -> dict[str, Any]:
        if self.db_func_tool is None:
            return {"status": "failed", "error": "Database connection is unavailable."}
        runtime = self._runtime_db_context()
        result = self.db_func_tool.read_query(
            f"EXPLAIN {sql.rstrip(';')}", datasource=runtime.get("datasource", ""), database=runtime.get("database", "")
        )
        if not result.success:
            return {"status": "failed", "error": str(result.error or "Warehouse EXPLAIN failed.")}
        return {
            "status": "success",
            "datasource": runtime.get("datasource", ""),
            "database": runtime.get("database", ""),
        }

    @staticmethod
    def _tool_succeeded(result: Any) -> bool:
        return (
            result.get("success", 1) in (1, True)
            if isinstance(result, dict)
            else getattr(result, "success", False) in (1, True)
        )

    @staticmethod
    def _tool_error(result: Any) -> str:
        return (
            str(result.get("error") or result.get("result") or "unknown error")
            if isinstance(result, dict)
            else str(getattr(result, "error", None) or getattr(result, "result", None) or "unknown error")
        )

    @tool_contract(access="read", direct_call=True, category="semantic_tools")
    def list_existing_osi_semantic_models(self) -> FuncToolResult:
        """Inspect existing models before selecting an authoring target."""
        result = self.osi_target_tools.list_existing_osi_semantic_models()
        if self._tool_succeeded(result):
            self._existing_models_checked = True
        return result

    def _existing_models_check_required(self) -> Optional[FuncToolResult]:
        if self._existing_models_checked:
            return None
        return FuncToolResult(
            success=0,
            error=(
                "Inspect subject/semantic_models for the active datasource with "
                "list_existing_osi_semantic_models before planning or binding a target."
            ),
            result={"code": "existing_semantic_models_check_required"},
        )

    def _target_already_selected(self) -> Optional[FuncToolResult]:
        selected = self.osi_target_state.selected
        if selected is None:
            return None
        return FuncToolResult(
            success=0,
            error=("A semantic-model target is already selected for this run. Continue authoring against that target."),
            result={
                "code": "semantic_model_target_already_selected",
                "semantic_model_name": selected.get("semantic_model_name"),
                "semantic_model_file": selected.get("semantic_model_file"),
            },
        )

    @tool_contract(access="restricted", direct_call=True, category="semantic_tools")
    def plan_osi_semantic_model_target(
        self,
        semantic_model_name: str = "",
        business_domain: str = "",
        fact_tables: Optional[List[str]] = None,
        dimension_tables: Optional[List[str]] = None,
    ) -> FuncToolResult:
        """Select a reused or new target after checking existing models.

        Success selects the target for this run; continue authoring without calling bind.

        Args:
            semantic_model_name: Exact existing name to reuse/repair, or the new model name.
            business_domain: Business domain used to find a suitable existing model.
            fact_tables: Physical fact tables needed by the requested outputs.
            dimension_tables: Physical dimension tables needed by the requested outputs.
        """
        required = self._existing_models_check_required()
        if required is not None:
            return required
        selected = self._target_already_selected()
        if selected is not None:
            return selected
        return self.osi_target_tools.plan_osi_semantic_model_target(
            semantic_model_name=semantic_model_name,
            business_domain=business_domain,
            fact_tables=fact_tables,
            dimension_tables=dimension_tables,
        )

    @tool_contract(access="write", direct_call=False, category="semantic_tools")
    def bind_osi_semantic_model_target(
        self,
        semantic_model_file: str = "",
        semantic_model_name: str = "",
    ) -> FuncToolResult:
        """Select an existing target after checking existing models; select only once.

        Args:
            semantic_model_file: Exact existing model path. Supply this or semantic_model_name.
            semantic_model_name: Exact existing name; when both selectors are supplied they must agree.
        """
        required = self._existing_models_check_required()
        if required is not None:
            return required
        selected = self._target_already_selected()
        if selected is not None:
            return selected
        return self.osi_target_tools.bind_osi_semantic_model_target(
            semantic_model_file=semantic_model_file,
            semantic_model_name=semantic_model_name,
        )

    def _target_was_mutated(self) -> bool:
        state = self.osi_target_state
        return bool(
            state.target_mutated
            or state.touched_metric_names
            or state.touched_dataset_names
            or state.planned_dataset_names
            or state.artifact_snapshot_path
        )

    @staticmethod
    def _metric_definitions(content: bytes) -> list[Any]:
        if not content:
            return []
        try:
            document = yaml.safe_load(content.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise RuntimeError(f"Cannot inspect semantic-model metrics for scope enforcement: {exc}") from exc
        models = [document] if isinstance(document, dict) and "datasets" in document else None
        if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
            return []
        metrics = models[0].get("metrics") or []
        return metrics if isinstance(metrics, list) else []

    def _enforce_authoring_scope(self) -> None:
        """Reject and roll back metric mutations made during a datasets-only run."""
        if self.authoring_scope != "datasets":
            return
        state = self.osi_target_state
        if not state.artifact_snapshot_path or state.artifact_snapshot_content is None:
            return
        target_path = state.artifact_snapshot_path
        try:
            current_content = Path(target_path).read_bytes()
        except OSError as exc:
            raise RuntimeError(f"Cannot inspect the selected semantic model for scope enforcement: {exc}") from exc
        if self._metric_definitions(
            self._scope_metric_baseline if self._scope_metric_baseline is not None else state.artifact_snapshot_content
        ) == self._metric_definitions(current_content):
            return
        rolled_back = bool(self.semantic_model_workspace and self.semantic_model_workspace.rollback_failed_authoring())
        rollback_suffix = " The semantic-model YAML was restored." if rolled_back else ""
        raise RuntimeError(
            "Datasets-only semantic_modeling cannot create, update, or delete metrics." + rollback_suffix
        )

    def _selected_osi_artifact(self) -> tuple[str, str, str]:
        state = self.osi_target_state
        selected = state.selected
        if state.last_error_code:
            raise RuntimeError("The semantic-model target remains unresolved after a failed selection.")
        if selected is None:
            raise RuntimeError("semantic_modeling must plan or bind one exact semantic-model target before completion.")

        semantic_model_file = str(selected.get("semantic_model_file") or "").strip()
        resolved = str(selected.get("absolute_path") or "").strip()
        model_name = str(selected.get("semantic_model_name") or "").strip()
        if not semantic_model_file or not resolved or not model_name:
            raise RuntimeError("The selected semantic-model target is incomplete.")
        if self.path_allowed is not None and not self.path_allowed(Path(resolved)):
            raise PermissionError("Semantic model is outside scoped_context.project_files")
        try:
            state.require_selected_revision(resolved)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        if not self.generation_tools:
            raise RuntimeError("Semantic-model generation tools are unavailable.")
        declared_names = self.generation_tools.extract_osi_model_names(resolved)
        if declared_names != [model_name]:
            raise RuntimeError(
                f"The selected OSI artifact must declare exactly model {model_name!r}; "
                f"it declares {declared_names!r}. Restore the exact selected name in {semantic_model_file!r} "
                "before validating or querying it."
            )
        return semantic_model_file, resolved, model_name

    def snapshot(self) -> dict[str, Any]:
        """Return the complete selected document for runtime logic, without tool previews."""
        public_path, resolved, model_name = self._selected_osi_artifact()
        if self.path_allowed is not None and not self.path_allowed(Path(resolved)):
            raise PermissionError("Semantic model is outside scoped_context.project_files")
        content = Path(resolved).read_bytes()
        return {
            "semantic_model_file": public_path,
            "semantic_model_name": model_name,
            "artifact_sha256": hashlib.sha256(content).hexdigest(),
            "document": yaml.safe_load(content),
            "business_validation": self.generation_evidence.business_verification_summary(resolved),
            "changed": self._target_was_mutated(),
        }

    def prepare_artifact(self) -> dict[str, Any]:
        """Apply host-owned identities before binding evidence to the final bytes."""
        self._enforce_authoring_scope()
        self.snapshot()
        _, resolved, _ = self._selected_osi_artifact()
        if self._reconciled_sha256:
            return self.snapshot()
        metric_uids = self.semantic_model_workspace.ensure_metric_uids(resolved)
        uid_sync_names = self.generation_tools.metric_names_needing_uid_sync(resolved, metric_uids)
        if uid_sync_names:
            self.osi_target_state.record_metric_touch(resolved, None, uid_sync_names)
        if self.authoring_scope == "datasets":
            self._scope_metric_baseline = Path(resolved).read_bytes()
        return self.snapshot()

    def validate_artifact(self) -> dict[str, Any]:
        """Validate the current artifact, preserving touched-metric compile requirements."""
        self._enforce_authoring_scope()
        self.snapshot()
        _, resolved, model_name = self._selected_osi_artifact()
        metric_names = self.generation_tools.extract_osi_metric_names(resolved)
        present_metrics, _ = self.osi_target_state.partition_touched_metrics(metric_names)
        validation_scope = "all" if metric_names else "semantic_model"

        artifact_validated = self.generation_evidence.semantic_artifact_validation_passed(
            model_name,
            resolved,
            required_scope=validation_scope,
        )
        metrics_compiled = self.generation_evidence.compiled_validation_passed(resolved, present_metrics)
        if not artifact_validated or (present_metrics and not metrics_compiled):
            if not getattr(self, "semantic_tools", None):
                raise RuntimeError(
                    "The selected semantic model needs validation, but validate_semantic is unavailable."
                )
            validation_result = self.semantic_tools.validate_semantic(
                scope=validation_scope,
                semantic_model_name=model_name,
            )
            self.generation_evidence.record_validation_result(validation_result)
            if not self._tool_succeeded(validation_result):
                raise RuntimeError(
                    f"validate_semantic failed before reconciling the semantic model: {self._tool_error(validation_result)}"
                )
            if not self.generation_evidence.record_semantic_artifact_validation(
                model_name,
                resolved,
                validation_scope=validation_scope,
            ):
                raise RuntimeError("Cannot bind semantic validation evidence to the planned OSI artifact.")

        if self.osi_target_state.touched_metric_names and not self.generation_evidence.compiled_validation_passed(
            resolved,
            present_metrics,
        ):
            raise RuntimeError(
                "validate_semantic did not produce current engine compile evidence for every present touched metric."
            )

        return self.snapshot()

    def reconcile_artifact(self) -> dict[str, Any]:
        """Project a prepared, validated revision into the KB; retain its rollback checkpoint."""
        prepared = self.prepare_artifact()
        snapshot = self.validate_artifact()
        _, resolved, _ = self._selected_osi_artifact()
        revision = snapshot["artifact_sha256"]
        with semantic_artifact_lock(Path(resolved), load_before_lock=False):
            if prepared["artifact_sha256"] != revision or self.snapshot()["artifact_sha256"] != revision:
                raise RuntimeError("The semantic model changed before knowledge-base reconciliation")
            metric_names = self.generation_tools.extract_osi_metric_names(resolved)
            present_metrics, _ = self.osi_target_state.partition_touched_metrics(metric_names)
            self.kb_sync_attempted = True
            sync_result = self.generation_tools.sync_osi_to_db(
                resolved,
                include_semantic_objects=True,
                include_metrics=bool(self.osi_target_state.touched_metric_names),
                metric_names_to_sync=set(present_metrics),
                compiled_metric_catalog=self.generation_evidence.compiled_metric_catalog(present_metrics),
            )
            if not sync_result.get("success"):
                raise RuntimeError(
                    f"OSI semantic model KB reconcile failed: {sync_result.get('error', 'unknown error')}"
                )

            self.generation_evidence.mark_kb_sync()
            self._reconciled_sha256 = revision
            return self.snapshot()

    def complete(self) -> dict[str, Any]:
        """Release rollback state only after the exact current revision reached the KB."""
        snapshot = self.snapshot()
        if not self._reconciled_sha256 or snapshot["artifact_sha256"] != self._reconciled_sha256:
            raise RuntimeError("The current semantic-model revision has not been reconciled to the KB.")
        self.osi_target_state.clear_artifact_snapshot()
        return snapshot

    def finalize(self) -> str:
        self.reconcile_artifact()
        return self.complete()["semantic_model_file"]
