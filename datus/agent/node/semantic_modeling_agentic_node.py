# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unified Dosi semantic-model and metric authoring node."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.configuration.agent_config import AgentConfig
from datus.schemas.semantic_agentic_node_models import SemanticModelingNodeResult, SemanticNodeInput
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SemanticModelingAgenticNode(GenMetricsAgenticNode):
    """Author Dosi datasets, metrics, or both against one selected model."""

    NODE_NAME = "semantic_modeling"
    result_class = SemanticModelingNodeResult

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        scope: Optional[str] = None,
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ):
        from datus.agent.node.semantic_authoring import resolve_semantic_adapter_type

        adapter = resolve_semantic_adapter_type(agent_config)
        if adapter != "dosi":
            raise ValueError("semantic_modeling is available only when semantic_adapter=dosi.")
        self._existing_models_checked = False
        super().__init__(
            agent_config=agent_config,
            execution_mode=execution_mode,
            scope=scope,
            is_subagent=is_subagent,
            session_id=session_id,
        )

    def setup_tools(self):
        """Compose the existing semantic-model and metric authoring surfaces."""
        if not self.agent_config:
            return

        self.tools = []
        self._setup_osi_target_tools()

        self._setup_db_tools(expose_tools=False)
        self._setup_semantic_discovery_tools()
        self._setup_generation_tools()
        self._setup_filesystem_tools()
        self._setup_semantic_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.info("Setup %s tools for %s: %s", len(self.tools), self.NODE_NAME, [tool.name for tool in self.tools])

    async def _before_stream(self, ctx: StreamRunContext) -> None:
        await super()._before_stream(ctx)
        self._existing_models_checked = False

    def _setup_osi_target_tools(self) -> None:
        """Expose existing-model inspection plus one planner/binder."""
        from datus.tools.func_tool import OsiSemanticModelTargetTools, trans_to_function_tool

        self.osi_target_tools = OsiSemanticModelTargetTools(
            self.agent_config,
            target_state=self.osi_target_state,
            generation_evidence=self.generation_evidence,
        )
        self.tools.extend(
            [
                trans_to_function_tool(self.list_existing_osi_semantic_models),
                trans_to_function_tool(self.plan_osi_semantic_model_target),
                trans_to_function_tool(self.bind_osi_semantic_model_target),
            ]
        )

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

    def plan_osi_semantic_model_target(
        self,
        semantic_model_name: str = "",
        business_domain: str = "",
        fact_tables: Optional[List[str]] = None,
        dimension_tables: Optional[List[str]] = None,
    ) -> FuncToolResult:
        """Plan a reused or new target only after checking existing models."""
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

    def bind_osi_semantic_model_target(
        self,
        semantic_model_file: str = "",
        semantic_model_name: str = "",
    ) -> FuncToolResult:
        """Bind an existing target only after checking existing models."""
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

    def _make_filesystem_tool(self, **kwargs):
        from datus.tools.func_tool.metric_filesystem_tools import SemanticModelingFilesystemFuncTool

        kwargs["filesystem_tool_cls"] = SemanticModelingFilesystemFuncTool
        return super()._make_filesystem_tool(**kwargs)

    def _semantic_modeling_mutation_guard(self, path: str):
        return self.osi_target_state.require_selected_path(path)

    def _record_semantic_modeling_mutation(self, path=None) -> None:
        self.generation_evidence.record_artifact_mutation(path)
        self.osi_target_state.record_selected_write()

    def _setup_filesystem_tools(self):
        try:
            self.filesystem_func_tool = self._make_filesystem_tool(
                mutation_guard=self._semantic_modeling_mutation_guard,
                mutation_callback=self._record_semantic_modeling_mutation,
            )
            filesystem_tools = [
                tool for tool in self.filesystem_func_tool.available_tools() if tool.name != "delete_file"
            ]
            self.tools.extend(filesystem_tools)
        except Exception as exc:
            logger.error("Failed to setup semantic-modeling filesystem tools: %s", exc)

    def _setup_generation_tools(self):
        """Prepare deterministic host-side validation and KB reconciliation."""
        try:
            from datus.agent.node.semantic_authoring import resolve_authoring_format

            authoring_format = resolve_authoring_format(self.agent_config)
            self.generation_tools = GenerationTools(
                self.agent_config,
                generation_evidence=self.generation_evidence,
                authoring_format=authoring_format,
                osi_target_state=self.osi_target_state,
                require_bound_osi_target=authoring_format == "osi",
            )
        except Exception as exc:
            logger.error("Failed to setup semantic-modeling generation tools: %s", exc)

    def _build_enhanced_message(
        self,
        user_input: SemanticNodeInput,
        extra_enhanced_parts: Optional[List[str]] = None,
    ) -> str:
        """Keep explicit target values as hints until existing models are checked."""
        parts = list(extra_enhanced_parts or [])
        semantic_model_name = str(getattr(user_input, "semantic_model_name", "") or "").strip()
        semantic_model_file = str(getattr(user_input, "semantic_model_file", "") or "").strip()
        if semantic_model_name or semantic_model_file:
            hints = ["## Semantic Model Selection Hint for This Turn"]
            if semantic_model_file:
                hints.append(f"- Requested semantic model file: `{semantic_model_file}`")
            if semantic_model_name:
                hints.append(f"- Requested semantic model name: `{semantic_model_name}`")
            hints.append("Verify the hint against the active datasource's existing models before selecting the target.")
            parts.append("\n".join(hints))
        return AgenticNode._build_enhanced_message(self, user_input, parts)

    @staticmethod
    def _parse_semantic_modeling_response(content: Any) -> tuple[Optional[str], Optional[str], Optional[str], str]:
        """Parse the unified node's small outcome envelope."""
        payload = content if isinstance(content, dict) else None
        if payload is None and isinstance(content, str) and content.strip():
            from datus.utils.json_utils import strip_json_str

            cleaned = strip_json_str(content)
            if cleaned:
                try:
                    import json_repair

                    parsed = json_repair.loads(cleaned)
                    payload = parsed if isinstance(parsed, dict) else None
                except Exception:
                    payload = None
        if payload is None:
            return None, None, None, ""
        status = payload.get("status")
        blocker_code = payload.get("blocker_code")
        skip_reason = payload.get("skip_reason")
        output = payload.get("output")
        return (
            status.strip().lower() if isinstance(status, str) else None,
            blocker_code.strip().lower() if isinstance(blocker_code, str) else None,
            skip_reason.strip().lower() if isinstance(skip_reason, str) else None,
            output if isinstance(output, str) else "",
        )

    def _build_success_result(self, ctx: StreamRunContext) -> SemanticModelingNodeResult:
        """Validate and reconcile the selected artifact after the LLM tool stream."""
        response_content = ctx.response_content
        if not response_content and ctx.last_successful_output:
            raw_output = ctx.last_successful_output.get("raw_output", "")
            response_content = (
                raw_output if isinstance(raw_output, dict) or raw_output else str(ctx.last_successful_output)
            )

        status, blocker_code, skip_reason, extracted_output = self._parse_semantic_modeling_response(response_content)
        if extracted_output:
            response_content = extracted_output
        if not isinstance(response_content, str):
            response_content = str(response_content) if response_content else ""

        if status not in {"generated", "skipped", "blocked"}:
            raise RuntimeError("semantic_modeling must return a supported status.")
        if blocker_code is not None and status != "blocked":
            raise RuntimeError("blocker_code is only valid when status='blocked'.")
        if skip_reason is not None and status != "skipped":
            raise RuntimeError("skip_reason is only valid when status='skipped'.")

        semantic_model_files: List[str] = []
        if status == "blocked":
            if blocker_code not in {
                "semantic_model_required",
                "semantic_model_selection_required",
                "semantic_model_target_invalid",
            }:
                raise RuntimeError("status='blocked' requires a supported semantic-model blocker_code.")
            if self._target_was_mutated():
                raise RuntimeError("semantic_modeling cannot return status='blocked' after changing the target.")
        elif status == "skipped":
            if skip_reason != "no_semantic_change":
                raise RuntimeError("status='skipped' requires skip_reason='no_semantic_change'.")
            if self._target_was_mutated():
                raise RuntimeError("semantic_modeling cannot return status='skipped' after changing the target.")
        else:
            semantic_model_files = [self._finalize_selected_osi_artifact()]

        tokens_used = 0
        if self.execution_mode == "interactive":
            tokens_used = self._extract_total_tokens(ctx.action_history_manager.get_actions())
        return SemanticModelingNodeResult(
            success=status != "blocked",
            error=response_content if status == "blocked" else None,
            response=response_content,
            semantic_models=semantic_model_files,
            tokens_used=int(tokens_used),
            status=status,
            blocker_code=blocker_code if status == "blocked" else None,
            skip_reason=skip_reason if status == "skipped" else None,
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
        try:
            state.require_selected_revision(resolved)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        if not self.generation_tools:
            raise RuntimeError("Semantic-model generation tools are unavailable.")
        if self.generation_tools.extract_osi_model_names(resolved) != [model_name]:
            raise RuntimeError("The selected OSI artifact no longer declares the selected semantic model.")
        return semantic_model_file, resolved, model_name

    def _finalize_selected_osi_artifact(self) -> str:
        """Validate and fully reconcile one target YAML before returning success."""
        semantic_model_file, resolved, model_name = self._selected_osi_artifact()
        metric_names = self.generation_tools.extract_osi_metric_names(resolved)
        validation_scope = "all" if metric_names else "semantic_model"

        artifact_validated = self.generation_evidence.semantic_artifact_validation_passed(
            model_name,
            resolved,
            required_scope=validation_scope,
        )
        if not artifact_validated:
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

        sync_result = self.generation_tools.sync_osi_to_db(
            resolved,
            include_semantic_objects=True,
            include_metrics=True,
        )
        if not sync_result.get("success"):
            raise RuntimeError(f"OSI semantic model KB reconcile failed: {sync_result.get('error', 'unknown error')}")

        self.generation_evidence.mark_kb_sync()
        self.osi_target_state.clear_artifact_snapshot()
        return semantic_model_file
