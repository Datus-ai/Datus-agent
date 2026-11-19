# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unified Dosi semantic-model and metric authoring node."""

from __future__ import annotations

from typing import Any, Final, Iterable, List, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.semantic_authoring_agentic_node import SemanticAuthoringAgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory
from datus.schemas.semantic_agentic_node_models import SemanticModelingNodeResult, SemanticNodeInput
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.registry.contracts import tool_contract
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_SUPPORTED_SEMANTIC_MODELING_STATUSES: Final[frozenset[str]] = frozenset({"generated", "skipped", "blocked"})
_SEMANTIC_MODELING_RESULT_RETRY_PROMPT: Final[str] = """Your semantic model changes have been preserved.
Return only one JSON object with this shape:
{
  "status": "generated",
  "blocker_code": null,
  "skip_reason": null,
  "output": "brief summary of reused and changed semantic objects"
}
Use status "blocked" or "skipped" only when appropriate under the original instructions.
You may use the available tools first if more work or verification is needed.
"""


class SemanticModelingResultRetryPolicy:
    """Retry once when semantic modeling returns an unsupported outcome status."""

    max_attempts: int = 2

    def __init__(self, node: "SemanticModelingAgenticNode"):
        self.node = node

    def reset(self, ctx: StreamRunContext) -> None:
        return None

    def should_retry(self, ctx: StreamRunContext) -> bool:
        response_content = self.node._response_content(ctx)
        status, _, _, _ = self.node._parse_semantic_modeling_response(response_content)
        return status not in _SUPPORTED_SEMANTIC_MODELING_STATUSES

    def next_prompt(self, ctx: StreamRunContext) -> str:
        return _SEMANTIC_MODELING_RESULT_RETRY_PROMPT

    def on_retry_actions(self, ctx: StreamRunContext) -> Iterable[ActionHistory]:
        return ()

    def finalise(self, ctx: StreamRunContext) -> None:
        return None


class SemanticModelingAgenticNode(SemanticAuthoringAgenticNode):
    """Author Dosi datasets, metrics, or both against one selected model."""

    NODE_NAME = "semantic_modeling"
    INCLUDE_OSI_CORE_SPEC = True
    result_class = SemanticModelingNodeResult

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        scope: Optional[str] = None,
        authoring_scope: Literal["datasets", "full"] = "full",
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ):
        from datus.utils.exceptions import DatusException, ErrorCode

        if authoring_scope not in {"datasets", "full"}:
            raise DatusException(
                ErrorCode.COMMON_UNSUPPORTED,
                message_args={"your_value": authoring_scope, "field_name": "authoring_scope"},
            )
        self.authoring_scope = authoring_scope
        super().__init__(
            agent_config=agent_config,
            execution_mode=execution_mode,
            scope=scope,
            is_subagent=is_subagent,
            session_id=session_id,
        )

    def setup_tools(self):
        """Compose shared authoring capabilities and the node's interactive channel."""
        from datus.agent.semantic_authoring_session import SemanticAuthoringSession

        self.authoring_session = SemanticAuthoringSession(
            self.agent_config,
            node_name=self.NODE_NAME,
            authoring_scope=self.authoring_scope,
            target_state=self.osi_target_state,
            generation_evidence=self.generation_evidence,
        )
        self.authoring_session.setup(
            filesystem_factory=self._make_filesystem_tool,
            runtime_db_context=self._semantic_runtime_db_context,
            filesystem_root=self._resolve_workspace_root(),
        )
        for name in (
            "osi_target_tools",
            "db_func_tool",
            "semantic_tools",
            "generation_tools",
            "filesystem_func_tool",
            "semantic_model_workspace",
            "semantic_model_editing_tools",
            "semantic_plan_file_tool",
            "semantic_verification_tools",
        ):
            setattr(self, name, getattr(self.authoring_session, name))
        self.tools = self.authoring_session.available_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()
        logger.info("Setup %s tools for %s: %s", len(self.tools), self.NODE_NAME, [tool.name for tool in self.tools])

    async def _before_stream(self, ctx: StreamRunContext) -> None:
        await super()._before_stream(ctx)
        self.authoring_session.reset_completion()

    @tool_contract(access="read", direct_call=True, category="semantic_tools")
    def list_existing_osi_semantic_models(self) -> FuncToolResult:
        """Inspect existing models before selecting an authoring target."""
        return self.authoring_session.list_existing_osi_semantic_models()

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
        return self.authoring_session.plan_osi_semantic_model_target(
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
        return self.authoring_session.bind_osi_semantic_model_target(
            semantic_model_file=semantic_model_file, semantic_model_name=semantic_model_name
        )

    def _build_enhanced_message(
        self,
        user_input: SemanticNodeInput,
        extra_enhanced_parts: Optional[List[str]] = None,
    ) -> str:
        """Keep explicit target values as hints until existing models are checked."""
        parts = list(extra_enhanced_parts or [])
        if self.authoring_scope == "datasets":
            parts.append(
                "## Authoring Scope\n"
                "This run is datasets-only. Create or update datasets, fields, relationships, and model metadata "
                "as needed. Do not create, update, or delete metrics. Existing metric definitions must remain "
                "semantically unchanged."
            )
        else:
            parts.append(
                "## Authoring Scope\n"
                "This run has full scope. Create or update datasets, fields, relationships, model metadata, and "
                "metrics as required by the request."
            )
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

    def _prepare_template_context(self, user_input: SemanticNodeInput) -> dict:
        context = super()._prepare_template_context(user_input)
        context["authoring_scope"] = self.authoring_scope
        return context

    def _get_retry_policy(self) -> SemanticModelingResultRetryPolicy:
        return SemanticModelingResultRetryPolicy(self)

    @staticmethod
    def _response_content(ctx: StreamRunContext) -> Any:
        response_content = ctx.response_content
        if not response_content and ctx.last_successful_output:
            raw_output = ctx.last_successful_output.get("raw_output", "")
            response_content = (
                raw_output if isinstance(raw_output, dict) or raw_output else str(ctx.last_successful_output)
            )
        return response_content

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
        response_content = self._response_content(ctx)

        status, blocker_code, skip_reason, extracted_output = self._parse_semantic_modeling_response(response_content)
        if extracted_output:
            response_content = extracted_output
        if not isinstance(response_content, str):
            response_content = str(response_content) if response_content else ""

        if status not in _SUPPORTED_SEMANTIC_MODELING_STATUSES:
            raise RuntimeError("semantic_modeling must return a supported status.")
        if blocker_code is not None and status != "blocked":
            raise RuntimeError("blocker_code is only valid when status='blocked'.")
        if skip_reason is not None and status != "skipped":
            raise RuntimeError("skip_reason is only valid when status='skipped'.")

        self._enforce_authoring_scope()

        semantic_model_files: List[str] = []
        if status == "blocked":
            if blocker_code not in {
                "semantic_model_required",
                "semantic_model_selection_required",
                "semantic_model_target_invalid",
                "semantic_model_inline_in_mapping",
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
            tokens_used = self._extract_total_tokens(ctx.get_run_actions())
        return SemanticModelingNodeResult(
            success=status != "blocked",
            error=response_content if status == "blocked" else None,
            response=response_content,
            semantic_models=semantic_model_files,
            tokens_used=int(tokens_used),
            status=status,
            blocker_code=blocker_code if status == "blocked" else None,
            skip_reason=skip_reason if status == "skipped" else None,
            business_validation=self.generation_evidence.business_verification_summary(
                self.osi_target_state.selected_path or ""
            ),
        )

    def _target_was_mutated(self) -> bool:
        return self.authoring_session._target_was_mutated()

    def _enforce_authoring_scope(self) -> None:
        """Reject and roll back metric mutations made during a datasets-only run."""
        return self.authoring_session._enforce_authoring_scope()

    def _selected_osi_artifact(self) -> tuple[str, str, str]:
        return self.authoring_session._selected_osi_artifact()

    def _finalize_selected_osi_artifact(self) -> str:
        """Validate and reconcile with the same host-owned session used by plugins."""
        self.authoring_session.semantic_tools = self.semantic_tools
        return self.authoring_session.finalize()
