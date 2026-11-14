# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared factory functions for creating interactive agentic nodes and their inputs.

Used by CLI print mode and interactive REPL to avoid duplicating node creation logic.
"""

from typing import TYPE_CHECKING, Callable, Literal, Optional

from datus.schemas.at_context import apply_at_context
from datus.utils.constants import HIDDEN_SYS_SUB_AGENTS, SYS_SUB_AGENTS
from datus.utils.loggings import get_logger
from datus.utils.node_utils import validate_node_class

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig


logger = get_logger(__name__)


def _configured_agentic_node_names(agent_config: "AgentConfig") -> frozenset[str]:
    agentic_nodes = getattr(agent_config, "agentic_nodes", None)
    if not agentic_nodes or not hasattr(agentic_nodes, "keys"):
        return frozenset()
    return frozenset(str(name) for name in agentic_nodes.keys())


def _visible_subagent_names(agent_config: "AgentConfig") -> frozenset[str]:
    visible_names = set(SYS_SUB_AGENTS - HIDDEN_SYS_SUB_AGENTS)
    agentic_nodes = getattr(agent_config, "agentic_nodes", None)
    if not agentic_nodes or not hasattr(agentic_nodes, "items"):
        return frozenset(visible_names)

    current_db = getattr(agent_config, "current_datasource", None)
    for name, sub_config in agentic_nodes.items():
        name = str(name)
        if name in SYS_SUB_AGENTS or name == "chat":
            continue
        if hasattr(sub_config, "model_dump"):
            sub_config = sub_config.model_dump()
        if isinstance(sub_config, dict):
            scoped_context = sub_config.get("scoped_context") or {}
            if hasattr(scoped_context, "model_dump"):
                scoped_context = scoped_context.model_dump()
            if isinstance(scoped_context, dict):
                scoped_datasource = scoped_context.get("datasource")
                if scoped_datasource and scoped_datasource != current_db:
                    continue
        visible_names.add(name)
    return frozenset(visible_names)


def create_interactive_node(
    subagent_name: Optional[str],
    agent_config: "AgentConfig",
    node_id_suffix: str = "",
    scope: Optional[str] = None,
    *,
    execution_mode: Literal["interactive", "workflow"] = "interactive",
    node_id: Optional[str] = None,
    session_id: Optional[str] = None,
    on_warning: Optional[Callable[[str], None]] = None,
):
    """Create an interactive agentic node based on subagent_name.

    Args:
        subagent_name: Name of the subagent, or None for default chat node.
        agent_config: Agent configuration.
        node_id_suffix: Suffix appended to node_id (e.g. "_cli", "_print").
        scope: Optional session scope for directory isolation.
        execution_mode: Node execution mode. CLI defaults to ``interactive``;
            API workflow runs pass ``workflow``.
        node_id: Override the auto-generated ``f"{subagent_name}{node_id_suffix}"``
            id. API path passes ``session_id`` here so the node's id matches the
            HTTP session.
        session_id: Optional resume target. When provided, forwarded into the
            node constructor so :meth:`AgenticNode.restore_plan_mode_state` runs
            before the first turn, rehydrating persisted plan-mode fields. When
            omitted, the node eagerly generates a fresh ``session_id`` in its
            ``__init__``.
        on_warning: Optional callback for warnings the caller should show its
            user. The CLI logs to file only, so a logged warning alone never
            reaches the terminal or the print-mode JSON stream.
    """
    if subagent_name:
        node_class_type = _resolve_node_class_type(subagent_name, agent_config)
        validate_node_class(node_class_type, subagent_name)

        if subagent_name == "semantic_modeling":
            from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

            return SemanticModelingAgenticNode(
                agent_config=agent_config, execution_mode=execution_mode, scope=scope, session_id=session_id
            )

        elif subagent_name == "gen_sql_summary":
            from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

            return SqlSummaryAgenticNode(
                node_name=subagent_name,
                agent_config=agent_config,
                execution_mode=execution_mode,
                scope=scope,
                session_id=session_id,
            )

        elif subagent_name == "gen_table" or node_class_type == "gen_table":
            from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

            return GenTableAgenticNode(
                agent_config=agent_config,
                execution_mode=execution_mode,
                node_name=subagent_name if node_class_type == "gen_table" else None,
                scope=scope,
                session_id=session_id,
            )

        elif subagent_name == "gen_job":
            from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

            return GenJobAgenticNode(
                agent_config=agent_config, execution_mode=execution_mode, scope=scope, session_id=session_id
            )

        elif subagent_name == "ontology_analyst" or node_class_type == "ontology_analyst":
            from datus.agent.node.ontology_analyst_agentic_node import OntologyAnalystAgenticNode
            from datus.configuration.node_type import NodeType

            return OntologyAnalystAgenticNode(
                node_id=node_id if node_id is not None else f"{subagent_name}{node_id_suffix}",
                description=f"Ontology question-answering node for {subagent_name}",
                node_type=NodeType.TYPE_ONTOLOGY_ANALYST,
                input_data=None,
                agent_config=agent_config,
                tools=None,
                node_name=subagent_name,
                scope=scope,
                execution_mode=execution_mode,
                session_id=session_id,
            )

        elif subagent_name == "ask_metrics" or node_class_type == "ask_metrics":
            from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode
            from datus.configuration.node_type import NodeType

            return AskMetricsAgenticNode(
                node_id=node_id if node_id is not None else f"{subagent_name}{node_id_suffix}",
                description=f"Metric question-answering node for {subagent_name}",
                node_type=NodeType.TYPE_ASK_METRICS,
                input_data=None,
                agent_config=agent_config,
                tools=None,
                node_name=subagent_name,
                scope=scope,
                execution_mode=execution_mode,
                session_id=session_id,
            )

        elif subagent_name == "gen_visual_report" or node_class_type == "gen_visual_report":
            from datus.agent.node.gen_visual_report_agentic_node import GenVisualReportAgenticNode

            return GenVisualReportAgenticNode(
                node_id=node_id if node_id is not None else f"{subagent_name}{node_id_suffix}",
                description=f"Visual report generation node for {subagent_name}",
                node_type="gen_visual_report",
                input_data=None,
                agent_config=agent_config,
                tools=None,
                node_name=subagent_name,
                scope=scope,
                execution_mode=execution_mode,
                session_id=session_id,
            )

        elif subagent_name == "gen_visual_dashboard" or node_class_type == "gen_visual_dashboard":
            from datus.agent.node.gen_visual_dashboard_agentic_node import GenVisualDashboardAgenticNode

            return GenVisualDashboardAgenticNode(
                node_id=node_id if node_id is not None else f"{subagent_name}{node_id_suffix}",
                description=f"Visual dashboard generation node for {subagent_name}",
                node_type="gen_visual_dashboard",
                input_data=None,
                agent_config=agent_config,
                tools=None,
                node_name=subagent_name,
                scope=scope,
                execution_mode=execution_mode,
                session_id=session_id,
            )

        elif subagent_name == "explore" or node_class_type == "explore":
            # Wire the read-only `explore` sub-agent into the interactive
            # path. `ExploreAgenticNode` is a first-class node type (see
            # datus/configuration/node_type.py TYPE_EXPLORE) that exposes
            # db_tools / context_search / filesystem in read-only mode. It's
            # used by the DAComp runner to chain explore -> dbt_layered so
            # the second agent sees pre-flight data-reality findings in its
            # user prompt instead of having to run the discovery itself.
            from datus.agent.node.explore_agentic_node import ExploreAgenticNode

            return ExploreAgenticNode(
                node_id=node_id if node_id is not None else f"{subagent_name}{node_id_suffix}",
                description=f"Explore node for {subagent_name}",
                node_type="explore",
                input_data=None,
                agent_config=agent_config,
                tools=None,
                node_name=subagent_name,
                execution_mode=execution_mode,
                session_id=session_id,
            )

        elif subagent_name == "gen_skill" or node_class_type == "gen_skill":
            from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

            return SkillCreatorAgenticNode(
                node_id=node_id if node_id is not None else f"{subagent_name}{node_id_suffix}",
                description=f"Skill generation node for {subagent_name}",
                node_type="gen_skill",
                input_data=None,
                agent_config=agent_config,
                tools=None,
                node_name=subagent_name if node_class_type == "gen_skill" else "gen_skill",
                execution_mode=execution_mode,
                scope=scope,
                session_id=session_id,
            )

        elif subagent_name == "feedback":
            from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

            return FeedbackAgenticNode(
                agent_config=agent_config,
                execution_mode=execution_mode,
                scope=scope,
                session_id=session_id,
            )

        elif subagent_name == "ask_report" or node_class_type == "ask_report":
            from datus.agent.node.ask_report_agentic_node import AskReportAgenticNode
            from datus.configuration.node_type import NodeType

            # ask_* subagents are conversational (ChatAgenticNode subclass),
            # so they reuse ``NodeType.TYPE_CHAT`` rather than minting a new
            # action type. The behavioural difference (artifact binding,
            # constrained filesystem root, prompt prefix) lives entirely on
            # the class; the node-level type just signals "chat-flavored".
            return AskReportAgenticNode(
                node_id=node_id if node_id is not None else f"{subagent_name}{node_id_suffix}",
                description=f"Ask-report follow-up node for {subagent_name}",
                node_type=NodeType.TYPE_CHAT,
                input_data=None,
                agent_config=agent_config,
                tools=None,
                node_name=subagent_name,
                scope=scope,
                execution_mode=execution_mode,
                session_id=session_id,
            )

        elif subagent_name == "ask_dashboard" or node_class_type == "ask_dashboard":
            from datus.agent.node.ask_dashboard_agentic_node import AskDashboardAgenticNode
            from datus.configuration.node_type import NodeType

            return AskDashboardAgenticNode(
                node_id=node_id if node_id is not None else f"{subagent_name}{node_id_suffix}",
                description=f"Ask-dashboard follow-up node for {subagent_name}",
                node_type=NodeType.TYPE_CHAT,
                input_data=None,
                agent_config=agent_config,
                tools=None,
                node_name=subagent_name,
                scope=scope,
                execution_mode=execution_mode,
                session_id=session_id,
            )

        else:
            # A configured agent scoped to another datasource is not available here,
            # so recognize only the names visible for the current datasource.
            visible_names = _visible_subagent_names(agent_config)
            if subagent_name not in SYS_SUB_AGENTS and subagent_name not in visible_names:
                available_names = ", ".join(sorted(visible_names))
                warning = (
                    f"Unknown subagent {subagent_name!r}; falling back to gen_sql. "
                    f"Available subagents: {available_names}"
                )
                logger.warning(warning)
                if on_warning is not None:
                    on_warning(warning)

            from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

            return GenSQLAgenticNode(
                node_id=node_id if node_id is not None else f"{subagent_name}{node_id_suffix}",
                description=f"SQL generation node for {subagent_name}",
                node_type="gen_sql",
                input_data=None,
                agent_config=agent_config,
                tools=None,
                node_name=subagent_name,
                scope=scope,
                execution_mode=execution_mode,
                session_id=session_id,
            )
    else:
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        return ChatAgenticNode(
            node_id=node_id if node_id is not None else f"chat{node_id_suffix}",
            description="Chat node for interactive mode",
            node_type="chat",
            input_data=None,
            agent_config=agent_config,
            tools=None,
            scope=scope,
            execution_mode=execution_mode,
            session_id=session_id,
        )


def create_node_input(
    user_message: str,
    node,
    catalog: Optional[str] = None,
    database: Optional[str] = None,
    db_schema: Optional[str] = None,
    scoped_tables=None,
    at_tables=None,
    at_metrics=None,
    at_sqls=None,
    external_knowledge: Optional[str] = None,
    context_hints=None,
    prompt_language: str = "en",
    plan_mode: bool = False,
    source_session_id: Optional[str] = None,
):
    """Create node input for *node*, then attach resolved @-context uniformly.

    The type dispatch lives in :func:`_build_typed_node_input`; @-context
    (``at_tables`` / ``at_metrics`` / ``at_sqls`` / ``external_knowledge``) is
    applied here through the single :func:`apply_at_context` choke point so no
    per-node branch can forget it. Any input inheriting
    :class:`~datus.schemas.at_context.AtContextInput` picks the context up;
    others are skipped silently.

    Args:
        user_message: The user's message.
        node: The target node instance (used for isinstance dispatch).
        catalog: Optional catalog name.
        database: Optional database name.
        db_schema: Optional schema name.
        at_tables: @-referenced tables.
        at_metrics: @-referenced metrics.
        at_sqls: @-referenced SQL queries.
        external_knowledge: @-referenced supplementary knowledge text.
        prompt_language: Language for prompts (default "en").
        plan_mode: Whether to enable plan mode.
        source_session_id: Source session the feedback node should copy from.
            Only consumed by :class:`FeedbackAgenticNode`.
    """
    node_input = _build_typed_node_input(
        user_message=user_message,
        node=node,
        catalog=catalog,
        database=database,
        db_schema=db_schema,
        scoped_tables=scoped_tables,
        prompt_language=prompt_language,
        plan_mode=plan_mode,
        source_session_id=source_session_id,
    )
    return apply_at_context(
        node_input,
        schemas=at_tables,
        metrics=at_metrics,
        reference_sql=at_sqls,
        external_knowledge=external_knowledge,
        context_hints=context_hints,
    )


def _build_typed_node_input(
    user_message: str,
    node,
    catalog: Optional[str] = None,
    database: Optional[str] = None,
    db_schema: Optional[str] = None,
    scoped_tables=None,
    prompt_language: str = "en",
    plan_mode: bool = False,
    source_session_id: Optional[str] = None,
):
    """Construct the node-type-specific input object (no @-context wiring)."""
    from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode
    from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode
    from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
    from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode
    from datus.agent.node.gen_visual_dashboard_agentic_node import GenVisualDashboardAgenticNode
    from datus.agent.node.gen_visual_report_agentic_node import GenVisualReportAgenticNode
    from datus.agent.node.ontology_analyst_agentic_node import OntologyAnalystAgenticNode
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

    if isinstance(
        node,
        (
            SemanticModelingAgenticNode,
            GenTableAgenticNode,
            GenJobAgenticNode,
        ),
    ):
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        return SemanticNodeInput(
            user_message=user_message,
            catalog=catalog,
            database=database,
            db_schema=db_schema,
            prompt_language=prompt_language,
        )

    elif isinstance(node, SqlSummaryAgenticNode):
        from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput

        return SqlSummaryNodeInput(
            user_message=user_message,
            catalog=catalog,
            database=database,
            db_schema=db_schema,
            prompt_language=prompt_language,
        )

    elif isinstance(node, OntologyAnalystAgenticNode):
        from datus.schemas.ontology_analyst_agentic_node_models import OntologyAnalystNodeInput

        return OntologyAnalystNodeInput(
            user_message=user_message,
            catalog=catalog,
            database=database,
            db_schema=db_schema,
        )

    elif isinstance(node, AskMetricsAgenticNode):
        from datus.schemas.ask_metrics_agentic_node_models import AskMetricsNodeInput

        return AskMetricsNodeInput(
            user_message=user_message,
            catalog=catalog,
            database=database,
            db_schema=db_schema,
        )

    elif isinstance(node, GenVisualReportAgenticNode):
        from datus.schemas.gen_visual_report_models import GenVisualReportNodeInput

        return GenVisualReportNodeInput(
            user_message=user_message,
            catalog=catalog,
            database=database,
            db_schema=db_schema,
        )

    elif isinstance(node, GenVisualDashboardAgenticNode):
        from datus.schemas.gen_visual_dashboard_models import GenVisualDashboardNodeInput

        return GenVisualDashboardNodeInput(
            user_message=user_message,
            catalog=catalog,
            database=database,
            db_schema=db_schema,
        )

    elif isinstance(node, GenSQLAgenticNode):
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        return GenSQLNodeInput(
            user_message=user_message,
            catalog=catalog,
            database=database,
            db_schema=db_schema,
            prompt_language=prompt_language,
            plan_mode=plan_mode,
        )

    # ExploreAgenticNode is a read-only data exploration node. It takes
    # ExploreNodeInput (user_message + optional database). This branch is
    # required for the DAComp runner's explore -> dbt_layered chain where
    # the runner instantiates an `explore` sub-agent via
    # create_interactive_node() and then needs a matching input type.
    from datus.agent.node.explore_agentic_node import ExploreAgenticNode

    if isinstance(node, ExploreAgenticNode):
        from datus.schemas.explore_agentic_node_models import ExploreNodeInput

        return ExploreNodeInput(
            user_message=user_message,
            database=database,
            scoped_tables=list(scoped_tables) if scoped_tables else None,
        )

    from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

    if isinstance(node, SkillCreatorAgenticNode):
        from datus.schemas.gen_skill_agentic_node_models import SkillCreatorNodeInput

        return SkillCreatorNodeInput(user_message=user_message)

    from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

    if isinstance(node, FeedbackAgenticNode):
        from datus.schemas.feedback_agentic_node_models import FeedbackNodeInput

        # CLI path leaves source_session_id=None because _copy_session_for_switch
        # seeds the feedback node's session directly. API/Gateway paths pass a
        # real source_session_id so the node copies the source session itself.
        return FeedbackNodeInput(
            user_message=user_message,
            database=database,
            source_session_id=source_session_id,
        )

    else:
        from datus.schemas.chat_agentic_node_models import ChatNodeInput

        return ChatNodeInput(
            user_message=user_message,
            catalog=catalog,
            database=database,
            db_schema=db_schema,
            plan_mode=plan_mode,
        )


def resolve_node_name(subagent_name: Optional[str]) -> str:
    """Return the canonical ``get_node_name()`` for a factory-produced node.

    Mirrors :func:`create_interactive_node` so callers that need the node name
    *before* construction (e.g. to compute a session_id prefix via
    :meth:`SessionManager.copy_session`) don't have to instantiate the node.

    For every subagent dispatched by :func:`create_interactive_node`, the
    resulting node's ``get_node_name()`` equals ``subagent_name`` (or ``"chat"``
    when ``subagent_name`` is ``None``). Aliased custom subagents are no
    exception: the factory threads ``subagent_name`` into ``node_name``,
    which becomes ``configured_node_name``.
    """
    return subagent_name or "chat"


def _resolve_node_class_type(subagent_name: str, agent_config: "AgentConfig") -> Optional[str]:
    """Resolve runtime node class from agent config for a subagent.

    ``node_class`` wins for older/custom yaml, while API-created agents store
    the selected runtime class under ``type``.
    """
    if hasattr(agent_config, "agentic_nodes") and agent_config.agentic_nodes:
        node_config = agent_config.agentic_nodes.get(subagent_name, {})
        if hasattr(node_config, "model_dump"):
            node_config = node_config.model_dump()
        if isinstance(node_config, dict):
            return node_config.get("node_class") or node_config.get("type")
    return None
