# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared factory functions for creating interactive agentic nodes and their inputs.

Used by CLI print mode and interactive REPL to avoid duplicating node creation logic.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig


def create_interactive_node(
    subagent_name: Optional[str],
    agent_config: "AgentConfig",
    node_id_suffix: str = "",
    scope: Optional[str] = None,
):
    """Create an interactive agentic node based on subagent_name.

    Args:
        subagent_name: Name of the subagent, or None for default chat node.
        agent_config: Agent configuration.
        node_id_suffix: Suffix appended to node_id (e.g. "_cli", "_print").
        scope: Optional session scope for directory isolation.
    """
    if subagent_name:
        node_class_type = _resolve_node_class_type(subagent_name, agent_config)

        if subagent_name == "gen_semantic_model":
            from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

            return GenSemanticModelAgenticNode(agent_config=agent_config, execution_mode="interactive", scope=scope)

        elif subagent_name == "gen_metrics":
            from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

            return GenMetricsAgenticNode(agent_config=agent_config, execution_mode="interactive", scope=scope)

        elif subagent_name == "gen_sql_summary":
            from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

            return SqlSummaryAgenticNode(
                node_name=subagent_name, agent_config=agent_config, execution_mode="interactive", scope=scope
            )

        elif subagent_name == "gen_ext_knowledge":
            from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode

            return GenExtKnowledgeAgenticNode(
                node_name=subagent_name, agent_config=agent_config, execution_mode="interactive", scope=scope
            )

        elif subagent_name == "gen_table" or node_class_type == "gen_table":
            from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

            return GenTableAgenticNode(
                agent_config=agent_config,
                execution_mode="interactive",
                node_name=subagent_name if node_class_type == "gen_table" else None,
            )

        elif subagent_name == "gen_job":
            from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

            return GenJobAgenticNode(agent_config=agent_config, execution_mode="interactive")

        elif subagent_name == "gen_report" or node_class_type == "gen_report":
            from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

            return GenReportAgenticNode(
                node_id=f"{subagent_name}{node_id_suffix}",
                description=f"Report generation node for {subagent_name}",
                node_type="gen_report",
                input_data=None,
                agent_config=agent_config,
                tools=None,
                node_name=subagent_name,
                scope=scope,
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
                node_id=f"{subagent_name}{node_id_suffix}",
                description=f"Explore node for {subagent_name}",
                node_type="explore",
                input_data=None,
                agent_config=agent_config,
                tools=None,
                node_name=subagent_name,
            )

        elif subagent_name == "gen_skill" or node_class_type == "gen_skill":
            from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

            return SkillCreatorAgenticNode(
                node_id=f"{subagent_name}{node_id_suffix}",
                description=f"Skill generation node for {subagent_name}",
                node_type="gen_skill",
                input_data=None,
                agent_config=agent_config,
                tools=None,
                node_name=subagent_name if node_class_type == "gen_skill" else "gen_skill",
            )

        elif subagent_name == "gen_dashboard" or node_class_type == "gen_dashboard":
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            return GenDashboardAgenticNode(
                agent_config=agent_config,
                execution_mode="interactive",
                node_id=f"{subagent_name}{node_id_suffix}",
                node_name=subagent_name if node_class_type == "gen_dashboard" else None,
                scope=scope,
            )

        elif subagent_name == "scheduler" or node_class_type == "scheduler":
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            return SchedulerAgenticNode(
                agent_config=agent_config,
                execution_mode="interactive",
                node_id=f"{subagent_name}{node_id_suffix}",
                node_name=subagent_name if node_class_type == "scheduler" else None,
                scope=scope,
            )

        elif subagent_name == "feedback":
            from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

            return FeedbackAgenticNode(
                agent_config=agent_config,
                execution_mode="interactive",
                scope=scope,
            )

        else:
            from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

            return GenSQLAgenticNode(
                node_id=f"{subagent_name}{node_id_suffix}",
                description=f"SQL generation node for {subagent_name}",
                node_type="gensql",
                input_data=None,
                agent_config=agent_config,
                tools=None,
                node_name=subagent_name,
                scope=scope,
            )
    else:
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        return ChatAgenticNode(
            node_id=f"chat{node_id_suffix}",
            description="Chat node for interactive mode",
            node_type="chat",
            input_data=None,
            agent_config=agent_config,
            tools=None,
            scope=scope,
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
    prompt_language: str = "en",
    plan_mode: bool = False,
    source_session_id: Optional[str] = None,
):
    """Create node input based on node type.

    Args:
        user_message: The user's message.
        node: The target node instance (used for isinstance dispatch).
        catalog: Optional catalog name.
        database: Optional database name.
        db_schema: Optional schema name.
        at_tables: @-referenced tables.
        at_metrics: @-referenced metrics.
        at_sqls: @-referenced SQL queries.
        prompt_language: Language for prompts (default "en").
        plan_mode: Whether to enable plan mode.
        source_session_id: Source session the feedback node should copy from.
            Only consumed by :class:`FeedbackAgenticNode`.
    """
    from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode
    from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode
    from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
    from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode
    from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
    from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
    from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode
    from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

    if isinstance(
        node,
        (
            GenSemanticModelAgenticNode,
            GenMetricsAgenticNode,
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

    elif isinstance(node, GenExtKnowledgeAgenticNode):
        from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput

        return ExtKnowledgeNodeInput(
            user_message=user_message,
            prompt_language=prompt_language,
            catalog=catalog,
            database=database,
            db_schema=db_schema,
        )

    elif isinstance(node, GenReportAgenticNode):
        from datus.schemas.gen_report_agentic_node_models import GenReportNodeInput

        return GenReportNodeInput(
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
            schemas=at_tables,
            metrics=at_metrics,
            reference_sql=at_sqls,
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
            schemas=at_tables,
            metrics=at_metrics,
            reference_sql=at_sqls,
            plan_mode=plan_mode,
        )


def _resolve_node_class_type(subagent_name: str, agent_config: "AgentConfig") -> Optional[str]:
    """Resolve node_class from agent config for a subagent."""
    if hasattr(agent_config, "agentic_nodes") and agent_config.agentic_nodes:
        node_config = agent_config.agentic_nodes.get(subagent_name, {})
        if hasattr(node_config, "model_dump"):
            node_config = node_config.model_dump()
        if isinstance(node_config, dict):
            return node_config.get("node_class")
    return None
