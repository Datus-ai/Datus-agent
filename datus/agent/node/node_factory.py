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
):
    """Create an interactive agentic node based on subagent_name.

    Args:
        subagent_name: Name of the subagent, or None for default chat node.
        agent_config: Agent configuration.
        node_id_suffix: Suffix appended to node_id (e.g. "_cli", "_print").
    """
    if subagent_name:
        node_class_type = _resolve_node_class_type(subagent_name, agent_config)

        if subagent_name == "gen_semantic_model":
            from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

            return GenSemanticModelAgenticNode(agent_config=agent_config, execution_mode="interactive")

        elif subagent_name == "gen_metrics":
            from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

            return GenMetricsAgenticNode(agent_config=agent_config, execution_mode="interactive")

        elif subagent_name == "gen_sql_summary":
            from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

            return SqlSummaryAgenticNode(
                node_name=subagent_name, agent_config=agent_config, execution_mode="interactive"
            )

        elif subagent_name == "gen_ext_knowledge":
            from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode

            return GenExtKnowledgeAgenticNode(
                node_name=subagent_name, agent_config=agent_config, execution_mode="interactive"
            )

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
        )


def create_node_input(
    user_message: str,
    node,
    catalog: Optional[str] = None,
    database: Optional[str] = None,
    db_schema: Optional[str] = None,
    at_tables=None,
    at_metrics=None,
    at_sqls=None,
    prompt_language: str = "en",
    plan_mode: bool = False,
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
    """
    from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode
    from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
    from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode
    from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
    from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
    from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

    if isinstance(node, (GenSemanticModelAgenticNode, GenMetricsAgenticNode)):
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
