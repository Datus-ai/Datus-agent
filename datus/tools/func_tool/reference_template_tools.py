# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
import json
from typing import Any, Dict, List, Optional

import jinja2
from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.storage.reference_template.store import ReferenceTemplateRAG
from datus.tools.func_tool.base import FuncToolResult, normalize_null, trans_to_function_tool
from datus.utils.loggings import get_logger
from datus.utils.mcp_decorators import mcp_tool, mcp_tool_class

logger = get_logger(__name__)


@mcp_tool_class(
    name="reference_template_tool",
    availability_property="has_reference_template_tools",
)
class ReferenceTemplateTools:
    @classmethod
    def create_dynamic(
        cls, agent_config: AgentConfig, sub_agent_name: Optional[str] = None
    ) -> "ReferenceTemplateTools":
        """Create ReferenceTemplateTools instance for dynamic mode.

        Args:
            agent_config: Agent configuration
            sub_agent_name: Optional sub-agent name

        Returns:
            ReferenceTemplateTools instance
        """
        return cls(agent_config, sub_agent_name=sub_agent_name)

    @classmethod
    def create_static(
        cls,
        agent_config: AgentConfig,
        sub_agent_name: Optional[str] = None,
        database_name: Optional[str] = None,
    ) -> "ReferenceTemplateTools":
        """Create ReferenceTemplateTools instance for static mode.

        Args:
            agent_config: Agent configuration
            sub_agent_name: Optional sub-agent name
            database_name: Optional database name (unused, for API compatibility)

        Returns:
            ReferenceTemplateTools instance
        """
        return cls(agent_config, sub_agent_name=sub_agent_name)

    def __init__(self, agent_config: AgentConfig, sub_agent_name: Optional[str] = None, db_func_tool=None):
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.reference_template_store = ReferenceTemplateRAG(agent_config, sub_agent_name)
        self.has_reference_templates = self.reference_template_store.get_reference_template_size() > 0
        self.db_func_tool = db_func_tool

    def available_tools(self) -> List[Tool]:
        tools = []
        if self.has_reference_templates:
            tools.append(trans_to_function_tool(self.search_reference_template))
            tools.append(trans_to_function_tool(self.get_reference_template))
            tools.append(trans_to_function_tool(self.render_reference_template))
            if self.db_func_tool:
                tools.append(trans_to_function_tool(self.execute_reference_template))
        return tools

    @mcp_tool(availability_check="has_reference_templates")
    def search_reference_template(
        self, query_text: str, subject_path: Optional[List[str]] = None, top_n: int = 5
    ) -> FuncToolResult:
        """
        Search for reference SQL templates using natural language queries.
        MUST call `list_subject_tree` first to get the subject path.

        **Application Guidance**: If matches are found, call `execute_reference_template` with the
        `subject_path`, `name`, and parameter values from results to render and execute in one step.
        Do NOT write SQL from scratch when a matching template exists.

        Args:
            query_text: The natural language query text representing the desired SQL intent.
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
            top_n: The number of top results to return (default 5).

        Returns:
            FuncToolResult with list of matching templates, each containing:
                - 'name': Template name
                - 'template': The raw Jinja2 SQL template
                - 'parameters': JSON string of parameter definitions with type metadata
                - 'summary': Brief description of what the template does
                - 'tags': Associated tags
        """
        subject_path = normalize_null(subject_path)
        try:
            result = self.reference_template_store.search_reference_templates(
                query_text=query_text,
                subject_path=subject_path,
                top_n=top_n,
                selected_fields=["name", "template", "parameters", "summary", "tags"],
            )
            return FuncToolResult(success=1, error=None, result=result)
        except Exception as e:
            logger.error(f"Failed to search reference templates for `{query_text}`: {e}")
            return FuncToolResult(success=0, error=str(e))

    @mcp_tool(availability_check="has_reference_templates")
    def get_reference_template(self, subject_path: List[str], name: str = "") -> FuncToolResult:
        """
        Get reference template detail by exact subject path and name.
        **IMPORTANT**: You MUST call `search_reference_template` first to discover valid subject_path and name values.
        Do NOT guess subject_path or name — they must come from search results.

        For dimension-type parameters, this tool enriches the response with `sample_values` —
        actual distinct values queried from the database — so you know exactly what values to use
        when calling `execute_reference_template`.

        Args:
            subject_path: Subject hierarchy path from search results (e.g., ['Finance', 'Revenue', 'Q1'])
            name: The exact name from search results.

        Returns:
            FuncToolResult with a single matching entry containing:
                - 'name': Template name
                - 'template': The raw Jinja2 SQL template
                - 'parameters': JSON string of parameter definitions with type metadata and sample_values
                - 'comment': Optional comment about the template
                - 'summary': Brief description of what the template does
                - 'tags': Associated tags
            Returns success=0 with error="No matched result" if not found.
        """
        name = normalize_null(name) or ""
        try:
            result = self.reference_template_store.get_reference_template_detail(
                subject_path=subject_path,
                name=name,
                selected_fields=["name", "template", "parameters", "comment", "summary", "tags"],
            )
            if len(result) > 0:
                return FuncToolResult(success=1, error=None, result=result[0])
            return FuncToolResult(success=0, error="No matched result", result=None)
        except Exception as e:
            logger.error(f"Failed to get reference template for `{'/'.join(subject_path)}/{name}`: {e}")
            return FuncToolResult(success=0, error=str(e))

    @mcp_tool(availability_check="has_reference_templates")
    def render_reference_template(self, subject_path: List[str], name: str, params: str) -> FuncToolResult:
        """
        Render a reference template with the given parameters to produce final SQL.
        The template is identified by subject_path + name, and rendered server-side using Jinja2.

        **Workflow**: First use `search_reference_template` or `get_reference_template` to find the template
        and its required parameters, then call this tool with appropriate parameter values.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
            name: The exact name of the reference template.
            params: JSON string of parameter key-value pairs to render the template.
                    Keys must match the template's parameter names.
                    Example: '{"start_date": "2024-01-01", "end_date": "2024-12-31", "region": "US"}'

        Returns:
            FuncToolResult with:
                - 'rendered_sql': The final rendered SQL string
                - 'template_name': Name of the template used
                - 'parameters_used': The parameters that were applied
            Returns success=0 with descriptive error if template not found or rendering fails.
        """
        name = normalize_null(name) or ""
        # Parse params from JSON string
        try:
            params_dict: Dict[str, Any] = json.loads(params) if isinstance(params, str) else params
        except (json.JSONDecodeError, TypeError) as e:
            return FuncToolResult(
                success=0,
                error=f"Invalid params format. Expected a JSON string like "
                f'{{"start_date": "2024-01-01", "region": "US"}}. Error: {e}',
            )
        try:
            # Fetch the template
            result = self.reference_template_store.get_reference_template_detail(
                subject_path=subject_path,
                name=name,
                selected_fields=["name", "template", "parameters"],
            )
            if not result:
                return FuncToolResult(success=0, error=f"Template not found: {'/'.join(subject_path)}/{name}")

            template_data = result[0]
            template_content = template_data.get("template", "")
            template_name = template_data.get("name", name)

            if not template_content:
                return FuncToolResult(success=0, error="Template content is empty")

            # Parse expected parameters for error reporting
            template_params_json = template_data.get("parameters", "[]")
            try:
                param_list = json.loads(template_params_json)
                expected_params = [p["name"] for p in param_list]
            except (json.JSONDecodeError, KeyError):
                expected_params = []

            provided_params = list(params_dict.keys())

            # Render the template using Jinja2 with strict undefined checking
            try:
                jinja_template = jinja2.Template(template_content, undefined=jinja2.StrictUndefined)
                rendered_sql = jinja_template.render(**params_dict)
            except jinja2.UndefinedError as e:
                # Provide detailed error message to help model retry with correct params
                missing = sorted(set(expected_params) - set(provided_params))
                return FuncToolResult(
                    success=0,
                    error=f"Rendering failed: {e}. "
                    f"Template '{template_name}' requires parameters: {expected_params}. "
                    f"You provided: {provided_params}. "
                    f"Missing parameters: {missing if missing else 'unknown'}. "
                    f"Please retry with all required parameters.",
                )
            except jinja2.TemplateSyntaxError as e:
                return FuncToolResult(success=0, error=f"Template syntax error: {e}")

            return FuncToolResult(
                success=1,
                error=None,
                result={
                    "rendered_sql": rendered_sql,
                    "template_name": template_name,
                    "parameters_used": params_dict,
                },
            )

        except Exception as e:
            logger.error(f"Failed to render reference template `{'/'.join(subject_path)}/{name}`: {e}")
            return FuncToolResult(success=0, error=str(e))

    @mcp_tool(availability_check="has_reference_templates")
    def execute_reference_template(
        self, subject_path: List[str], name: str, params: str, database: str = ""
    ) -> FuncToolResult:
        """
        **PREFERRED** way to use reference templates. Render a template with parameters and immediately
        execute the resulting SQL (read-only), returning query results in one step.

        **Workflow**: Call `search_reference_template` first to find the template and its parameters,
        then call this tool with the `subject_path`, `name`, and parameter values from search results.
        Do NOT write ad-hoc SQL when a matching template exists — use this tool instead.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
            name: The exact name of the reference template.
            params: JSON string of parameter key-value pairs to render the template.
                    Keys must match the template's parameter names.
                    Example: '{"start_date": "2024-01-01", "end_date": "2024-12-31", "region": "US"}'
            database: Optional database name for multi-database scenarios.

        Returns:
            FuncToolResult with:
                - 'rendered_sql': The SQL that was executed
                - 'template_name': Name of the template used
                - 'parameters_used': The parameters that were applied
                - 'query_result': The query execution result (rows)
            Returns success=0 with descriptive error if template not found, rendering fails,
            or query execution fails.
        """
        # Step 1: Render the template
        render_result = self.render_reference_template(subject_path, name, params)
        if render_result.success == 0:
            return render_result

        rendered_sql = render_result.result["rendered_sql"]
        template_name = render_result.result["template_name"]
        parameters_used = render_result.result["parameters_used"]

        # Step 2: Execute the rendered SQL via db_func_tool
        if not self.db_func_tool:
            return FuncToolResult(
                success=0,
                error="Database tools not available. Use `render_reference_template` to get the SQL, "
                "then execute it manually with `read_query`.",
            )

        try:
            exec_result = self.db_func_tool.read_query(rendered_sql, database=database)
            if exec_result.success == 0:
                return FuncToolResult(
                    success=0,
                    error=f"Template rendered successfully but query execution failed: {exec_result.error}",
                    result={
                        "rendered_sql": rendered_sql,
                        "template_name": template_name,
                        "parameters_used": parameters_used,
                    },
                )

            return FuncToolResult(
                success=1,
                error=None,
                result={
                    "rendered_sql": rendered_sql,
                    "template_name": template_name,
                    "parameters_used": parameters_used,
                    "query_result": exec_result.result,
                },
            )
        except Exception as e:
            logger.error(f"Failed to execute rendered template `{'/'.join(subject_path)}/{name}`: {e}")
            return FuncToolResult(
                success=0,
                error=f"Template rendered but execution failed: {e}",
                result={
                    "rendered_sql": rendered_sql,
                    "template_name": template_name,
                    "parameters_used": parameters_used,
                },
            )
