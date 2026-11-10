# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Tool access contracts for hosts that expose a restricted execution surface.

Keep these declarations with the agent release, rather than duplicating method
lists in every host. A new tool must declare its contract here; coverage tests
compare the catalog with the actual agent and MCP registries. Declarations are
not enforcement: hosts must install SQL and filesystem guards at invocation.

Family defaults are only used for capabilities a restricted host must never
execute. Mixed families require a method declaration, including new readers.
This module intentionally has no imports of tool implementations or services.
"""

from dataclasses import dataclass
from typing import Literal

Access = Literal["read", "write", "execute", "sql", "restricted"]


@dataclass(frozen=True)
class ToolCapability:
    access: Access
    direct_call: bool = True
    # A result-paging endpoint shares the owner's grant, not its identity.
    companion_of: str | None = None


# These families may create persistent state or execute arbitrary commands.
# The aliases include historic names accepted by host configuration files.
DENIED_TOOL_FAMILIES = frozenset(
    {
        "bash_tools",
        "scheduler_tools",
        "skill_tools",
        "skills",
        "memory_tools",
        "visual_report_tools",
        "visual_dashboard_tools",
        "export_tools",
        "bi_tools",
        "artifact_tools",
        "sub_agent_tools",
        "web_tool",
        "mcp",
        "tools",
    }
)


def _entries(category: str, access: Access, names: str, *, direct_call: bool = True) -> dict[str, ToolCapability]:
    return {f"{category}.{name}": ToolCapability(access, direct_call) for name in names.split()}


TOOL_CAPABILITIES: dict[str, ToolCapability] = {
    **_entries(
        "db_tools",
        "read",
        """
        describe_table list_databases list_schemas list_tables search_table get_datasource_identity
    """,
    ),
    **_entries("db_tools", "sql", "execute_sql"),
    **_entries(
        "db_tools",
        "write",
        """
        import_database_file transfer_query_result validate_ddl write_query execute_ddl
    """,
    ),
    # Caller-controlled local paths and authoring helpers are not suitable for
    # restricted hosts, even where the operation itself does not write.
    **_entries(
        "db_tools",
        "restricted",
        """
        load_file_as_table plan_datasource check_datasource_quality
        get_migration_capabilities suggest_table_layout
    """,
    ),
    **_entries("filesystem_tools", "read", "read_file read_image glob grep"),
    **_entries(
        "filesystem_tools",
        "write",
        """
        write_file edit_file create_file delete_file upsert_osi_datasets upsert_osi_metrics
        delete_osi_datasets delete_osi_metrics
    """,
    ),
    **_entries(
        "context_search_tools",
        "read",
        """
        get_reference_sql list_subject_tree search_metrics search_reference_sql search_semantic_objects
    """,
    ),
    **_entries("platform_doc_tools", "read", "get_document list_document_nav search_document"),
    **_entries(
        "reference_template_tools",
        "read",
        """
        execute_reference_template get_reference_template render_reference_template search_reference_template
    """,
    ),
    **_entries(
        "semantic_tools",
        "read",
        """
        attribution_analyze get_dataset get_metric get_dimensions list_metrics query_metrics
        describe_concept find_concept_paths select_records check_semantic_object_exists
        generate_sql_summary_id profile_semantic_model_evidence
    """,
    ),
    # Validation maintains in-process authoring evidence. It is available during
    # an agent turn, but is not an independently exposed public endpoint.
    **_entries("semantic_tools", "read", "validate_semantic", direct_call=False),
    **_entries(
        "semantic_tools",
        "write",
        """
        publish_semantic_model publish_metrics sync_osi_to_db bind_osi_semantic_model_target
    """,
    ),
    **_entries(
        "semantic_tools",
        "restricted",
        """
        list_existing_osi_semantic_models plan_osi_semantic_model_target verify_semantic_outputs
    """,
    ),
    **_entries("semantic_tools", "write", "write_semantic_model_plan_file"),
    **_entries("date_parsing_tools", "read", "parse_temporal_expressions", direct_call=False),
    **_entries("tool_result_tools", "read", "read_tool_result", direct_call=False),
    **_entries("lineage_tools", "read", "query_lineage"),
    **_entries("lineage_tools", "write", "upsert_lineage delete_lineage"),
    "semantic_tools.get_query_metrics_result": ToolCapability("read", companion_of="semantic_tools.query_metrics"),
}


def tool_capability(tool_id: str) -> ToolCapability | None:
    """Return the declared contract; unclassified mixed-family methods are unknown."""
    if tool_id.split(".", 1)[0] in DENIED_TOOL_FAMILIES:
        return ToolCapability("restricted", direct_call=False)
    return TOOL_CAPABILITIES.get(tool_id)
