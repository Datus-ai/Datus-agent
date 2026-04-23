"""
Stateless service for agent CRUD operations.

Handles listing, creating, and editing sub-agents. Builtin agents are resolved
from the BUILTIN_SUBAGENTS set; custom agents are persisted in agent.yml.
"""

import os
from pathlib import Path
from typing import Optional

from datus.api.models.agent_models import CreateAgentInput, EditAgentInput
from datus.api.models.base_models import Result
from datus.configuration.agent_config import AgentConfig
from datus.prompts.prompt_manager import PromptManager
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.func_tool.database import DBFuncTool
from datus.tools.func_tool.platform_doc_search import PlatformDocSearchTool
from datus.tools.func_tool.sub_agent_task_tool import BUILTIN_SUBAGENT_DESCRIPTIONS
from datus.utils.constants import HIDDEN_SYS_SUB_AGENTS, SYS_SUB_AGENTS
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Valid tool categories and their methods, derived from tool classes in datus-agent.
# Matches GenSQLAgenticNode._setup_tool_pattern() categories.

VALID_TOOL_METHODS: dict[str, set[str]] = {
    "db_tools": set(DBFuncTool.all_tools_name()),
    "context_search_tools": set(ContextSearchTools.all_tools_name()),
    "date_parsing_tools": {"parse_temporal_expressions"},
    "filesystem_tools": {
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
    },
    "platform_doc_tools": set(PlatformDocSearchTool.all_tools_name()),
}

VALID_TOOL_CATEGORIES = set(VALID_TOOL_METHODS.keys())

BUILTIN_SUBAGENTS = SYS_SUB_AGENTS - HIDDEN_SYS_SUB_AGENTS

# Tool reference for each agent type
SUBAGENT_TOOL_REFERENCE: dict[str, list[str]] = {
    "gen_sql": list(VALID_TOOL_METHODS.keys()),
    "gen_report": list(VALID_TOOL_METHODS.keys()),
}


def _validate_tools(tools: list[str]) -> list[str]:
    """Validate tool patterns and return list of invalid ones.

    Valid formats:
      - "db_tools"              (exact category)
      - "db_tools.*"            (wildcard — all methods in category)
      - "db_tools.list_tables"  (specific method)
    """
    invalid = []
    for pattern in tools:
        pattern = pattern.strip()
        if not pattern:
            continue
        # Exact category match: "db_tools"
        if pattern in VALID_TOOL_CATEGORIES:
            continue
        if "." in pattern:
            category, method = pattern.split(".", 1)
            if category not in VALID_TOOL_CATEGORIES:
                invalid.append(pattern)
                continue
            # Wildcard: "db_tools.*"
            if method == "*":
                continue
            # Specific method: "db_tools.list_tables"
            if method not in VALID_TOOL_METHODS[category]:
                invalid.append(pattern)
                continue
        else:
            invalid.append(pattern)
    return invalid


def _save_agentic_nodes(agent_config: AgentConfig, nodes: dict) -> None:
    """Save agentic_nodes back to agent.yml."""
    import yaml

    config_path = Path(agent_config.home) / "agent.yml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw["agentic_nodes"] = nodes
    with open(config_path, "w") as f:
        yaml.dump(raw, f, allow_unicode=True, default_flow_style=False)


class AgentService:
    """Service for Agent API operations.

    Handles agent management (CRUD) and subagent chat with SSE streaming.
    """

    def __init__(self):
        """Initialize AgentService."""
        pass

    @staticmethod
    def get_use_tools(agent_type: str) -> Result[dict]:
        """Return available tools for a given agent type."""
        if agent_type not in SUBAGENT_TOOL_REFERENCE:
            return Result(
                success=False,
                errorCode="INVALID_AGENT_TYPE",
                errorMessage=f"Unknown agent_type '{agent_type}'. Must be one of: {', '.join(SUBAGENT_TOOL_REFERENCE)}",
            )
        return Result(success=True, data={"tools": SUBAGENT_TOOL_REFERENCE[agent_type]})

    async def get_agent(
        self,
        agent_id: str,
        agent_config: AgentConfig,
    ) -> Result[dict]:
        """Return agent configuration matching IAgentInfo."""

        # 1. Check builtin agents
        if agent_id in BUILTIN_SUBAGENTS:
            return Result(
                success=True,
                data={
                    "agent": {
                        "id": agent_id,
                        "name": agent_id,
                        "type": "builtin",
                    }
                },
            )

        # 2. Query custom sub-agent from agent.yml (dict keyed by name, treated as id)
        agentic_nodes = agent_config.agentic_nodes or {}
        agent = agentic_nodes.get(agent_id)
        if not agent:
            return Result(success=False, errorCode="AGENT_NOT_FOUND", errorMessage=f"Agent '{agent_id}' not found")

        # 3. Read prompt template content from project template file
        agent_type = agent.get("type", "gen_sql")
        prompt_content = self._read_prompt_template(
            agent_name=agent_id,
            agent_type=agent_type,
            version=agent.get("prompt_version"),
            agent_config=agent_config,
        )

        return Result(
            success=True,
            data={
                "agent": {
                    "id": agent_id,
                    "name": agent_id,
                    "type": agent_type,
                    "description": agent.get("description", ""),
                    "system_prompt": prompt_content or agent.get("prompt_template", ""),
                    "tools": agent.get("tools", []),
                    "rules": agent.get("rules", []),
                    "catalogs": agent.get("catalogs", []),
                    "subjects": agent.get("subjects", []),
                }
            },
        )

    async def list_agents(self, agent_config: AgentConfig) -> Result[dict]:
        """List all agents available for this project."""

        # 1. Builtin agents
        builtin = [
            {
                "id": name,
                "name": name,
                "type": "builtin",
                "description": BUILTIN_SUBAGENT_DESCRIPTIONS.get(name, ""),
            }
            for name in sorted(BUILTIN_SUBAGENTS)
        ]

        # 2. Custom sub-agents from agent.yml
        agentic_nodes = agent_config.agentic_nodes or {}
        custom = [
            {
                "id": name,
                "name": name,
                "type": node.get("type", "gen_sql"),
                "description": node.get("description", ""),
            }
            for name, node in sorted(agentic_nodes.items())
        ]

        return Result(success=True, data={"agents": builtin + custom})

    # Map sub-agent type to builtin prompt template base name
    _TYPE_TO_TEMPLATE = {
        "gen_sql": "gen_sql_system",
        "gen_report": "gen_report_system",
        "chat": "chat_system",
    }

    _prompt_manager = PromptManager()

    async def create_agent(
        self,
        request: CreateAgentInput,
        agent_config: AgentConfig,
    ) -> Result[dict]:
        """Create a new custom sub-agent."""

        # Validate tools
        if request.tools:
            invalid = _validate_tools(request.tools)
            if invalid:
                return Result(
                    success=False,
                    errorCode="INVALID_TOOLS",
                    errorMessage=f"Invalid tool(s): {', '.join(invalid)}. Valid categories: {', '.join(sorted(VALID_TOOL_CATEGORIES))}",
                )

        # Check name not taken
        agentic_nodes = agent_config.agentic_nodes or {}
        if request.name in agentic_nodes or request.name in BUILTIN_SUBAGENTS:
            return Result(
                success=False,
                errorCode="AGENT_ALREADY_EXISTS",
                errorMessage=f"Agent '{request.name}' already exists",
            )

        # Create new agent entry (dict keyed by name, which acts as the id)
        agent_entry = {
            "type": request.type or "gen_sql",
            "description": request.description or "",
            "tools": request.tools or [],
            "catalogs": request.catalogs or [],
            "subjects": request.subjects or [],
            "rules": request.rules or [],
        }
        if request.prompt_template:
            agent_entry["prompt_template"] = request.prompt_template
        if request.prompt_version:
            agent_entry["prompt_version"] = request.prompt_version

        # Save to agent.yml
        agentic_nodes[request.name] = agent_entry
        _save_agentic_nodes(agent_config, agentic_nodes)

        # Copy the builtin prompt template to the project's template directory (non-fatal)
        try:
            self._copy_prompt_template(
                agent_type=request.type or "gen_sql",
                agent_name=request.name,
                version=request.prompt_version,
                agent_config=agent_config,
            )
        except Exception:
            logger.warning(f"Failed to copy prompt template for agent '{request.name}' (non-fatal)", exc_info=True)

        return Result(success=True, data={"name": request.name, "id": request.name})

    def _resolve_template_version(self, template_base: str, version: Optional[str] = None) -> Optional[str]:
        """Resolve prompt template version via PromptManager; returns latest if version is None."""
        if version:
            return version
        return self._prompt_manager.get_latest_version(template_base)

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        """Sanitize a string for safe use as a path component (no traversal)."""
        # Take only the basename to strip any directory separators
        safe = Path(value.replace(" ", "_")).name
        # Reject empty or dot-only names
        if not safe or safe in (".", ".."):
            raise ValueError(f"Invalid path component: {value!r}")
        return safe

    def _copy_prompt_template(
        self,
        agent_type: str,
        agent_name: str,
        version: Optional[str],
        agent_config: AgentConfig,
    ) -> None:
        """Copy the builtin prompt template for the agent type to the workspace template dir."""
        template_base = self._TYPE_TO_TEMPLATE.get(agent_type, "gen_sql_system")
        safe_name = self._sanitize_path_component(agent_name)
        try:
            source_path = self._prompt_manager._get_template_path(template_base)
        except FileNotFoundError:
            logger.warning(f"Builtin template '{template_base}' not found, skipping copy")
            return

        safe_version = self._sanitize_path_component(version) if version else version
        template_dir = Path(agent_config.home) / "template"
        os.makedirs(template_dir, exist_ok=True)
        target_file = template_dir / f"{safe_name}_system_{safe_version}.j2"
        if not target_file.resolve().is_relative_to(template_dir.resolve()):
            raise ValueError(f"Path escapes template directory: {target_file}")
        if not target_file.exists():
            content = source_path.read_text(encoding="utf-8")
            target_file.write_text(content, encoding="utf-8")
            logger.info(f"Copied prompt template: {source_path.name} -> {target_file}")

    def _read_prompt_template(
        self,
        agent_name: str,
        agent_type: str,
        version: Optional[str],
        agent_config: AgentConfig,
    ) -> str:
        """Read prompt template content. Checks workspace dir first, falls back to builtin."""
        template_base = self._TYPE_TO_TEMPLATE.get(agent_type, "gen_sql_system")
        safe_name = self._sanitize_path_component(agent_name)
        resolved = self._resolve_template_version(template_base, version)

        if resolved:
            safe_resolved = self._sanitize_path_component(resolved)
            # Try workspace template first
            template_dir = Path(agent_config.home) / "template"
            target_file = template_dir / f"{safe_name}_system_{safe_resolved}.j2"
            try:
                if target_file.exists():
                    return target_file.read_text(encoding="utf-8")
            except Exception:
                logger.warning(f"Failed to read project template '{target_file}'", exc_info=True)

            # Fallback: read builtin via PromptManager
            try:
                source_path = self._prompt_manager._get_template_path(template_base, resolved)
                return source_path.read_text(encoding="utf-8")
            except (FileNotFoundError, Exception):
                logger.warning(f"Failed to read builtin template '{template_base}' v{resolved}", exc_info=True)

        return ""

    def _save_prompt_template(
        self,
        agent_name: str,
        version: Optional[str],
        content: str,
        agent_config: AgentConfig,
    ) -> None:
        """Write prompt template content to the project's template file."""
        if not content:
            return
        safe_name = self._sanitize_path_component(agent_name)
        resolved = self._sanitize_path_component(version or "1.0")
        template_dir = Path(agent_config.home) / "template"
        os.makedirs(template_dir, exist_ok=True)
        target_file = template_dir / f"{safe_name}_system_{resolved}.j2"
        if not target_file.resolve().is_relative_to(template_dir.resolve()):
            raise ValueError(f"Path escapes template directory: {target_file}")
        target_file.write_text(content, encoding="utf-8")
        logger.info(f"Saved prompt template: {target_file}")

    async def edit_agent(
        self,
        request: EditAgentInput,
        agent_config: AgentConfig,
    ) -> Result[dict]:
        """Edit an existing custom sub-agent."""

        # Validate tools
        if request.tools:
            invalid = _validate_tools(request.tools)
            if invalid:
                return Result(
                    success=False,
                    errorCode="INVALID_TOOLS",
                    errorMessage=f"Invalid tool(s): {', '.join(invalid)}. Valid categories: {', '.join(sorted(VALID_TOOL_CATEGORIES))}",
                )

        # Find the agent (dict keyed by name, treated as id)
        agentic_nodes = agent_config.agentic_nodes or {}
        if request.id not in agentic_nodes:
            return Result(
                success=False,
                errorCode="AGENT_NOT_FOUND",
                errorMessage=f"Agent '{request.id}' not found",
            )

        agent = agentic_nodes[request.id]

        # If prompt_template content is provided, save to template file
        prompt_content = request.prompt_template
        if prompt_content is not None:
            version = request.prompt_version or agent.get("prompt_version")
            try:
                self._save_prompt_template(
                    agent_name=request.id,
                    version=version,
                    content=prompt_content,
                    agent_config=agent_config,
                )
            except Exception:
                logger.warning(f"Failed to save prompt template for agent '{request.id}' (non-fatal)", exc_info=True)

        # Update only provided fields (name is the dict key and acts as id, so exclude it)
        update_data = request.model_dump(exclude={"id", "name", "prompt_template"}, exclude_none=True)
        if not update_data and prompt_content is None:
            return Result(success=True, data={"name": request.id, "id": request.id})

        # Merge update data into the agent entry
        agent.update(update_data)

        # Save back to agent.yml
        _save_agentic_nodes(agent_config, agentic_nodes)

        return Result(success=True, data={"name": request.id, "id": request.id})
