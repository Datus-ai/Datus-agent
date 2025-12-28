# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScopedContextLists(BaseModel):
    tables: List[str] = Field(default_factory=list, description="Normalized table identifiers")
    metrics: List[str] = Field(default_factory=list, description="Normalized metric identifiers")
    sqls: List[str] = Field(default_factory=list, description="Normalized sql identifiers")

    def any(self) -> bool:
        return bool(self.tables or self.metrics or self.sqls)


class ScopedContext(BaseModel):
    namespace: Optional[str] = Field(default_factory=str, description="The namespace corresponding to the data source")
    tables: Optional[str] = Field(default=None, init=True, description="Tables to be used by sub-agents")
    metrics: Optional[str] = Field(default=None, init=True, description="Metrics to be used by sub-agents")
    sqls: Optional[str] = Field(default=None, init=True, description="Reference SQL to be used by sub-agents")

    @property
    def is_empty(self) -> bool:
        return not self.tables and not self.metrics and not self.sqls

    def as_lists(self) -> ScopedContextLists:
        def _split(value: Optional[str]) -> List[str]:
            if not value:
                return []
            tokens = [token.strip() for token in str(value).replace("\n", ",").split(",")]
            seen = set()
            normalized: List[str] = []
            for token in tokens:
                if token and token not in seen:
                    normalized.append(token)
                    seen.add(token)
            return normalized

        return ScopedContextLists(
            tables=_split(self.tables),
            metrics=_split(self.metrics),
            sqls=_split(self.sqls),
        )


class SubAgentConfig(BaseModel):
    system_prompt: str = Field("", init=True, description="Name of sub agent")
    agent_description: Optional[str] = Field(default=None, init=True, description="Description of sub agent")
    node_class: Optional[str] = Field(
        default=None,
        init=True,
        description="Node class type for custom subagents: 'gen_sql' (default) or 'gen_report'",
    )
    tools: str = Field(default="", init=True, description="Native tools to be used by sub-agents")
    mcp: str = Field(default="", init=True, description="MCP tools to be used by sub-agents")
    scoped_context: Optional[ScopedContext] = Field(
        default=None, init=True, description="Scoped context for sub-agents"
    )
    rules: List[str] = Field(default_factory=list, init=True, description="Rules to be used by sub-agents")
    prompt_version: str = Field(default="1.0", init=True, description="System Prompt version")
    prompt_language: str = Field(default="en", init=True, description="System Prompt language")
    scoped_kb_path: Optional[str] = Field(default=None, init=True, description="Path to scoped KB storage")

    class Config:
        populate_by_name = True

    def has_scoped_context(self) -> bool:
        return self.scoped_context and not self.scoped_context.is_empty

    def has_scoped_context_by(self, attr_name: str) -> bool:
        if self.scoped_context and hasattr(self.scoped_context, attr_name):
            return True
        return False

    def is_in_namespace(self, namespace: str) -> bool:
        return self.has_scoped_context() and namespace == self.scoped_context.namespace

    def as_payload(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "system_prompt": self.system_prompt,
            "prompt_version": self.prompt_version,
            "prompt_language": self.prompt_language,
            "agent_description": self.agent_description,
            "tools": self.tools,
            "mcp": self.mcp,
            "rules": list(self.rules or []),
        }

        if self.scoped_kb_path:
            payload["scoped_kb_path"] = self.scoped_kb_path

        if self.has_scoped_context():
            self.scoped_context.namespace = namespace
            scoped_context = self.scoped_context.model_dump(exclude_none=True)
            payload["scoped_context"] = scoped_context

        return payload

    @property
    def tool_list(self) -> List[str]:
        if not self.tools or not self.tools.strip():
            return []
        return [tool.strip() for tool in self.tools.split(",") if tool.strip()]
