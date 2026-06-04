# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for orchestrator issue lifecycle tools."""

import pytest

from datus.tools.func_tool.orchestrator_tools import OrchestratorIssueTools


@pytest.mark.ci
class TestOrchestratorIssueTools:
    def test_available_tools_exposes_issue_lifecycle_tools(self) -> None:
        tools = OrchestratorIssueTools().available_tools()

        assert {tool.name for tool in tools} == {
            "create_issue_comment",
            "update_issue_status",
            "request_human_input",
            "mark_blocked",
            "finish_mission",
        }
        assert all(tool.strict_json_schema is False for tool in tools)

    @pytest.mark.asyncio
    async def test_tool_returns_proxy_required_error(self) -> None:
        tool = next(tool for tool in OrchestratorIssueTools().available_tools() if tool.name == "finish_mission")

        result = await tool.on_invoke_tool(
            None,
            '{"outcome":"completed","summary":"done","next_status":"In Review"}',
        )

        assert result["success"] == 0
        assert "must be proxied" in result["error"]
        assert result["result"] == {
            "issue_id": None,
            "outcome": "completed",
            "summary": "done",
            "next_status": "In Review",
            "artifacts": [],
            "validation": [],
        }
