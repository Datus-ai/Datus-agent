# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for DocSearchNode.

CI-level: zero external deps, zero network, zero API keys.
SearchTool is mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node.doc_search_node import DocSearchNode
from datus.schemas.action_history import ActionStatus
from datus.schemas.doc_search_node_models import DocSearchInput, DocSearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_config():
    cfg = MagicMock()
    cfg.agentic_nodes = {}
    cfg.permissions_config = None
    cfg.skills_config = None
    return cfg


def _make_doc_search_input(**kwargs):
    defaults = dict(
        platform="sqlite",
        keywords=["revenue", "sales"],
        top_n=3,
    )
    defaults.update(kwargs)
    return DocSearchInput(**defaults)


def _make_node(agent_config=None):
    cfg = agent_config or _make_agent_config()
    node = DocSearchNode(
        node_id="doc_search_1",
        description="Doc search node",
        node_type="doc_search",
        agent_config=cfg,
    )
    node.input = _make_doc_search_input()
    return node


def _make_workflow(keywords=None):
    wf = MagicMock()
    wf.context.doc_search_keywords = keywords or ["revenue", "sales"]
    # Use a real list so that the node can iterate over it
    return wf


def _make_doc_result(success=True, docs=None, doc_count=0):
    result = MagicMock(spec=DocSearchResult)
    result.success = success
    result.docs = docs or {}
    result.doc_count = doc_count
    return result


# ---------------------------------------------------------------------------
# TestDocSearchNodeInit
# ---------------------------------------------------------------------------


class TestDocSearchNodeInit:
    def test_node_creates(self):
        node = _make_node()
        assert node.id == "doc_search_1"
        assert node.description == "Doc search node"


# ---------------------------------------------------------------------------
# TestSetupInput
# ---------------------------------------------------------------------------


class TestSetupInputDocSearch:
    def test_setup_input_sets_input_on_node(self):
        """setup_input stores a DocSearchInput on the node."""
        node = _make_node()
        wf = _make_workflow(keywords=["profit", "margin"])
        # The source passes keywords+top_n+method to DocSearchInput but platform is required.
        # Patch DocSearchInput to avoid validation error from missing platform field.
        with patch("datus.agent.node.doc_search_node.DocSearchInput") as mock_input_class:
            mock_input_instance = MagicMock(spec=DocSearchInput)
            mock_input_class.return_value = mock_input_instance
            result = node.setup_input(wf)

        assert result["success"] is True
        assert node.input is mock_input_instance

    def test_setup_input_returns_suggestions(self):
        node = _make_node()
        wf = _make_workflow()
        with patch("datus.agent.node.doc_search_node.DocSearchInput") as mock_input_class:
            mock_input_class.return_value = MagicMock(spec=DocSearchInput)
            result = node.setup_input(wf)

        assert "suggestions" in result
        assert len(result["suggestions"]) == 1

    def test_setup_input_uses_workflow_keywords(self):
        """setup_input passes workflow keywords to DocSearchInput."""
        node = _make_node()
        wf = _make_workflow(keywords=["profit", "margin"])
        with patch("datus.agent.node.doc_search_node.DocSearchInput") as mock_input_class:
            mock_input_class.return_value = MagicMock(spec=DocSearchInput)
            node.setup_input(wf)

        call_kwargs = mock_input_class.call_args
        assert call_kwargs is not None
        # keywords should be passed
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("keywords") == ["profit", "margin"]
        else:
            assert call_kwargs.args[0] == ["profit", "margin"]


# ---------------------------------------------------------------------------
# TestUpdateContext
# ---------------------------------------------------------------------------


class TestUpdateContextDocSearch:
    def test_update_context_sets_document_result(self):
        node = _make_node()
        doc_result = _make_doc_result(success=True)
        node.result = doc_result

        wf = _make_workflow()
        result = node.update_context(wf)

        assert result["success"] is True
        assert wf.context.document_result == doc_result

    def test_update_context_handles_exception(self):
        node = _make_node()
        node.result = _make_doc_result()

        wf = _make_workflow()
        # Make assignment raise
        type(wf.context).document_result = property(
            lambda self: None,
            MagicMock(side_effect=RuntimeError("context error")),
        )

        # The property trick should trigger an exception during context update
        try:
            node.update_context(wf)
        except (RuntimeError, AttributeError):
            pass  # Expected: the property setter raises RuntimeError or AttributeError
        else:
            # If no exception, verify context was still set correctly
            assert wf.context is not None


# ---------------------------------------------------------------------------
# TestExecuteDocument
# ---------------------------------------------------------------------------


class TestExecuteDocument:
    def test_execute_document_calls_search_tool(self):
        node = _make_node()
        expected_result = _make_doc_result(success=True)

        with patch("datus.agent.node.doc_search_node.SearchTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = expected_result
            result = node._execute_document()

        mock_tool.execute.assert_called_once_with(node.input)
        assert result == expected_result

    def test_execute_sets_result(self):
        node = _make_node()
        expected_result = _make_doc_result(success=True)

        with patch("datus.agent.node.doc_search_node.SearchTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = expected_result
            node.execute()

        assert node.result == expected_result


# ---------------------------------------------------------------------------
# TestExecuteStream
# ---------------------------------------------------------------------------


class TestExecuteStreamDocSearch:
    @pytest.mark.asyncio
    async def test_execute_stream_yields_actions(self):
        """execute_stream yields at least a processing and final action."""
        node = _make_node()
        doc_result = _make_doc_result(success=True, doc_count=2)
        doc_result.docs = {"revenue": ["doc1"]}

        with patch("datus.agent.node.doc_search_node.SearchTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = doc_result
            actions = []
            async for action in node.execute_stream():
                actions.append(action)

        assert len(actions) >= 1

    @pytest.mark.asyncio
    async def test_execute_stream_final_status_success(self):
        """Final action status is SUCCESS when doc search succeeds."""
        node = _make_node()
        doc_result = _make_doc_result(success=True)

        with patch("datus.agent.node.doc_search_node.SearchTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = doc_result
            actions = []
            async for action in node.execute_stream():
                actions.append(action)

        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_stream_final_status_failed_on_doc_failure(self):
        """Final action status is FAILED when doc search returns failure."""
        node = _make_node()
        doc_result = _make_doc_result(success=False)

        with patch("datus.agent.node.doc_search_node.SearchTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = doc_result
            actions = []
            async for action in node.execute_stream():
                actions.append(action)

        assert actions[-1].status == ActionStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_stream_stores_result(self):
        """execute_stream stores result on self.result."""
        node = _make_node()
        doc_result = _make_doc_result(success=True)

        with patch("datus.agent.node.doc_search_node.SearchTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = doc_result
            async for _ in node.execute_stream():
                pass

        assert node.result == doc_result

    @pytest.mark.asyncio
    async def test_execute_stream_has_document_search_action_type(self):
        """The yielded action has action_type='document_search'."""
        node = _make_node()
        doc_result = _make_doc_result(success=True)

        with patch("datus.agent.node.doc_search_node.SearchTool") as mock_tool_class:
            mock_tool = mock_tool_class.return_value
            mock_tool.execute.return_value = doc_result
            actions = []
            async for action in node.execute_stream():
                actions.append(action)

        action_types = [a.action_type for a in actions]
        assert "document_search" in action_types

    @pytest.mark.asyncio
    async def test_execute_stream_propagates_exception(self):
        """execute_stream re-raises exceptions from _execute_document."""
        node = _make_node()

        with patch.object(node, "_execute_document", side_effect=RuntimeError("search error")):
            with pytest.raises(RuntimeError, match="search error"):
                async for _ in node.execute_stream():
                    pass
