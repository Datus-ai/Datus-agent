# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.tools.search_tools.search_tool — _get_document_store."""

from unittest.mock import MagicMock, patch

from datus.tools.search_tools.search_tool import SearchTool


class TestGetDocumentStore:
    """Tests for SearchTool._get_document_store."""

    def test_returns_store_when_has_data(self):
        """Should return the store when it has data."""
        mock_store = MagicMock()
        mock_store.has_data.return_value = True

        with patch("datus.tools.search_tools.search_tool.document_store", return_value=mock_store):
            tool = SearchTool.__new__(SearchTool)
            result = tool._get_document_store("test_platform")

        assert result is mock_store

    def test_returns_none_when_no_data(self):
        """Should return None when the store has no data."""
        mock_store = MagicMock()
        mock_store.has_data.return_value = False

        with patch("datus.tools.search_tools.search_tool.document_store", return_value=mock_store):
            tool = SearchTool.__new__(SearchTool)
            result = tool._get_document_store("test_platform")

        assert result is None
