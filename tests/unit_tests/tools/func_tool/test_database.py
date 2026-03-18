"""
Test cases for DBFuncTool compressor model_name initialization.
"""

from unittest.mock import Mock, patch

from datus.tools.func_tool.database import DBFuncTool


class TestDBFuncToolCompressorModelName:
    """Verify that DBFuncTool uses agent_config's model name for DataCompressor."""

    def test_compressor_uses_agent_config_model(self):
        """When agent_config is provided, compressor should use its active model name."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []

        mock_config = Mock()
        mock_config.active_model.return_value.model = "claude-sonnet-4"

        with patch("datus.tools.func_tool.database.SchemaWithValueRAG") as mock_rag, patch(
            "datus.tools.func_tool.database.SemanticModelRAG"
        ) as mock_sem:
            mock_rag.return_value.schema_store.table_size.return_value = 0
            mock_sem.return_value.get_size.return_value = 0
            tool = DBFuncTool(mock_connector, agent_config=mock_config)

        assert tool.compressor.model_name == "claude-sonnet-4"

    def test_compressor_defaults_without_agent_config(self):
        """When agent_config is None, compressor should fall back to gpt-3.5-turbo."""
        mock_connector = Mock()
        mock_connector.dialect = "sqlite"
        mock_connector.get_databases.return_value = []

        with patch("datus.tools.func_tool.database.SchemaWithValueRAG"), patch(
            "datus.tools.func_tool.database.SemanticModelRAG"
        ):
            tool = DBFuncTool(mock_connector)

        assert tool.compressor.model_name == "gpt-3.5-turbo"
