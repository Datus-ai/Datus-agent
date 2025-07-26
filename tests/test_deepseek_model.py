import os

import pytest
from agents import set_tracing_disabled
from dotenv import load_dotenv

from datus.configuration.agent_config import DbConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.models.deepseek_model import DeepSeekModel
from datus.tools.mcp_server import MCPServer
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config

logger = get_logger(__name__)
set_tracing_disabled(True)


class TestDeepSeekModel:
    """Test suite for the DeepSeekModel class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment before each test method."""
        load_dotenv()
        config = load_acceptance_config()
        self.model = DeepSeekModel(config.active_model())

    def test_initialization_ark_r1(self):
        """Test initialization with DeepSeek ARK R1 model."""
        config = load_agent_config()
        model = DeepSeekModel(config["deepseek-ark-r1"])
        
        result = model.generate("Hello", max_tokens=50)
        
        assert result is not None, "Response should not be None"
        assert isinstance(result, str), "Response should be a string"
        assert len(result) > 0, "Response should not be empty"
        
        logger.debug(f"ARK R1 response: {result}")

    def test_initialization_deepseek_r1(self):
        """Test initialization with DeepSeek R1 model."""
        config = load_agent_config()
        model = DeepSeekModel(config["deepseek-r1"])
        
        result = model.generate("Hello", max_tokens=50)
        
        assert result is not None, "Response should not be None"
        assert isinstance(result, str), "Response should be a string"
        assert len(result) > 0, "Response should not be empty"
        
        logger.debug(f"R1 response: {result}")

    def test_initialization_deepseek_v3(self):
        """Test initialization with DeepSeek V3 model."""
        config = load_agent_config()
        model = DeepSeekModel(config["deepseek-v3"])
        
        result = model.generate("Hello", max_tokens=50)
        
        assert result is not None, "Response should not be None"
        assert isinstance(result, str), "Response should be a string"
        assert len(result) > 0, "Response should not be empty"
        
        logger.debug(f"V3 response: {result}")

    def test_generate(self):
        """Test basic text generation functionality."""
        result = self.model.generate("Hello", temperature=0.5, max_tokens=100)
        
        assert result is not None, "Response should not be None"
        assert isinstance(result, str), "Response should be a string"
        assert len(result) > 0, "Response should not be empty"
        
        logger.debug(f"Generated response: {result}")

    def test_generate_with_json_output(self):
        """Test JSON output generation."""
        result = self.model.generate_with_json_output(
            "Respond with a JSON object containing a greeting message"
        )
        
        assert result is not None, "Response should not be None"
        assert isinstance(result, dict), "Response should be a dictionary"
        assert len(result) > 0, "Response should not be empty"
        
        logger.debug(f"JSON response: {result}")

    def test_generate_with_system_prompt(self):
        """Test generation with system and user prompts."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond in JSON format with 'question' and 'answer' fields."
            },
            {
                "role": "user", 
                "content": "How many r's are in 'strawberry'?"
            }
        ]
        
        result = self.model.generate_with_json_output(messages)
        
        assert result is not None, "Response should not be None"
        assert isinstance(result, dict), "Response should be a dictionary"
        assert len(result) > 0, "Response should not be empty"
        
        logger.debug(f"System prompt response: {result}")

    @pytest.mark.asyncio
    async def test_generate_with_mcp(self):
        """Test MCP integration with SQLite."""
        instructions = """You are a SQLite expert. Your task is to:
        1. Understand the user's question about data analysis
        2. Generate appropriate SQL queries for SQLite
        3. Execute the queries using the provided tools
        4. Present the results in a clear and concise manner
        
        Output format: {
            "sql": "SELECT * FROM table LIMIT 10",
            "result": "Results here...",
            "explanation": "Explanation here..."
        }"""
        
        question = "database_type='sqlite' task='Create a simple products table and show sample data'"
        test_db_path = "./test_deepseek_mcp.db"
        mcp_server = MCPServer.get_sqlite_mcp_server(db_path=test_db_path)
        
        result = await self.model.generate_with_mcp(
            prompt=question,
            output_type=str,
            mcp_servers={"sqlite": mcp_server},
            instruction=instructions,
        )
        
        assert result is not None, "MCP response should not be None"
        assert "content" in result, "Response should contain content"
        assert "sql_contexts" in result, "Response should contain sql_contexts"
        
        logger.debug(f"MCP response: {result.get('content', '')}")

    @pytest.mark.asyncio
    async def test_generate_with_mcp_stream(self):
        """Test MCP streaming functionality with SQLite."""
        instructions = """You are a SQLite expert. Analyze the database and provide insights.
        
        Output format: {
            "sql": "SELECT COUNT(*) FROM table",
            "result": "Count results here...",
            "explanation": "Table analysis explanation..."
        }"""
        
        question = "database_type='sqlite' task='Analyze the database structure and provide basic statistics'"
        test_db_path = "./test_deepseek_stream.db"
        mcp_server = MCPServer.get_sqlite_mcp_server(db_path=test_db_path)
        
        action_count = 0
        async for action in self.model.generate_with_mcp_stream(
            prompt=question,
            output_type=str,
            mcp_servers={"sqlite": mcp_server},
            instruction=instructions,
        ):
            action_count += 1
            assert action is not None, "Stream action should not be None"
            logger.debug(f"Stream action {action_count}: {type(action)}")
        
        assert action_count > 0, "Should receive at least one streaming action"

    # Acceptance Tests for Performance Validation
    @pytest.mark.acceptance
    def test_generate_acceptance(self):
        """Acceptance test for basic generation performance."""
        prompts = [
            "Explain machine learning in one sentence.",
            "What is the capital of France?",
            "Write a haiku about programming."
        ]
        
        for prompt in prompts:
            result = self.model.generate(prompt, max_tokens=100)
            
            assert result is not None, f"Response should not be None for prompt: {prompt}"
            assert isinstance(result, str), "Response should be a string"
            assert len(result) > 0, "Response should not be empty"
            logger.debug(f"Acceptance test prompt: {prompt[:30]}... -> Response length: {len(result)}")

    @pytest.mark.acceptance
    @pytest.mark.asyncio
    async def test_generate_with_mcp_acceptance(self):
        """Acceptance test for MCP functionality with comprehensive scenarios."""
        test_scenarios = [
            {
                "task": "Create a users table with sample data",
                "expected_keywords": ["CREATE", "TABLE", "users", "INSERT"]
            },
            {
                "task": "Count all records in any existing tables",
                "expected_keywords": ["SELECT", "COUNT"]
            },
            {
                "task": "Show database schema information",
                "expected_keywords": ["sqlite_master", "schema"]
            }
        ]
        
        instructions = """You are a SQLite expert. Execute the requested database operations and provide clear results."""
        test_db_path = "./test_deepseek_acceptance.db"
        mcp_server = MCPServer.get_sqlite_mcp_server(db_path=test_db_path)
        
        for i, scenario in enumerate(test_scenarios):
            question = f"database_type='sqlite' task='{scenario['task']}'"
            
            result = await self.model.generate_with_mcp(
                prompt=question,
                output_type=str,
                mcp_servers={"sqlite": mcp_server},
                instruction=instructions,
            )
            
            assert result is not None, f"MCP response should not be None for scenario {i+1}"
            assert "content" in result, f"Response should contain content for scenario {i+1}"
            
            content = str(result.get('content', '')).lower()
            keyword_found = any(keyword.lower() in content for keyword in scenario['expected_keywords'])
            assert keyword_found, f"Response should contain relevant SQL keywords for scenario {i+1}: {scenario['expected_keywords']}"
            
            logger.debug(f"Acceptance scenario {i+1} completed: {scenario['task']}")

    @pytest.mark.acceptance
    @pytest.mark.asyncio
    async def test_generate_with_mcp_stream_acceptance(self):
        """Acceptance test for MCP streaming performance and reliability."""
        instructions = """You are a SQLite expert. Perform comprehensive database analysis with detailed explanations."""
        
        complex_scenarios = [
            "Create a comprehensive e-commerce database schema with products, customers, and orders tables",
            "Analyze data relationships and provide statistical insights",
            "Generate sample data and perform complex queries with joins"
        ]
        
        for i, scenario in enumerate(complex_scenarios):
            question = f"database_type='sqlite' task='{scenario}'"
            test_db_path = f"./test_deepseek_stream_acceptance_{i}.db"
            mcp_server = MCPServer.get_sqlite_mcp_server(db_path=test_db_path)
            
            action_count = 0
            total_content_length = 0
            
            async for action in self.model.generate_with_mcp_stream(
                prompt=question,
                output_type=str,
                mcp_servers={"sqlite": mcp_server},
                instruction=instructions,
            ):
                action_count += 1
                assert action is not None, f"Stream action should not be None for scenario {i+1}"
                
                # Track content if available
                if hasattr(action, 'content') and action.content:
                    total_content_length += len(str(action.content))
                
                logger.debug(f"Acceptance stream scenario {i+1}, action {action_count}: {type(action)}")
            
            assert action_count > 0, f"Should receive at least one streaming action for scenario {i+1}"
            logger.debug(f"Acceptance stream scenario {i+1} completed: {action_count} actions, {total_content_length} total content length")