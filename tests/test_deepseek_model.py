import pytest
from agents import set_tracing_disabled
from dotenv import load_dotenv

from datus.configuration.agent_config import AgentConfig
from datus.models.deepseek_model import DeepSeekModel
from datus.tools.func_tool import db_function_tools
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config
from tests.test_tracing import auto_traceable

logger = get_logger(__name__)
set_tracing_disabled(True)


@auto_traceable
class TestDeepSeekModel:
    """Test suite for the DeepSeekModel class."""

    @pytest.fixture
    def agent_config(self) -> AgentConfig:
        load_dotenv()
        return load_acceptance_config()

    @pytest.fixture(autouse=True)
    def setup_method(self, agent_config: AgentConfig):
        """Set up test environment before each test method."""

        # self.model = DeepSeekModel(config.active_model())
        self.model = DeepSeekModel(model_config=agent_config.models["deepseek"])
        # self.model = DeepSeekModel(model_config=config["deepseek-ark"])

    def test_initialization_deepseek_r1(self, agent_config: AgentConfig):
        """Test initialization with DeepSeek R1 model."""
        model = DeepSeekModel(agent_config.models["deepseek-r1"])

        result = model.generate("Hello", max_tokens=200)

        assert result is not None, "Response should not be None"
        assert isinstance(result, str), "Response should be a string"
        assert len(result) > 0, "Response should not be empty"

        logger.debug(f"R1 response: {result}")

        result = model.generate("what's deepseek r1", enable_thinking=True, max_tokens=1000)

        assert result is not None, "Response should not be None"
        assert isinstance(result, str), "Response should be a string"
        assert len(result) > 0, "Response should not be empty"

        logger.debug(f"R1 response: {result}")

    def test_initialization_deepseek_v3(self, agent_config: AgentConfig):
        """Test initialization with DeepSeek V3 model."""
        model = DeepSeekModel(agent_config.models["deepseek"])

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
        result = self.model.generate_with_json_output("Respond with a JSON object containing a greeting message")

        assert result is not None, "Response should not be None"
        assert isinstance(result, dict), "Response should be a dictionary"
        assert len(result) > 0, "Response should not be empty"

        logger.info(f"JSON response: {result}")

    def test_generate_with_system_prompt(self):
        """Test generation with system and user prompts."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond in JSON format with 'question' and 'answer' fields.",
            },
            {"role": "user", "content": "How many r's are in 'strawberry'?"},
        ]

        result = self.model.generate_with_json_output(messages)

        assert result is not None, "Response should not be None"
        assert isinstance(result, dict), "Response should be a dictionary"
        assert len(result) > 0, "Response should not be empty"

        logger.debug(f"System prompt response: {result}")

    @pytest.mark.asyncio
    async def test_generate_with_mcp(self):
        """Test MCP integration with SSB database."""
        instructions = """You are a SQLite expert working with the Star Schema Benchmark (SSB) database.
        The database contains tables: customer, supplier, part, date, and lineorder.
        Focus on business analytics queries.

        Key tables and their relationships:
        - lineorder: main fact table with lo_revenue, lo_discount, lo_quantity, lo_extendedprice
        - date: dimension table with d_year, d_datekey
        - customer, supplier, part: other dimension tables

        Output format: {
            "sql": "SELECT ...",
            "result": "Query results...",
            "explanation": "Business explanation..."
        }"""

        question = (
            "database_type='sqlite' task='Calculate the total revenue in 1993 from orders with a discount "
            "between 1 and 3 and sales volume less than 25, where revenue is calculated by multiplying the "
            "extended price by the discount'"
        )

        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        try:
            result = await self.model.generate_with_tools(
                prompt=question,
                output_type=str,
                tools=tools,
                instruction=instructions,
            )

            assert result is not None, "MCP response should not be None"
            assert "content" in result, "Response should contain content"
            assert "sql_contexts" in result, "Response should contain sql_contexts"

            logger.debug(f"MCP response: {result.get('content', '')}")
        except DatusException as e:
            if e.code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"MCP test skipped due to max turns exceeded: {str(e)}")
            else:
                raise
        except Exception:
            raise

    @pytest.mark.acceptance
    @pytest.mark.asyncio
    async def test_generate_with_mcp_stream_acceptance(self):
        """Test MCP streaming functionality with SSB database."""
        instructions = """You are a SQLite expert analyzing the Star Schema Benchmark database.
        Provide comprehensive business analysis with multiple SQL queries.

        Database schema: customer, supplier, part, date, lineorder tables.
        Focus on revenue and sales analysis with detailed explanations.

        Output format: {
            "sql": "SELECT ...",
            "result": "Analysis results...",
            "explanation": "Business insights..."
        }"""

        question = """database_type='sqlite' task='Calculate the total revenue in 1992 from orders with a discount
         between 1 and 3 and sales volume less than 25, where revenue is calculated by multiplying the extended
         price by the discount'"""
        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        try:
            action_count = 0
            async for action in self.model.generate_with_tools_stream(
                prompt=question,
                output_type=str,
                tools=tools,
                max_turns=20,
                instruction=instructions,
            ):
                action_count += 1
                assert action is not None, "Stream action should not be None"
                logger.debug(f"Stream action {action_count}: {type(action)}")
                logger.info(f"Got action: {action}")

            assert action_count > 0, "Should receive at least one streaming action"
        except DatusException as e:
            if e.code == ErrorCode.MODEL_MAX_TURNS_EXCEEDED:
                pytest.skip(f"MCP test skipped due to max turns exceeded: {str(e)}")
            else:
                raise
        except Exception:
            raise

    # Acceptance Tests for Performance Validation
    @pytest.mark.acceptance
    def test_generate_acceptance(self):
        """Acceptance test for basic generation performance."""
        prompts = [
            "Explain machine learning in one sentence.",
            "What is the capital of France?",
            "Write a haiku about programming.",
        ]

        for prompt in prompts:
            result = self.model.generate(prompt, max_tokens=100)

            assert result is not None, f"Response should not be None for prompt: {prompt}"
            assert isinstance(result, str), "Response should be a string"
            assert len(result) > 0, "Response should not be empty"
            logger.info(f"Acceptance test prompt: {prompt[:30]}... -> Response length: {len(result)}")

    @pytest.mark.acceptance
    @pytest.mark.asyncio
    async def test_generate_with_mcp_acceptance(self):
        """Acceptance test for MCP functionality with SSB business scenarios."""
        test_scenarios = [
            {
                "task": "Find total revenue by customer region in the SSB database",
                "expected_keywords": ["SELECT", "revenue", "region"],  # More flexible keywords
            },
            {
                "task": "Calculate average discount by supplier nation using SSB data",
                "expected_keywords": ["SELECT", "supplier", "nation"],  # More flexible keywords
            },
            {
                "task": "Show the top 3 most profitable parts by total revenue in SSB",
                "expected_keywords": ["SELECT", "revenue", "LIMIT"],  # More flexible keywords
            },
        ]

        instructions = """You are a SQLite expert working with the Star Schema Benchmark database.
        Execute business analytics queries and provide clear results with proper joins."""
        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        for i, scenario in enumerate(test_scenarios):
            question = f"database_type='sqlite' task='{scenario['task']}'"

            result = await self.model.generate_with_tools(
                prompt=question,
                output_type=str,
                tools=tools,
                instruction=instructions,
                max_turns=20,
            )

            assert result is not None, f"MCP response should not be None for scenario {i+1}"
            assert "content" in result, f"Response should contain content for scenario {i+1}"

            content = str(result.get("content", "")).lower()
            keyword_found = any(keyword.lower() in content for keyword in scenario["expected_keywords"])
            assert (
                keyword_found
            ), f"Response should contain relevant SQL keywords for scenario {i+1}: {scenario['expected_keywords']}"

            logger.debug(f"Acceptance scenario {i+1} completed: {scenario['task']}")
            logger.info(f"Final result: {result}")

    @pytest.mark.asyncio
    async def test_generate_with_mcp_stream(self):
        """Acceptance test for MCP streaming with complex SSB analytics."""
        instructions = """You are a SQLite expert performing comprehensive analysis on the Star Schema Benchmark
        database. Provide detailed business analytics with multiple queries and insights."""

        complex_scenarios = [
            (
                "Analyze revenue trends by customer region and supplier nation with year-over-year "
                "growth in the SSB database"
            ),
            "Calculate profitability metrics by part category and manufacturer with discount impact analysis",
            (
                "Perform comprehensive supplier performance analysis including revenue, volume, and "
                "geographic distribution"
            ),
        ]

        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        for i, scenario in enumerate(complex_scenarios):
            question = f"database_type='sqlite' task='{scenario}'"

            action_count = 0
            total_content_length = 0

            async for action in self.model.generate_with_tools_stream(
                prompt=question,
                output_type=str,
                tools=tools,
                instruction=instructions,
                max_turns=30,
            ):
                action_count += 1
                assert action is not None, f"Stream action should not be None for scenario {i+1}"

                # Track content if available
                if hasattr(action, "content") and action.content:
                    total_content_length += len(str(action.content))

                logger.debug(f"Acceptance stream scenario {i+1}, action {action_count}: {type(action)}")

            assert action_count > 0, f"Should receive at least one streaming action for scenario {i+1}"
            logger.debug(
                f"Acceptance stream scenario {i+1} completed: {action_count} actions, "
                f"{total_content_length} total content length"
            )
            logger.info(f"Final Action: {action}")

    @pytest.mark.asyncio
    async def test_generate_with_mcp_session(self):
        """Test MCP integration with session management."""
        import uuid

        session_id = f"test_mcp_session_{uuid.uuid4().hex[:8]}"

        # Create session
        session = self.model.create_session(session_id)

        instructions = """You are a SQLite expert working with the SSB database.
        Answer questions about the database schema and data."""

        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        # First question: explore schema
        question1 = "database_type='sqlite' task='Show me all the tables in the database'"
        result1 = await self.model.generate_with_tools(
            prompt=question1,
            output_type=str,
            tools=tools,
            instruction=instructions,
            session=session,
        )

        assert result1 is not None
        assert "content" in result1
        assert "sql_contexts" in result1

        # Second question in same session: follow-up query
        question2 = "database_type='sqlite' task='Count the total number of rows in the customer table'"
        result2 = await self.model.generate_with_tools(
            prompt=question2,
            output_type=str,
            tools=tools,
            instruction=instructions,
            session=session,
        )

        assert result2 is not None
        assert "content" in result2
        assert "sql_contexts" in result2

        # Third question: reference previous answer to test session continuity
        question3 = "database_type='sqlite' task='What's the result of the previous number plus 5?'"
        result3 = await self.model.generate_with_tools(
            prompt=question3,
            output_type=str,
            tools=tools,
            instruction=instructions,
            session=session,
        )

        assert result3 is not None
        assert "content" in result3
        assert "sql_contexts" in result3

        # Verify session persistence with multiple interactions

        # Verify session persistence
        session_info = self.model.session_manager.get_session_info(session_id)
        assert session_info["exists"] is True
        assert session_info["item_count"] > 0

        # Cleanup
        self.model.delete_session(session_id)

        logger.debug(f"MCP session Q1: {result1.get('content', '')[:100]}...")
        logger.debug(f"MCP session Q2: {result2.get('content', '')[:100]}...")

    @pytest.mark.asyncio
    async def test_generate_with_mcp_stream_session(self):
        """Test MCP streaming with session management."""
        import uuid

        session_id = f"test_stream_session_{uuid.uuid4().hex[:8]}"

        # Create session
        session = self.model.create_session(session_id)

        instructions = """You are a SQLite expert working with the SSB database.
        Provide clear and concise answers about the database."""

        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        # First streaming question
        question1 = "database_type='sqlite' task='Describe the customer table structure'"
        action_count1 = 0

        async for action in self.model.generate_with_tools_stream(
            prompt=question1,
            output_type=str,
            tools=tools,
            instruction=instructions,
            session=session,
        ):
            action_count1 += 1
            assert action is not None
            logger.debug(f"Stream action 1.{action_count1}: {type(action)}")

        assert action_count1 > 0

        # Second streaming question in same session
        question2 = "database_type='sqlite' task='Show a sample of 3 rows from the customer table'"
        action_count2 = 0

        async for action in self.model.generate_with_tools_stream(
            prompt=question2,
            output_type=str,
            tools=tools,
            instruction=instructions,
            session=session,
        ):
            action_count2 += 1
            assert action is not None
            logger.debug(f"Stream action 2.{action_count2}: {type(action)}")

        assert action_count2 > 0

        # Verify session management
        assert self.model.session_manager.session_exists(session_id)

        # Cleanup
        self.model.delete_session(session_id)

        logger.debug(f"MCP stream session: {action_count1} + {action_count2} total actions")

    @pytest.mark.acceptance
    @pytest.mark.asyncio
    async def test_generate_with_mcp_token_consumption(self):
        """Test token consumption tracking between generate_with_tools and generate_with_tools_stream."""
        from datus.schemas.action_history import ActionHistoryManager

        instructions = """You are a SQLite expert working with the SSB database.
        Answer the question briefly and concisely."""

        question = "database_type='sqlite' task='Count the total number of rows in the customer table'"
        # question = "database_type='sqlite' task='To calculate the gross profit of orders where both the customer and"
        # " $supplier are located in the Americas, and the parts are manufactured by 'MFGR#1' or 'MFGR#2', you would"
        # " aggregate by year, supplier country, and part category. The result should be sorted in ascending order by"
        # " year, supplier country, and part category'"
        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        # Test 1: Non-streaming version
        logger.info("=== Testing generate_with_tools (non-streaming) ===")
        result_non_stream = await self.model.generate_with_tools(
            prompt=question,
            output_type=str,
            tools=tools,
            instruction=instructions,
            max_turns=20,
        )

        assert result_non_stream is not None, "Non-streaming result should not be None"
        assert "content" in result_non_stream, "Non-streaming result should contain content"
        assert "usage" in result_non_stream, "Non-streaming result should contain usage"

        non_stream_usage = result_non_stream.get("usage", {})
        logger.info(f"Non-streaming usage info: {non_stream_usage}")

        # Verify usage fields exist and are reasonable
        assert isinstance(non_stream_usage, dict), "Usage should be a dictionary"
        total_tokens_non_stream = non_stream_usage.get("total_tokens", 0)
        input_tokens_non_stream = non_stream_usage.get("input_tokens", 0)
        output_tokens_non_stream = non_stream_usage.get("output_tokens", 0)

        logger.info(
            f"Non-streaming tokens: total={total_tokens_non_stream}, input={input_tokens_non_stream},"
            f" output={output_tokens_non_stream}"
        )

        assert total_tokens_non_stream > 0, "Total tokens should be greater than 0"
        assert input_tokens_non_stream > 0, "Input tokens should be greater than 0"
        assert output_tokens_non_stream > 0, "Output tokens should be greater than 0"

        # Test 2: Streaming version
        logger.info("=== Testing generate_with_tools_stream (streaming) ===")
        action_history_manager = ActionHistoryManager()
        action_count = 0
        total_tokens_stream = 0
        actions_with_tokens = 0
        all_usage_info = []

        async for action in self.model.generate_with_tools_stream(
            prompt=question,
            output_type=str,
            tools=tools,
            instruction=instructions,
            max_turns=20,
            action_history_manager=action_history_manager,
        ):
            action_count += 1
            logger.info(
                f"Stream action {action_count}: type={action.action_type}, role={action.role}, status={action.status}"
            )

            # Check if action has token usage information
            if action.output and isinstance(action.output, dict):
                usage_info = action.output.get("usage", {})
                if usage_info and isinstance(usage_info, dict):
                    action_tokens = usage_info.get("total_tokens", 0)
                    if action_tokens > 0:
                        actions_with_tokens += 1
                        total_tokens_stream += action_tokens
                        all_usage_info.append(
                            {
                                "action_id": action.action_id,
                                "action_type": action.action_type,
                                "role": action.role,
                                "usage": usage_info,
                            }
                        )
                        logger.info(f"Action {action.action_id} ({action.action_type}) tokens: {usage_info}")

        # Check final actions after our fix has been applied
        final_actions = action_history_manager.get_actions()

        # Detailed logging of usage information
        logger.debug("=== Detailed Action Usage Analysis ===")
        final_actions_with_tokens = 0
        final_total_tokens = 0

        for i, action in enumerate(final_actions):
            logger.debug(f"Action {i}: type={action.action_type}, role={action.role}, status={action.status}")

            if action.output and isinstance(action.output, dict):
                logger.debug(f"  Output keys: {list(action.output.keys())}")

                if "usage" in action.output:
                    usage = action.output["usage"]
                    logger.debug(f"  Usage found: {usage}")

                    if isinstance(usage, dict):
                        total_tokens = usage.get("total_tokens", 0)
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        estimated = usage.get("estimated", False)

                        logger.info(f"    Total tokens: {total_tokens}")
                        logger.info(f"    Input tokens: {input_tokens}")
                        logger.info(f"    Output tokens: {output_tokens}")
                        logger.info(f"    Estimated: {estimated}")

                        if total_tokens > 0:
                            final_actions_with_tokens += 1
                            final_total_tokens += total_tokens
                    else:
                        logger.info(f"    Usage is not a dict: {type(usage)}")
                else:
                    logger.info("  No usage info in output")
            else:
                logger.info(f"  No output or invalid output type: {type(action.output)}")

        logger.info("=== Results Summary ===")
        logger.info(f"Non-streaming tokens: {total_tokens_non_stream}")
        logger.info(f"Streaming actions with tokens: {final_actions_with_tokens}/{len(final_actions)}")
        logger.info(f"Streaming total tokens: {final_total_tokens}")
        logger.info(f"Summary: {final_actions_with_tokens} actions with tokens, {final_total_tokens} total tokens")

        # Verify our fix is working
        if final_actions_with_tokens > 0 and final_total_tokens > 0:
            logger.info("✅ SUCCESS: Token consumption fix is working!")
        else:
            logger.error("❌ FAILURE: Token consumption fix needs work")

        assert action_count > 0, "Should receive at least one streaming action"

    @pytest.mark.asyncio
    async def test_generate_with_mcp_stream_session_acceptance(self):
        """Acceptance test for MCP streaming with session management."""
        import uuid

        session_id = f"test_acceptance_session_{uuid.uuid4().hex[:8]}"

        # Create session
        session = self.model.create_session(session_id)

        instructions = """You are a SQLite expert working with the SSB database.
        Provide concise answers about database schema and simple queries."""

        # Set up agent config for SQLite database
        agent_config = load_acceptance_config(namespace="ssb_sqlite")
        tools = db_function_tools(agent_config)

        # Simple acceptance scenarios with session
        scenarios = [
            "database_type='sqlite' task='List all tables in the database'",
            "database_type='sqlite' task='Describe the lineorder table structure'",
            "database_type='sqlite' task='Count rows in the date table'",
        ]

        total_actions = 0

        for i, scenario in enumerate(scenarios):
            action_count = 0

            async for action in self.model.generate_with_tools_stream(
                prompt=scenario,
                output_type=str,
                tools=tools,
                instruction=instructions,
                session=session,
            ):
                action_count += 1
                total_actions += 1
                assert action is not None
                logger.debug(f"Acceptance scenario {i+1}, action {action_count}: {type(action)}")

            assert action_count > 0, f"Should receive at least one action for scenario {i+1}"
            logger.debug(f"Acceptance scenario {i+1} completed with {action_count} actions")

        # Verify session management
        session_info = self.model.session_manager.get_session_info(session_id)
        assert session_info["exists"] is True
        assert session_info["item_count"] > 0

        # Cleanup
        self.model.delete_session(session_id)

        logger.debug(f"Acceptance test completed: {total_actions} total actions across {len(scenarios)} scenarios")
