import time
from typing import AsyncGenerator, Dict, Optional

from agents import SQLiteSession

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.agentic_node_models import AgenticInput, AgenticResult
from datus.tools.mcp_server import MCPServer
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class AgenticNode(Node):
    """
    AgenticNode provides general-purpose chat functionality with tool access.
    This node is designed to work independently of workflows and supports
    multi-turn conversations with context management.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_history_manager = None
        self.session: Optional[SQLiteSession] = None

    def execute(self):
        """Execute the agentic chat in non-streaming mode"""
        # For non-streaming, we'll use a simple synchronous execution
        import asyncio

        try:
            # Create action history manager for this execution
            action_history_manager = ActionHistoryManager()

            # Run the streaming version and collect the final result
            result = asyncio.run(self._execute_async(action_history_manager))
            self.result = result
            return result
        except Exception as e:
            logger.error(f"Agentic chat execution error: {str(e)}")
            return AgenticResult(
                success=False,
                error=str(e),
                response="I encountered an error while processing your request.",
                session_id=self.input.session_id,
            )

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute agentic chat with streaming support"""
        if action_history_manager is None:
            action_history_manager = ActionHistoryManager()

        self.action_history_manager = action_history_manager

        try:
            async for action in self._agentic_chat_stream(action_history_manager):
                yield action

        except Exception as e:
            logger.error(f"Agentic chat streaming error: {str(e)}")
            error_action = ActionHistory(
                action_id="agentic_chat_error",
                role=ActionRole.SYSTEM,
                messages=f"Error in agentic chat: {str(e)}",
                action_type="error",
                status=ActionStatus.FAILED,
                output={"error": str(e)},
            )
            yield error_action

    async def _execute_async(self, action_history_manager: ActionHistoryManager) -> AgenticResult:
        """Async execution for non-streaming mode"""
        start_time = time.time()
        actions_taken = []

        try:
            # Process the streaming execution and collect the final result
            async for action in self._agentic_chat_stream(action_history_manager):
                if action.action_type not in ["message"]:
                    actions_taken.append(action.action_type)

            # Extract final response from action history
            response, sql = self._extract_final_result(action_history_manager)
            execution_time = time.time() - start_time

            return AgenticResult(
                success=True,
                response=response,
                sql=sql,
                session_id=self.input.session_id,
                actions_taken=actions_taken,
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Async execution error: {str(e)}")
            return AgenticResult(
                success=False,
                error=str(e),
                response="I encountered an error while processing your request.",
                session_id=self.input.session_id,
                execution_time=time.time() - start_time,
            )

    async def _agentic_chat_stream(
        self, action_history_manager: ActionHistoryManager
    ) -> AsyncGenerator[ActionHistory, None]:
        """Core streaming chat implementation"""
        if not self.model:
            logger.error("Model not available for agentic chat")
            return

        try:
            # Setup session
            self.session = self.model.create_session(self.input.session_id)

            # Create initial setup action
            setup_action = ActionHistory(
                action_id="setup_agentic_chat",
                role=ActionRole.WORKFLOW,
                messages="Setting up agentic chat with tool access",
                action_type="setup",
                input={
                    "session_id": self.input.session_id,
                    "message": self.input.message,
                    "database_name": self.input.database_name,
                },
                status=ActionStatus.SUCCESS,
            )
            yield setup_action

            # Setup MCP servers for tool access
            mcp_servers = {}
            if self.input.database_name:
                db_config = self.agent_config.current_db_config(self.input.database_name)
                db_mcp_server = MCPServer.get_db_mcp_server(db_config)
                mcp_servers[self.input.database_name] = db_mcp_server

            # Get filesystem MCP server if available
            try:
                fs_mcp_server = MCPServer.get_filesystem_mcp_server()
                if fs_mcp_server:
                    mcp_servers["filesystem"] = fs_mcp_server
            except Exception as e:
                logger.debug(f"Filesystem MCP server not available: {e}")

            # Setup system instruction for agentic chat
            system_instruction = self._build_system_instruction()

            # Stream the conversation with tools
            async for action in self.model.generate_with_tools_stream(
                prompt=self.input.message,
                mcp_servers=mcp_servers,
                instruction=system_instruction,
                output_type=str,
                max_turns=10,  # Allow up to 10 turns
                session=self.session,
                action_history_manager=action_history_manager,
            ):
                yield action

        except Exception as e:
            logger.error(f"Agentic chat streaming error: {str(e)}")
            raise

    def _build_system_instruction(self) -> str:
        """Build system instruction for agentic chat"""
        instruction = """You are a helpful AI assistant specialized in data engineering and SQL tasks. 
You have access to database tools and filesystem tools to help users with their data-related questions.

Key capabilities:
- Execute SQL queries and analyze database schemas
- Read and analyze files
- Generate SQL queries from natural language
- Debug and fix SQL issues
- Explore database structures and relationships

When responding:
1. Be concise and helpful
2. If you generate SQL, include it in your response
3. Use tools when necessary to gather information or execute queries
4. Explain your reasoning when solving complex problems

Always format your final response as JSON with this structure:
{
    "response": "your detailed response to the user",
    "sql": "any SQL query you generated (optional)"
}

If no SQL is involved, omit the "sql" field."""

        # Add context if available
        if self.input.database_name:
            instruction += f"\n\nYou are currently working with the '{self.input.database_name}' database."

        return instruction

    def _extract_final_result(self, action_history_manager: ActionHistoryManager) -> tuple[str, Optional[str]]:
        """Extract the final response and SQL from action history"""
        response = "I processed your request."
        sql = None

        # Look for the final assistant message
        for action in reversed(action_history_manager.actions):
            if action.role == ActionRole.ASSISTANT and action.action_type == "message":
                if action.output and "raw_output" in action.output:
                    raw_output = action.output["raw_output"]
                    try:
                        # Try to parse as JSON first
                        parsed = llm_result2json(raw_output)
                        if parsed and "response" in parsed:
                            response = parsed["response"]
                            sql = parsed.get("sql")
                            break
                    except Exception:
                        # If JSON parsing fails, use raw output as response
                        response = raw_output
                        break
                elif action.messages:
                    response = action.messages
                    break

        return response, sql

    def setup_input(self, workflow: Workflow) -> Dict:
        """Setup input for agentic chat node (minimal workflow integration)"""
        # For agentic chat, we mostly work independently
        # But we can optionally use workflow context if provided
        if self.input is None:
            self.input = AgenticInput(
                message="Hello! How can I help you with your data tasks today?", session_id="default"
            )

        # Inherit database name from workflow if not set
        if workflow and workflow.task and not self.input.database_name:
            self.input.database_name = workflow.task.database_name

        return {"success": True, "message": "Agentic chat input setup complete"}

    def update_context(self, workflow: Workflow) -> Dict:
        """Update workflow context with agentic chat results (minimal integration)"""
        # Agentic chat works independently, but we can optionally
        # add useful information back to workflow context

        if self.result and self.result.success and self.result.sql:
            # If we generated SQL, add it to the workflow's SQL context
            from datus.schemas.node_models import SQLContext

            sql_context = SQLContext(
                sql_query=self.result.sql,
                explanation=f"Generated via agentic chat: {self.result.response[:100]}...",
                sql_return="",  # Will be filled if executed
                sql_error="",
                row_count=0,
            )

            if workflow and workflow.context:
                workflow.context.sql_contexts.append(sql_context)

        return {"success": True, "message": "Agentic chat context updated"}
