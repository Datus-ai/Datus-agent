import json
import os
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import yaml
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.mcp import MCPServerStdio
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI
from pydantic import AnyUrl

from datus.configuration.agent_config import ModelConfig
from datus.models.base import LLMBaseModel
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
MAX_INPUT_DEEPSEEK = 52000  # 57344 - buffer of ~5000 tokens

set_tracing_disabled(True)


class DeepSeekModel(LLMBaseModel):
    """
    Implementation of the BaseModel for DeepSeek's API.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        **kwargs,
    ):
        super().__init__(model_config, **kwargs)

        self.api_key = model_config.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key must be provided or set as DEEPSEEK_API_KEY environment variable")

        self.api_base = model_config.base_url or os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.model_name = model_config.model
        # Initialize OpenAI client and langsmith wrapper
        logger.debug(f"Using DeepSeek model: {self.model_name} base Url: {self.api_base}")
        from langsmith.wrappers import wrap_openai

        self.client = wrap_openai(OpenAI(api_key=self.api_key, base_url=self.api_base))

        # Store reference to workflow and current node for trace saving
        self.workflow = None
        self.current_node = None

    def _save_llm_trace(self, prompt: Any, response_content: str, reasoning_content: Any = None):
        """Save LLM input/output trace to YAML file if tracing is enabled.

        Args:
            prompt: The input prompt (str or list of messages)
            response_content: The response content from the model
            reasoning_content: Optional reasoning content for reasoning models
        """
        if not self.model_config.save_llm_trace:
            return

        try:
            # Get workflow and node context from current execution
            if (
                not hasattr(self, "workflow")
                or not self.workflow
                or not hasattr(self, "current_node")
                or not self.current_node
            ):
                logger.debug("No workflow or node context available for trace saving")
                return

            # Create trace directory
            trajectory_dir = Path(self.workflow.global_config.trajectory_dir)
            task_id = self.workflow.task.id
            trace_dir = trajectory_dir / task_id
            trace_dir.mkdir(parents=True, exist_ok=True)

            # Parse prompt to separate system and user content
            system_prompt = ""
            user_prompt = ""

            if isinstance(prompt, list):
                # Handle message format like [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
                for message in prompt:
                    if message.get("role") == "system":
                        system_prompt = message.get("content", "")
                    elif message.get("role") == "user":
                        user_prompt = message.get("content", "")
            else:
                # Handle string prompt - put it all in user_prompt
                user_prompt = str(prompt)

            # Create trace data
            trace_data = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "reason_content": reasoning_content or "",
                "output_content": response_content,
            }

            # Save to YAML file named after node ID
            trace_file = trace_dir / f"{self.current_node.id}.yml"
            with open(trace_file, "w", encoding="utf-8") as f:
                yaml.dump(trace_data, f, default_flow_style=False, allow_unicode=True, indent=2, sort_keys=False)

            logger.debug(f"LLM trace saved to {trace_file}")

        except Exception as e:
            logger.error(f"Failed to save LLM trace: {str(e)}")

    def set_context(self, workflow=None, current_node=None):
        """Set workflow and node context for trace saving.

        Args:
            workflow: Current workflow instance
            current_node: Current node instance
        """
        self.workflow = workflow
        self.current_node = current_node

    def generate(self, prompt: Any, **kwargs) -> str:
        """Generate a response from the DeepSeek model.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            The generated text response
        """
        # Merge default parameters with any provided kwargs
        params = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 5000),
            "top_p": 1.0,
            **kwargs,
        }

        # Create messages format expected by OpenAI
        if type(prompt) is list:
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        # Call the OpenAI API
        response = self.client.chat.completions.create(messages=messages, **params)

        # Get response content
        response_content = response.choices[0].message.content

        # Check for reasoning content (for deepseek-reasoner and similar models)
        reasoning_content = None
        if hasattr(response.choices[0].message, "reasoning_content"):
            reasoning_content = response.choices[0].message.reasoning_content
        elif hasattr(response, "reasoning_content"):
            reasoning_content = response.reasoning_content

        # Save trace if enabled
        self._save_llm_trace(prompt, response_content, reasoning_content)

        # Log the response
        logger.debug(f"Model response: {response_content}")

        return response_content

    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """
        Generate a response and ensure it conforms to the provided JSON schema.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            A dictionary representing the JSON response
        """
        # Add instructions to format the response as JSON according to the schema
        # json_prompt = f"{prompt}\n\nRespond with a JSON object that conforms
        # to the following schema:\n{json.dumps(json_schema, indent=2)}"

        # Generate the response
        response_text = self.generate(prompt, response_format={"type": "json_object"}, **kwargs)

        # Parse the JSON response
        try:
            json_result = json.loads(response_text)
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON from the response
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    json_result = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    json_result = {}
            else:
                json_result = {}

        # Note: trace is already saved in self.generate() call above
        return json_result

    def generate_with_tools(self, prompt: str, tools: List[Any], **kwargs) -> Dict:
        # flow control and context cache here
        pass

    async def generate_with_mcp(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: type[Any],
        max_turns: int = 10,
        **kwargs,
    ) -> Dict:
        """Generate a response using multiple MCP (Machine Conversation Protocol) servers.

        Args:
            prompt: The input prompt to send to the model
            mcp_servers: Dictionary of MCP servers to use for execution
            instruction: The instruction for the agent
            output_type: The type of output expected from the agent
            max_turns: Maximum number of conversation turns
            **kwargs: Additional parameters for the agent

        Returns:
            The result from the MCP agent execution with content and sql_contexts
        """

        # Custom JSON encoder to handle special types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        json._default_encoder = CustomJSONEncoder()

        # Initialize reasoning content list to track the entire MCP conversation process
        reasoning_steps = []

        # Create async OpenAI client
        logger.debug(f"Creating async OpenAI client with base_url: {self.api_base}, model: {self.model_name}")
        async_client = wrap_openai(
            AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        )

        model_params = {"model": self.model_name}
        async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)

        # Define the agent instructions
        logger.debug("Starting run_agent")
        try:
            # Use context manager to manage multiple MCP servers
            from datus.models.mcp_utils import multiple_mcp_servers

            async with multiple_mcp_servers(mcp_servers) as connected_servers:
                logger.debug("MCP servers started successfully")

                agent = Agent(
                    name=kwargs.pop("agent_name", "MCP_Agent"),
                    instructions=instruction,
                    mcp_servers=list(connected_servers.values()),
                    output_type=output_type,
                    model=async_model,
                )
                logger.debug(f"Agent created with name: {agent.name}, {output_type}")

                result = await Runner.run(agent, input=prompt, max_turns=max_turns)

                logger.info(f"deepseek mcp run Result: {result}")
                # Build the result
                final_result = {
                    "content": result.final_output,
                    "sql_contexts": extract_sql_contexts(result),
                }

                # Create reasoning content from the full interaction list
                reasoning_content = None
                if hasattr(result, "to_input_list"):
                    try:
                        # Pass the raw list of interactions
                        reasoning_content = result.to_input_list()
                    except Exception as e:
                        logger.error(f"Error getting reasoning content list: {e}")
                        # Fallback to a simple string representation
                        reasoning_content = str(result.to_input_list())

                logger.debug(f"Reasoning content: {reasoning_content}")
                self._save_llm_trace(
                    prompt=prompt,
                    response_content=result.final_output,
                    reasoning_content=reasoning_content,
                )
                return final_result
        except Exception as e:
            logger.error(f"Error in run_agent: {str(e)}")
            # Save trace even on error
            full_reasoning_content = "\n".join(reasoning_steps)
            self._save_llm_trace(
                prompt=f"Instruction: {instruction}\n\nUser Prompt: {prompt}",
                response_content=f"ERROR: {str(e)}",
                reasoning_content=full_reasoning_content,
            )
            raise

    async def generate_with_mcp_stream(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        action_history_manager: Optional[ActionHistoryManager] = None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate a response using multiple MCP servers with streaming support.

        Args:
            prompt: The input prompt to send to the model
            mcp_servers: Dictionary of MCP servers to use for execution
            instruction: The instruction for the agent
            output_type: The type of output expected from the agent
            max_turns: Maximum number of conversation turns
            action_history_manager: Optional action history manager
            **kwargs: Additional parameters for the agent

        Yields:
            ActionHistory objects representing the streaming function calls
        """
        if action_history_manager is None:
            action_history_manager = ActionHistoryManager()

        # Custom JSON encoder to handle special types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, AnyUrl):
                    return str(obj)
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                return super().default(obj)

        json._default_encoder = CustomJSONEncoder()

        # Don't create setup action here - let the calling function handle it

        try:
            # Create async OpenAI client
            logger.debug(f"Creating async OpenAI client with base_url: {self.api_base}, model: {self.model_name}")
            async_client = wrap_openai(
                AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                )
            )

            model_params = {"model": self.model_name}
            async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)

            # Use context manager to manage multiple MCP servers
            from datus.models.mcp_utils import multiple_mcp_servers

            async with multiple_mcp_servers(mcp_servers) as connected_servers:
                logger.debug("MCP servers started successfully")

                agent = Agent(
                    name=kwargs.pop("agent_name", "MCP_Agent"),
                    instructions=instruction,
                    mcp_servers=list(connected_servers.values()),
                    output_type=output_type,
                    model=async_model,
                )
                logger.debug(f"Agent created with name: {agent.name}, {output_type}")

                # Start streaming execution
                logger.debug(f"Running agent with streaming, max_turns: {max_turns}")
                result = Runner.run_streamed(agent, input=prompt, max_turns=max_turns)

                function_call_count = 0

                logger.debug("Starting streaming loop, checking if result is complete...")
                while not result.is_complete:
                    logger.debug(f"Result not complete, entering stream_events() loop")
                    async for event in result.stream_events():
                        logger.debug(f"Received streaming event: {type(event).__name__}")

                        # Log event attributes for debugging
                        if hasattr(event, "type"):
                            logger.debug(f"Event type: {event.type}")
                        else:
                            logger.debug(f"Event has no 'type' attribute. Event attributes: {dir(event)}")

                        # Simplified logging - only log non-raw events
                        if hasattr(event, "type") and event.type != "raw_response_event":
                            logger.debug(f"NON_RAW_EVENT: {event.type}")

                        # Process different event types
                        if hasattr(event, "type"):
                            if event.type == "run_item_stream_event":
                                logger.debug("Processing run_item_stream_event")
                                # Check if this is a function call event
                                if hasattr(event, "item") and hasattr(event.item, "type"):
                                    logger.debug(f"Run item type: {event.item.type}")
                                    if event.item.type == "tool_call_item":
                                        logger.debug("Processing tool_call_item in run_item_stream_event")
                                        # Function call started - extract real call_id from raw_item
                                        function_call_count += 1
                                        call_id = None
                                        function_name = None
                                        arguments = None

                                        if hasattr(event.item, "raw_item"):
                                            raw_item = event.item.raw_item
                                            if hasattr(raw_item, "call_id"):
                                                call_id = raw_item.call_id
                                            if hasattr(raw_item, "name"):
                                                function_name = raw_item.name
                                            if hasattr(raw_item, "arguments"):
                                                arguments = raw_item.arguments

                                        tool_call_action = ActionHistory(
                                            action_id=call_id or str(uuid.uuid4()),
                                            role=ActionRole.MODEL,
                                            thought=f"Database function call: {function_name or 'unknown'}",
                                            action_type=ActionType.FUNCTION_CALL,
                                            input={
                                                "function_name": function_name,
                                                "arguments": arguments,
                                                "call_id": call_id,
                                            },
                                            timestamp=datetime.now().isoformat(),
                                        )
                                        action_history_manager.add_action(tool_call_action)
                                        logger.debug(
                                            "Created and yielding tool_call_action: {}".format(tool_call_action.thought)
                                        )
                                        yield tool_call_action

                                    elif event.item.type == "tool_call_output_item":
                                        logger.debug("Processing tool_call_output_item in run_item_stream_event")
                                        # Function call completed - find matching action by call_id
                                        call_id = None
                                        if hasattr(event.item, "raw_item") and hasattr(event.item.raw_item, "call_id"):
                                            call_id = event.item.raw_item.call_id

                                        # Find the matching action by call_id
                                        matching_action = None
                                        if call_id:
                                            matching_action = action_history_manager.find_action_by_id(call_id)

                                        if matching_action:
                                            # Update the action with the result
                                            output_data = self._extract_tool_call_output(event)
                                            matching_action.output = output_data

                                            # Determine success and reflection
                                            success = output_data.get("success", True)
                                            function_name = (
                                                matching_action.input.get("function_name", "")
                                                if matching_action.input
                                                else ""
                                            )
                                            matching_action.reflection = (
                                                ("✅ Function executed successfully" if success else "❌ Function failed")
                                                + ": "
                                                + str(function_name)
                                            )

                                            logger.debug(
                                                "Updated matching action with output, yielding: {}".format(
                                                    matching_action.reflection
                                                )
                                            )
                                            # Yield the updated action
                                            yield matching_action
                                        else:
                                            logger.warning(
                                                "Received tool_call_output but no matching action found for call_id: {}".format(call_id))

                                    elif event.item.type == "message_output_item":
                                        logger.debug("Processing message_output_item - extracting final result")
                                        # This is the final message output, extract the actual SQL
                                        if hasattr(event.item, "raw_item") and hasattr(event.item.raw_item, "content"):
                                            content = event.item.raw_item.content
                                            if content and len(content) > 0:
                                                # content[0] is a ResponseOutputText object, access .text attribute
                                                if isinstance(content, list) and len(content) > 0:
                                                    text_content = (
                                                        content[0].text
                                                        if hasattr(content[0], "text")
                                                        else str(content[0])
                                                    )
                                                else:
                                                    text_content = str(content)
                                                logger.debug("Final message content: {}".format(text_content))

                                                # Try to extract SQL from JSON content
                                                try:
                                                    if text_content.strip().startswith("{"):
                                                        json_content = json.loads(text_content)
                                                        final_sql = json_content.get("sql", "")
                                                        explanation = json_content.get("explanation", "")

                                                        if final_sql:
                                                            # Create final result action with actual SQL
                                                            final_action = ActionHistory(
                                                                action_id=str(uuid.uuid4()),
                                                                role=ActionRole.MODEL,
                                                                thought="Final SQL query generated",
                                                                action_type=ActionType.FUNCTION_CALL,
                                                                input={
                                                                    "reasoning_task": "SQL generation from natural language",
                                                                    "total_function_calls": function_call_count,
                                                                },
                                                                output={
                                                                    "success": True,
                                                                    "final_sql_query": final_sql,
                                                                    "explanation": explanation,
                                                                    "reasoning_complete": True,
                                                                },
                                                                reflection=f"✅ Generated final SQL: {final_sql}",
                                                                timestamp=datetime.now().isoformat(),
                                                            )
                                                            action_history_manager.add_action(final_action)
                                                            yield final_action
                                                except Exception as e:
                                                    logger.debug(f"Could not parse JSON content: {e}")
                                                    # Fallback to text content
                                                    final_action = ActionHistory(
                                                        action_id=str(uuid.uuid4()),
                                                        role=ActionRole.MODEL,
                                                        thought="Final reasoning output",
                                                        action_type=ActionType.CHAT,
                                                        input={"reasoning_task": "SQL generation"},
                                                        output={"success": True, "content": text_content},
                                                        reflection="Reasoning completed with text output",
                                                        timestamp=datetime.now().isoformat(),
                                                    )
                                                    action_history_manager.add_action(final_action)
                                                    yield final_action
                                    else:
                                        logger.debug(f"Unknown run_item_stream_event type: {event.item.type}")
                            else:
                                logger.debug(f"Ignoring event type: {event.type}")
                        else:
                            logger.debug("Event has no type attribute, skipping")

                logger.debug("Exited streaming loop - result is complete")

                # Final result is now handled in message_output_item processing above
                # No need to create duplicate final action here

        except Exception as e:
            logger.error(f"Error in streaming MCP execution: {str(e)}")

            # Create error action for streaming output
            error_action = ActionHistory(
                action_id=str(uuid.uuid4()),
                role=ActionRole.WORKFLOW,
                thought="Error occurred during MCP streaming",
                action_type=ActionType.FUNCTION_CALL,
                input={"error_type": type(e).__name__},
                output={"error": str(e), "success": False},
                reflection=f"❌ Streaming execution failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
            )
            action_history_manager.add_action(error_action)
            yield error_action

            # Re-raise the exception for proper error handling
            raise

    def _extract_tool_call_input(self, event) -> Dict[str, Any]:
        """Extract input data from tool call event."""
        try:
            logger.debug(f"Extracting tool call input from event: {type(event).__name__}")
            if hasattr(event, "item"):
                logger.debug(f"Event has item: {type(event.item).__name__}")
                if hasattr(event.item, "input"):
                    input_data = event.item.input
                    logger.debug(f"Input data type: {type(input_data)}, value: {input_data}")
                    if isinstance(input_data, dict):
                        # Look for SQL query in the input
                        sql_query = ""
                        if "query" in input_data:
                            sql_query = input_data["query"]
                            logger.debug(f"Found SQL query in 'query' field: {sql_query}")
                        elif "sql" in input_data:
                            sql_query = input_data["sql"]
                            logger.debug(f"Found SQL query in 'sql' field: {sql_query}")
                        elif isinstance(input_data, dict) and len(input_data) == 1:
                            # If there's only one key, it might be the SQL query
                            sql_query = list(input_data.values())[0]
                            logger.debug(f"Found SQL query as single value: {sql_query}")
                        else:
                            logger.debug(f"No SQL query found in input_data keys: {list(input_data.keys())}")

                        result = {
                            "sql_query": sql_query,
                            "raw_input": input_data,
                        }
                        logger.debug(f"Extracted input result: {result}")
                        return result
                    else:
                        result = {"raw_input": str(input_data)}
                        logger.debug(f"Non-dict input data, returning: {result}")
                        return result
                else:
                    logger.debug("Event item has no 'input' attribute")
            else:
                logger.debug("Event has no 'item' attribute")
            return {}
        except Exception as e:
            logger.error(f"Error extracting tool call input: {e}")
            return {"extraction_error": str(e)}

    def _extract_tool_call_output(self, event) -> Dict[str, Any]:
        """Extract output data from tool call output event."""
        try:
            logger.debug(f"Extracting tool call output from event: {type(event).__name__}")
            if hasattr(event, "item"):
                logger.debug(f"Event has item: {type(event.item).__name__}")
                if hasattr(event.item, "output"):
                    output_data = event.item.output
                    logger.debug(f"Output data type: {type(output_data)}, value: {output_data}")
                    if isinstance(output_data, dict):
                        result = {
                            "success": True,
                            "sql_return": output_data.get("result", ""),
                            "row_count": output_data.get("row_count", 0),
                            "raw_output": output_data,
                        }
                        logger.debug(f"Extracted dict output result: {result}")
                        return result
                    else:
                        result = {
                            "success": True,
                            "sql_return": str(output_data),
                            "raw_output": output_data,
                        }
                        logger.debug(f"Extracted non-dict output result: {result}")
                        return result
                else:
                    logger.debug("Event item has no 'output' attribute")
            else:
                logger.debug("Event has no 'item' attribute")
            result = {"success": False, "error": "No output data found"}
            logger.debug(f"No output data found, returning: {result}")
            return result
        except Exception as e:
            logger.error(f"Error extracting tool call output: {e}")
            return {"success": False, "error": str(e)}

    def _extract_function_call_input(self, event) -> Dict[str, Any]:
        """Extract input data from function call event in run_item_stream_event format."""
        try:
            logger.debug(f"Extracting function call input from run_item_stream_event: {type(event).__name__}")
            if hasattr(event, "item"):
                logger.debug(f"Event has item: {type(event.item).__name__}")
                if hasattr(event.item, "input"):
                    input_data = event.item.input
                    logger.debug(f"Input data type: {type(input_data)}, value: {input_data}")
                    if isinstance(input_data, dict):
                        # Look for SQL query in the input
                        sql_query = ""
                        if "query" in input_data:
                            sql_query = input_data["query"]
                            logger.debug(f"Found SQL query in 'query' field: {sql_query}")
                        elif "sql" in input_data:
                            sql_query = input_data["sql"]
                            logger.debug(f"Found SQL query in 'sql' field: {sql_query}")
                        elif isinstance(input_data, dict) and len(input_data) == 1:
                            # If there's only one key, it might be the SQL query
                            sql_query = list(input_data.values())[0]
                            logger.debug(f"Found SQL query as single value: {sql_query}")
                        else:
                            logger.debug(f"No SQL query found in input_data keys: {list(input_data.keys())}")

                        result = {
                            "sql_query": sql_query,
                            "raw_input": input_data,
                        }
                        logger.debug(f"Extracted input result: {result}")
                        return result
                    else:
                        result = {"raw_input": str(input_data)}
                        logger.debug(f"Non-dict input data, returning: {result}")
                        return result
                else:
                    logger.debug("Event item has no 'input' attribute")
            else:
                logger.debug("Event has no 'item' attribute")
            return {}
        except Exception as e:
            logger.error(f"Error extracting function call input: {e}")
            return {"extraction_error": str(e)}

    def _extract_function_call_output(self, event) -> Dict[str, Any]:
        """Extract output data from function call output event in run_item_stream_event format."""
        try:
            logger.debug(f"Extracting function call output from run_item_stream_event: {type(event).__name__}")
            if hasattr(event, "item"):
                logger.debug(f"Event has item: {type(event.item).__name__}")
                if hasattr(event.item, "output"):
                    output_data = event.item.output
                    logger.debug(f"Output data type: {type(output_data)}, value: {output_data}")
                    if isinstance(output_data, dict):
                        result = {
                            "success": True,
                            "sql_return": output_data.get("result", ""),
                            "row_count": output_data.get("row_count", 0),
                            "raw_output": output_data,
                        }
                        logger.debug(f"Extracted dict output result: {result}")
                        return result
                    else:
                        result = {
                            "success": True,
                            "sql_return": str(output_data),
                            "raw_output": output_data,
                        }
                        logger.debug(f"Extracted non-dict output result: {result}")
                        return result
                else:
                    logger.debug("Event item has no 'output' attribute")
            else:
                logger.debug("Event has no 'item' attribute")
            result = {"success": False, "error": "No output data found"}
            logger.debug(f"No output data found, returning: {result}")
            return result
        except Exception as e:
            logger.error(f"Error extracting function call output: {e}")
            return {"success": False, "error": str(e)}

    def token_count(self, prompt: str) -> int:
        """Estimate the number of tokens in a text using the deepseek tokenizer.

        Args:
            prompt (str): The text to count the tokens of.

        Returns:
            int: The number of tokens in the text.
        """
        return int(len(prompt) * 0.3 + 0.5)

    def max_tokens(self) -> int:
        return MAX_INPUT_DEEPSEEK
