import copy
import json
import os
import re
from datetime import date, datetime
from typing import Any, Dict, List

import anthropic
import httpx
from agents import Agent, OpenAIChatCompletionsModel, Runner
from agents.mcp import MCPServerStdio
from langsmith.wrappers import wrap_anthropic, wrap_openai
from openai import AsyncOpenAI, OpenAI
from pydantic import AnyUrl

from datus.models.base import LLMBaseModel
from datus.models.mcp_result_extractors import extract_sql_contexts
from datus.models.mcp_utils import multiple_mcp_servers
from datus.schemas.node_models import SQLContext
from datus.utils.loggings import get_logger

logger = get_logger("claude_model")


def wrap_prompt_cache(messages):
    messages_copy = copy.deepcopy(messages)
    msg_size = len(messages_copy)
    content = messages_copy[msg_size - 1]["content"]
    cnt_size = len(content)
    if isinstance(content, list):
        content[cnt_size - 1]["cache_control"] = {"type": "ephemeral"}

    return messages_copy


def convert_tools_for_anthropic(mcp_tools):
    anthropic_tools = []

    for tool in mcp_tools:
        anthropic_tool = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema,
        }

        # Rename inputSchema's 'properties' to match Anthropic's convention if needed
        if "properties" in anthropic_tool["input_schema"]:
            for _, prop_value in anthropic_tool["input_schema"]["properties"].items():
                if "description" not in prop_value and "desc" in prop_value:
                    prop_value["description"] = prop_value.pop("desc")

        if hasattr(tool, "annotations") and tool.annotations:
            anthropic_tool["annotations"] = tool.annotations

        anthropic_tools.append(anthropic_tool)

    # add tool cache
    tool_size = len(anthropic_tools)
    anthropic_tools[tool_size - 1]["cache_control"] = {"type": "ephemeral"}
    return anthropic_tools


class ClaudeModel(LLMBaseModel):
    """
    Implementation of the BaseModel for Claude's API.
    """

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.api_base = model_config.base_url
        self.model_name = model_config.model
        # fix it, remove os.env with model_config
        self.api_key = model_config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        logger.debug(f"Using Claude model: {self.model_name} base Url: {self.api_base}")

        self.client = wrap_openai(OpenAI(api_key=self.api_key, base_url=self.api_base + "/v1"))

        # Optional proxy configuration - only use if environment variable is set
        proxy_url = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
        if proxy_url:
            proxy_client = httpx.Client(
                transport=httpx.HTTPTransport(proxy=httpx.Proxy(url=proxy_url)),
                timeout=60.0,
            )
            self.anthropic_client = wrap_anthropic(
                anthropic.Anthropic(
                    api_key=self.api_key,
                    base_url=self.api_base if self.api_base else None,
                    http_client=proxy_client,
                )
            )
        else:
            self.anthropic_client = wrap_anthropic(
                anthropic.Anthropic(
                    api_key=self.api_key,
                    base_url=self.api_base if self.api_base else None,
                )
            )

    def generate(self, prompt: Any, **kwargs) -> str:
        """Generate a response from the Claude model.

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
            "max_tokens": kwargs.get("max_tokens", 3000),
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

        # Log the response
        logger.debug(f"Model response: {response.choices[0].message.content}")

        return response.choices[0].message.content

    def fix_sql_in_json_string(self, raw_json_str: str):
        match = re.search(r'"sql"\s*:\s*"(.+?)"\s*,\s*"tables"', raw_json_str, re.DOTALL)
        if not match:
            raise ValueError("No sql found")

        raw_sql = match.group(1)
        escaped_sql = raw_sql.replace('"', r"\"")
        fixed_json_str = raw_json_str.replace(raw_sql, escaped_sql)

        return fixed_json_str

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
        # json_prompt = f"{prompt}\n\nRespond with a JSON object that
        # conforms to the following schema:\n{json.dumps(json_schema, indent=2)}"

        # Generate the response
        response_text = self.generate(prompt, response_format={"type": "json_object"}, **kwargs)

        # Parse the JSON response
        try:
            return json.loads(response_text, strict=False)
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON from the response
            fixed_response_text = self.fix_sql_in_json_string(response_text)
            try:
                return json.loads(fixed_response_text, strict=False)
            except json.JSONDecodeError:
                pass
            return {}

    def generate_with_tools(self, prompt: str, tools: List[Any], **kwargs) -> Dict:
        # flow control and context cache here
        pass

    async def generate_with_mcp(
        self,
        prompt: str,
        mcp_servers: Dict[str, MCPServerStdio],
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        **kwargs,
    ) -> Dict:
        """Generate a response using multiple MCP (Machine Conversation Protocol) servers.

        Args:
            prompt: The input prompt to send to the model
            mcp_servers: Dictionary of MCP servers to use for execution
            instruction: The instruction for the agent
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

        client = "anthropic"
        # TODO use config to switch client
        if client == "openai":
            # Create async OpenAI client
            logger.debug(f"Creating async OpenAI client with base_url: " f"{self.api_base}, model: {self.model_name}")
            async_client = wrap_openai(
                AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                )
            )

            model_params = {"model": self.model_name}
            async_model = OpenAIChatCompletionsModel(**model_params, openai_client=async_client)

            logger.debug("Starting run_agent with OpenAI")
            try:
                # Use context manager to manage multiple MCP servers
                async with multiple_mcp_servers(mcp_servers) as connected_servers:
                    logger.debug("MCP servers started successfully")

                    agent = Agent(
                        name=kwargs.pop("agent_name", "MCP_Agent"),
                        instructions=instruction,
                        mcp_servers=list(connected_servers.values()),
                        output_type=output_type,
                        model=async_model,
                    )
                    logger.debug(f"Agent created with name: {agent.name}")

                    logger.debug(f"Running agent with max_turns: {max_turns}")
                    result = await Runner.run(agent, input=prompt, max_turns=max_turns)

                    logger.debug("Agent execution completed")
                    # Wrap in object so .content and .sql_contexts are accessible
                    return {
                        "content": result.final_output,
                        "sql_contexts": extract_sql_contexts(result),
                    }
            except Exception as e:
                logger.error(f"Error in run_agent: {str(e)}")
                raise

        elif client == "anthropic":
            # Create async Anthropic client
            logger.debug(
                f"Creating async Anthropic client with base_url: " f"{self.api_base}, model: {self.model_name}"
            )
            try:
                all_tools = []

                # Use context manager to manage multiple MCP servers
                async with multiple_mcp_servers(mcp_servers) as connected_servers:
                    # Get all tools
                    for server_name, connected_server in connected_servers.items():
                        try:
                            mcp_tools = await connected_server.list_tools()
                            all_tools.extend(mcp_tools)
                            logger.debug(f"Retrieved {len(mcp_tools)} tools from {server_name}")
                        except Exception as e:
                            logger.error(f"Error getting tools from {server_name}: {str(e)}")
                            continue

                    logger.debug(f"Retrieved {len(all_tools)} tools from MCP servers")

                    tools = convert_tools_for_anthropic(all_tools)
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": f"{instruction}\n\n{prompt}"}],
                        }
                    ]
                    tool_call_cache = {}
                    sql_contexts = []
                    final_content = ""

                    # Execute conversation loop
                    for turn in range(max_turns):
                        logger.debug(f"Turn {turn + 1}/{max_turns}")

                        response = self.anthropic_client.messages.create(
                            model=self.model_name,
                            system=instruction,
                            messages=wrap_prompt_cache(messages),
                            tools=tools,
                            max_tokens=kwargs.get("max_tokens", 20480),
                            temperature=kwargs.get("temperature", 0.7),
                        )

                        message = response.content

                        # If no tool calls, conversation is complete
                        if not any(block.type == "tool_use" for block in message):
                            # Save final text response
                            final_content = "\n".join([block.text for block in message if block.type == "text"])
                            logger.debug(f"No tool calls, conversation completed: {final_content}")
                            break

                        for block in message:
                            if block.type == "tool_use":
                                logger.debug(f"Executing tool: {block.name} with input: {block.input}")
                                tool_executed = False

                                for server_name, connected_server in connected_servers.items():
                                    try:
                                        tmp_tools = await connected_server.list_tools()
                                        if any(tool.name == block.name for tool in tmp_tools):
                                            tool_result = await connected_server.call_tool(
                                                tool_name=block.name,
                                                arguments=json.loads(json.dumps(block.input)),
                                            )
                                            tool_call_cache[block.id] = tool_result
                                            tool_executed = True
                                            logger.debug(f"Tool {block.name} executed successfully on {server_name}")
                                            break
                                    except Exception as e:
                                        logger.error(f"Error executing tool {block.name} on {server_name}: {str(e)}")
                                        continue

                                if not tool_executed:
                                    logger.error(f"Tool {block.name} could not be executed on any server")

                        for block in message:
                            content = []
                            if block.type == "text":
                                content.append({"type": "text", "content": block.text})
                            elif block.type == "tool_use":
                                content.append(
                                    {
                                        "type": "tool_use",
                                        "id": block.id,
                                        "name": block.name,
                                        "input": block.input,
                                    }
                                )
                                messages.append({"role": "assistant", "content": content})

                                if block.id in tool_call_cache:
                                    sql_result = tool_call_cache[block.id].content[0].text
                                    # Use "Error" to determine whether the execution was successful,
                                    # because there's no way to judge it within MCP.
                                    if "Error" not in sql_result and block.name == "read_query":
                                        sql_context = SQLContext(
                                            sql_query=block.input["query"],
                                            sql_return=sql_result,
                                            row_count=None,
                                        )
                                        sql_contexts.append(sql_context)
                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "tool_result",
                                                    "tool_use_id": block.id,
                                                    "content": sql_result,
                                                }
                                            ],
                                        }
                                    )
                                else:
                                    # If tool execution failed, add error message
                                    error_message = f"Tool {block.name} execution failed"
                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "tool_result",
                                                    "tool_use_id": block.id,
                                                    "content": error_message,
                                                }
                                            ],
                                        }
                                    )
                            else:
                                raise ValueError("Unknown block")

                    logger.debug("Agent execution completed")
                    return {"content": final_content, "sql_contexts": sql_contexts}

            except Exception as e:
                logger.error(f"Error in generate_with_mcp: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported client: {client}")

    def token_count(self, prompt: str) -> int:
        """Estimate the number of tokens in a text using a simple approximation.

        Args:
            prompt (str): The text to count the tokens of.

        Returns:
            int: The estimated number of tokens in the text.
        """
        # Claude uses a similar tokenization scheme to GPT-3
        # We can use a simple approximation of ~4 characters per token
        return int(len(prompt) / 4 + 0.5)

    def set_context(self, workflow=None, current_node=None):
        pass
