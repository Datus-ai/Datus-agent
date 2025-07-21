import copy
import json
import re
from typing import Any, AsyncGenerator, Dict, Optional

import anthropic
from agents import Agent, RunContextWrapper, Usage
from langsmith.wrappers import wrap_anthropic

from datus.models.base import LLMBaseModel
from datus.models.client_factory import ClientFactory, ModelConfigHelper
from datus.models.common_mixins import JSONParsingMixin, MCPMixin, StreamingMixin
from datus.models.mcp_utils import multiple_mcp_servers
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.node_models import SQLContext
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def wrap_prompt_cache(messages):
    """Add cache control to messages."""
    messages_copy = copy.deepcopy(messages)
    msg_size = len(messages_copy)
    content = messages_copy[msg_size - 1]["content"]
    cnt_size = len(content)
    if isinstance(content, list):
        content[cnt_size - 1]["cache_control"] = {"type": "ephemeral"}
    return messages_copy


def convert_tools_for_anthropic(mcp_tools):
    """Convert MCP tools to Anthropic format."""
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
    if anthropic_tools:
        anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
    return anthropic_tools


class ClaudeModel(LLMBaseModel, JSONParsingMixin, MCPMixin, StreamingMixin):
    """Implementation of the BaseModel for Claude's API."""

    def __init__(self, model_config):
        super().__init__(model_config)
        self.api_base = ModelConfigHelper.get_base_url(model_config)
        self.model_name = model_config.model
        self.api_key = ModelConfigHelper.get_api_key(model_config, "ANTHROPIC_API_KEY")

        logger.debug(f"Using Claude model: {self.model_name} base Url: {self.api_base}")

        # OpenAI-compatible client
        base_url = f"{self.api_base}/v1" if self.api_base else None
        self.client = ClientFactory.create_openai_client(
            api_key=self.api_key,
            base_url=base_url,
            proxy_url=ModelConfigHelper.get_proxy_url(),
        )

        # Native Anthropic client
        self.anthropic_client = self._create_anthropic_client()

    def _create_anthropic_client(self):
        """Create Anthropic client with proxy support."""
        proxy_url = ModelConfigHelper.get_proxy_url()
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.api_base if self.api_base else None,
        }

        if proxy_url:
            import httpx

            proxy_client = httpx.Client(
                transport=httpx.HTTPTransport(proxy=httpx.Proxy(url=proxy_url)),
                timeout=60.0,
            )
            client_kwargs["http_client"] = proxy_client

        return wrap_anthropic(anthropic.Anthropic(**client_kwargs))

    def generate(self, prompt: Any, **kwargs) -> str:
        """Generate a response from the Claude model."""
        base_params = {
            "model": self.model_name,
            "temperature": 0.7,
            "max_tokens": 3000,
            "top_p": 1.0,
        }
        params = ModelConfigHelper.merge_params(base_params, **kwargs)
        messages = ModelConfigHelper.prepare_messages(prompt)

        response = self.client.chat.completions.create(messages=messages, **params)
        logger.debug(f"Model response: {response.choices[0].message.content}")
        return response.choices[0].message.content

    def fix_sql_in_json_string(self, raw_json_str: str):
        """Fix SQL escaping in JSON string."""
        match = re.search(r'"sql"\s*:\s*"(.+?)"\s*,\s*"tables"', raw_json_str, re.DOTALL)
        if not match:
            raise ValueError("No sql found")

        raw_sql = match.group(1)
        escaped_sql = raw_sql.replace('"', r"\"")
        return raw_json_str.replace(raw_sql, escaped_sql)

    def generate_with_json_output(self, prompt: Any, **kwargs) -> Dict:
        """Generate a response and ensure it conforms to the provided JSON schema."""
        response_text = self.generate(prompt, response_format={"type": "json_object"}, **kwargs)
        return self.parse_json_response(response_text, self.fix_sql_in_json_string)

    def generate_with_tools(self, prompt: str, tools: list, **kwargs) -> Dict:
        """Generate with tools - placeholder for future implementation."""
        # TODO: Implement tool-based generation
        return {"content": "Tool-based generation not implemented yet"}

    async def generate_with_mcp(
        self,
        prompt: str,
        mcp_servers: dict,
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        **kwargs,
    ) -> Dict:
        """Generate a response using multiple MCP servers."""
        self.setup_json_encoder()

        client = "anthropic"
        if client == "openai":

            def async_model_factory(**factory_kwargs):
                base_url = f"{self.api_base}/v1" if self.api_base else None
                return ClientFactory.create_async_model(
                    model_name=self.model_name,
                    api_key=self.api_key,
                    base_url=base_url,
                )

            return await self.generate_with_mcp_base(
                prompt=prompt,
                mcp_servers=mcp_servers,
                instruction=instruction,
                output_type=output_type,
                max_turns=max_turns,
                async_model_factory=async_model_factory,
                **kwargs,
            )
        elif client == "anthropic":
            return await self._generate_with_anthropic_mcp(prompt, mcp_servers, instruction, max_turns, **kwargs)
        else:
            raise ValueError(f"Unsupported client: {client}")

    async def _generate_with_anthropic_mcp(
        self, prompt: str, mcp_servers: dict, instruction: str, max_turns: int, **kwargs
    ) -> Dict:
        """Generate using native Anthropic client with MCP."""
        try:
            all_tools = []
            async with multiple_mcp_servers(mcp_servers) as connected_servers:
                # Get all tools
                for server_name, connected_server in connected_servers.items():
                    try:
                        agent = Agent(name="mcp-tools-agent")
                        run_context = RunContextWrapper(context=None, usage=Usage())
                        mcp_tools = await connected_server.list_tools(run_context, agent)
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
                        final_content = "\n".join([block.text for block in message if block.type == "text"])
                        logger.debug(f"No tool calls, conversation completed: {final_content}")
                        break

                    # Process tool calls
                    await self._process_tool_calls(message, connected_servers, tool_call_cache, messages)

                    # Extract SQL contexts
                    for block in message:
                        if block.type == "tool_use" and block.id in tool_call_cache:
                            sql_result = tool_call_cache[block.id].content[0].text
                            if "Error" not in sql_result and block.name == "read_query":
                                sql_context = SQLContext(
                                    sql_query=block.input["query"],
                                    sql_return=sql_result,
                                    row_count=None,
                                )
                                sql_contexts.append(sql_context)

                logger.debug("Agent execution completed")
                return {"content": final_content, "sql_contexts": sql_contexts}

        except Exception as e:
            logger.error(f"Error in generate_with_mcp: {str(e)}")
            raise

    async def _process_tool_calls(self, message, connected_servers, tool_call_cache, messages):
        """Process tool calls in Anthropic MCP."""
        for block in message:
            if block.type == "tool_use":
                logger.debug(f"Executing tool: {block.name} with input: {block.input}")
                tool_executed = False

                for server_name, connected_server in connected_servers.items():
                    try:
                        agent = Agent(name="mcp-claude-agent")
                        run_context = RunContextWrapper(context=None, usage=Usage())
                        tmp_tools = await connected_server.list_tools(run_context, agent)
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

        # Add tool results to messages
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

    async def generate_with_mcp_stream(
        self,
        prompt: str,
        mcp_servers: dict,
        instruction: str,
        output_type: dict,
        max_turns: int = 10,
        action_history_manager: Optional[ActionHistoryManager] = None,
        **kwargs,
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate a response using multiple MCP servers with streaming."""

        def async_model_factory(**factory_kwargs):
            base_url = f"{self.api_base}/v1" if self.api_base else None
            return ClientFactory.create_async_model(
                model_name=self.model_name,
                api_key=self.api_key,
                base_url=base_url,
            )

        async for action in self.generate_with_mcp_stream_base(
            prompt=prompt,
            mcp_servers=mcp_servers,
            instruction=instruction,
            output_type=output_type,
            max_turns=max_turns,
            action_history_manager=action_history_manager,
            async_model_factory=async_model_factory,
            **kwargs,
        ):
            yield action

    def set_context(self, workflow=None, current_node=None):
        """Set workflow and node context."""
        pass

    def token_count(self, prompt: str) -> int:
        """Estimate the number of tokens in a text."""
        # Claude uses a similar tokenization scheme to GPT-3
        return int(len(prompt) / 4 + 0.5)
