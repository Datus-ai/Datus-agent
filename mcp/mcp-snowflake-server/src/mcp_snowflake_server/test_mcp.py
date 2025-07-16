import argparse
import asyncio
import json
import os

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    RunContextWrapper,
    Usage,
    set_default_openai_client,
    set_trace_processors,
    set_tracing_disabled,
)
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import OpenAIAgentsTracingProcessor, wrap_openai
from openai import AsyncOpenAI
from pydantic import AnyUrl

load_dotenv(override=True)

set_tracing_disabled(True)


async def test(model_name):
    model = get_model(model_name)
    agent = Agent(name="Assistant", instructions="你是一名助人为乐的助手。", model=model)
    result = await Runner.run(agent, "who are you?")
    print(result.final_output)


from datetime import date, datetime


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, AnyUrl):
            return str(obj)
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


def get_model(model_name):
    if model_name == "deepseek-v3":  # v3-0324
        client = wrap_openai(AsyncOpenAI(base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_O_API_KEY")))
        model = OpenAIChatCompletionsModel(model="deepseek-chat", openai_client=client)
    elif model_name == "deepseek-r1":
        # deepseek-r1 doesn't support function calling in deepseek official api, but works in ark deepseek-r1 model
        client = wrap_openai(AsyncOpenAI(base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_O_API_KEY")))
        model = OpenAIChatCompletionsModel(model="deepseek-reasoner", openai_client=client)
    elif model_name == "openai-4.1":
        client = wrap_openai(AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        model = OpenAIChatCompletionsModel(model="gpt-4.1", openai_client=client)
    elif model_name == "openai-o4-mini":
        client = wrap_openai(AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        model = OpenAIChatCompletionsModel(model="o4-mini", openai_client=client)
    elif model_name == "claude-3.7":
        client = wrap_openai(
            AsyncOpenAI(base_url="https://api.anthropic.com/v1", api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
        model = OpenAIChatCompletionsModel(model="claude-3-7-sonnet-20250219", openai_client=client)
    elif model_name == "seed-1.5-thinking":
        client = wrap_openai(
            AsyncOpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=os.getenv("ARK_API_KEY"))
        )
        model = OpenAIChatCompletionsModel(model="ep-20250417143015-qkxhv", openai_client=client)
    else:
        raise ValueError(f"unsupported model: {model_name}")

    set_default_openai_client(client)
    return model


SQL_AGENT_INSTRUCTIONS = """You are a snowflake expert. Your task is to:
1. Understand the user's question about data analysis
2. Generate appropriate SQL queries
3. Execute the queries using the provided tools
4. Present the results in a clear and concise manner
*Enclose all column names in double quotes to comply with Snowflake syntax requirements and avoid grammar errors.* When referencing table names in Snowflake SQL, you must include both the database_name and schema_name."""

# from langsmith import trace


# @trace(project_name="MCP_test")
@traceable
async def run(model_name, max_turns=10, question=None):
    if question is None:
        question = "Based on the most recent refresh date, identify the top-ranked rising search term for the week that is exactly one year prior to the latest available week in the dataset"

    model = get_model(model_name)
    json._default_encoder = CustomJSONEncoder()
    snowflake_server_path = os.getenv("SNOWFLAKE_SERVER_PATH", "/path/to/snowflake-server")

    async with MCPServerStdio(
        params={
            "command": "uv",
            "args": [
                "--directory",
                snowflake_server_path,
                "run",
                "mcp_snowflake_server",
                "--log_dir",
                f"{snowflake_server_path}/logs/",
                "--log_level",
                "DEBUG",
            ],
        },
    ) as server:
        # Create minimal agent and run context for the new interface
        temp_agent = Agent(name='test-agent')
        run_context = RunContextWrapper(context=None, usage=Usage())
        tools = await server.list_tools(run_context, temp_agent)

        sql_agent = Agent(
            name="MCP_snowflake_test", instructions=SQL_AGENT_INSTRUCTIONS, mcp_servers=[server], model=model
        )

        result = await Runner.run(sql_agent, input=question, max_turns=max_turns)
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v3",
        choices=["deepseek-v3", "deepseek-r1", "openai-4.1", "openai-o4-mini", "claude-3.7", "seed-1.5-thinking"],
        help="Choose a model",
    )
    parser.add_argument("--test", action="store_true", help="test connection")
    parser.add_argument("--max-turns", type=int, default=10, help="max turns")
    parser.add_argument("--question", type=str, help="Query you want to ask")
    args = parser.parse_args()

    if args.test:
        asyncio.run(test(args.model))
    else:
        set_trace_processors([OpenAIAgentsTracingProcessor()])
        asyncio.run(run(args.model, args.max_turns, args.question))
