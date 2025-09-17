from typing import Literal, Type, Union

try:
    from langsmith.client import RUN_TYPE_T
except ImportError:
    RUN_TYPE_T = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]
from openai import AsyncOpenAI, OpenAI

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def create_openai_client(
    cls: Type[Union[OpenAI, AsyncOpenAI]], api_key: str, base_url: str
) -> Union[OpenAI, AsyncOpenAI]:
    client = cls(api_key=api_key, base_url=base_url)
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except ImportError:
        logger.warning("langsmith wrapper not available")
        return client


def optional_traceable(name: str = "", run_type: RUN_TYPE_T = "chain"):
    def decorator(func):
        try:
            from langsmith import traceable

            # Use provided run_name or fallback to function name
            trace_name = name or getattr(func, "__name__", "agent_operation")

            # Directly apply the traceable decorator to the original function and return it
            return traceable(name=trace_name, run_type=run_type)(func)
        except ImportError:
            # If langsmith is not available, just return the original function
            return func

    return decorator
