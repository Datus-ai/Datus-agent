# -*- coding: utf-8 -*-
"""Flow-controlled agent wrapper that respects execution state."""

from agents import Agent

from datus.cli.execution_state import ExecutionState, execution_controller
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class FlowControlledAgent:
    """Agent wrapper that respects execution flow control."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.original_run = agent.run

    async def run(self, *args, **kwargs):
        """Run the agent with flow control."""
        # Check execution state before proceeding
        state = await execution_controller.get_state()

        if state == ExecutionState.PAUSED:
            logger.info("Agent execution paused, waiting for resume...")
            await execution_controller.wait_for_resume()
            logger.info("Agent execution resumed")

        # Run the original agent method
        return await self.original_run(*args, **kwargs)

    def patch_agent(self):
        """Patch the agent to use flow-controlled run method."""
        self.agent.run = self.run
        return self.agent

    @classmethod
    def wrap_agent(cls, agent: Agent) -> Agent:
        """Wrap an agent with flow control."""
        wrapper = cls(agent)
        return wrapper.patch_agent()
