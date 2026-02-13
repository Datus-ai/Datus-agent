import copy

import pytest

from datus.configuration.agent_config import AgentConfig
from tests.conftest import load_acceptance_config


@pytest.fixture
def agent_config() -> AgentConfig:
    """Load acceptance config for bird_school namespace.

    Creates a shallow copy of agentic_nodes to prevent test mutations
    (e.g. adding SubAgentConfig entries) from leaking into the
    configuration_manager cache and polluting subsequent tests.
    """
    config = load_acceptance_config(namespace="bird_school")
    config.rag_base_path = "tests/data"
    config.agentic_nodes = copy.deepcopy(config.agentic_nodes)
    return config


@pytest.fixture
def snowflake_config() -> AgentConfig:
    """Load acceptance config for snowflake namespace."""
    return load_acceptance_config(namespace="snowflake")
