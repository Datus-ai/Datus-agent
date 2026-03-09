import copy
import shutil
from pathlib import Path

import pytest

from datus.configuration.agent_config import AgentConfig
from tests.conftest import load_acceptance_config

NIGHTLY_SUB_AGENT_NAMES = ["nightly_test", "nightly_n7_test"]


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


@pytest.fixture(autouse=True)
def cleanup_nightly_sub_agent_data(agent_config):
    """Ensure sub-agent artifacts are cleaned up after each test, even on failure.

    Bootstrap tests write LanceDB indexes and other artifacts under
    ``{rag_base_path}/sub_agents/{name}/``.  Without this fixture a test
    failure would skip the manual ``_cleanup()`` call and leave stale data
    in the repo checkout, polluting subsequent runs.
    """
    yield
    for name in NIGHTLY_SUB_AGENT_NAMES:
        sub_agent_dir = Path(agent_config.rag_base_path) / "sub_agents" / name
        if sub_agent_dir.exists():
            shutil.rmtree(sub_agent_dir, ignore_errors=True)


@pytest.fixture
def snowflake_config() -> AgentConfig:
    """Load acceptance config for snowflake namespace."""
    return load_acceptance_config(namespace="snowflake")
