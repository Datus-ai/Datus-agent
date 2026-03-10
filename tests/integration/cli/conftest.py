import time
from argparse import Namespace

import pytest

from tests.conftest import TEST_CONF_DIR


@pytest.fixture
def mock_args():
    """Provides default mock arguments for initializing DatusCLI."""
    return Namespace(
        history_file="~/.datus/reference_sql",
        debug=False,
        namespace="bird_school",
        database="california_schools",
        config=str(TEST_CONF_DIR / "agent.yml"),
        storage_path="tests/data",
    )


def wait_for_agent(cli, timeout=120):
    """Wait for agent to be ready with timeout."""
    start_time = time.time()
    while not cli.agent_ready:
        if time.time() - start_time > timeout:
            pytest.fail("Agent initialization timed out.")
        time.sleep(0.5)
