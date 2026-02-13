# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared test fixtures for unit tests.

Design principle: NO mock except LLM.
- Real AgentConfig (from config dict)
- Real SQLite database (in tmp_path)
- Real db_manager_instance (connecting to real SQLite)
- Real Storage/RAG (LanceDB in tmp_path)
- Real Tools (DBFuncTool, ContextSearchTools, etc.)
- Real PromptManager (using built-in templates)
- Real PathManager

The ONLY allowed mock: LLMBaseModel.create_model -> returns MockLLMModel
"""

import os
import shutil
from unittest.mock import patch

import pytest

from datus.configuration.agent_config import AgentConfig, NodeConfig
from tests.unit_tests.mock_llm_model import MockLLMModel


def pytest_collection_modifyitems(items):
    """Automatically mark all tests under unit_tests/ with the 'ci' marker."""
    for item in items:
        if "unit_tests" in str(item.fspath):
            item.add_marker(pytest.mark.ci)


# ---------------------------------------------------------------------------
# Singleton cleanup
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_global_singletons():
    """Clean up global singletons to avoid cross-test pollution.

    autouse=False -- use explicitly when needed.
    """
    yield

    # Clean up db_manager_instance
    import datus.tools.db_tools.db_manager as db_mgr_mod

    if db_mgr_mod._INSTANCE is not None:
        try:
            db_mgr_mod._INSTANCE.close()
        except Exception:
            pass
        db_mgr_mod._INSTANCE = None

    # Clean up StorageCache
    from datus.storage.cache import clear_cache

    clear_cache()

    # Note: Do NOT reset path_manager here -- AgentConfig.__init__ calls
    # path_manager.update_home() which keeps it in sync.


# ---------------------------------------------------------------------------
# SQLite database setup helper
# ---------------------------------------------------------------------------

CALIFORNIA_SCHOOLS_DB = os.path.join(
    os.path.dirname(__file__), "..", "..", "sample_data", "california_schools", "california_schools.sqlite"
)


def _copy_california_schools_db(dest_path: str) -> None:
    """Copy california_schools.sqlite into the test directory."""
    shutil.copy2(CALIFORNIA_SCHOOLS_DB, dest_path)


# ---------------------------------------------------------------------------
# Real AgentConfig fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def real_agent_config(tmp_path, reset_global_singletons):
    """Create a fully real AgentConfig backed by a real SQLite database.

    Includes:
    - home = tmp_path
    - target = "mock"
    - models with a mock OpenAI config
    - namespace "test_ns" with california_schools.sqlite ("california_schools")
    - agentic_nodes config for chat, gensql, gen_ext_knowledge, compare,
      gen_sql_summary, gen_metrics, gen_semantic_model, gen_report
    """
    db_path = os.path.join(str(tmp_path), "california_schools.sqlite")
    _copy_california_schools_db(db_path)

    # Create workspace subdirectory for filesystem tools
    os.makedirs(os.path.join(str(tmp_path), "workspace"), exist_ok=True)

    config_kwargs = {
        "home": str(tmp_path),
        "target": "mock",
        "models": {
            "mock": {
                "type": "openai",
                "api_key": "mock-api-key",
                "model": "mock-model",
                "base_url": "http://localhost:0",
            },
        },
        "namespace": {
            "test_ns": {
                "type": "sqlite",
                "dbs": [
                    {
                        "uri": db_path,
                        "name": "california_schools",
                    }
                ],
            },
        },
        "agentic_nodes": {
            "chat": {
                "system_prompt": "chat",
                "tools": "db_tools.*,context_search_tools.*",
                "max_turns": 5,
            },
            "gensql": {
                "system_prompt": "gensql",
                "tools": "db_tools.*",
                "max_turns": 5,
            },
            "gen_ext_knowledge": {
                "system_prompt": "gen_ext_knowledge",
                "max_turns": 5,
            },
            "compare": {
                "system_prompt": "compare",
                "tools": "db_tools.*",
                "max_turns": 5,
            },
            "gen_sql_summary": {
                "system_prompt": "gen_sql_summary",
                "max_turns": 5,
            },
            "gen_metrics": {
                "system_prompt": "gen_metrics",
                "max_turns": 5,
            },
            "gen_semantic_model": {
                "system_prompt": "gen_semantic_model",
                "tools": "db_tools.*",
                "max_turns": 5,
            },
            "gen_report": {
                "system_prompt": "gen_report",
                "tools": "db_tools.*,context_search_tools.*",
                "max_turns": 5,
            },
        },
    }

    nodes: dict[str, NodeConfig] = {}
    agent_config = AgentConfig(nodes=nodes, **config_kwargs)

    # Set current namespace and database
    agent_config.current_namespace = "test_ns"
    agent_config.current_database = "california_schools"

    return agent_config


# ---------------------------------------------------------------------------
# Mock LLM create_model fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_create():
    """The ONLY allowed mock: patch LLMBaseModel.create_model to return MockLLMModel.

    Returns the MockLLMModel instance so tests can call model.reset(responses=[...])
    to configure LLM responses.
    """
    mock_model = MockLLMModel()
    with patch("datus.models.base.LLMBaseModel.create_model", return_value=mock_model):
        yield mock_model
