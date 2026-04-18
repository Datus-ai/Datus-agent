# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared test fixtures for unit tests.

Design principle: NO mock except LLM.
- Real AgentConfig (from config dict)
- Real SQLite database (in tmp_path)
- Real db_manager_instance (connecting to real SQLite)
- Real Storage/RAG (vector store in tmp_path)
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


@pytest.fixture(autouse=True)
def _isolate_project_cwd(monkeypatch, tmp_path):
    """Run every unit test in a per-test isolated working directory.

    Two effects:

    1. ``load_project_override`` (see ``agent_config_loader._apply_project_override``)
       reads ``{cwd}/.datus/config.yml`` unconditionally. On a developer workstation
       that file typically pins ``target:`` to whatever model the human is using,
       which will not be present in the stub ``agent.yml`` fixtures the tests load.
       Without this isolation every test that reaches ``load_agent_config`` crashes
       with ``Unexcepted value of target``.

    2. ``AgentConfig.__init__`` derives ``project_root`` from ``os.getcwd()`` when
       the caller doesn't pass one. Pinning CWD to a fresh tmp dir keeps
       implicit ``{project_root}/subject`` paths, sharded session/data
       directories, and similar from leaking the real repo into a test's
       storage layout.

    The fixture is function-scoped so each test gets its own clean dir and
    monkeypatch restores the original CWD on teardown. ``tests/data`` and
    ``tests/conf`` loaders in this suite already resolve via
    ``Path(__file__).resolve().parents[...]``, so they're unaffected.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True, scope="session")
def _disable_langsmith_tracing():
    """Disable LangSmith/LangChain tracing for the unit test session only.

    Scoped to tests/unit_tests/ so integration/nightly/regression suites that
    intentionally exercise real tracing pipelines remain unaffected.
    Overrides any inherited env vars so UT runs never upload traces, even when
    the developer's shell has LANGSMITH_TRACING=true or an API key set.
    """
    saved = {
        k: os.environ.get(k)
        for k in (
            "LANGCHAIN_API_KEY",
            "LANGSMITH_API_KEY",
            "LANGCHAIN_ENDPOINT",
            "LANGSMITH_ENDPOINT",
            "LANGSMITH_TRACING",
            "LANGCHAIN_TRACING_V2",
        )
    }
    for key in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT"):
        os.environ.pop(key, None)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Singleton cleanup
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_global_singletons():
    """Clean up global singletons to avoid cross-test pollution.

    autouse=False -- use explicitly when needed.
    """
    from datus.utils.path_manager import reset_path_manager

    reset_path_manager()
    yield

    # Reset db_manager factory (if set)
    from datus.tools.db_tools.db_manager import set_db_manager_factory

    set_db_manager_factory(None)

    # Clean up storage registry
    from datus.storage.registry import clear_storage_registry

    clear_storage_registry()

    # Clear the context-local home used by implicit path-manager callers.
    reset_path_manager()


# ---------------------------------------------------------------------------
# SQLite database setup helper
# ---------------------------------------------------------------------------

CALIFORNIA_SCHOOLS_DB = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "sample_data",
    "california_schools",
    "california_schools.sqlite",
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
        "service": {
            "databases": {
                "california_schools": {
                    "type": "sqlite",
                    "uri": db_path,
                    "name": "california_schools",
                    "default": True,
                },
            },
            "bi_tools": {},
            "schedulers": {},
        },
        "project_root": str(tmp_path / "workspace"),
        "storage": {},
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
            "explore": {
                "system_prompt": "explore",
                "max_turns": 15,
            },
            "gen_table": {
                "system_prompt": "gen_table",
                "tools": "db_tools.*",
                "max_turns": 10,
            },
        },
    }

    nodes: dict[str, NodeConfig] = {}
    agent_config = AgentConfig(nodes=nodes, **config_kwargs)

    # Set current database (was: current_namespace = "test_ns")
    agent_config.current_database = "california_schools"

    # Capture cwd before yielding, in case a test changes it
    project_root = os.getcwd()

    yield agent_config

    # Cleanup: storage backends with empty data_dir create datus_db* in cwd
    for name in os.listdir(project_root):
        if name.startswith("datus_db"):
            shutil.rmtree(os.path.join(project_root, name), ignore_errors=True)


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
