# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared fixtures for integration tests.

Provides reusable AgentConfig, SkillManager, and PermissionManager fixtures
that load from tests/conf/agent.yml — mirroring the real agent startup flow.
"""

import copy
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools import SkillConfig, SkillFuncTool, SkillManager

TESTS_ROOT = Path(__file__).parent.parent  # tests/
CONF_DIR = TESTS_ROOT / "conf"
SKILLS_DIR = TESTS_ROOT / "data" / "skills"

# Real LLM integration test paths
REAL_SKILLS_DIR = Path.home() / ".datus" / "skills"
REAL_SQLITE_DB = Path.home() / ".datus" / "benchmark" / "california_schools" / "california_schools.sqlite"


@dataclass(frozen=True)
class RequiredPostgresqlStorage:
    """Provisioned PostgreSQL RDB + pgvector test environments."""

    rdb_config: Any
    vector_config: Any


@dataclass(frozen=True)
class P0ExternalSources:
    """Source checkouts required by the deterministic P0 integration suites."""

    storage_adapters: Path
    sql_policies: Path
    datus_plugins: Path

    @property
    def superset_plugin(self) -> Path:
        return self.datus_plugins / "datus-superset-plugin"


@dataclass(frozen=True)
class ManagedPluginRuntime:
    """Isolated managed plugin store exercised through fresh CLI processes."""

    home: Path
    datus_executable: Path

    def run(
        self,
        *args: str,
        check: bool = True,
        cwd: Path | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        # Isolating HOME must not also discard uv's dependency/build cache;
        # otherwise every managed source install re-downloads its wheelhouse.
        env.setdefault("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))
        env["HOME"] = str(self.home)
        return subprocess.run(
            [str(self.datus_executable), *args],
            check=check,
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
            timeout=timeout,
        )

    def install(self, source: Path) -> subprocess.CompletedProcess[str]:
        return self.run("plugin", "install", f"src:{source}")


def _entry_points_for(group: str, name: str) -> list[Any]:
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return list(entry_points.select(group=group, name=name))
    return [entry_point for entry_point in entry_points.get(group, []) if entry_point.name == name]


def _required_test_environment(group: str, name: str):
    matches = _entry_points_for(group, name)
    if len(matches) != 1:
        pytest.fail(
            f"P0 requires exactly one {group}:{name} entry point, found {len(matches)}",
            pytrace=False,
        )
    try:
        return matches[0].load()()
    except Exception as exc:  # noqa: BLE001 - fail with the external provider's concrete error.
        pytest.fail(f"P0 could not load {group}:{name}: {exc}", pytrace=False)


@pytest.fixture(scope="session")
def required_postgresql_storage() -> RequiredPostgresqlStorage:
    """Start real PostgreSQL and pgvector providers, failing instead of skipping."""
    rdb_environment = _required_test_environment("datus.storage.rdb.testing", "postgresql")
    vector_environment = _required_test_environment("datus.storage.vector.testing", "postgresql")
    started = []
    try:
        for environment in (rdb_environment, vector_environment):
            started.append(environment)
            environment.setup()
        rdb_config = rdb_environment.get_config()
        vector_config = vector_environment.get_config()
        if rdb_config.backend_type != "postgresql" or vector_config.backend_type != "postgresql":
            pytest.fail(
                "P0 PostgreSQL providers must both report backend_type=postgresql",
                pytrace=False,
            )
        yield RequiredPostgresqlStorage(rdb_config=rdb_config, vector_config=vector_config)
    except Exception as exc:
        pytest.fail(f"P0 PostgreSQL/pgvector setup failed: {exc}", pytrace=False)
    finally:
        teardown_errors = []
        for environment in reversed(started):
            try:
                environment.teardown()
            except Exception as exc:  # noqa: BLE001 - report every provider cleanup error.
                teardown_errors.append(str(exc))
        if teardown_errors:
            pytest.fail(f"P0 PostgreSQL/pgvector teardown failed: {'; '.join(teardown_errors)}", pytrace=False)


def _resolve_external_root() -> Path:
    configured = os.environ.get("EXTERNAL_REPOS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]
    for candidate in repo_root.parents:
        if (candidate / "datus-storage-adapters").is_dir() and (candidate / "Datus-Plugins").is_dir():
            return candidate
    pytest.fail("P0 external repository root was not found; set EXTERNAL_REPOS_ROOT", pytrace=False)


@pytest.fixture(scope="session")
def p0_external_sources() -> P0ExternalSources:
    """Require the three P0 source checkouts used by Nightly."""
    root = _resolve_external_root()
    sources = P0ExternalSources(
        storage_adapters=root / "datus-storage-adapters",
        sql_policies=root / "datus-sql-policies",
        datus_plugins=root / "Datus-Plugins",
    )
    required = {
        "datus-storage-adapters": sources.storage_adapters,
        "datus-sql-policies": sources.sql_policies,
        "Datus-Plugins": sources.datus_plugins,
        "datus-superset-plugin": sources.superset_plugin,
    }
    missing = [f"{name} ({path})" for name, path in required.items() if not path.is_dir()]
    if missing:
        pytest.fail(f"P0 external source checkout(s) missing: {', '.join(missing)}", pytrace=False)
    return sources


@pytest.fixture
def managed_plugin_runtime(tmp_path) -> ManagedPluginRuntime:
    """Return a clean managed store; every command runs in a new process."""
    executable = Path(sys.executable).with_name("datus")
    if not executable.is_file():
        pytest.fail(f"Datus CLI executable not found beside test Python: {executable}", pytrace=False)
    home = tmp_path / "home"
    home.mkdir()
    return ManagedPluginRuntime(home=home, datus_executable=executable)


# ── AgentConfig fixtures ──


@pytest.fixture(scope="module")
def agent_config(tmp_path_factory) -> AgentConfig:
    """Load AgentConfig from a temp copy of tests/conf/agent.yml.

    Uses tmp copy so tests never mutate the source config.
    The config includes skills, permissions, and agentic_nodes sections.
    """
    src = CONF_DIR / "agent.yml"
    tmp_dir = tmp_path_factory.mktemp("skill_conf")
    tmp_cfg = tmp_dir / "agent.yml"
    shutil.copy2(src, tmp_cfg)
    config = load_agent_config(
        config=str(tmp_cfg),
        datasource="bird_school",
        reload=True,
        force=True,
        yes=True,
    )
    return config


# ── SkillConfig fixtures ──


@pytest.fixture
def skill_config() -> SkillConfig:
    """SkillConfig pointing to tests/data/skills."""
    return SkillConfig(directories=[str(SKILLS_DIR)])


@pytest.fixture
def skill_config_with_extra(tmp_path) -> tuple[SkillConfig, Path]:
    """SkillConfig with two directories: real skills + a tmp dir for dynamic tests."""
    extra_dir = tmp_path / "extra_skills"
    extra_dir.mkdir()
    return SkillConfig(directories=[str(SKILLS_DIR), str(extra_dir)]), extra_dir


# ── PermissionManager fixtures ──


@pytest.fixture
def perm_deny_admin() -> PermissionConfig:
    """Permission config that denies admin-* skills."""
    return PermissionConfig(
        default_permission=PermissionLevel.ALLOW,
        rules=[
            PermissionRule(tool="skills", pattern="admin-*", permission=PermissionLevel.DENY),
        ],
    )


@pytest.fixture
def perm_ask_sql() -> PermissionConfig:
    """Permission config that requires ASK for sql-* skills."""
    return PermissionConfig(
        default_permission=PermissionLevel.ALLOW,
        rules=[
            PermissionRule(tool="skills", pattern="sql-*", permission=PermissionLevel.ASK),
        ],
    )


@pytest.fixture
def perm_deny_admin_with_node_override() -> tuple:
    """Global DENY admin + node override that ALLOWs admin for school_all."""
    global_config = PermissionConfig(
        default_permission=PermissionLevel.ALLOW,
        rules=[
            PermissionRule(tool="skills", pattern="admin-*", permission=PermissionLevel.DENY),
        ],
    )
    node_overrides = {
        "school_all": PermissionConfig(
            rules=[
                PermissionRule(tool="skills", pattern="admin-*", permission=PermissionLevel.ALLOW),
            ],
        ),
    }
    return global_config, node_overrides


# ── SkillManager fixtures ──


@pytest.fixture
def skill_manager(skill_config) -> SkillManager:
    """SkillManager without permissions (discovers all skills)."""
    return SkillManager(config=skill_config)


def pytest_collection_modifyitems(items):
    """Automatically mark all tests under integration/ with the 'integration' marker.

    Also reorder gen_* agent tests to respect logical dependencies:
    semantic_model → metrics
    (metrics reference measures defined in semantic models)
    """
    for item in items:
        if "integration" in str(item.fspath) and "unit_tests" not in str(item.fspath):
            item.add_marker(pytest.mark.integration)

    # Reorder gen_* agent tests by dependency (lower = runs first)
    gen_test_order = {
        "test_gen_semantic_model_agentic": 0,
        "test_gen_metrics_agentic": 1,
    }

    gen_items = [(i, item) for i, item in enumerate(items) if item.fspath.purebasename in gen_test_order]
    if gen_items:
        indices = [i for i, _ in gen_items]
        sorted_gen = sorted(
            [item for _, item in gen_items],
            key=lambda x: (gen_test_order[x.fspath.purebasename], x.name),
        )
        for idx, item in zip(indices, sorted_gen):
            items[idx] = item


@pytest.fixture
def skill_manager_with_perms(skill_config, perm_deny_admin) -> SkillManager:
    """SkillManager with permission enforcement (admin-* denied)."""
    perm_manager = PermissionManager(global_config=perm_deny_admin)
    return SkillManager(config=skill_config, permission_manager=perm_manager)


# ── SkillFuncTool fixtures ──


@pytest.fixture
def skill_func_tool(skill_manager) -> SkillFuncTool:
    """SkillFuncTool for the chatbot node (no permissions)."""
    return SkillFuncTool(manager=skill_manager, node_name="chatbot")


# ── Real LLM integration test fixtures ──


@pytest.fixture(scope="module")
def llm_agent_config(tmp_path_factory) -> AgentConfig:
    """Load AgentConfig for real LLM integration tests.

    Uses tests/conf/agent_llm_skill.yml with california_schools database
    and real skills from ~/.datus/skills/.

    Skips if prerequisites missing:
    - DEEPSEEK_API_KEY not set
    - california_schools.sqlite not found
    - ~/.datus/skills/report-generator/ not found
    """
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")
    if not REAL_SQLITE_DB.exists():
        pytest.skip(f"SQLite database not found: {REAL_SQLITE_DB}")
    if not (REAL_SKILLS_DIR / "report-generator" / "SKILL.md").exists():
        pytest.skip(f"report-generator skill not found: {REAL_SKILLS_DIR}")

    src = CONF_DIR / "agent_llm_skill.yml"
    tmp_dir = tmp_path_factory.mktemp("llm_skill_conf")
    tmp_cfg = tmp_dir / "agent_llm_skill.yml"
    shutil.copy2(src, tmp_cfg)

    config = load_agent_config(
        config=str(tmp_cfg),
        datasource="california_schools",
        reload=True,
        force=True,
        yes=True,
    )
    return config


# ── CLI shared fixtures ──


@pytest.fixture
def mock_args():
    """Provides default mock arguments for initializing DatusCLI."""
    return Namespace(
        history_file="~/.datus/reference_sql",
        debug=False,
        datasource="bird_school",
        database="california_schools",
        config=str(CONF_DIR / "agent.yml"),
        storage_path="tests/data",
    )


def wait_for_agent(cli, timeout=120):
    """Wait for agent to be ready with timeout."""
    start_time = time.time()
    while not cli.agent_ready:
        if time.time() - start_time > timeout:
            pytest.fail("Agent initialization timed out.")
        time.sleep(0.5)


# ── Sub-agent cleanup fixtures ──

NIGHTLY_SUB_AGENT_NAMES = ["nightly_test", "nightly_n7_test"]


@pytest.fixture
def nightly_agent_config(tmp_path) -> AgentConfig:
    """Load acceptance config for nightly sub-agent tests.

    Function-scoped with deepcopy of agentic_nodes to prevent test mutations
    from leaking into the configuration_manager cache. The SQLite benchmark
    datasource is also function-scoped so adapters that create support tables
    cannot mutate the shared ``~/benchmark`` fixture.
    """
    from tests.conftest import isolate_bird_sqlite_databases, load_acceptance_config

    config = load_acceptance_config(datasource="bird_school")
    isolate_bird_sqlite_databases(config, tmp_path, ("california_schools",))
    config.rag_base_path = "tests/data"
    config.agentic_nodes = copy.deepcopy(config.agentic_nodes)
    return config


@pytest.fixture
def cleanup_sub_agent_data(nightly_agent_config):
    """Clean up sub-agent artifacts before and after each test, even on failure.

    Bootstrap tests write LanceDB indexes and other artifacts under
    ``{rag_base_path}/sub_agents/{name}/``. This fixture ensures stale data
    from interrupted runs is removed before the test starts, and cleaned up
    after each test run.
    """

    # Confine deletions to the test data tree so a misconfigured/empty
    # rag_base_path can never resolve rmtree onto real user data.
    safe_base = (TESTS_ROOT / "data").resolve()

    def _cleanup():
        for name in NIGHTLY_SUB_AGENT_NAMES:
            sub_agent_dir = (Path(nightly_agent_config.rag_base_path) / "sub_agents" / name).resolve()
            if safe_base not in sub_agent_dir.parents:
                pytest.fail(f"Refusing to rmtree outside test data tree: {sub_agent_dir}")
            if sub_agent_dir.exists():
                # safe: dir is asserted under TESTS_ROOT/data above
                shutil.rmtree(sub_agent_dir, ignore_errors=True)  # audit-noqa: rmtree_outside_tmp

    _cleanup()
    yield
    _cleanup()
