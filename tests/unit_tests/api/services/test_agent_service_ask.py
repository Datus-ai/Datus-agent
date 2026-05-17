"""Unit tests for the ``ask_report`` / ``ask_dashboard`` create_agent flow.

Pins:

* SUBAGENT_TOOL_REFERENCE exposes default tools for both new types and
  excludes filesystem writes (read-only consultant invariant).
* create_agent validates ``artifact_slug`` is supplied, matches the slug
  regex, and that the matching ``reports/<slug>`` or ``dashboards/<slug>``
  directory exists under ``project_root``.
* create_agent persists ``artifact_slug`` on the agentic_nodes entry —
  this is the load-bearing field that the node reads at startup to
  resolve the binding.
* Same-artifact uniqueness: only one ``ask_*`` agent per (type, slug)
  can be registered; second attempt returns ``ARTIFACT_ALREADY_BOUND``.

Reuses the ``real_agent_config`` / ``agent_yml_with_singleton`` fixtures
from the existing agent_service test module so the persistence path
(ConfigurationManager.save → re-read) is exercised end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datus.api.models.agent_models import CreateAgentInput
from datus.api.services.agent_service import SUBAGENT_TOOL_REFERENCE, AgentService

pytestmark = pytest.mark.asyncio


@pytest.fixture
def agent_yml_with_singleton(real_agent_config):
    """Same shape as the sibling ``test_agent_service.py`` fixture — duplicated
    here because pytest fixtures defined in another test module are not
    visible across modules (would need to live in ``conftest.py``).

    Pre-seeds an empty ``agent.yml`` at the resolved home and installs the
    ``ConfigurationManager`` singleton at that path so ``_save_agentic_nodes``
    writes land in the tmp_path tree.
    """
    from datus.configuration import agent_config_loader

    home = real_agent_config.path_manager.datus_home
    cfg_path = home / "agent.yml"
    if not cfg_path.exists():
        cfg_path.write_text("agent: {}\n", encoding="utf-8")
    agent_config_loader.configuration_manager(config_path=str(cfg_path), reload=True)
    yield cfg_path
    agent_config_loader.CONFIGURATION_MANAGER = None


# --------------------------------------------------------------------------- #
# SUBAGENT_TOOL_REFERENCE                                                     #
# --------------------------------------------------------------------------- #


class TestSubagentToolReference:
    """The two new types must show up with read-only tool default sets."""

    def test_ask_report_registered(self):
        assert "ask_report" in SUBAGENT_TOOL_REFERENCE

    def test_ask_dashboard_registered(self):
        assert "ask_dashboard" in SUBAGENT_TOOL_REFERENCE

    @pytest.mark.parametrize("ask_type", ["ask_report", "ask_dashboard"])
    def test_default_tools_include_read_only_filesystem(self, ask_type):
        defaults = SUBAGENT_TOOL_REFERENCE[ask_type]["default_tools"]
        # read_file / glob / grep are essential for the LLM to navigate the
        # artifact directory by hand.
        assert "filesystem_tools.read_file" in defaults
        assert "filesystem_tools.glob" in defaults
        assert "filesystem_tools.grep" in defaults

    @pytest.mark.parametrize("ask_type", ["ask_report", "ask_dashboard"])
    def test_default_tools_exclude_filesystem_writes(self, ask_type):
        """Read-only consultant invariant: no write/edit/delete in defaults."""
        defaults = SUBAGENT_TOOL_REFERENCE[ask_type]["default_tools"]
        assert "filesystem_tools.write_file" not in defaults
        assert "filesystem_tools.edit_file" not in defaults
        assert "filesystem_tools.delete_file" not in defaults
        # bash_tools is meant for general-purpose execution — has no place
        # in a read-only ask agent default config.
        assert not any(d.startswith("bash_tools") for d in defaults)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _mkdir_artifact(project_root: str, kind: str, slug: str) -> Path:
    """Create the ``reports/<slug>/`` or ``dashboards/<slug>/`` directory
    expected by the validator."""
    kind_dir = "reports" if kind == "report" else "dashboards"
    path = Path(project_root) / kind_dir / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# create_agent — validation & persistence                                     #
# --------------------------------------------------------------------------- #


class TestCreateAskAgent:
    async def test_missing_artifact_slug_rejected(self, real_agent_config, agent_yml_with_singleton):
        """ask_report without artifact_slug fails before any filesystem write."""
        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(name="ask_no_slug", type="ask_report"),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "ARTIFACT_SLUG_REQUIRED"

    @pytest.mark.parametrize(
        "bad_slug",
        [
            "Has-Hyphen",  # uppercase / hyphens not in pattern
            "has space",
            "中文",
            "x" * 81,  # over length cap
        ],
    )
    async def test_invalid_slug_rejected(self, real_agent_config, agent_yml_with_singleton, bad_slug):
        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(name="ask_bad_slug", type="ask_report", artifact_slug=bad_slug),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "INVALID_ARTIFACT_SLUG"

    async def test_missing_artifact_dir_rejected(self, real_agent_config, agent_yml_with_singleton):
        """ask_report fails when reports/<slug>/ does not exist on disk."""
        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(
                name="ask_ghost",
                type="ask_report",
                artifact_slug="not_a_real_report",
            ),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "ARTIFACT_NOT_FOUND"

    async def test_create_ask_report_success_persists_slug(self, real_agent_config, agent_yml_with_singleton):
        """Happy path: artifact exists → agent persists with artifact_slug."""
        import yaml

        _mkdir_artifact(real_agent_config.project_root, "report", "demo_report")

        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(
                name="ask_demo_report",
                type="ask_report",
                artifact_slug="demo_report",
                description="Follow-up consultant for demo_report",
            ),
            real_agent_config,
        )
        assert result.success is True
        assert result.data["name"] == "ask_demo_report"

        with open(agent_yml_with_singleton) as f:
            raw = yaml.safe_load(f)
        entry = raw["agent"]["agentic_nodes"]["ask_demo_report"]
        # artifact_slug lives at the top level so the node reads it via
        # self.node_config without any wrapper extraction.
        assert entry["artifact_slug"] == "demo_report"
        assert entry["type"] == "ask_report"

    async def test_create_ask_dashboard_success(self, real_agent_config, agent_yml_with_singleton):
        """Mirror happy path for ask_dashboard — different root directory."""
        _mkdir_artifact(real_agent_config.project_root, "dashboard", "demo_dash")

        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(
                name="ask_demo_dash",
                type="ask_dashboard",
                artifact_slug="demo_dash",
            ),
            real_agent_config,
        )
        assert result.success is True

    async def test_same_artifact_double_bind_rejected(self, real_agent_config, agent_yml_with_singleton):
        """A given (type, slug) pair accepts exactly one ask_* agent."""
        _mkdir_artifact(real_agent_config.project_root, "report", "twins")

        svc = AgentService()
        first = await svc.create_agent(
            CreateAgentInput(
                name="ask_twins_a",
                type="ask_report",
                artifact_slug="twins",
            ),
            real_agent_config,
        )
        assert first.success is True

        second = await svc.create_agent(
            CreateAgentInput(
                name="ask_twins_b",
                type="ask_report",
                artifact_slug="twins",
            ),
            real_agent_config,
        )
        assert second.success is False
        assert second.errorCode == "ARTIFACT_ALREADY_BOUND"

    async def test_different_type_same_slug_allowed(self, real_agent_config, agent_yml_with_singleton):
        """Uniqueness is per (type, slug), not slug alone — a report and a
        dashboard happen to share a slug? Both bindings are valid."""
        _mkdir_artifact(real_agent_config.project_root, "report", "shared_slug")
        _mkdir_artifact(real_agent_config.project_root, "dashboard", "shared_slug")

        svc = AgentService()
        r1 = await svc.create_agent(
            CreateAgentInput(
                name="ask_shared_report",
                type="ask_report",
                artifact_slug="shared_slug",
            ),
            real_agent_config,
        )
        r2 = await svc.create_agent(
            CreateAgentInput(
                name="ask_shared_dash",
                type="ask_dashboard",
                artifact_slug="shared_slug",
            ),
            real_agent_config,
        )
        assert r1.success is True and r2.success is True
