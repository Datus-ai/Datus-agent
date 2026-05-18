"""Unit tests for ``AskReportAgenticNode`` / ``AskDashboardAgenticNode``.

Pins the node-level invariants we depend on at runtime:

* ``BaseArtifactAskAgenticNode._resolve_artifact_root`` fails loud (rather
  than degrading silently) when ``artifact_slug`` is missing / malformed /
  unresolvable on disk / outside ``project_root``.
* On a healthy binding, the resolver records both the slug and an absolute
  artifact path under ``project_root/<kind>s/<slug>/``.
* The filesystem tool override anchors at the artifact root, so the LLM's
  ``read_file`` / ``glob`` / ``grep`` reach is constrained to one artifact.
* The artifact-context preamble rendered into the system prompt includes
  the manifest name, the intent.md body, the expected directory tree
  (with kind-specific branches: ``insights.json`` only for reports),
  and the seven load-bearing behavioral rules. ``interpretation.json``
  and ``suggested_questions.json`` are intentionally NOT loaded into
  the preamble (the first was removed; the second is reserved for UI
  chips to avoid anchoring the LLM on a fixed question set).

We instantiate the nodes directly (bypassing ``node_factory``) so the test
focuses on the binding / context-injection layer without dragging in the
chat-level setup overhead. The chat conversational loop itself is already
covered by ``test_chat_agentic_node.py`` and unaffected by ask_*.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datus.agent.node.ask_dashboard_agentic_node import AskDashboardAgenticNode
from datus.agent.node.ask_report_agentic_node import AskReportAgenticNode
from datus.utils.exceptions import DatusException

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _seed_artifact(project_root: str, kind: str, slug: str, *, with_analysis: bool = True) -> Path:
    """Materialize a minimal ``reports/<slug>/`` (or dashboard) on disk.

    Includes a manifest with ``name`` / ``description`` / ``datasources``
    plus, when ``with_analysis=True``, ``analysis/intent.md`` — the
    single anchor file the node preloads. Other analysis files
    (insights, suggested_questions, subject_refs) are intentionally
    omitted: the node fetches insights on demand via ``read_file``,
    suggested_questions belong to the UI chip layer (not the LLM
    context), and subject_refs is present-iff-non-empty.
    """
    kind_dir = "reports" if kind == "report" else "dashboards"
    root = Path(project_root) / kind_dir / slug
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    (root / "queries").mkdir(parents=True, exist_ok=True)
    (root / "render").mkdir(parents=True, exist_ok=True)

    manifest = {
        "slug": slug,
        "name": f"Demo {kind.title()}",
        "description": "Smoke-test artifact used by ask_* node unit tests.",
        "kind": kind,
        "created_at": "2026-05-17T00:00:00Z",
        "datasources": ["test_ds"],
        "key_tables": ["Account", "Person"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    if with_analysis:
        (root / "analysis" / "intent.md").write_text(
            "### [2026-05-17T00:00:00Z] mode: new\n> investigate Q3 anomalies\n",
            encoding="utf-8",
        )
    return root


def _register_ask_agent(agent_config, *, name: str, kind: str, slug: str) -> None:
    """Insert an ask_* agentic_nodes entry so node_config lookup succeeds."""
    agent_type = "ask_report" if kind == "report" else "ask_dashboard"
    if not hasattr(agent_config, "agentic_nodes") or agent_config.agentic_nodes is None:
        agent_config.agentic_nodes = {}
    agent_config.agentic_nodes[name] = {
        "type": agent_type,
        "artifact_slug": slug,
        "agent_description": f"Ask consultant for {slug}",
        "tools": "db_tools.*,filesystem_tools.read_file",
        "rules": [],
        "max_turns": 5,
    }


def _make_ask_report_node(agent_config, *, name: str = "ask_demo_report", slug: str = "demo_report"):
    """Build an AskReportAgenticNode pre-bound to a freshly-seeded artifact."""
    _seed_artifact(agent_config.project_root, "report", slug)
    _register_ask_agent(agent_config, name=name, kind="report", slug=slug)
    return AskReportAgenticNode(
        node_id=f"{name}_test",
        description="test ask_report node",
        node_type="chat",
        agent_config=agent_config,
        node_name=name,
    )


def _make_ask_dashboard_node(agent_config, *, name: str = "ask_demo_dash", slug: str = "demo_dash"):
    _seed_artifact(agent_config.project_root, "dashboard", slug)
    _register_ask_agent(agent_config, name=name, kind="dashboard", slug=slug)
    return AskDashboardAgenticNode(
        node_id=f"{name}_test",
        description="test ask_dashboard node",
        node_type="chat",
        agent_config=agent_config,
        node_name=name,
    )


# --------------------------------------------------------------------------- #
# Artifact binding resolution                                                 #
# --------------------------------------------------------------------------- #


class TestArtifactBinding:
    def test_missing_artifact_slug_raises(self, real_agent_config):
        """Node config without artifact_slug → DatusException at init."""
        _register_ask_agent(real_agent_config, name="ask_no_slug", kind="report", slug="anything")
        # Erase the slug from the agentic_nodes entry to simulate a bad config.
        real_agent_config.agentic_nodes["ask_no_slug"].pop("artifact_slug")

        with pytest.raises(DatusException):
            AskReportAgenticNode(
                node_id="x",
                description="d",
                node_type="chat",
                agent_config=real_agent_config,
                node_name="ask_no_slug",
            )

    def test_malformed_slug_raises(self, real_agent_config):
        _register_ask_agent(real_agent_config, name="ask_bad", kind="report", slug="Bad-Slug")
        with pytest.raises(DatusException):
            AskReportAgenticNode(
                node_id="x",
                description="d",
                node_type="chat",
                agent_config=real_agent_config,
                node_name="ask_bad",
            )

    def test_missing_artifact_dir_raises(self, real_agent_config):
        # Register WITHOUT seeding the directory on disk.
        _register_ask_agent(real_agent_config, name="ask_ghost", kind="report", slug="ghost_report")
        with pytest.raises(DatusException):
            AskReportAgenticNode(
                node_id="x",
                description="d",
                node_type="chat",
                agent_config=real_agent_config,
                node_name="ask_ghost",
            )

    def test_healthy_binding_records_path(self, real_agent_config):
        node = _make_ask_report_node(real_agent_config)
        assert node._artifact_slug == "demo_report"
        assert node._artifact_root is not None
        assert node._artifact_root.is_dir()
        # Path is under project_root — verifies traversal defence didn't trip.
        assert str(node._artifact_root).startswith(str(Path(real_agent_config.project_root).resolve()))

    def test_dashboard_binding_uses_dashboards_root(self, real_agent_config):
        node = _make_ask_dashboard_node(real_agent_config)
        assert node._artifact_root.name == "demo_dash"
        assert node._artifact_root.parent.name == "dashboards"

    def test_symlink_redirect_within_project_root_rejected(self, real_agent_config):
        """Defence-in-depth: a symlink at ``reports/<slug>`` redirecting to
        a sibling directory inside ``project_root`` still lives "inside"
        the project, so the old startswith check passed it through. The
        tightened check compares the resolved path against the unresolved
        expected location, catching any redirection — including inside the
        project root."""
        project_root = Path(real_agent_config.project_root)
        # A real sibling directory the symlink will point to.
        other_dir = project_root / "reports" / "actual_target"
        other_dir.mkdir(parents=True, exist_ok=True)
        # The slug we register points to a symlink redirected to that sibling.
        slug = "redirect_slug"
        symlink_path = project_root / "reports" / slug
        symlink_path.parent.mkdir(parents=True, exist_ok=True)
        symlink_path.symlink_to(other_dir, target_is_directory=True)
        _register_ask_agent(real_agent_config, name="ask_redirect", kind="report", slug=slug)

        with pytest.raises(DatusException):
            AskReportAgenticNode(
                node_id="x",
                description="d",
                node_type="chat",
                agent_config=real_agent_config,
                node_name="ask_redirect",
            )


# --------------------------------------------------------------------------- #
# Filesystem tool anchoring                                                   #
# --------------------------------------------------------------------------- #


class TestFilesystemAnchoring:
    def test_filesystem_tool_root_equals_artifact_root(self, real_agent_config):
        node = _make_ask_report_node(real_agent_config)
        # ``filesystem_func_tool.root_path`` is what gates read_file / glob.
        assert Path(node.filesystem_func_tool.root_path).resolve() == node._artifact_root.resolve()


# --------------------------------------------------------------------------- #
# Anchor files preload                                                        #
# --------------------------------------------------------------------------- #


class TestAnchorFilePreload:
    def test_intent_loaded(self, real_agent_config):
        node = _make_ask_report_node(real_agent_config)
        assert "Q3 anomalies" in node._artifact_intent_md

    def test_interpretation_not_attribute(self, real_agent_config):
        """``_artifact_interpretation`` was removed along with the
        interpretation.json file; the attribute should no longer exist
        on the node so accidental readers fail loud."""
        node = _make_ask_report_node(real_agent_config)
        assert not hasattr(node, "_artifact_interpretation")

    def test_missing_intent_degrades_silently(self, real_agent_config):
        """When intent.md is absent, init still succeeds and the cached
        value stays empty (prompt template branches on emptiness)."""
        _seed_artifact(real_agent_config.project_root, "report", "no_anchors", with_analysis=False)
        _register_ask_agent(real_agent_config, name="ask_no_anchor", kind="report", slug="no_anchors")
        node = AskReportAgenticNode(
            node_id="x",
            description="d",
            node_type="chat",
            agent_config=real_agent_config,
            node_name="ask_no_anchor",
        )
        assert node._artifact_intent_md == ""
        # Manifest still loaded — it's not in analysis/.
        assert node._artifact_manifest["slug"] == "no_anchors"


# --------------------------------------------------------------------------- #
# Prompt rendering                                                            #
# --------------------------------------------------------------------------- #


class TestArtifactContextBlock:
    def test_report_block_includes_insights_in_tree(self, real_agent_config):
        node = _make_ask_report_node(real_agent_config)
        block = node._render_artifact_context_block()
        assert "Demo Report" in block  # manifest name
        assert "demo_report" in block  # slug
        assert "Q3 anomalies" in block  # intent.md
        # Directory tree branches on artifact_kind — report shows insights.
        assert "insights.json" in block
        # Brief sidecar replaced reasoning sidecar in the tree.
        assert "brief.json" in block
        assert "reasoning.json" not in block
        # Behavioral rules are present and number 7.
        assert "Ground in existing analysis first" in block
        assert "No artifact mutations" in block

    def test_report_block_includes_key_tables(self, real_agent_config):
        """``manifest.key_tables`` (code-aggregated by finalize) must be
        surfaced in the preamble so the LLM skips ``list_tables`` /
        ``describe_table`` round-trips when answering schema-shape
        questions or planning a new SQL on related tables."""
        node = _make_ask_report_node(real_agent_config)
        block = node._render_artifact_context_block()
        assert "Tables referenced" in block
        assert "Account" in block
        assert "Person" in block

    def test_report_block_excludes_interpretation_and_suggested(self, real_agent_config):
        """interpretation.json was removed; suggested_questions.json is
        UI-chip data and must not leak into the system prompt where it
        would anchor the LLM toward a fixed question set."""
        node = _make_ask_report_node(real_agent_config)
        block = node._render_artifact_context_block()
        assert "interpretation.json" not in block
        assert "suggested_questions.json" not in block

    def test_dashboard_block_excludes_insights(self, real_agent_config):
        node = _make_ask_dashboard_node(real_agent_config)
        block = node._render_artifact_context_block()
        # Dashboard tree omits insights.json because dashboards have no
        # static conclusions to surface.
        assert "insights.json" not in block
        # Dashboard-specific rule about runtime data is present.
        assert "no precomputed data" in block
        # Template suffix shows .sql.j2 not .sql.
        assert ".sql.j2" in block

    def test_block_directs_user_to_gen_visual_for_modifications(self, real_agent_config):
        """Rule 2 — read-only consultant points modifications at the gen_visual_* agent."""
        node = _make_ask_report_node(real_agent_config)
        block = node._render_artifact_context_block()
        assert "gen_visual_report" in block
