"""Unit tests for ``AskReportAgenticNode`` / ``AskDashboardAgenticNode``.

Pins the node-level invariants we depend on at runtime:

* ``BaseArtifactAskAgenticNode._resolve_artifact_binding_early`` resolves
  the artifact from either an in-memory ``artifact_blob`` injected into the
  agentic_nodes entry (backend / SaaS path) or, for kinds with
  ``BLOB_REQUIRED = False``, the on-disk ``<kind>/<slug>/`` directory.
  Failures (missing slug, malformed slug, unresolvable disk path,
  symlink redirection, blob required but absent) raise ``DatusException``
  at init.
* The filesystem tool is anchored correctly per source:
  - blob source ⇒ :class:`MemoryFilesystemFuncTool` (no disk),
  - disk source ⇒ :class:`FilesystemFuncTool` rooted at the artifact dir.
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
from datus.tools.func_tool.memory_filesystem_tools import MemoryFilesystemFuncTool
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


def _register_ask_agent(
    agent_config,
    *,
    name: str,
    kind: str,
    slug: str,
    blob: dict | None = None,
) -> None:
    """Insert an ask_* agentic_nodes entry so node_config lookup succeeds.

    When ``blob`` is provided, it's stored under ``artifact_blob`` to mirror
    what ``datus_backend.config_loader._build_agentic_nodes_dict`` injects
    after looking up the latest ``VisualReportVersion`` for the slug.
    """
    agent_type = "ask_report" if kind == "report" else "ask_dashboard"
    if not hasattr(agent_config, "agentic_nodes") or agent_config.agentic_nodes is None:
        agent_config.agentic_nodes = {}
    entry = {
        "type": agent_type,
        "artifact_slug": slug,
        "agent_description": f"Ask consultant for {slug}",
        "tools": "db_tools.*,filesystem_tools.read_file",
        "rules": [],
        "max_turns": 5,
    }
    if blob is not None:
        entry["artifact_blob"] = blob
    agent_config.agentic_nodes[name] = entry


def _blob_from_disk(project_root: str, kind: str, slug: str) -> dict:
    """Build a ``{manifest, files}`` blob from a previously-seeded disk tree.

    Mirrors the production wire shape produced by
    ``datus_backend.services.report_service.publish``:

    * ``manifest`` carries the parsed ``manifest.json`` contents (structured
      dict, not a string).
    * ``files`` is a flat list of ``{path, content}`` entries under
      ``render/`` / ``queries/`` / ``analysis/`` **only** —
      ``manifest.json`` is intentionally NOT duplicated here.

    AskNode bridges this asymmetry by synthesizing ``manifest.json`` back
    into the in-memory file map at init time so ``read_file("manifest.json")``
    keeps working from the LLM's perspective.
    """
    kind_dir = "reports" if kind == "report" else "dashboards"
    root = Path(project_root) / kind_dir / slug
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    files: list[dict] = []
    # Production's ``_iter_artifact_files`` walks only the three known
    # subdirs and drops files outside them. Match that here so blob-mode
    # tests are exercising the same shape AskReport sees in SaaS.
    for sub in ("render", "queries", "analysis"):
        sub_root = root / sub
        if not sub_root.is_dir():
            continue
        for f in sorted(sub_root.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(root).as_posix()
            files.append({"path": rel, "content": f.read_text(encoding="utf-8")})
    return {"manifest": manifest, "files": files}


def _make_ask_report_node(agent_config, *, name: str = "ask_demo_report", slug: str = "demo_report"):
    """Build an AskReportAgenticNode against a published-blob fixture.

    Mirrors production: backend's ``config_loader`` snapshots the latest
    published version into ``artifact_blob`` and AskReport runs against
    that. We seed the disk tree only as a convenient way to construct the
    blob via :func:`_blob_from_disk` — the node never touches it.
    """
    _seed_artifact(agent_config.project_root, "report", slug)
    blob = _blob_from_disk(agent_config.project_root, "report", slug)
    _register_ask_agent(agent_config, name=name, kind="report", slug=slug, blob=blob)
    return AskReportAgenticNode(
        node_id=f"{name}_test",
        description="test ask_report node",
        node_type="chat",
        agent_config=agent_config,
        node_name=name,
    )


def _make_ask_dashboard_node(agent_config, *, name: str = "ask_demo_dash", slug: str = "demo_dash"):
    """Build an AskDashboardAgenticNode against the on-disk fallback path.

    Dashboards have ``BLOB_REQUIRED = False`` until the publish flow lands,
    so they exercise the legacy on-disk binding.
    """
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
    """Binding resolution invariants common to both kinds."""

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

    @pytest.mark.parametrize(
        "degenerate_blob",
        [
            pytest.param({}, id="empty_dict"),
            pytest.param({"files": [{"path": "a", "content": "x"}]}, id="manifest_missing"),
            pytest.param({"manifest": {}, "files": [{"path": "a", "content": "x"}]}, id="manifest_empty"),
            pytest.param({"manifest": {"slug": "x"}, "files": []}, id="files_empty"),
            pytest.param({"manifest": {"slug": "x"}, "files": "not a list"}, id="files_wrong_type"),
            pytest.param({"manifest": "string", "files": []}, id="manifest_wrong_type"),
        ],
    )
    def test_report_degenerate_blob_fails_loud(self, real_agent_config, degenerate_blob):
        """Degenerate blob shapes must NOT silently bind to an empty
        filesystem — they trip the same BLOB_REQUIRED branch as a
        missing blob so the publish half-bound state is visible at init.
        """
        _seed_artifact(real_agent_config.project_root, "report", "degenerate")
        _register_ask_agent(
            real_agent_config,
            name="ask_degenerate",
            kind="report",
            slug="degenerate",
            blob=degenerate_blob,
        )
        with pytest.raises(DatusException):
            AskReportAgenticNode(
                node_id="x",
                description="d",
                node_type="chat",
                agent_config=real_agent_config,
                node_name="ask_degenerate",
            )

    def test_dashboard_degenerate_blob_falls_back_to_disk(self, real_agent_config):
        """For BLOB_REQUIRED=False kinds, a degenerate blob behaves the
        same as a missing blob: fall back to the on-disk artifact root.
        Guards against the empty-blob path silently winning over a
        perfectly valid disk tree."""
        _seed_artifact(real_agent_config.project_root, "dashboard", "deg_dash")
        _register_ask_agent(
            real_agent_config,
            name="ask_deg_dash",
            kind="dashboard",
            slug="deg_dash",
            blob={"manifest": {"slug": "deg_dash"}, "files": []},
        )
        node = AskDashboardAgenticNode(
            node_id="x",
            description="d",
            node_type="chat",
            agent_config=real_agent_config,
            node_name="ask_deg_dash",
        )
        # Disk fallback engaged: in-memory file map untouched, disk root
        # populated and pointing at the seeded dashboard tree.
        assert node._artifact_files is None
        assert node._artifact_root.name == "deg_dash"

    def test_report_without_blob_raises_fail_loud(self, real_agent_config):
        """``ask_report`` declares ``BLOB_REQUIRED = True``. Half-bound
        state (subagent exists, no published version → config_loader didn't
        attach ``artifact_blob``) must fail at init rather than silently
        falling back to a disk path that the backend may not even have
        access to."""
        # Disk dir exists, but no blob — simulates "subagent created but
        # report never finished publishing".
        _seed_artifact(real_agent_config.project_root, "report", "no_blob")
        _register_ask_agent(real_agent_config, name="ask_no_blob", kind="report", slug="no_blob")
        with pytest.raises(DatusException):
            AskReportAgenticNode(
                node_id="x",
                description="d",
                node_type="chat",
                agent_config=real_agent_config,
                node_name="ask_no_blob",
            )

    def test_report_with_blob_loads_from_memory(self, real_agent_config):
        """Healthy report binding: blob loaded, no disk root set."""
        node = _make_ask_report_node(real_agent_config)
        assert node._artifact_slug == "demo_report"
        # Blob mode: in-memory file map populated with the expected files
        # (anchors plus the seeded manifest), disk root not touched. We
        # assert on the file keys directly rather than a bare ``is not None``
        # so a future bug where the map is built but empty also fails.
        assert set(node._artifact_files.keys()) >= {"manifest.json", "analysis/intent.md"}
        assert node._artifact_root is None
        # Manifest came through the structured blob path, not a JSON re-decode.
        assert node._artifact_manifest["slug"] == "demo_report"

    def test_blob_synthesizes_manifest_json_from_structured_form(self, real_agent_config):
        """Production blob carries ``manifest`` structured and omits
        ``manifest.json`` from ``files[]`` (no on-wire duplication). But
        the LLM-facing tool surface advertises ``manifest.json`` as a
        readable file — the prompt preamble even prints it in the
        directory tree — so blob mode must synthesize it back from
        ``manifest`` to keep ``read_file("manifest.json")`` working.

        Regression test for the bug where in-memory ask sessions saw
        "File not found" on ``manifest.json`` while disk sessions could
        read it normally.
        """
        # Blob shape mirrors production: manifest as a dict, files[]
        # WITHOUT manifest.json.
        manifest_dict = {
            "slug": "no_root_file",
            "name": "Synth Test",
            "description": "d",
            "kind": "report",
            "created_at": "2026-05-17T00:00:00Z",
        }
        blob = {
            "manifest": manifest_dict,
            "files": [{"path": "analysis/intent.md", "content": "## intent\n"}],
        }
        _register_ask_agent(real_agent_config, name="ask_synth", kind="report", slug="no_root_file", blob=blob)
        node = AskReportAgenticNode(
            node_id="x",
            description="d",
            node_type="chat",
            agent_config=real_agent_config,
            node_name="ask_synth",
        )
        # MemoryFilesystemFuncTool serves manifest.json as JSON of the structured form,
        # round-trippable back to the original dict so the LLM sees the
        # same field structure regardless of source path.
        assert "manifest.json" in node._artifact_files
        round_trip = json.loads(node._artifact_files["manifest.json"])
        assert round_trip == manifest_dict
        # Also verify the LLM-facing surface: read_file("manifest.json")
        # round-trips the same way (catches a regression where the file
        # is in the dict but the tool path filters it out).
        res = node.filesystem_func_tool.read_file("manifest.json")
        assert res.success == 1
        assert json.loads(res.result) == manifest_dict

    def test_blob_does_not_overwrite_explicit_manifest_json_entry(self, real_agent_config):
        """If a future backend explicitly includes ``manifest.json`` in
        ``files[]`` (e.g. wire-format drift), don't shadow it with a
        re-serialized copy — the on-wire content wins. Guards against
        a subtle drift where the LLM would see formatting differences
        between the structured manifest and the file body."""
        explicit_body = '{"hand": "crafted", "slug": "explicit"}'
        blob = {
            "manifest": {"slug": "explicit", "name": "Explicit"},
            "files": [
                {"path": "manifest.json", "content": explicit_body},
                {"path": "analysis/intent.md", "content": "x"},
            ],
        }
        _register_ask_agent(real_agent_config, name="ask_explicit", kind="report", slug="explicit", blob=blob)
        node = AskReportAgenticNode(
            node_id="x",
            description="d",
            node_type="chat",
            agent_config=real_agent_config,
            node_name="ask_explicit",
        )
        assert node._artifact_files["manifest.json"] == explicit_body

    def test_blob_malformed_entries_skipped(self, real_agent_config):
        """The blob wire-format is owned by the backend. Garbage entries
        (non-dict, missing path/content, non-string content) are skipped
        silently so unrelated drift doesn't break the conversation —
        missing files still surface as ``read_file: File not found``."""
        bad_blob = {
            "manifest": {"slug": "noisy", "name": "Noisy"},
            "files": [
                {"path": "ok.md", "content": "real file"},
                "not a dict",  # ignored
                {"path": "", "content": "empty path"},  # ignored
                {"path": "no_content.md"},  # ignored
                {"path": "binary.bin", "content": 42},  # ignored
            ],
        }
        _register_ask_agent(real_agent_config, name="ask_noisy", kind="report", slug="noisy", blob=bad_blob)
        node = AskReportAgenticNode(
            node_id="x",
            description="d",
            node_type="chat",
            agent_config=real_agent_config,
            node_name="ask_noisy",
        )
        # ``manifest.json`` is synthesized from the structured manifest
        # (covered by its own test); here we just verify that everything
        # else in the malformed ``files[]`` is dropped — i.e. only the
        # one valid entry survives alongside the synthesized manifest.
        assert set(node._artifact_files.keys()) == {"ok.md", "manifest.json"}
        assert node._artifact_files["ok.md"] == "real file"

    # --- Disk-fallback path lives on dashboard until publish lands ---

    def test_dashboard_binding_uses_dashboards_root(self, real_agent_config):
        """``ask_dashboard`` has ``BLOB_REQUIRED = False`` so it still
        resolves from disk under ``dashboards/<slug>/``."""
        node = _make_ask_dashboard_node(real_agent_config)
        # Concrete path-shape assertions (name + parent) — also implicitly
        # confirms ``_artifact_root`` is a populated Path rather than None.
        assert node._artifact_root.name == "demo_dash"
        assert node._artifact_root.parent.name == "dashboards"
        # Disk path → no in-memory file map.
        assert node._artifact_files is None

    def test_dashboard_missing_disk_dir_raises(self, real_agent_config):
        """Disk path still fails loud when the directory is missing."""
        _register_ask_agent(real_agent_config, name="ask_ghost_dash", kind="dashboard", slug="ghost_dash")
        with pytest.raises(DatusException):
            AskDashboardAgenticNode(
                node_id="x",
                description="d",
                node_type="chat",
                agent_config=real_agent_config,
                node_name="ask_ghost_dash",
            )

    def test_dashboard_symlink_redirect_within_project_root_rejected(self, real_agent_config):
        """Defence-in-depth on the disk path: a symlink redirecting the
        artifact dir to a sibling directory inside ``project_root`` is
        rejected by comparing the resolved path against the unresolved
        expected location. Migrated to dashboard since the disk binding
        is now dashboard-only."""
        project_root = Path(real_agent_config.project_root)
        other_dir = project_root / "dashboards" / "actual_target"
        other_dir.mkdir(parents=True, exist_ok=True)
        slug = "redirect_slug"
        symlink_path = project_root / "dashboards" / slug
        symlink_path.parent.mkdir(parents=True, exist_ok=True)
        symlink_path.symlink_to(other_dir, target_is_directory=True)
        _register_ask_agent(real_agent_config, name="ask_redirect_dash", kind="dashboard", slug=slug)

        with pytest.raises(DatusException):
            AskDashboardAgenticNode(
                node_id="x",
                description="d",
                node_type="chat",
                agent_config=real_agent_config,
                node_name="ask_redirect_dash",
            )

    def test_dashboard_with_blob_uses_memory_path(self, real_agent_config):
        """Dashboard isn't required to carry a blob today, but if one is
        injected the node must still prefer it over disk so the future
        publish flow can drop in without touching this class. (Catches
        regressions where someone hardcodes ``ARTIFACT_KIND == "report"``
        as the gate.)"""
        _seed_artifact(real_agent_config.project_root, "dashboard", "demo_dash2")
        blob = _blob_from_disk(real_agent_config.project_root, "dashboard", "demo_dash2")
        _register_ask_agent(real_agent_config, name="ask_dash_blob", kind="dashboard", slug="demo_dash2", blob=blob)
        node = AskDashboardAgenticNode(
            node_id="x",
            description="d",
            node_type="chat",
            agent_config=real_agent_config,
            node_name="ask_dash_blob",
        )
        # Concrete content check — the manifest from the blob must be the
        # one we built from disk, proving the blob path won over the disk
        # path rather than both silently activating.
        assert node._artifact_manifest.get("slug") == "demo_dash2"
        assert "manifest.json" in node._artifact_files
        assert node._artifact_root is None


# --------------------------------------------------------------------------- #
# Filesystem tool anchoring                                                   #
# --------------------------------------------------------------------------- #


class TestFilesystemAnchoring:
    def test_report_filesystem_tool_is_memory_fs(self, real_agent_config):
        """Report runs against MemoryFilesystemFuncTool so the LLM can never reach the
        underlying disk — even if a stale report directory happens to
        live next to the running backend."""
        node = _make_ask_report_node(real_agent_config)
        assert isinstance(node.filesystem_func_tool, MemoryFilesystemFuncTool)
        # ``root_path`` is the human-readable label (read by a debug log
        # in ChatAgenticNode), not a real filesystem path.
        assert node.filesystem_func_tool.root_path == "in-memory:demo_report"

    def test_report_memory_fs_serves_seeded_files(self, real_agent_config):
        """Cross-component contract: files put into the blob round-trip
        through the LLM-facing ``read_file`` surface."""
        node = _make_ask_report_node(real_agent_config)
        res = node.filesystem_func_tool.read_file("manifest.json")
        assert res.success == 1
        assert "Demo Report" in res.result

    def test_blob_branch_forwards_kwargs(self, real_agent_config):
        """``_make_filesystem_tool`` must forward caller-supplied kwargs
        in blob mode the same way it does in disk mode — otherwise any
        future per-tool wiring routed through the helper would be
        silently dropped only on blob-bound agents. We exercise this by
        invoking the helper directly with a sentinel kwarg; the kwarg
        lands in ``tool_params`` via ``BaseTool.__init__``.
        """
        node = _make_ask_report_node(real_agent_config)
        sentinel = object()
        tool = node._make_filesystem_tool(_test_marker=sentinel)
        assert isinstance(tool, MemoryFilesystemFuncTool)
        # BaseTool absorbs unknown kwargs into ``tool_params`` — verifying
        # the round-trip proves forwarding works without coupling to any
        # specific kwarg the caller might add in the future.
        assert tool.tool_params.get("_test_marker") is sentinel

    def test_dashboard_filesystem_tool_anchored_at_disk_root(self, real_agent_config):
        """Dashboard keeps the legacy disk-rooted tool until its publish
        flow lands. ``filesystem_func_tool.root_path`` is what gates
        ``read_file`` / ``glob`` reach there."""
        node = _make_ask_dashboard_node(real_agent_config)
        assert not isinstance(node.filesystem_func_tool, MemoryFilesystemFuncTool)
        assert Path(node.filesystem_func_tool.root_path).resolve() == node._artifact_root.resolve()


# --------------------------------------------------------------------------- #
# Anchor files preload                                                        #
# --------------------------------------------------------------------------- #


class TestAnchorFilePreload:
    def test_intent_loaded_from_blob(self, real_agent_config):
        """In blob mode the intent comes from the in-memory file map, not
        a disk read."""
        node = _make_ask_report_node(real_agent_config)
        assert "Q3 anomalies" in node._artifact_intent_md

    def test_intent_loaded_from_disk_for_dashboard(self, real_agent_config):
        """Disk path still preloads ``analysis/intent.md`` for the kinds
        that haven't moved to blob mode yet."""
        node = _make_ask_dashboard_node(real_agent_config)
        assert "Q3 anomalies" in node._artifact_intent_md

    def test_interpretation_not_attribute(self, real_agent_config):
        """``_artifact_interpretation`` was removed along with the
        interpretation.json file; the attribute should no longer exist
        on the node so accidental readers fail loud."""
        node = _make_ask_report_node(real_agent_config)
        assert not hasattr(node, "_artifact_interpretation")

    def test_missing_intent_degrades_silently_blob_mode(self, real_agent_config):
        """When intent.md is absent from the blob, init still succeeds
        and the cached value stays empty (prompt template branches on
        emptiness). The manifest still comes through via the structured
        blob path.

        We add a render file so the blob has at least one entry in
        ``files[]`` — an empty ``files`` list trips the degenerate-blob
        validator (covered separately in
        ``test_report_degenerate_blob_fails_loud``)."""
        _seed_artifact(real_agent_config.project_root, "report", "no_anchors", with_analysis=False)
        (Path(real_agent_config.project_root) / "reports" / "no_anchors" / "render" / "app.jsx").write_text(
            "export default function App(){return null}", encoding="utf-8"
        )
        blob = _blob_from_disk(real_agent_config.project_root, "report", "no_anchors")
        _register_ask_agent(real_agent_config, name="ask_no_anchor", kind="report", slug="no_anchors", blob=blob)
        node = AskReportAgenticNode(
            node_id="x",
            description="d",
            node_type="chat",
            agent_config=real_agent_config,
            node_name="ask_no_anchor",
        )
        assert node._artifact_intent_md == ""
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

    def test_report_block_advertises_in_memory_source(self, real_agent_config):
        """Blob mode advertises an in-memory source line instead of a disk
        ``Root:`` line so the LLM (and a human reading the prompt) knows
        the artifact came from a frozen published snapshot."""
        node = _make_ask_report_node(real_agent_config)
        block = node._render_artifact_context_block()
        assert "in-memory snapshot" in block
        # Disk root must NOT leak into the prompt — would mislead the LLM
        # and (in SaaS) expose an irrelevant backend path. Match the
        # rendered markdown form rather than a generic "Root:" so we
        # catch the actual production wording.
        assert "**Root**" not in block

    def test_dashboard_block_advertises_disk_root(self, real_agent_config):
        """Disk mode still surfaces the artifact root path so CLI users
        can correlate prompt context with what's under ``dashboards/``."""
        node = _make_ask_dashboard_node(real_agent_config)
        block = node._render_artifact_context_block()
        assert "**Root**" in block
        assert "in-memory snapshot" not in block
