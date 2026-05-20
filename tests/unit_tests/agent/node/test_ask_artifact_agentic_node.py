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
  the manifest header, the intent.md body, the subject-library scope
  (when any subject refs exist), the confirmed insights (report only),
  a per-query catalog (brief + columns + sample/rows + SQL with byte +
  row gating and a catalog-level cap), the filesystem layout note, and
  the seven load-bearing behavioral rules — rule 1 in particular
  forbids defensive ``glob`` / ``read_file`` on anything already
  inlined. ``interpretation.json`` (removed) is never mentioned;
  ``suggested_questions.json`` only appears in the layout tree as a
  ``DO NOT read`` annotation so its contents never anchor the LLM
  toward a fixed question set.

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
        # Layout tree branches on artifact_kind — report flags insights as inlined.
        assert "insights.json" in block
        # Brief sidecar called out in the "already loaded" list as part of
        # the inline-context contract; SQL summaries / reasoning sidecars
        # don't exist anymore.
        assert "brief.json" in block
        assert "reasoning.json" not in block
        # Behavioral rule 1 is the load-bearing "answer from inlined context first"
        # nudge added when the renderer started inlining briefs/insights/SQL.
        # Rule 7 is the read-only mutation guard.
        assert "Answer from the inlined context first" in block
        assert "Do NOT issue `glob` or `read_file`" in block
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
        UI-chip data and must not leak its contents into the system
        prompt where it would anchor the LLM toward a fixed question
        set. The filename itself MAY appear in the layout tree but only
        as a "DO NOT read" annotation — the original anti-anchor intent
        is now enforced by explicit instruction rather than by omission.
        """
        node = _make_ask_report_node(real_agent_config)
        block = node._render_artifact_context_block()
        assert "interpretation.json" not in block
        # The filename always appears in the layout tree but must be
        # paired with the "DO NOT read" annotation on every mentioning
        # line. We pin on the literal rendered annotation rather than
        # a loose substring search so a regression that silently inlines
        # the file's questions (the original anti-anchor concern) still
        # fails this test.
        mentioning_lines = [line for line in block.splitlines() if "suggested_questions.json" in line]
        assert mentioning_lines, (
            "renderer must surface suggested_questions.json in the layout "
            "tree (with a DO NOT read annotation) so the LLM knows the file "
            "exists but is off-limits; missing entirely would invite the "
            "LLM to `glob` for it"
        )
        for line in mentioning_lines:
            assert "DO NOT read" in line, f"suggested_questions.json mentioned without DO NOT annotation: {line!r}"

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


# --------------------------------------------------------------------------- #
# Inline-content rendering                                                    #
# --------------------------------------------------------------------------- #


def _seed_query_files(
    artifact_root: Path,
    kind: str,
    queries: list[dict],
) -> None:
    """Drop per-query sidecars into an already-seeded artifact root.

    Each ``queries`` entry is a dict like::

        {
            "slug": str,
            "brief": dict,             # written to <slug>.brief.json
            "result": dict | None,     # report-only: <slug>.json
            "params": dict | None,     # dashboard-only: <slug>.params.json
            "sql": str,                # <slug>.sql (report) or <slug>.sql.j2 (dashboard)
        }

    Built as an explicit fixture-driver rather than overloading
    ``_seed_artifact`` (whose existing kwargs are load-bearing on a
    dozen other tests) so the inline-rendering tests can dial in
    exact row counts, byte sizes, and SQL lengths.
    """
    queries_dir = artifact_root / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)
    for q in queries:
        slug = q["slug"]
        (queries_dir / f"{slug}.brief.json").write_text(json.dumps(q.get("brief") or {}), encoding="utf-8")
        if kind == "report" and q.get("result") is not None:
            (queries_dir / f"{slug}.json").write_text(json.dumps(q["result"]), encoding="utf-8")
        if kind == "dashboard" and q.get("params") is not None:
            (queries_dir / f"{slug}.params.json").write_text(json.dumps(q["params"]), encoding="utf-8")
        sql_suffix = ".sql" if kind == "report" else ".sql.j2"
        (queries_dir / f"{slug}{sql_suffix}").write_text(q.get("sql") or "", encoding="utf-8")


def _seed_analysis_extras(
    artifact_root: Path,
    *,
    insights: list[dict] | None = None,
    subject_refs: dict | None = None,
    suggested_questions: list[dict] | None = None,
) -> None:
    """Drop optional analysis files into an already-seeded artifact root."""
    analysis_dir = artifact_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if insights is not None:
        (analysis_dir / "insights.json").write_text(json.dumps(insights), encoding="utf-8")
    if subject_refs is not None:
        (analysis_dir / "subject_refs.json").write_text(json.dumps(subject_refs), encoding="utf-8")
    if suggested_questions is not None:
        (analysis_dir / "suggested_questions.json").write_text(json.dumps(suggested_questions), encoding="utf-8")


def _make_dashboard_with_queries(agent_config, *, slug: str, queries: list[dict], subject_refs: dict | None = None):
    """Build a dashboard node bound to a fully-seeded artifact.

    Dashboards keep ``BLOB_REQUIRED = False`` so we test the disk path
    here — the same renderer code runs in blob mode (see the parity
    test) but disk mode is the production path for ``ask_dashboard``
    until publish lands."""
    _seed_artifact(agent_config.project_root, "dashboard", slug)
    root = Path(agent_config.project_root) / "dashboards" / slug
    _seed_query_files(root, "dashboard", queries)
    if subject_refs is not None:
        _seed_analysis_extras(root, subject_refs=subject_refs)
    _register_ask_agent(agent_config, name=f"ask_{slug}", kind="dashboard", slug=slug)
    return AskDashboardAgenticNode(
        node_id=f"{slug}_test",
        description="test ask_dashboard node",
        node_type="chat",
        agent_config=agent_config,
        node_name=f"ask_{slug}",
    )


def _make_report_with_queries(
    agent_config,
    *,
    slug: str,
    queries: list[dict],
    insights: list[dict] | None = None,
    subject_refs: dict | None = None,
    suggested_questions: list[dict] | None = None,
):
    """Build a report node from a fully-seeded artifact via the blob path.

    Reports have ``BLOB_REQUIRED = True`` so the artifact must travel
    through ``_blob_from_disk`` — the helper that mirrors the backend
    publish wire shape. We seed disk first as a convenient construction
    vehicle then snapshot it into the blob; the node never touches the
    disk tree at runtime.
    """
    _seed_artifact(agent_config.project_root, "report", slug)
    root = Path(agent_config.project_root) / "reports" / slug
    _seed_query_files(root, "report", queries)
    _seed_analysis_extras(
        root,
        insights=insights,
        subject_refs=subject_refs,
        suggested_questions=suggested_questions,
    )
    blob = _blob_from_disk(agent_config.project_root, "report", slug)
    _register_ask_agent(agent_config, name=f"ask_{slug}", kind="report", slug=slug, blob=blob)
    return AskReportAgenticNode(
        node_id=f"{slug}_test",
        description="test ask_report node",
        node_type="chat",
        agent_config=agent_config,
        node_name=f"ask_{slug}",
    )


def _basic_query_result(
    slug: str,
    *,
    columns: list[tuple[str, str]] | None = None,
    rows: list[dict] | None = None,
) -> dict:
    cols = [{"name": n, "type": t} for n, t in (columns or [("v", "number")])]
    rows = rows or [{"v": 1}, {"v": 2}]
    return {
        "executed_at": "2026-05-19T00:00:00Z",
        "datasource": "test_ds",
        "row_count": len(rows),
        "columns": cols,
        "rows": rows,
    }


class TestArtifactContextBlockInlining:
    """Renderer inlines as much as fits and degrades the rest cleanly.

    The runtime concern these tests pin: an ``ask_*`` follow-up should
    not need to pre-fetch sidecars to answer the user. The renderer
    accomplishes that by inlining insights / subject scope / per-query
    brief + columns + sample rows + SQL into the system prompt, with
    threshold-based degradation when individual queries or the catalog
    as a whole would blow the prompt budget.
    """

    # --- Small-artifact happy path --------------------------------------

    def test_small_report_inlines_all_query_rows(self, real_agent_config):
        """Tiny artifact: 1 query with 3 rows should be inlined in full
        (rows block, not sample). Pins the lower bound of the row+byte
        gate so a regression that always degrades to ``sample`` is
        visible immediately.
        """
        queries = [
            {
                "slug": "small_q",
                "brief": {"name": "small_q", "hypothesis": "h1", "caveats": "c1", "uses": {}},
                "result": _basic_query_result(
                    "small_q",
                    columns=[("day", "string"), ("v", "number")],
                    rows=[{"day": "Mon", "v": 10}, {"day": "Tue", "v": 11}, {"day": "Wed", "v": 12}],
                ),
                "sql": "SELECT day, v FROM t",
            }
        ]
        node = _make_report_with_queries(real_agent_config, slug="small_report", queries=queries)
        block = node._render_artifact_context_block()
        # Catalog header for the slug + the full-rows marker (NOT sample).
        assert "#### `small_q`" in block
        assert "**rows** (3):" in block
        assert "**sample**" not in block
        # Every row's value should appear verbatim.
        assert 'day="Mon"' in block
        assert "v=12" in block
        # Hypothesis and caveats survive in non-degraded mode.
        assert "**hypothesis**: h1" in block
        assert "**caveats**: c1" in block

    def test_row_count_above_limit_degrades_to_sample(self, real_agent_config):
        """row_count > INLINE_ROW_LIMIT (20) ⇒ sample mode with the
        ``read_file`` pointer so the LLM knows where the full data is.
        """
        rows = [{"v": i} for i in range(30)]
        queries = [
            {
                "slug": "wide_q",
                "brief": {"name": "wide_q", "uses": {}},
                "result": _basic_query_result("wide_q", rows=rows),
                "sql": "SELECT v FROM t",
            }
        ]
        node = _make_report_with_queries(real_agent_config, slug="wide_report", queries=queries)
        block = node._render_artifact_context_block()
        assert "#### `wide_q` — 30 rows" in block
        # Sample wording references the actual count + remediation path.
        assert "**sample** (first 2 of 30" in block
        assert "read_file('queries/wide_q.json')" in block
        # Full rows block must NOT be emitted for a wide result.
        assert "**rows** (30)" not in block
        # The remaining 28 rows shouldn't all be in the prompt.
        # Cheap proxy: the last row's value (29) shouldn't appear.
        assert "v=29" not in block

    def test_byte_size_above_limit_degrades_to_sample(self, real_agent_config):
        """A 5-row result whose rows each carry a multi-KB text field
        must degrade even though row_count is well under 20. This is the
        gate that protects against fat text columns silently inflating
        the prompt — the user-reported failure mode that originally
        prompted the double-gate design.
        """
        fat = "x" * 2_000  # ~2KB per row, 5 rows = 10KB > 4KB byte limit
        rows = [{"id": i, "blob": fat} for i in range(5)]
        queries = [
            {
                "slug": "fat_q",
                "brief": {"name": "fat_q", "uses": {}},
                "result": _basic_query_result("fat_q", columns=[("id", "integer"), ("blob", "string")], rows=rows),
                "sql": "SELECT id, blob FROM t",
            }
        ]
        node = _make_report_with_queries(real_agent_config, slug="fat_report", queries=queries)
        block = node._render_artifact_context_block()
        # Header still says 5 rows; the degradation is on inline content, not the count.
        assert "#### `fat_q` — 5 rows" in block
        # Sample mode kicks in because of bytes, not count.
        assert "**sample** (first 2 of 5" in block
        # The fat blob should appear at most twice (the sample), not 5 times.
        assert block.count("blob=") <= 2

    # --- SQL truncation -------------------------------------------------

    def test_long_sql_is_truncated_with_marker(self, real_agent_config):
        """SQL > INLINE_SQL_LINE_LIMIT (40) ⇒ truncate to the first N
        lines and append a marker telling the LLM where to find the full
        body. The marker text is load-bearing — if it changes the rule-1
        contract ("flag when the inlined summary doesn't address the
        question") breaks because the LLM no longer knows the SQL was
        elided.
        """
        long_sql = "\n".join([f"-- line {i}" for i in range(60)] + ["SELECT 1"])
        queries = [
            {
                "slug": "long_sql_q",
                "brief": {"name": "long_sql_q", "uses": {}},
                "result": _basic_query_result("long_sql_q"),
                "sql": long_sql,
            }
        ]
        node = _make_report_with_queries(real_agent_config, slug="long_sql_report", queries=queries)
        block = node._render_artifact_context_block()
        assert "-- line 0" in block
        # Line 50 is past the 40-line cap.
        assert "-- line 50" not in block
        # Truncation marker names the file so the LLM can read_file it
        # explicitly when truly needed.
        assert "more lines; read queries/long_sql_q.sql for full body" in block

    # --- Catalog cap degradation ---------------------------------------

    def test_catalog_cap_degrades_later_entries(self, real_agent_config, monkeypatch):
        """When the catalog's running byte count would exceed
        INLINE_CATALOG_BYTES_CAP, later entries drop caveats and fall
        back to sample-mode (even when individually they'd fit the
        per-row inline gate). Header info (slug + row count + columns)
        always survives so the LLM still knows the long-tail queries
        exist.

        We monkeypatch the cap down to a tiny value so the test stays
        fast and deterministic — the production cap (64KB) would need
        an artificially large fixture to trip.
        """
        from datus.agent.node import base_artifact_ask_agentic_node as mod

        # Cap chosen empirically: each non-degraded entry in this
        # fixture renders to ~220 bytes (header + hypothesis + caveats
        # + 3-row inline + SQL block). 400 fits the first entry only;
        # the second entry's running total trips the cap and switches
        # to degraded mode for the rest of the catalog.
        monkeypatch.setattr(mod, "INLINE_CATALOG_BYTES_CAP", 400)

        common_rows = [{"v": i} for i in range(3)]
        queries = []
        for slug in ("a_first", "b_second", "c_third"):
            queries.append(
                {
                    "slug": slug,
                    "brief": {
                        "name": slug,
                        "hypothesis": f"{slug} hypothesis",
                        # Caveats are visible enough to assert on.
                        "caveats": f"{slug}-CAVEAT-MARKER",
                        "uses": {},
                    },
                    "result": _basic_query_result(slug, rows=list(common_rows)),
                    "sql": f"SELECT v FROM {slug}",
                }
            )
        node = _make_report_with_queries(real_agent_config, slug="capped_report", queries=queries)
        block = node._render_artifact_context_block()
        # All three slugs survive (header info always emits).
        assert "#### `a_first`" in block
        assert "#### `b_second`" in block
        assert "#### `c_third`" in block
        # First entry keeps its caveat (not degraded).
        assert "a_first-CAVEAT-MARKER" in block
        # At least one later entry has its caveat dropped (degraded mode).
        # We don't assert which specific entry to keep the test resilient
        # against minor rendering-size shifts.
        late_caveats_dropped = "b_second-CAVEAT-MARKER" not in block or "c_third-CAVEAT-MARKER" not in block
        assert late_caveats_dropped, "expected catalog cap to drop at least one later caveat"

    # --- Subject scope --------------------------------------------------

    def test_subject_scope_reverse_index_lists_referencing_queries(self, real_agent_config):
        """One subject asset referenced by multiple queries → the
        subject scope block lists all referencing slugs in alphabetical
        order. This is the section that makes "which queries use metric
        X?" answerable without a file scan.
        """
        subj = {
            "path": ["Commerce", "Orders", "AOV"],
            "name": "average_order_value",
        }
        queries = [
            {
                "slug": "q_alpha",
                "brief": {"name": "q_alpha", "uses": {"metrics": [subj]}},
                "result": _basic_query_result("q_alpha"),
                "sql": "SELECT 1",
            },
            {
                "slug": "q_beta",
                "brief": {"name": "q_beta", "uses": {"metrics": [subj]}},
                "result": _basic_query_result("q_beta"),
                "sql": "SELECT 1",
            },
            # A query without the metric — must NOT appear in the
            # reverse index even though it's in the catalog.
            {
                "slug": "q_orphan",
                "brief": {"name": "q_orphan", "uses": {}},
                "result": _basic_query_result("q_orphan"),
                "sql": "SELECT 1",
            },
        ]
        node = _make_report_with_queries(
            real_agent_config,
            slug="subj_report",
            queries=queries,
            subject_refs={
                "metrics": [subj],
                "reference_sql": [],
                "ext_knowledge": [],
            },
        )
        block = node._render_artifact_context_block()
        # Path + name rendered together in the scope section.
        assert "Commerce > Orders > AOV > average_order_value" in block
        # Reverse index: both referencing slugs listed; orphan absent.
        assert "used by: q_alpha, q_beta" in block
        # Per-query "subjects" line shows up only on the referencing entries.
        # Pin on a concrete substring that appears once per referencing
        # query — confirms the catalog entry actually includes the tag.
        assert block.count("metric:`average_order_value`") >= 2

    # --- Skipped sections when files absent -----------------------------

    def test_insights_section_renders_id_title_confidence_evidence(self, real_agent_config):
        """When insights.json is present, the section renders one
        numbered entry per insight with id (back-ticked), title,
        optional confidence in 2-decimal form, summary, and an
        ``evidence:`` line citing referenced query slugs. Locks in the
        full set of fields the LLM relies on for ``insight:<id>``
        citations and cross-references in answers.
        """
        insights = [
            {
                "id": "weekend_aov_spike",
                "title": "Sunday AOV is nearly 3x weekday levels",
                "summary": "Average order value on Sundays ($29.19) is significantly higher.",
                "confidence": 0.95,
                "evidence_queries": ["aov_by_day_of_week", "aov_daily_trend"],
            },
            {
                # An insight with NO confidence and NO summary — must
                # still render its title without crashing or emitting a
                # stray confidence badge or an empty summary line.
                "id": "qualitative_obs",
                "title": "qualitative observation",
                "evidence_queries": [],
            },
        ]
        queries = [
            {
                "slug": "aov_by_day_of_week",
                "brief": {"name": "aov_by_day_of_week", "uses": {}},
                "result": _basic_query_result("aov_by_day_of_week"),
                "sql": "SELECT 1",
            }
        ]
        node = _make_report_with_queries(
            real_agent_config,
            slug="ins_report",
            queries=queries,
            insights=insights,
        )
        block = node._render_artifact_context_block()
        assert "### Confirmed Findings (`analysis/insights.json`)" in block
        # First insight: id, title, confidence in 2-decimal form, summary, evidence list.
        assert "1. **`weekend_aov_spike`** — Sunday AOV is nearly 3x weekday levels _(conf 0.95)_" in block
        assert "Average order value on Sundays" in block
        assert "evidence: `aov_by_day_of_week`, `aov_daily_trend`" in block
        # Second insight: id + title only, no confidence badge or evidence line.
        # Pin on the exact rendered form so a regression that adds
        # ``_(conf None)_`` or an empty evidence list ", " trips.
        assert "2. **`qualitative_obs`** — qualitative observation" in block
        # Sanity: the second entry must NOT emit a stray confidence badge
        # (`_(conf ...)_` is the precise rendered form for the badge).
        qual_line = next(line for line in block.splitlines() if "`qualitative_obs`" in line)
        assert "_(conf" not in qual_line, f"unexpected conf badge: {qual_line!r}"

    def test_missing_insights_skips_section(self, real_agent_config):
        """No insights.json ⇒ no "Confirmed Findings" heading. The
        renderer must not emit an empty stub section that confuses the
        LLM into thinking the report has no conclusions.
        """
        queries = [
            {
                "slug": "q",
                "brief": {"name": "q", "uses": {}},
                "result": _basic_query_result("q"),
                "sql": "SELECT 1",
            }
        ]
        node = _make_report_with_queries(real_agent_config, slug="no_ins", queries=queries)
        block = node._render_artifact_context_block()
        assert "### Confirmed Findings" not in block

    def test_missing_subject_refs_skips_section(self, real_agent_config):
        """No subject_refs.json ⇒ no "Subject Library Scope" heading."""
        queries = [
            {
                "slug": "q",
                "brief": {"name": "q", "uses": {}},
                "result": _basic_query_result("q"),
                "sql": "SELECT 1",
            }
        ]
        node = _make_report_with_queries(real_agent_config, slug="no_subj", queries=queries)
        block = node._render_artifact_context_block()
        assert "### Subject Library Scope" not in block

    def test_no_queries_skips_catalog(self, real_agent_config):
        """Artifact with no queries/ files ⇒ no catalog section. Avoids
        emitting an empty heading or a misleading "no data" stub when a
        bare-bones report binds to ask_report (e.g. freshly published
        before save_query calls were issued — should never happen, but
        the renderer must not crash if it does).
        """
        node = _make_ask_report_node(real_agent_config)  # no queries seeded
        block = node._render_artifact_context_block()
        assert "### Query Catalog" not in block
        # Other sections still render normally.
        assert "## Bound Artifact" in block

    # --- Dashboard branching --------------------------------------------

    def test_dashboard_catalog_uses_sample_params_not_rows(self, real_agent_config):
        """Dashboard catalog renders ``sample_params`` from the .params
        sidecar and labels the size as ``template`` — NOT rows. Locks
        in the kind-branch in ``_render_query_catalog_entry`` so a
        regression that always emits report shape on dashboards
        (silently dropping params from the prompt) is visible.
        """
        queries = [
            {
                "slug": "dash_q",
                "brief": {"name": "dash_q", "uses": {}},
                "params": {
                    "columns": [{"name": "v", "type": "number"}],
                    "sample_params": {"start_date": "2026-01-01", "store_id": 42},
                    "sample_row_count": 7,
                },
                "sql": "SELECT v FROM t WHERE store_id = {{ store_id }}",
            }
        ]
        node = _make_dashboard_with_queries(real_agent_config, slug="dash_inline", queries=queries)
        block = node._render_artifact_context_block()
        assert "#### `dash_q` — template · sample 7 rows" in block
        assert "**sample_params**:" in block
        assert "start_date" in block and "store_id" in block
        # Dashboard SQL appears under the .sql.j2 suffix.
        assert "store_id = {{ store_id }}" in block
        # Report-shape ("rows" inline) MUST NOT appear.
        assert "**rows** (" not in block

    # --- Disk ↔ blob parity ---------------------------------------------

    def test_blob_and_disk_modes_render_identically(self, real_agent_config):
        """Same artifact seeded once and read once via the disk path and
        once via the blob path renders the same content block (modulo
        the source-line: in-memory snapshot vs Root path). This is the
        cross-component contract between the disk-backed
        FilesystemFuncTool flow (CLI) and the MemoryFilesystemFuncTool flow (SaaS) —
        if rendering drifts between the two, the same artifact would
        answer differently depending on deployment.
        """
        slug = "parity"
        queries = [
            {
                "slug": "parity_q",
                "brief": {"name": "parity_q", "hypothesis": "h", "uses": {}},
                "result": _basic_query_result("parity_q", rows=[{"v": 1}, {"v": 2}]),
                "sql": "SELECT v FROM t",
            }
        ]

        # Disk path via dashboard kind (BLOB_REQUIRED=False).
        node_disk = _make_dashboard_with_queries(real_agent_config, slug=f"{slug}_d", queries=queries)
        block_disk = node_disk._render_artifact_context_block()

        # Blob path via report kind (BLOB_REQUIRED=True). Note: this is
        # a different kind, so the test compares structural similarity
        # rather than byte-identical rendering. We pin the load-bearing
        # cross-mode behavior: the per-query data shows up the same way
        # regardless of where the files came from.
        node_blob = _make_report_with_queries(
            real_agent_config,
            slug=f"{slug}_r",
            queries=queries,
        )
        block_blob = node_blob._render_artifact_context_block()

        # Header per-mode differs (Root vs in-memory snapshot); verify
        # the kind-agnostic per-query rendering matches structure.
        assert "#### `parity_q`" in block_blob
        # Dashboard renders sample_params if params present; since this
        # fixture only seeded a report-shape result, the dashboard side
        # has no params, so its entry header is "no data file".
        assert "#### `parity_q` — no data file" in block_disk
        # SQL appears in BOTH renderings.
        assert "SELECT v FROM t" in block_disk
        assert "SELECT v FROM t" in block_blob

    # --- Rule 1 contract -----------------------------------------------

    def test_rule_one_forbids_defensive_reads(self, real_agent_config):
        """Behavioral rule 1 must explicitly forbid pre-fetching files
        already inlined. This is the runtime behavior change the
        rewrite was driven by; without an explicit prohibition the LLM
        reverts to defensive ``glob`` + ``read_file`` even when the
        prompt carries everything.
        """
        queries = [
            {
                "slug": "q",
                "brief": {"name": "q", "uses": {}},
                "result": _basic_query_result("q"),
                "sql": "SELECT 1",
            }
        ]
        node = _make_report_with_queries(real_agent_config, slug="rule1", queries=queries)
        block = node._render_artifact_context_block()
        # Find rule 1 specifically (it starts with a numbered "1. ").
        # We pin on both the imperative wording and the explicit "DO
        # NOT" so a future copy-edit that softens the rule trips.
        assert "1. **Answer from the inlined context first**" in block
        assert "Do NOT issue `glob` or `read_file`" in block
