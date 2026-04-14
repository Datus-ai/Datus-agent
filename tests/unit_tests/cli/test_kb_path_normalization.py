# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
End-to-end CI tests for the knowledge_base_home / FilesystemFuncTool re-scope
refactor. Validates three contracts:

  1. FilesystemFuncTool.config.root_path equals knowledge_base_home for each of
     the four generation nodes that route through generation_hooks /
     generation_tools.
  2. write_file → read_file round-trips work for naked filenames the LLM may
     emit (the silent normalizer rewrites the path under {kind_subdir}/{db}/).
  3. The same normalization rules drive end_semantic_model_generation /
     end_metric_generation path resolution, so a file written via the tool is
     findable by the downstream KB sync regardless of whether the LLM included
     the typed prefix.

Mocking strategy: NO mock except LLM (matches the convention in
``tests/unit_tests/conftest.py``). Uses the existing ``real_agent_config``
fixture which mirrors the structure of ``tests/conf/agent.yml`` while keeping
the home / KB rooted at ``tmp_path``.
"""

import os
from pathlib import Path

import pytest

from datus.cli.generation_hooks import (
    GenerationHooks,
    make_kb_path_normalizer,
    normalize_kb_relative_path,
)
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

# ---------------------------------------------------------------------------
# Pure-function tests for normalize_kb_relative_path
# ---------------------------------------------------------------------------


class TestNormalizeKbRelativePath:
    def test_prepends_when_prefix_missing(self):
        assert (
            normalize_kb_relative_path("orders.yaml", "semantic", "school_db")
            == "semantic_models/school_db/orders.yaml"
        )

    def test_prepends_for_sql_summary(self):
        assert (
            normalize_kb_relative_path("q_001.yaml", "sql_summary", "school_db") == "sql_summaries/school_db/q_001.yaml"
        )

    def test_prepends_for_ext_knowledge(self):
        assert (
            normalize_kb_relative_path("notes.yaml", "ext_knowledge", "school_db")
            == "ext_knowledge/school_db/notes.yaml"
        )

    def test_metric_kind_co_locates_with_semantic_models(self):
        """metrics live under semantic_models/{db}/metrics/ — same root as semantic."""
        assert (
            normalize_kb_relative_path("metrics/orders_metrics.yaml", "metric", "school_db")
            == "semantic_models/school_db/metrics/orders_metrics.yaml"
        )

    def test_idempotent_when_prefix_already_correct(self):
        already = "semantic_models/school_db/orders.yaml"
        assert normalize_kb_relative_path(already, "semantic", "school_db") == already

    def test_passes_through_paths_in_other_namespaces(self):
        """Explicit cross-namespace authoring is preserved."""
        path = "semantic_models/other_db/orders.yaml"
        assert normalize_kb_relative_path(path, "semantic", "school_db") == path

    def test_passes_through_paths_in_other_kinds(self):
        """semantic node may want to read peer artifacts; do not rewrite their prefix."""
        path = "sql_summaries/school_db/q_001.yaml"
        assert normalize_kb_relative_path(path, "semantic", "school_db") == path

    def test_absolute_paths_unchanged(self):
        assert normalize_kb_relative_path("/abs/path/orders.yaml", "semantic", "school_db") == "/abs/path/orders.yaml"

    def test_empty_path_unchanged(self):
        assert normalize_kb_relative_path("", "semantic", "school_db") == ""

    def test_dot_path_unchanged(self):
        """list_directory(".") for the workspace root must not be munged."""
        assert normalize_kb_relative_path(".", "semantic", "school_db") == "."

    def test_parent_traversal_unchanged(self):
        """Top-level traversal — leave alone so the sandbox check rejects it."""
        assert normalize_kb_relative_path("../../etc/passwd", "semantic", "school_db") == "../../etc/passwd"

    def test_unknown_kind_unchanged(self):
        assert normalize_kb_relative_path("orders.yaml", "unknown", "school_db") == "orders.yaml"

    def test_missing_namespace_unchanged(self):
        assert normalize_kb_relative_path("orders.yaml", "semantic", None) == "orders.yaml"


# ---------------------------------------------------------------------------
# make_kb_path_normalizer factory: lazy namespace resolution
# ---------------------------------------------------------------------------


class TestMakeKbPathNormalizer:
    def test_uses_default_kind_when_file_type_missing(self, real_agent_config):
        normalizer = make_kb_path_normalizer(real_agent_config, default_kind="semantic")
        assert normalizer("orders.yaml", None) == f"semantic_models/{real_agent_config.current_namespace}/orders.yaml"

    def test_file_type_overrides_default_kind(self, real_agent_config):
        normalizer = make_kb_path_normalizer(real_agent_config, default_kind="semantic")
        assert (
            normalizer("q_001.yaml", "sql_summary") == f"sql_summaries/{real_agent_config.current_namespace}/q_001.yaml"
        )

    def test_file_type_aliases_recognized(self, real_agent_config):
        """LLM may pass file_type='semantic_model' or 'metric' — both must map correctly."""
        normalizer = make_kb_path_normalizer(real_agent_config, default_kind=None)
        ns = real_agent_config.current_namespace
        assert normalizer("orders.yaml", "semantic_model") == f"semantic_models/{ns}/orders.yaml"
        assert normalizer("metrics/x.yaml", "metric") == f"semantic_models/{ns}/metrics/x.yaml"
        assert normalizer("notes.yaml", "ext_knowledge") == f"ext_knowledge/{ns}/notes.yaml"

    def test_namespace_resolved_at_call_time(self):
        """Sub-agent switches mid-session must be honored — closure rebinds each call."""

        class _MutableCfg:
            def __init__(self, ns: str):
                self.current_namespace = ns

        cfg = _MutableCfg("ns_x")
        normalizer = make_kb_path_normalizer(cfg, default_kind="semantic")
        assert normalizer("orders.yaml", None) == "semantic_models/ns_x/orders.yaml"
        cfg.current_namespace = "ns_y"
        assert normalizer("orders.yaml", None) == "semantic_models/ns_y/orders.yaml"


# ---------------------------------------------------------------------------
# FilesystemFuncTool with a normalizer: write → read round-trip
# ---------------------------------------------------------------------------


class TestFilesystemFuncToolNormalizerRoundTrip:
    def _make(self, tmp_path: Path, kind: str, namespace: str):
        kb_root = tmp_path / "kb"
        kb_root.mkdir()

        class _Cfg:
            def __init__(self, ns: str):
                self.current_namespace = ns

        tool = FilesystemFuncTool(
            root_path=str(kb_root),
            path_normalizer=make_kb_path_normalizer(_Cfg(namespace), default_kind=kind),
        )
        return tool, kb_root

    def test_naked_filename_lands_in_typed_subdir(self, tmp_path):
        tool, kb_root = self._make(tmp_path, "semantic", "school_db")
        result = tool.write_file("orders.yml", "id: orders\n", file_type="semantic_model")
        assert result.success == 1
        on_disk = kb_root / "semantic_models" / "school_db" / "orders.yml"
        assert on_disk.is_file()
        assert on_disk.read_text() == "id: orders\n"

    def test_read_naked_filename_after_naked_write(self, tmp_path):
        """Reproduces: LLM forgets prefix on both write and subsequent read."""
        tool, kb_root = self._make(tmp_path, "semantic", "school_db")
        tool.write_file("orders.yml", "payload\n", file_type="semantic_model")

        # Same naked path used by the LLM on a follow-up turn must resolve.
        read_result = tool.read_file("orders.yml")
        assert read_result.success == 1
        assert read_result.result == "payload\n"

    def test_read_with_full_prefix_after_naked_write(self, tmp_path):
        """LLM that recalls the prefix on read must also succeed."""
        tool, kb_root = self._make(tmp_path, "semantic", "school_db")
        tool.write_file("orders.yml", "payload\n", file_type="semantic_model")

        read_result = tool.read_file("semantic_models/school_db/orders.yml")
        assert read_result.success == 1
        assert read_result.result == "payload\n"

    def test_edit_file_after_naked_write(self, tmp_path):
        """edit_file must also see the file the prior write_file created."""
        tool, _ = self._make(tmp_path, "sql_summary", "school_db")
        tool.write_file("q_001.yaml", "name: original\n", file_type="sql_summary")

        edit_result = tool.edit_file(
            "q_001.yaml",
            [{"oldText": "original", "newText": "edited"}],
        )
        assert edit_result.success == 1, edit_result.error
        assert tool.read_file("q_001.yaml").result == "name: edited\n"

    def test_write_with_full_prefix_does_not_double_prefix(self, tmp_path):
        tool, kb_root = self._make(tmp_path, "semantic", "school_db")
        tool.write_file(
            "semantic_models/school_db/customers.yml",
            "id: customers\n",
            file_type="semantic_model",
        )
        on_disk = kb_root / "semantic_models" / "school_db" / "customers.yml"
        assert on_disk.is_file()
        assert not (kb_root / "semantic_models" / "school_db" / "semantic_models").exists()

    def test_cross_kind_read_works(self, tmp_path):
        """A semantic-mode tool must still be able to read peer sql_summaries by full path."""
        tool, kb_root = self._make(tmp_path, "semantic", "school_db")

        # Pre-seed a file under sql_summaries that semantic node should be able to read.
        peer = kb_root / "sql_summaries" / "school_db" / "q_001.yaml"
        peer.parent.mkdir(parents=True)
        peer.write_text("peer content\n")

        read_result = tool.read_file("sql_summaries/school_db/q_001.yaml")
        assert read_result.success == 1
        assert read_result.result == "peer content\n"


# ---------------------------------------------------------------------------
# Hook + tool agreement: end_semantic_model_generation can find naked-write files
# ---------------------------------------------------------------------------


class TestHookAndToolAgreement:
    def test_resolve_path_finds_naked_file_after_normalized_write(self, tmp_path, real_agent_config):
        """
        Simulate the full LLM flow:
          1. FilesystemFuncTool.write_file("orders.yml", file_type="semantic")
             → file lands at {kb_home}/semantic_models/{ns}/orders.yml
          2. LLM calls end_semantic_model_generation(["orders.yml"])
          3. Hook._resolve_path("orders.yml", "semantic") MUST resolve to the
             same on-disk path, otherwise the KB-sync step would FileNotFoundError.
        """
        kb_root = Path(str(real_agent_config.path_manager.knowledge_base_home))

        tool = FilesystemFuncTool(
            root_path=str(kb_root),
            path_normalizer=make_kb_path_normalizer(real_agent_config, default_kind="semantic"),
        )
        write_result = tool.write_file("orders.yml", "id: orders\n", file_type="semantic_model")
        assert write_result.success == 1

        # The hook side resolves the LLM-reported naked path.
        hooks = GenerationHooks(broker=None, agent_config=real_agent_config)
        resolved = hooks._resolve_path("orders.yml", "semantic")

        # Both should point at the same physical file.
        on_disk = kb_root / "semantic_models" / real_agent_config.current_namespace / "orders.yml"
        assert os.path.realpath(resolved) == os.path.realpath(str(on_disk))
        assert Path(resolved).is_file()

    def test_extract_filepaths_resolves_relative_entries_against_kb_home(self, tmp_path, real_agent_config):
        """end_semantic_model_generation result-list extraction also routes through normalizer."""
        kb_root = Path(str(real_agent_config.path_manager.knowledge_base_home))

        # Put a real file in place so the assertion is resolvable.
        target = kb_root / "semantic_models" / real_agent_config.current_namespace / "orders.yml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("data\n")

        hooks = GenerationHooks(broker=None, agent_config=real_agent_config)
        result = {"result": {"semantic_model_files": ["orders.yml"]}}
        paths = hooks._extract_filepaths_from_result(result)
        assert len(paths) == 1
        assert os.path.realpath(paths[0]) == os.path.realpath(str(target))


# ---------------------------------------------------------------------------
# Node-level wiring: each of the 4 generation nodes scopes the FilesystemFuncTool
# to knowledge_base_home and installs a kb path_normalizer.
# ---------------------------------------------------------------------------


class TestGenerationNodeFilesystemRootPath:
    """
    Black-box assertions: after instantiating each generation node, its
    ``filesystem_func_tool`` must be sandboxed to ``knowledge_base_home`` and
    must carry a configured normalizer (so naked-filename writes are corrected).

    Uses ``real_agent_config`` (mirrors tests/conf/agent.yml) — its home is
    tmp_path so the assertion has a stable expected value.
    """

    @pytest.fixture
    def expected_kb_home(self, real_agent_config):
        return str(real_agent_config.path_manager.knowledge_base_home)

    def test_gen_semantic_model_node_root_is_kb_home(self, real_agent_config, mock_llm_create, expected_kb_home):
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        node = GenSemanticModelAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        assert node.filesystem_func_tool is not None
        assert node.filesystem_func_tool.config.root_path == expected_kb_home
        assert node.filesystem_func_tool._path_normalizer is not None
        # And the normalizer routes a naked write to the typed sub-dir.
        ns = real_agent_config.current_namespace
        assert node.filesystem_func_tool._path_normalizer("orders.yml", None) == f"semantic_models/{ns}/orders.yml"

    def test_sql_summary_node_root_is_kb_home(self, real_agent_config, mock_llm_create, expected_kb_home):
        from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

        node = SqlSummaryAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
            node_name="gen_sql_summary",
        )
        assert node.filesystem_func_tool is not None
        assert node.filesystem_func_tool.config.root_path == expected_kb_home
        assert node.filesystem_func_tool._path_normalizer is not None
        ns = real_agent_config.current_namespace
        assert node.filesystem_func_tool._path_normalizer("q_001.yaml", None) == f"sql_summaries/{ns}/q_001.yaml"

    def test_gen_metrics_node_root_is_kb_home(self, real_agent_config, mock_llm_create, expected_kb_home):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        assert node.filesystem_func_tool is not None
        assert node.filesystem_func_tool.config.root_path == expected_kb_home
        assert node.filesystem_func_tool._path_normalizer is not None
        ns = real_agent_config.current_namespace
        # metric kind co-locates under semantic_models/
        assert (
            node.filesystem_func_tool._path_normalizer("metrics/orders.yaml", None)
            == f"semantic_models/{ns}/metrics/orders.yaml"
        )

    def test_gen_ext_knowledge_node_root_is_kb_home(self, real_agent_config, mock_llm_create, expected_kb_home):
        from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode

        node = GenExtKnowledgeAgenticNode(
            node_name="gen_ext_knowledge",
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        assert node.filesystem_func_tool is not None
        assert node.filesystem_func_tool.config.root_path == expected_kb_home
        assert node.filesystem_func_tool._path_normalizer is not None
        ns = real_agent_config.current_namespace
        assert node.filesystem_func_tool._path_normalizer("notes.yaml", None) == f"ext_knowledge/{ns}/notes.yaml"
