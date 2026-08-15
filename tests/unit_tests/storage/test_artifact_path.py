# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/storage/artifact_path.py."""

import os
from unittest.mock import patch

import pytest

from datus.storage.artifact_path import normalize_kb_relative_path, resolve_kb_sandbox_path


class TestNormalizeKbRelativePath:
    def test_prepends_when_prefix_missing(self):
        assert normalize_kb_relative_path("orders.yaml", "semantic") == "semantic_models/orders.yaml"

    def test_prepends_for_sql_summary(self):
        assert normalize_kb_relative_path("q_001.yaml", "sql_summary") == "sql_summaries/q_001.yaml"

    def test_metric_kind_co_locates_with_semantic_models(self):
        """metrics live under semantic_models/metrics/ — same root as semantic."""
        assert (
            normalize_kb_relative_path("metrics/orders_metrics.yaml", "metric")
            == "semantic_models/metrics/orders_metrics.yaml"
        )

    def test_idempotent_when_prefix_already_correct(self):
        already = "semantic_models/orders.yaml"
        assert normalize_kb_relative_path(already, "semantic") == already

    def test_passes_through_paths_in_other_kinds(self):
        path = "sql_summaries/q_001.yaml"
        assert normalize_kb_relative_path(path, "semantic") == path

    def test_absolute_paths_unchanged(self):
        assert normalize_kb_relative_path("/abs/path/orders.yaml", "semantic") == "/abs/path/orders.yaml"

    def test_empty_path_unchanged(self):
        assert normalize_kb_relative_path("", "semantic") == ""

    def test_dot_path_unchanged(self):
        assert normalize_kb_relative_path(".", "semantic") == "."

    def test_parent_traversal_unchanged(self):
        assert normalize_kb_relative_path("../../etc/passwd", "semantic") == "../../etc/passwd"

    def test_unknown_kind_unchanged(self):
        assert normalize_kb_relative_path("orders.yaml", "unknown") == "orders.yaml"


class TestResolveKbSandboxPath:
    def test_empty_path_returns_none(self, tmp_path):
        assert resolve_kb_sandbox_path("", "sql_summary", str(tmp_path)) is None

    def test_bare_filename_is_prefixed_under_sandbox(self, tmp_path):
        kb = tmp_path
        resolved = resolve_kb_sandbox_path("q_001.yaml", "sql_summary", str(kb))
        assert resolved == os.path.normpath(str(kb / "sql_summaries" / "q_001.yaml"))

    def test_fully_prefixed_relative_path_passes_through(self, tmp_path):
        kb = tmp_path
        resolved = resolve_kb_sandbox_path("sql_summaries/q.yaml", "sql_summary", str(kb))
        assert resolved == os.path.normpath(str(kb / "sql_summaries" / "q.yaml"))

    def test_absolute_path_inside_sandbox_accepted(self, tmp_path):
        kb = tmp_path
        (kb / "sql_summaries").mkdir(parents=True)
        inside = kb / "sql_summaries" / "q.yaml"
        inside.write_text("x")
        resolved = resolve_kb_sandbox_path(str(inside), "sql_summary", str(kb))
        assert os.path.realpath(resolved) == os.path.realpath(str(inside))

    def test_absolute_path_outside_sandbox_rejected(self, tmp_path):
        """A fabricated absolute path outside the sandbox must be refused so
        callers never sync an arbitrary on-disk file."""
        assert resolve_kb_sandbox_path("/etc/passwd", "sql_summary", str(tmp_path)) is None

    def test_traversal_escape_rejected(self, tmp_path):
        """``../../etc/passwd`` resolves outside the sandbox → rejected."""
        assert resolve_kb_sandbox_path("../../etc/passwd", "sql_summary", str(tmp_path)) is None

    def test_unknown_kind_no_containment_check(self, tmp_path):
        """For an unknown kind we cannot compute a sandbox — fall back to
        just normalizing against knowledge_base_dir."""
        resolved = resolve_kb_sandbox_path("foo.yaml", "unknown", str(tmp_path))
        assert resolved == os.path.normpath(str(tmp_path / "foo.yaml"))

    def test_commonpath_value_error_fails_closed(self, tmp_path):
        """Simulate os.path.commonpath raising (e.g. mixed drives on
        Windows) — the resolver must fail closed with None."""
        with patch("datus.storage.artifact_path.os.path.commonpath", side_effect=ValueError("mixed drives")):
            assert resolve_kb_sandbox_path("q.yaml", "sql_summary", str(tmp_path)) is None

    def test_rejects_symlink_that_escapes_sandbox(self, tmp_path):
        """A path inside the sandbox that symlinks outside must be rejected."""
        kb = tmp_path / "kb"
        sandbox = kb / "sql_summaries"
        sandbox.mkdir(parents=True)
        outside = tmp_path / "outside.yaml"
        outside.write_text("x")
        link = sandbox / "q.yaml"
        try:
            # Setup-only guard: the OSError comes from symlink creation on
            # platforms without the privilege (e.g. Windows sans Developer
            # Mode), never from the resolver under test.
            link.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"platform cannot create symlinks: {exc}")  # audit-noqa: try_except_skip
        assert resolve_kb_sandbox_path("sql_summaries/q.yaml", "sql_summary", str(kb)) is None
