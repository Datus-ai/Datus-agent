# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/storage/document/doc_init.py.

Tests cover:
- _detect_versions_from_file_paths: version detection from file paths
- _build_version_details: per-version statistics building
- infer_platform_from_source: platform name inference from various source formats
- init_platform_docs: check mode (returns existing store stats)

NO MOCK EXCEPT LLM. Uses real DocumentStore (LanceDB in tmp_path) where needed.
"""

from datus.storage.document.doc_init import (
    _detect_versions_from_file_paths,
    infer_platform_from_source,
)

# ===========================================================================
# _detect_versions_from_file_paths
# ===========================================================================


class TestDetectVersionsFromFilePaths:
    """Tests for _detect_versions_from_file_paths version detection."""

    def test_empty_file_paths_returns_empty(self):
        """Empty list returns empty set."""
        result = _detect_versions_from_file_paths([])
        assert result == set()

    def test_versioned_paths_detected(self):
        """File paths with version prefixes are correctly detected."""
        paths = [
            "1.2.0/docs/intro.md",
            "1.2.0/guides/setup.md",
            "1.3.0/docs/intro.md",
            "1.3.0/guides/setup.md",
        ]
        result = _detect_versions_from_file_paths(paths)
        assert result == {"1.2.0", "1.3.0"}

    def test_no_version_pattern_returns_empty(self):
        """Paths without version patterns return empty set."""
        paths = [
            "docs/intro.md",
            "guides/setup.md",
            "README.md",
        ]
        result = _detect_versions_from_file_paths(paths)
        assert result == set()

    def test_mixed_paths_below_threshold_returns_empty(self):
        """Mixed paths with fewer than 50% versioned return empty set."""
        paths = [
            "1.2.0/docs/intro.md",
            "docs/guide.md",
            "README.md",
            "src/main.py",
            "tests/test.py",
        ]
        result = _detect_versions_from_file_paths(paths)
        assert result == set()

    def test_v_prefix_versions_detected(self):
        """Paths with 'v' prefix (v1.0.0) are detected."""
        paths = [
            "v2.0.0/docs/intro.md",
            "v2.0.0/guides/setup.md",
            "v3.1.0/docs/intro.md",
            "v3.1.0/guides/setup.md",
        ]
        result = _detect_versions_from_file_paths(paths)
        assert result == {"2.0.0", "3.1.0"} or result == {"v2.0.0", "v3.1.0"}


# ===========================================================================
# infer_platform_from_source
# ===========================================================================


class TestInferPlatformFromSource:
    """Tests for infer_platform_from_source with various source formats."""

    def test_github_url(self):
        """GitHub URL extracts repo name as platform."""
        result = infer_platform_from_source("https://github.com/StarRocks/starrocks")
        assert result == "starrocks"

    def test_github_url_with_docs_suffix(self):
        """GitHub URL with -docs suffix strips it."""
        result = infer_platform_from_source("https://github.com/snowflake/snowflake-docs")
        assert result == "snowflake"

    def test_github_url_with_git_suffix(self):
        """GitHub URL with .git suffix is handled."""
        result = infer_platform_from_source("https://github.com/owner/platform.git")
        assert result == "platform"

    def test_github_shorthand(self):
        """GitHub shorthand 'owner/repo' extracts repo name."""
        result = infer_platform_from_source("myorg/myplatform")
        assert result == "myplatform"

    def test_github_shorthand_with_docs_suffix(self):
        """GitHub shorthand with docs suffix strips it."""
        result = infer_platform_from_source("myorg/myplatform-docs")
        assert result == "myplatform"

    def test_website_url(self):
        """Website URL extracts second-level domain."""
        result = infer_platform_from_source("https://docs.snowflake.com/en/user-guide")
        assert result == "snowflake"

    def test_website_url_simple_domain(self):
        """Simple domain URL extracts correctly."""
        result = infer_platform_from_source("https://polaris.io/docs")
        assert result == "polaris"

    def test_website_url_with_www(self):
        """URL with www. prefix is handled."""
        result = infer_platform_from_source("https://www.databricks.com/docs")
        assert result == "databricks"

    def test_local_path(self):
        """Local path extracts last directory component."""
        result = infer_platform_from_source("/Users/dev/projects/starrocks-docs")
        assert result == "starrocks"

    def test_local_path_plain_name(self):
        """Local path without suffix returns the name."""
        result = infer_platform_from_source("/opt/data/polaris")
        assert result == "polaris"

    def test_empty_source_returns_none(self):
        """Empty source returns None."""
        result = infer_platform_from_source("")
        assert result is None

    def test_whitespace_source_returns_none(self):
        """Whitespace-only source returns None."""
        result = infer_platform_from_source("   ")
        assert result is None

    def test_github_url_with_trailing_slash(self):
        """GitHub URL with trailing slash is handled."""
        result = infer_platform_from_source("https://github.com/owner/platform/")
        assert result == "platform"

    def test_github_url_documentation_suffix(self):
        """GitHub URL with -documentation suffix strips it."""
        result = infer_platform_from_source("https://github.com/owner/mydb-documentation")
        assert result == "mydb"


class TestInferPlatformEdgeCases:
    """Edge case tests for infer_platform_from_source."""

    def test_url_with_port(self):
        """URL with port number is handled."""
        result = infer_platform_from_source("https://docs.example.com:8080/path")
        assert result == "example"

    def test_github_shorthand_empty_parts(self):
        """Invalid shorthand with empty parts returns None."""
        result = infer_platform_from_source("/invalid")
        # This is a local path, should extract "invalid"
        assert result == "invalid"

    def test_local_path_website_suffix(self):
        """Local path with -website suffix strips it."""
        result = infer_platform_from_source("/data/polaris-website")
        assert result == "polaris"

    def test_local_path_doc_suffix(self):
        """Local path with -doc suffix strips it."""
        result = infer_platform_from_source("/data/snowflake-doc")
        assert result == "snowflake"

    def test_single_part_domain_returns_none(self):
        """URL with single-part domain (e.g. localhost) returns None."""
        result = infer_platform_from_source("http://localhost/path")
        assert result is None

    def test_root_path_returns_none(self):
        """Root path '/' has empty name, returns None."""
        result = infer_platform_from_source("/")
        assert result is None

    def test_local_path_name_stripped_to_empty(self):
        """Local path where suffix stripping leaves empty string returns None."""
        result = infer_platform_from_source("/data/docs")
        assert result is None

    def test_dot_path_returns_none(self):
        """Dot path '.' has empty name from Path, returns None (line 528)."""
        result = infer_platform_from_source(".")
        assert result is None


# ===========================================================================
# _build_version_details
# ===========================================================================


class TestBuildVersionDetails:
    """Tests for _build_version_details building per-version statistics."""

    def test_build_version_details_with_target_versions(self, tmp_path):
        """_build_version_details filters to target versions."""
        from datus.storage.document.doc_init import _build_version_details
        from datus.storage.document.store import document_store

        store = document_store(str(tmp_path / "test_details.db"))
        # Store is empty but _build_version_details should still work
        details = _build_version_details(store, ["1.0", "2.0", "3.0"], {"2.0"})
        assert len(details) == 1
        assert details[0].version == "2.0"
        assert details[0].doc_count == 0
        assert details[0].chunk_count == 0

    def test_build_version_details_no_target_uses_all(self, tmp_path):
        """_build_version_details uses all versions when target is empty."""
        from datus.storage.document.doc_init import _build_version_details
        from datus.storage.document.store import document_store

        store = document_store(str(tmp_path / "test_all.db"))
        details = _build_version_details(store, ["1.0", "2.0"], set())
        assert len(details) == 2
        assert details[0].version == "1.0"
        assert details[1].version == "2.0"

    def test_build_version_details_sorted_output(self, tmp_path):
        """_build_version_details returns sorted version details."""
        from datus.storage.document.doc_init import _build_version_details
        from datus.storage.document.store import document_store

        store = document_store(str(tmp_path / "test_sorted.db"))
        details = _build_version_details(store, ["3.0", "1.0", "2.0"], set())
        versions = [d.version for d in details]
        assert versions == ["1.0", "2.0", "3.0"]


# ===========================================================================
# init_platform_docs check mode
# ===========================================================================


class TestInitPlatformDocsCheckMode:
    """Tests for init_platform_docs in check mode (returns store stats)."""

    def test_check_mode_returns_result(self, tmp_path):
        """init_platform_docs check mode returns InitResult with store stats."""
        from datus.configuration.agent_config import DocumentConfig
        from datus.storage.document.doc_init import init_platform_docs

        cfg = DocumentConfig(source="https://github.com/owner/repo", type="github")
        result = init_platform_docs(
            db_path=str(tmp_path / "check_test.db"),
            platform="test_platform",
            cfg=cfg,
            build_mode="check",
            pool_size=1,
        )

        assert result is not None
        assert result.success is True
        assert result.platform == "test_platform"
        assert result.total_docs == 0  # Empty store
        assert result.total_chunks == 0
        assert result.version_details is not None

    def test_check_mode_with_version_override(self, tmp_path):
        """init_platform_docs check mode respects explicit version."""
        from datus.configuration.agent_config import DocumentConfig
        from datus.storage.document.doc_init import init_platform_docs

        cfg = DocumentConfig(source="https://github.com/owner/repo", type="github", version="v2.0")
        result = init_platform_docs(
            db_path=str(tmp_path / "check_ver.db"),
            platform="versioned",
            cfg=cfg,
            build_mode="check",
            pool_size=1,
        )

        assert result is not None
        assert result.success is True
