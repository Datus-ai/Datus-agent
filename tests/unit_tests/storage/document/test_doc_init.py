# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.document.doc_init."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from datus.storage.document.doc_init import (
    _VERSION_PATH_RE,
    InitResult,
    VersionStats,
    _build_version_details,
    _delete_existing_versions,
    _detect_versions_from_file_paths,
    _detect_versions_from_paths,
    _make_empty_result,
    infer_platform_from_source,
)

# ---------------------------------------------------------------------------
# VersionStats / InitResult dataclasses
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestVersionStats:
    """Tests for VersionStats dataclass."""

    def test_creation(self):
        """VersionStats holds version, doc_count, chunk_count."""
        vs = VersionStats(version="1.0", doc_count=10, chunk_count=50)
        assert vs.version == "1.0"
        assert vs.doc_count == 10
        assert vs.chunk_count == 50


@pytest.mark.ci
class TestInitResult:
    """Tests for InitResult dataclass."""

    def test_creation_minimal(self):
        """InitResult can be created with required fields."""
        r = InitResult(
            platform="test",
            version="1.0",
            source="https://example.com",
            total_docs=5,
            total_chunks=20,
            success=True,
            errors=[],
            duration_seconds=1.5,
        )
        assert r.platform == "test"
        assert r.success is True
        assert r.version_details is None

    def test_creation_with_version_details(self):
        """InitResult can include version_details."""
        vd = [VersionStats(version="1.0", doc_count=3, chunk_count=15)]
        r = InitResult(
            platform="test",
            version="1.0",
            source="local",
            total_docs=3,
            total_chunks=15,
            success=True,
            errors=[],
            duration_seconds=0.5,
            version_details=vd,
        )
        assert len(r.version_details) == 1
        assert r.version_details[0].version == "1.0"


# ---------------------------------------------------------------------------
# _VERSION_PATH_RE
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestVersionPathRegex:
    """Tests for the _VERSION_PATH_RE pattern."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("1.3.0", "1.3.0"),
            ("v1.3.0", "1.3.0"),
            ("2.0", "2.0"),
            ("v2.0", "2.0"),
            ("1.2.3-beta", "1.2.3-beta"),
            ("v1.2.3-rc.1", "1.2.3-rc.1"),
        ],
    )
    def test_matches_version_strings(self, path, expected):
        """Regex matches valid version strings."""
        m = _VERSION_PATH_RE.match(path)
        assert m is not None
        assert m.group(1) == expected

    @pytest.mark.parametrize(
        "path",
        ["docs", "README.md", "src", "api-guide", "v1-api"],
    )
    def test_rejects_non_version_strings(self, path):
        """Regex does not match non-version paths."""
        m = _VERSION_PATH_RE.match(path)
        assert m is None


# ---------------------------------------------------------------------------
# _detect_versions_from_paths
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestDetectVersionsFromPaths:
    """Tests for _detect_versions_from_paths."""

    def test_empty_paths(self):
        """Empty list returns empty set."""
        assert _detect_versions_from_paths([]) == set()

    def test_all_version_paths(self):
        """All paths are version-like: returns version set."""
        result = _detect_versions_from_paths(["1.3.0", "1.2.0", "v2.0.0"])
        assert result == {"1.3.0", "1.2.0", "2.0.0"}

    def test_mixed_paths_returns_empty(self):
        """If not all paths are version-like, returns empty set."""
        result = _detect_versions_from_paths(["1.3.0", "docs", "README.md"])
        assert result == set()

    def test_nested_version_paths(self):
        """Version is extracted from first path segment."""
        result = _detect_versions_from_paths(["1.3.0/docs/intro.md", "1.2.0/guides/setup.md"])
        assert result == {"1.3.0", "1.2.0"}

    def test_single_version_path(self):
        """Single version path returns its version."""
        result = _detect_versions_from_paths(["v1.0.0"])
        assert result == {"1.0.0"}


# ---------------------------------------------------------------------------
# _detect_versions_from_file_paths
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestDetectVersionsFromFilePaths:
    """Tests for _detect_versions_from_file_paths."""

    def test_empty_file_paths(self):
        """Empty list returns empty set."""
        assert _detect_versions_from_file_paths([]) == set()

    def test_all_versioned_file_paths(self):
        """All files under version dirs returns version set."""
        fps = ["1.3.0/docs/intro.md", "1.3.0/guides/setup.md", "1.2.0/docs/intro.md"]
        result = _detect_versions_from_file_paths(fps)
        assert result == {"1.3.0", "1.2.0"}

    def test_no_versioned_file_paths(self):
        """Non-version paths return empty set."""
        fps = ["docs/intro.md", "guides/setup.md"]
        result = _detect_versions_from_file_paths(fps)
        assert result == set()

    def test_below_threshold_returns_empty(self):
        """If less than 50% of paths are versioned, returns empty set."""
        fps = ["1.3.0/intro.md", "docs/guide.md", "api/ref.md", "README.md"]
        result = _detect_versions_from_file_paths(fps)
        assert result == set()

    def test_exactly_50_percent(self):
        """50% versioned paths meets the threshold."""
        fps = ["1.3.0/intro.md", "docs/guide.md"]
        result = _detect_versions_from_file_paths(fps)
        assert result == {"1.3.0"}


# ---------------------------------------------------------------------------
# _build_version_details
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestBuildVersionDetails:
    """Tests for _build_version_details."""

    def test_with_target_versions(self):
        """Returns only target versions, sorted."""
        mock_store = MagicMock()
        mock_store.get_stats_by_version.side_effect = lambda v: {
            "doc_count": 5,
            "total_chunks": 25,
        }

        result = _build_version_details(mock_store, ["1.0", "2.0", "3.0"], {"2.0", "1.0"})

        assert len(result) == 2
        assert result[0].version == "1.0"
        assert result[1].version == "2.0"

    def test_without_target_versions(self):
        """Without targets, uses all_versions sorted."""
        mock_store = MagicMock()
        mock_store.get_stats_by_version.return_value = {"doc_count": 3, "total_chunks": 10}

        result = _build_version_details(mock_store, ["3.0", "1.0", "2.0"], set())

        assert len(result) == 3
        assert [r.version for r in result] == ["1.0", "2.0", "3.0"]

    def test_empty_versions(self):
        """No versions returns empty list."""
        mock_store = MagicMock()

        result = _build_version_details(mock_store, [], set())

        assert result == []


# ---------------------------------------------------------------------------
# _make_empty_result
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestMakeEmptyResult:
    """Tests for _make_empty_result helper."""

    def test_creates_success_result(self):
        """Creates a zero-count success result."""
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = _make_empty_result("test", "1.0", "https://example.com", start)

        assert result.platform == "test"
        assert result.version == "1.0"
        assert result.total_docs == 0
        assert result.total_chunks == 0
        assert result.success is True
        assert result.duration_seconds > 0

    def test_with_errors(self):
        """Errors are included."""
        start = datetime.now(timezone.utc)
        result = _make_empty_result("test", "1.0", "local", start, errors=["No docs"])

        assert len(result.errors) == 1
        assert "No docs" in result.errors[0]

    def test_missing_version_uses_unknown(self):
        """Empty version defaults to 'unknown'."""
        start = datetime.now(timezone.utc)
        result = _make_empty_result("test", "", "local", start)

        assert result.version == "unknown"


# ---------------------------------------------------------------------------
# _delete_existing_versions
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestDeleteExistingVersions:
    """Tests for _delete_existing_versions helper."""

    def test_single_version(self):
        """Single version mode calls delete_docs once."""
        mock_store = MagicMock()
        mock_store.delete_docs.return_value = 10

        _delete_existing_versions(mock_store, "1.0", set())

        mock_store.delete_docs.assert_called_once_with(version="1.0")

    def test_multi_version(self):
        """Multi-version mode calls delete_docs for each path version."""
        mock_store = MagicMock()
        mock_store.delete_docs.return_value = 5

        _delete_existing_versions(mock_store, "1.0", {"1.0", "2.0"})

        assert mock_store.delete_docs.call_count == 2

    def test_no_deletions(self):
        """When delete_docs returns 0/None, no error."""
        mock_store = MagicMock()
        mock_store.delete_docs.return_value = 0

        _delete_existing_versions(mock_store, "1.0", set())

        mock_store.delete_docs.assert_called_once()


# ---------------------------------------------------------------------------
# infer_platform_from_source
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInferPlatformFromSource:
    """Tests for the infer_platform_from_source function."""

    def test_empty_source(self):
        """Empty source returns None."""
        assert infer_platform_from_source("") is None
        assert infer_platform_from_source("   ") is None

    def test_github_url(self):
        """GitHub URL extracts repo name."""
        result = infer_platform_from_source("https://github.com/snowflakedb/snowflake-docs")
        assert result == "snowflake"

    def test_github_url_with_git_suffix(self):
        """GitHub URL with .git suffix."""
        result = infer_platform_from_source("https://github.com/duckdb/duckdb.git")
        assert result == "duckdb"

    def test_github_shorthand(self):
        """owner/repo shorthand."""
        result = infer_platform_from_source("snowflakedb/snowflake-docs")
        assert result == "snowflake"

    def test_github_shorthand_no_suffix(self):
        """owner/repo with no -docs suffix."""
        result = infer_platform_from_source("apache/spark")
        assert result == "spark"

    def test_website_url(self):
        """Website URL extracts domain name."""
        result = infer_platform_from_source("https://docs.snowflake.com/en/guides")
        assert result == "snowflake"

    def test_website_url_www(self):
        """Website URL with www prefix."""
        result = infer_platform_from_source("https://www.example.com/docs")
        assert result == "example"

    def test_local_path(self):
        """Local path extracts directory name."""
        result = infer_platform_from_source("/path/to/starrocks-docs")
        assert result == "starrocks"

    def test_local_path_plain_name(self):
        """Local directory without -docs suffix."""
        result = infer_platform_from_source("/path/to/duckdb")
        assert result == "duckdb"

    def test_trailing_slash_stripped(self):
        """Trailing slash is handled."""
        result = infer_platform_from_source("https://docs.postgresql.org/")
        assert result == "postgresql"

    def test_github_url_with_path(self):
        """GitHub URL with extra path components."""
        result = infer_platform_from_source("https://github.com/apache/spark/tree/main/docs")
        assert result == "spark"

    def test_local_path_documentation_suffix(self):
        """Local path with -documentation suffix."""
        result = infer_platform_from_source("/data/mysql-documentation")
        assert result == "mysql"
