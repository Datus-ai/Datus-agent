# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.document.doc_init."""

import warnings
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

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
    init_platform_docs,
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


# ---------------------------------------------------------------------------
# init_platform_docs db_path deprecation
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitPlatformDocsDbPathDeprecation:
    """Tests for db_path deprecation warning in init_platform_docs."""

    def test_db_path_emits_deprecation_warning(self):
        """Passing db_path should emit a DeprecationWarning."""
        mock_cfg = MagicMock()
        mock_cfg.source = "/some/path"
        mock_cfg.type = "local"
        mock_cfg.version = "v1"
        mock_cfg.paths = []
        mock_cfg.chunk_size = 512
        mock_cfg.chunk_overlap = 50
        mock_cfg.include_patterns = []
        mock_cfg.exclude_patterns = []

        with (
            warnings.catch_warnings(record=True) as w,
            patch("datus.storage.document.doc_init.document_store") as mock_store_fn,
        ):
            warnings.simplefilter("always")
            mock_store = MagicMock()
            mock_store.get_stats.return_value = {"versions": [], "total_chunks": 0, "doc_count": 0}
            mock_store_fn.return_value = mock_store

            init_platform_docs(
                platform="test_deprecation",
                cfg=mock_cfg,
                build_mode="check",
                db_path="/old/path",
            )

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "db_path is deprecated" in str(deprecation_warnings[0].message)

    def test_no_db_path_no_warning(self):
        """Not passing db_path should not emit a DeprecationWarning."""
        mock_cfg = MagicMock()
        mock_cfg.source = "/some/path"
        mock_cfg.type = "local"
        mock_cfg.version = "v1"
        mock_cfg.paths = []
        mock_cfg.chunk_size = 512
        mock_cfg.chunk_overlap = 50
        mock_cfg.include_patterns = []
        mock_cfg.exclude_patterns = []

        with (
            warnings.catch_warnings(record=True) as w,
            patch("datus.storage.document.doc_init.document_store") as mock_store_fn,
        ):
            warnings.simplefilter("always")
            mock_store = MagicMock()
            mock_store.get_stats.return_value = {"versions": [], "total_chunks": 0, "doc_count": 0}
            mock_store_fn.return_value = mock_store

            init_platform_docs(
                platform="test_no_deprecation",
                cfg=mock_cfg,
                build_mode="check",
            )

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0


# ---------------------------------------------------------------------------
# init_platform_docs — emit callback + cancel_check
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitPlatformDocsEmit:
    """Tests for emit callback and cancel_check in init_platform_docs."""

    def _mock_cfg(self, source="/some/path", source_type="local", version="v1"):
        cfg = MagicMock()
        cfg.source = source
        cfg.type = source_type
        cfg.version = version
        cfg.paths = []
        cfg.chunk_size = 512
        cfg.include_patterns = []
        cfg.exclude_patterns = []
        return cfg

    @patch("datus.storage.document.doc_init.document_store")
    def test_emit_none_backward_compatible(self, mock_store_fn):
        """init_platform_docs with emit=None works identically to before."""
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {"versions": [], "total_chunks": 0, "doc_count": 0}
        mock_store_fn.return_value = mock_store

        result = init_platform_docs(
            platform="test_compat",
            cfg=self._mock_cfg(),
            build_mode="check",
            emit=None,
        )
        assert result.success is True

    @patch("datus.storage.document.doc_init.document_store")
    def test_emit_receives_task_started(self, mock_store_fn):
        """emit callback receives task_started event."""
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {"versions": ["v1"], "total_chunks": 5, "doc_count": 2}
        mock_store_fn.return_value = mock_store

        events = []
        result = init_platform_docs(
            platform="test_emit",
            cfg=self._mock_cfg(),
            build_mode="check",
            emit=lambda e: events.append(e),
        )
        assert result.success is True
        # Should have received at least a task_started event
        stages = [e.stage for e in events]
        assert "task_started" in stages

    @patch("datus.storage.document.doc_init.document_store")
    def test_cancel_check_before_processing(self, mock_store_fn):
        """cancel_check returning True causes early return."""
        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store

        events = []
        result = init_platform_docs(
            platform="test_cancel",
            cfg=self._mock_cfg(),
            build_mode="overwrite",
            emit=lambda e: events.append(e),
            cancel_check=lambda: True,
        )
        assert "Cancelled" in result.errors

    @patch("datus.storage.document.doc_init.document_store")
    def test_cancel_check_false_continues(self, mock_store_fn):
        """cancel_check returning False allows normal processing."""
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {"versions": [], "total_chunks": 0, "doc_count": 0}
        mock_store_fn.return_value = mock_store

        result = init_platform_docs(
            platform="test_no_cancel",
            cfg=self._mock_cfg(),
            build_mode="check",
            cancel_check=lambda: False,
        )
        assert result.success is True

    @patch("datus.storage.document.doc_init.document_store")
    def test_emit_task_failed_on_store_error(self, mock_store_fn):
        """emit receives task_failed when document_store raises."""
        from datus.utils.exceptions import DatusException, ErrorCode

        mock_store_fn.side_effect = DatusException(ErrorCode.COMMON_VALIDATION_FAILED)

        events = []
        result = init_platform_docs(
            platform="test_store_err",
            cfg=self._mock_cfg(),
            build_mode="check",
            emit=lambda e: events.append(e),
        )
        assert result.success is True  # _make_empty_result sets success=True
        stages = [e.stage for e in events]
        assert "task_failed" in stages

    @patch("datus.storage.document.doc_init.document_store")
    def test_check_mode_emits_no_task_failed(self, mock_store_fn):
        """Check mode emits task_started and does not emit task_failed."""
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {"versions": ["v1", "v2"], "total_chunks": 30, "doc_count": 6}
        mock_store.get_stats_by_version.return_value = {"doc_count": 3, "total_chunks": 15}
        mock_store_fn.return_value = mock_store

        events = []
        result = init_platform_docs(
            platform="test_check_completed",
            cfg=self._mock_cfg(),
            build_mode="check",
            emit=lambda e: events.append(e),
        )

        assert result.success is True
        # check mode returns early — emit only gets task_started (no task_completed from the helper)
        # The important assertion is that no task_failed was emitted
        stages = [e.stage for e in events]
        assert "task_failed" not in stages
        assert "task_started" in stages

    @patch("datus.storage.document.doc_init.document_store")
    def test_emit_task_failed_on_processing_exception(self, mock_store_fn):
        """emit receives task_failed with exception_type when processing raises."""
        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store

        # Use overwrite + local source so we enter the processing branch
        cfg = self._mock_cfg(source="/some/path", source_type="local")

        # Patch LocalFetcher to raise during fetch
        from unittest.mock import patch as inner_patch

        with inner_patch("datus.storage.document.doc_init.LocalFetcher") as mock_fetcher_cls:
            mock_fetcher = MagicMock()
            mock_fetcher.fetch.side_effect = ValueError("disk read error")
            mock_fetcher_cls.return_value = mock_fetcher

            events = []
            result = init_platform_docs(
                platform="test_proc_exc",
                cfg=cfg,
                build_mode="overwrite",
                emit=lambda e: events.append(e),
            )

        assert result.success is False
        stages = [e.stage for e in events]
        assert "task_failed" in stages
        # Verify the exception type and error message were captured in the event
        failed_events = [e for e in events if e.stage == "task_failed"]
        assert len(failed_events) >= 1
        assert failed_events[0].exception_type == "ValueError"
        assert "disk read error" in (failed_events[0].error or "")

    @patch("datus.storage.document.doc_init.document_store")
    def test_cancel_check_true_in_overwrite_emits_task_failed(self, mock_store_fn):
        """cancel_check=lambda: True during overwrite emits task_failed before processing."""
        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store

        events = []
        result = init_platform_docs(
            platform="test_cancel_emit",
            cfg=self._mock_cfg(),
            build_mode="overwrite",
            emit=lambda e: events.append(e),
            cancel_check=lambda: True,
        )

        # Result should contain "Cancelled" in errors
        assert "Cancelled" in result.errors

        # emit must have received a task_failed event
        stages = [e.stage for e in events]
        assert "task_failed" in stages

        # Verify the task_failed event has the cancellation error
        failed_events = [e for e in events if e.stage == "task_failed"]
        assert any("Cancelled" in (e.error or "") for e in failed_events)


# ---------------------------------------------------------------------------
# init_platform_docs — overwrite processing branches
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitPlatformDocsOverwrite:
    """Tests for overwrite-mode processing branches in init_platform_docs."""

    def _mock_cfg(self, source="/some/path", source_type="local", version="v1"):
        cfg = MagicMock()
        cfg.source = source
        cfg.type = source_type
        cfg.version = version
        cfg.paths = []
        cfg.chunk_size = 512
        cfg.include_patterns = []
        cfg.exclude_patterns = []
        return cfg

    @patch("datus.storage.document.doc_init.StreamingDocProcessor")
    @patch("datus.storage.document.doc_init.LocalFetcher")
    @patch("datus.storage.document.doc_init.document_store")
    def test_overwrite_local_source_processes_documents(self, mock_store_fn, mock_fetcher_cls, mock_processor_cls):
        """Overwrite with local source fetches docs, processes them, and returns success."""
        from datus.storage.document.streaming_processor import ProcessingStats

        mock_store = MagicMock()
        mock_store.delete_docs.return_value = 0
        mock_store_fn.return_value = mock_store

        mock_fetcher = MagicMock()
        mock_doc = MagicMock()
        mock_doc.version = "v1"
        mock_fetcher.fetch.return_value = [mock_doc] * 5
        mock_fetcher_cls.return_value = mock_fetcher

        mock_processor = MagicMock()
        mock_stats = ProcessingStats()
        mock_stats.increment(docs=5, chunks=20)
        mock_processor.process_local.return_value = mock_stats
        mock_processor_cls.return_value = mock_processor

        events = []
        result = init_platform_docs(
            platform="test_local",
            cfg=self._mock_cfg(source="/some/path", source_type="local", version="v1"),
            build_mode="overwrite",
            emit=lambda e: events.append(e),
        )

        mock_fetcher_cls.assert_called_once()
        mock_fetcher.fetch.assert_called_once()
        mock_processor.process_local.assert_called_once()
        mock_store.delete_docs.assert_called_once_with(version="v1")

        assert result.success is True
        assert result.total_docs == 5
        assert result.total_chunks == 20

        stages = [e.stage for e in events]
        assert "task_started" in stages
        assert "task_validated" in stages
        assert "task_completed" in stages

    @patch("datus.storage.document.doc_init.LocalFetcher")
    @patch("datus.storage.document.doc_init.document_store")
    def test_overwrite_local_no_documents_returns_empty(self, mock_store_fn, mock_fetcher_cls):
        """Overwrite with local source returning no docs gives total_docs=0 and error."""
        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store

        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = []
        mock_fetcher_cls.return_value = mock_fetcher

        result = init_platform_docs(
            platform="test_local_empty",
            cfg=self._mock_cfg(source="/empty/path", source_type="local", version="v1"),
            build_mode="overwrite",
        )

        assert result.total_docs == 0
        assert any("No documents found" in err for err in result.errors)

    @patch("datus.storage.document.doc_init.StreamingDocProcessor")
    @patch("datus.storage.document.doc_init.GitHubFetcher")
    @patch("datus.storage.document.doc_init.document_store")
    def test_overwrite_github_source_processes_files(self, mock_store_fn, mock_fetcher_cls, mock_processor_cls):
        """Overwrite with github source calls collect_metadata and process_github."""
        from datus.storage.document.streaming_processor import ProcessingStats

        mock_store = MagicMock()
        mock_store.delete_docs.return_value = 0
        mock_store_fn.return_value = mock_store

        mock_fetcher = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.file_paths = ["docs/intro.md", "docs/setup.md", "docs/ref.md"]
        mock_metadata.version = "v2"
        mock_fetcher.collect_metadata.return_value = mock_metadata
        mock_fetcher_cls.return_value = mock_fetcher

        mock_processor = MagicMock()
        mock_stats = ProcessingStats()
        mock_stats.increment(docs=3, chunks=12)
        mock_processor.process_github.return_value = mock_stats
        mock_processor_cls.return_value = mock_processor

        result = init_platform_docs(
            platform="test_github",
            cfg=self._mock_cfg(source="owner/repo", source_type="github", version="v2"),
            build_mode="overwrite",
        )

        mock_fetcher.collect_metadata.assert_called_once()
        mock_processor.process_github.assert_called_once()
        assert result.total_docs == 3
        assert result.total_chunks == 12

    @patch("datus.storage.document.doc_init.StreamingDocProcessor")
    @patch("datus.storage.document.doc_init.WebFetcher")
    @patch("datus.storage.document.doc_init.document_store")
    def test_overwrite_website_source_processes_urls(self, mock_store_fn, mock_fetcher_cls, mock_processor_cls):
        """Overwrite with website source calls process_website."""
        from datus.storage.document.streaming_processor import ProcessingStats

        mock_store = MagicMock()
        mock_store.delete_docs.return_value = 0
        mock_store_fn.return_value = mock_store

        mock_fetcher = MagicMock()
        mock_fetcher._detect_version_from_url.return_value = "v3"
        mock_fetcher_cls.return_value = mock_fetcher

        mock_processor = MagicMock()
        mock_stats = ProcessingStats()
        mock_stats.increment(docs=10, chunks=40)
        mock_processor.process_website.return_value = mock_stats
        mock_processor_cls.return_value = mock_processor

        result = init_platform_docs(
            platform="test_website",
            cfg=self._mock_cfg(source="https://docs.example.com", source_type="website", version=""),
            build_mode="overwrite",
        )

        mock_processor.process_website.assert_called_once()
        assert result.total_docs == 10
        assert result.total_chunks == 40

    @patch("datus.storage.document.doc_init.StreamingDocProcessor")
    @patch("datus.storage.document.doc_init.LocalFetcher")
    @patch("datus.storage.document.doc_init.document_store")
    def test_index_creation_after_processing(self, mock_store_fn, mock_fetcher_cls, mock_processor_cls):
        """After processing with chunks > 0, create_indices is called on the store."""
        from datus.storage.document.streaming_processor import ProcessingStats

        mock_store = MagicMock()
        mock_store.delete_docs.return_value = 0
        mock_store_fn.return_value = mock_store

        mock_fetcher = MagicMock()
        mock_doc = MagicMock()
        mock_doc.version = "v1"
        mock_fetcher.fetch.return_value = [mock_doc]
        mock_fetcher_cls.return_value = mock_fetcher

        mock_processor = MagicMock()
        mock_stats = ProcessingStats()
        mock_stats.increment(docs=1, chunks=5)
        mock_processor.process_local.return_value = mock_stats
        mock_processor_cls.return_value = mock_processor

        init_platform_docs(
            platform="test_index",
            cfg=self._mock_cfg(),
            build_mode="overwrite",
        )

        mock_store.create_indices.assert_called_once()

    @patch("datus.storage.document.doc_init.StreamingDocProcessor")
    @patch("datus.storage.document.doc_init.LocalFetcher")
    @patch("datus.storage.document.doc_init.document_store")
    def test_index_creation_error_logged(self, mock_store_fn, mock_fetcher_cls, mock_processor_cls):
        """When create_indices raises, error is captured in stats but result is still success."""
        from datus.storage.document.streaming_processor import ProcessingStats

        mock_store = MagicMock()
        mock_store.delete_docs.return_value = 0
        mock_store.create_indices.side_effect = RuntimeError("index build failed")
        mock_store_fn.return_value = mock_store

        mock_fetcher = MagicMock()
        mock_doc = MagicMock()
        mock_doc.version = "v1"
        mock_fetcher.fetch.return_value = [mock_doc]
        mock_fetcher_cls.return_value = mock_fetcher

        mock_processor = MagicMock()
        mock_stats = ProcessingStats()
        mock_stats.increment(docs=1, chunks=5)
        mock_processor.process_local.return_value = mock_stats
        mock_processor_cls.return_value = mock_processor

        result = init_platform_docs(
            platform="test_index_err",
            cfg=self._mock_cfg(),
            build_mode="overwrite",
        )

        mock_store.create_indices.assert_called_once()
        assert any("Index error" in err for err in result.errors)
        # total_chunks > 0 so success is True despite index error
        assert result.success is True

    @patch("datus.storage.document.doc_init.StreamingDocProcessor")
    @patch("datus.storage.document.doc_init.LocalFetcher")
    @patch("datus.storage.document.doc_init.document_store")
    def test_cancel_check_between_phases(self, mock_store_fn, mock_fetcher_cls, mock_processor_cls):
        """cancel_check True after validation (second call) triggers early return with Cancelled."""
        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store

        mock_fetcher = MagicMock()
        mock_doc = MagicMock()
        mock_doc.version = "v1"
        mock_fetcher.fetch.return_value = [mock_doc] * 3
        mock_fetcher_cls.return_value = mock_fetcher

        # cancel_check: first call (before any processing) returns False,
        # second call (after task_validated, before delete+process) returns True
        call_count = {"n": 0}

        def cancel_check():
            call_count["n"] += 1
            return call_count["n"] >= 2

        events = []
        result = init_platform_docs(
            platform="test_cancel_between",
            cfg=self._mock_cfg(source="/some/path", source_type="local", version="v1"),
            build_mode="overwrite",
            emit=lambda e: events.append(e),
            cancel_check=cancel_check,
        )

        assert "Cancelled" in result.errors
        # Cancellation must skip both deletion and processing
        mock_store.delete_docs.assert_not_called()
        mock_processor_cls.return_value.process_local.assert_not_called()
        stages = [e.stage for e in events]
        assert "task_failed" in stages
