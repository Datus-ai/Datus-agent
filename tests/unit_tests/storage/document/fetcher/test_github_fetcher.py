# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.document.fetcher.github_fetcher."""

import base64
import re
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# Guard: PyGithub may not be installed
try:
    import github  # noqa: F401

    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

pytestmark = [
    pytest.mark.ci,
    pytest.mark.skipif(not GITHUB_AVAILABLE, reason="PyGithub not installed"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fetcher(**kwargs):
    """Create a GitHubFetcher with mocked PyGithub internals."""
    from datus.storage.document.fetcher.rate_limiter import RateLimiter

    rl = RateLimiter()
    rl.wait = MagicMock(return_value=0.0)

    defaults = dict(platform="testplatform", rate_limiter=rl, pool_size=1, token="fake-token")
    defaults.update(kwargs)

    with patch("datus.storage.document.fetcher.github_fetcher.Github") as MockGithub:
        mock_github_instance = MagicMock()
        MockGithub.return_value = mock_github_instance

        from datus.storage.document.fetcher.github_fetcher import GitHubFetcher

        fetcher = GitHubFetcher(**defaults)
        fetcher._github = mock_github_instance

    return fetcher


def _make_content_file(path, name, content_text, file_type="file", encoding="base64"):
    """Create a mock GitHub ContentFile object."""
    mock = MagicMock()
    mock.path = path
    mock.name = name
    mock.type = file_type
    mock.encoding = encoding
    mock.content = (
        base64.b64encode(content_text.encode("utf-8")).decode("utf-8") if encoding == "base64" else content_text
    )
    mock.sha = "abc123"
    mock.size = len(content_text)
    return mock


def _make_directory(path, name):
    """Create a mock directory ContentFile."""
    mock = MagicMock()
    mock.path = path
    mock.name = name
    mock.type = "dir"
    return mock


# ---------------------------------------------------------------------------
# _detect_version_from_path (static method)
# ---------------------------------------------------------------------------


class TestDetectVersionFromPath:
    """Tests for GitHubFetcher._detect_version_from_path."""

    def test_version_in_releases_dir(self):
        """Should detect version from releases/ directory."""
        from datus.storage.document.fetcher.github_fetcher import GitHubFetcher

        assert GitHubFetcher._detect_version_from_path("releases/0.2.0/foo.md") == "0.2.0"

    def test_version_with_v_prefix(self):
        """Should detect version with v prefix."""
        from datus.storage.document.fetcher.github_fetcher import GitHubFetcher

        assert GitHubFetcher._detect_version_from_path("content/releases/v3.4.0/bar.md") == "3.4.0"

    def test_version_with_beta_suffix(self):
        """Should detect version with beta suffix."""
        from datus.storage.document.fetcher.github_fetcher import GitHubFetcher

        assert GitHubFetcher._detect_version_from_path("versioned_docs/1.2.3-beta/baz.md") == "1.2.3-beta"

    def test_no_version_returns_none(self):
        """No version in path returns None."""
        from datus.storage.document.fetcher.github_fetcher import GitHubFetcher

        assert GitHubFetcher._detect_version_from_path("docs/guide/intro.md") is None

    def test_two_part_version(self):
        """Should detect two-part versions like 1.2."""
        from datus.storage.document.fetcher.github_fetcher import GitHubFetcher

        assert GitHubFetcher._detect_version_from_path("docs/1.2/guide.md") == "1.2"


# ---------------------------------------------------------------------------
# _VERSION_DIR_RE
# ---------------------------------------------------------------------------


class TestVersionDirRegex:
    """Tests for the _VERSION_DIR_RE pattern."""

    @pytest.mark.parametrize(
        "dirname, expected",
        [
            ("1.2.0", True),
            ("v3.4.0", True),
            ("1.0.0-beta", True),
            ("v2.1", True),
            ("docs", False),
            ("src", False),
            ("", False),
        ],
    )
    def test_version_dir_match(self, dirname, expected):
        """Should match version directory names."""
        from datus.storage.document.fetcher.github_fetcher import GitHubFetcher

        result = GitHubFetcher._VERSION_DIR_RE.match(dirname)
        assert bool(result) == expected


# ---------------------------------------------------------------------------
# _is_doc_file (inherited from BaseFetcher)
# ---------------------------------------------------------------------------


class TestIsDocFile:
    """Tests for _is_doc_file used in _collect_file_paths."""

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("README.md", True),
            ("guide.rst", True),
            ("page.html", True),
            ("notes.txt", True),
            ("main.py", False),
            ("data.csv", False),
            ("image.png", False),
        ],
    )
    def test_doc_file_detection(self, filename, expected):
        """Should correctly identify documentation files."""
        f = _make_fetcher()
        assert f._is_doc_file(filename) == expected


# ---------------------------------------------------------------------------
# _detect_content_type (inherited from BaseFetcher)
# ---------------------------------------------------------------------------


class TestDetectContentType:
    """Tests for content type detection."""

    @pytest.mark.parametrize(
        "filename, content, expected",
        [
            ("guide.md", "# Title", "markdown"),
            ("page.html", "<html>", "html"),
            ("doc.rst", "=====", "rst"),
            ("README", "# Heading", "markdown"),
        ],
    )
    def test_content_type_by_extension(self, filename, content, expected):
        """Should detect content type from file extension."""
        f = _make_fetcher()
        assert f._detect_content_type(filename, content) == expected


# ---------------------------------------------------------------------------
# _detect_default_branch
# ---------------------------------------------------------------------------


class TestDetectDefaultBranch:
    """Tests for _detect_default_branch."""

    def test_uses_repo_default_branch(self):
        """Should use repo.default_branch when available."""
        f = _make_fetcher()
        mock_repo = MagicMock()
        mock_repo.default_branch = "develop"
        assert f._detect_default_branch(mock_repo) == "develop"

    def test_falls_back_to_main(self):
        """When default_branch raises, should try 'main'."""
        f = _make_fetcher()
        mock_repo = MagicMock()
        type(mock_repo).default_branch = PropertyMock(side_effect=Exception("API error"))
        mock_repo.get_branch.return_value = MagicMock()
        assert f._detect_default_branch(mock_repo) == "main"

    def test_falls_back_to_master(self):
        """When 'main' branch doesn't exist, should try 'master'."""
        from github import GithubException

        f = _make_fetcher()
        mock_repo = MagicMock()
        type(mock_repo).default_branch = PropertyMock(side_effect=Exception("API error"))

        def branch_side_effect(name):
            if name == "main":
                raise GithubException(404, {"message": "Not found"}, None)
            return MagicMock()

        mock_repo.get_branch.side_effect = branch_side_effect
        assert f._detect_default_branch(mock_repo) == "master"

    def test_ultimate_fallback_main(self):
        """When all branches fail, default to 'main'."""
        from github import GithubException

        f = _make_fetcher()
        mock_repo = MagicMock()
        type(mock_repo).default_branch = PropertyMock(side_effect=Exception("API error"))
        mock_repo.get_branch.side_effect = GithubException(404, {"message": "Not found"}, None)
        assert f._detect_default_branch(mock_repo) == "main"


# ---------------------------------------------------------------------------
# _detect_version
# ---------------------------------------------------------------------------


class TestDetectVersion:
    """Tests for _detect_version."""

    def test_uses_latest_release(self):
        """Should prefer the latest release tag."""
        f = _make_fetcher()
        mock_repo = MagicMock()
        mock_release = MagicMock()
        mock_release.tag_name = "v2.0.0"
        mock_repo.get_latest_release.return_value = mock_release
        assert f._detect_version(mock_repo, "main") == "v2.0.0"

    def test_falls_back_to_releases_list(self):
        """When get_latest_release fails, should try releases list."""
        from github import GithubException

        f = _make_fetcher()
        mock_repo = MagicMock()
        mock_repo.get_latest_release.side_effect = GithubException(404, {"message": "Not found"}, None)

        mock_release = MagicMock()
        mock_release.tag_name = "v1.5.0"
        mock_releases = MagicMock()
        mock_releases.__getitem__ = MagicMock(return_value=[mock_release])
        mock_repo.get_releases.return_value = mock_releases

        assert f._detect_version(mock_repo, "main") == "v1.5.0"

    def test_branch_name_when_not_default(self):
        """Should use branch name when not main/master and no releases/tags."""
        from github import GithubException

        f = _make_fetcher()
        mock_repo = MagicMock()
        mock_repo.get_latest_release.side_effect = GithubException(404, {}, None)
        mock_repo.get_releases.side_effect = GithubException(404, {}, None)
        mock_repo.get_tags.side_effect = GithubException(404, {}, None)

        result = f._detect_version(mock_repo, "feature-branch")
        assert result == "feature-branch"

    def test_date_fallback_for_main(self):
        """Should fall back to date for main/master branches."""
        from github import GithubException

        f = _make_fetcher()
        mock_repo = MagicMock()
        mock_repo.get_latest_release.side_effect = GithubException(404, {}, None)
        mock_repo.get_releases.side_effect = GithubException(404, {}, None)
        mock_repo.get_tags.side_effect = GithubException(404, {}, None)

        result = f._detect_version(mock_repo, "main")
        assert re.match(r"\d{4}-\d{2}-\d{2}", result)


# ---------------------------------------------------------------------------
# _collect_file_paths
# ---------------------------------------------------------------------------


class TestCollectFilePaths:
    """Tests for _collect_file_paths."""

    def test_collects_doc_files(self):
        """Should collect documentation files from a directory."""
        f = _make_fetcher()
        mock_repo = MagicMock()

        md_file = _make_content_file("docs/guide.md", "guide.md", "# Guide")
        txt_file = _make_content_file("docs/notes.txt", "notes.txt", "Notes")
        py_file = _make_content_file("docs/main.py", "main.py", "print()")

        mock_repo.get_contents.return_value = [md_file, txt_file, py_file]

        paths = f._collect_file_paths(mock_repo, "docs", "main")
        assert "docs/guide.md" in paths
        assert "docs/notes.txt" in paths
        assert "docs/main.py" not in paths

    def test_recurses_into_directories(self):
        """Should recurse into subdirectories."""
        f = _make_fetcher()
        mock_repo = MagicMock()

        subdir = _make_directory("docs/sub", "sub")
        deep_file = _make_content_file("docs/sub/deep.md", "deep.md", "# Deep")

        def get_contents_side_effect(path, ref=None):
            if path == "docs":
                return [subdir]
            elif path == "docs/sub":
                return [deep_file]
            return []

        mock_repo.get_contents.side_effect = get_contents_side_effect

        paths = f._collect_file_paths(mock_repo, "docs", "main")
        assert "docs/sub/deep.md" in paths

    def test_single_file_path(self):
        """When get_contents returns a single file (not a list), should handle it."""
        f = _make_fetcher()
        mock_repo = MagicMock()

        single_file = _make_content_file("README.md", "README.md", "# Readme")
        mock_repo.get_contents.return_value = single_file  # Not a list

        paths = f._collect_file_paths(mock_repo, "README.md", "main")
        assert "README.md" in paths


# ---------------------------------------------------------------------------
# _fetch_file
# ---------------------------------------------------------------------------


class TestFetchFile:
    """Tests for _fetch_file."""

    def test_fetch_file_success(self):
        """Should fetch and decode a base64-encoded file."""
        f = _make_fetcher()
        mock_repo = MagicMock()

        content = _make_content_file("docs/guide.md", "guide.md", "# Guide Content")
        mock_repo.get_contents.return_value = content

        doc = f._fetch_file(mock_repo, "docs/guide.md", "main", "v1.0", "owner/repo")
        assert doc is not None
        assert doc.raw_content == "# Guide Content"
        assert doc.platform == "testplatform"
        assert doc.version == "v1.0"
        assert doc.content_type == "markdown"
        assert doc.metadata["sha"] == "abc123"
        assert doc.metadata["branch"] == "main"
        assert "github.com/owner/repo/blob/main/docs/guide.md" in doc.source_url

    def test_fetch_file_non_base64_encoding(self):
        """Should handle non-base64 encoded content."""
        f = _make_fetcher()
        mock_repo = MagicMock()

        content = _make_content_file("docs/guide.md", "guide.md", "# Raw Content", encoding="utf-8")
        mock_repo.get_contents.return_value = content

        doc = f._fetch_file(mock_repo, "docs/guide.md", "main", "v1.0", "owner/repo")
        assert doc is not None
        # For non-base64, it uses content.content directly (which is the raw text)
        assert doc.raw_content == "# Raw Content"

    def test_fetch_file_directory_returns_none(self):
        """Getting a directory instead of file should return None."""
        f = _make_fetcher()
        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = [MagicMock(), MagicMock()]  # List = directory

        doc = f._fetch_file(mock_repo, "docs/", "main", "v1.0", "owner/repo")
        assert doc is None

    def test_fetch_file_404_returns_none(self):
        """404 error should return None."""
        from github import GithubException

        f = _make_fetcher()
        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = GithubException(404, {"message": "Not found"}, None)

        doc = f._fetch_file(mock_repo, "missing.md", "main", "v1.0", "owner/repo")
        assert doc is None

    def test_fetch_file_other_github_error_returns_none(self):
        """Non-404 GitHub errors should return None."""
        from github import GithubException

        f = _make_fetcher()
        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = GithubException(500, {"message": "Server error"}, None)

        doc = f._fetch_file(mock_repo, "docs/guide.md", "main", "v1.0", "owner/repo")
        assert doc is None

    def test_fetch_file_unicode_error_returns_none(self):
        """UnicodeDecodeError should return None."""
        f = _make_fetcher()
        mock_repo = MagicMock()

        content = MagicMock()
        content.encoding = "base64"
        content.content = base64.b64encode(b"\x80\x81\x82").decode("ascii")
        mock_repo.get_contents.return_value = content

        # Force UnicodeDecodeError by making decode fail
        with patch("base64.b64decode", return_value=b"\x80\x81\x82"):
            doc = f._fetch_file(mock_repo, "binary.md", "main", "v1.0", "owner/repo")
        assert doc is None


# ---------------------------------------------------------------------------
# _discover_version_directories
# ---------------------------------------------------------------------------


class TestDiscoverVersionDirectories:
    """Tests for _discover_version_directories."""

    def test_finds_version_dirs(self):
        """Should discover version-named directories."""
        f = _make_fetcher()
        mock_repo = MagicMock()

        dir1 = MagicMock()
        dir1.type = "dir"
        dir1.name = "1.0.0"

        dir2 = MagicMock()
        dir2.type = "dir"
        dir2.name = "2.1.0"

        non_version = MagicMock()
        non_version.type = "dir"
        non_version.name = "docs"

        a_file = MagicMock()
        a_file.type = "file"
        a_file.name = "README.md"

        mock_repo.get_contents.return_value = [dir1, dir2, non_version, a_file]

        result = f._discover_version_directories(mock_repo, "versioned-docs")
        assert result == ["1.0.0", "2.1.0"]

    def test_returns_empty_for_no_versions(self):
        """Should return empty list when no version directories found."""
        f = _make_fetcher()
        mock_repo = MagicMock()

        non_version = MagicMock()
        non_version.type = "dir"
        non_version.name = "docs"

        mock_repo.get_contents.return_value = [non_version]

        result = f._discover_version_directories(mock_repo, "main")
        assert result == []

    def test_handles_non_list_response(self):
        """Should return empty list when get_contents returns a single item."""
        f = _make_fetcher()
        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = MagicMock()  # Not a list

        result = f._discover_version_directories(mock_repo, "main")
        assert result == []

    def test_handles_github_exception(self):
        """Should return empty list on GitHub API error."""
        from github import GithubException

        f = _make_fetcher()
        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = GithubException(500, {}, None)

        result = f._discover_version_directories(mock_repo, "main")
        assert result == []


# ---------------------------------------------------------------------------
# GitHubFetchMetadata
# ---------------------------------------------------------------------------


class TestGitHubFetchMetadata:
    """Tests for GitHubFetchMetadata dataclass."""

    def test_metadata_creation(self):
        """Should create metadata with default nav_map."""
        from datus.storage.document.fetcher.github_fetcher import GitHubFetchMetadata

        meta = GitHubFetchMetadata(
            repo=MagicMock(),
            branch="main",
            version="v1.0",
            source="owner/repo",
            file_paths=["docs/guide.md"],
        )
        assert meta.branch == "main"
        assert meta.version == "v1.0"
        assert meta.source == "owner/repo"
        assert meta.file_paths == ["docs/guide.md"]
        assert meta.nav_map == {}


# ---------------------------------------------------------------------------
# fetch_single
# ---------------------------------------------------------------------------


class TestFetchSingle:
    """Tests for fetch_single."""

    def test_requires_repo_name(self):
        """Should raise ValueError when repo_name is not provided."""
        f = _make_fetcher()
        with pytest.raises(ValueError, match="repo_name is required"):
            f.fetch_single("docs/guide.md")

    def test_fetch_single_with_github_ref(self):
        """Should use github_ref as branch when set."""
        f = _make_fetcher(github_ref="v3.4.0")

        mock_repo = MagicMock()
        f._github.get_repo.return_value = mock_repo

        content = _make_content_file("docs/guide.md", "guide.md", "# Guide")
        mock_repo.get_contents.return_value = content

        doc = f.fetch_single("docs/guide.md", repo_name="owner/repo")
        assert doc is not None
        assert doc.version == "3.4.0"  # Extracted from github_ref "v3.4.0"

    def test_fetch_single_with_non_version_github_ref(self):
        """Should detect version when github_ref is not a version string."""
        f = _make_fetcher(github_ref="versioned-docs")

        mock_repo = MagicMock()
        f._github.get_repo.return_value = mock_repo

        # Set up version detection to fall back to date
        from github import GithubException

        mock_repo.get_latest_release.side_effect = GithubException(404, {}, None)
        mock_repo.get_releases.side_effect = GithubException(404, {}, None)
        mock_repo.get_tags.side_effect = GithubException(404, {}, None)

        content = _make_content_file("docs/guide.md", "guide.md", "# Guide")
        mock_repo.get_contents.return_value = content

        doc = f.fetch_single("docs/guide.md", repo_name="owner/repo")
        assert doc is not None
        # version should be "versioned-docs" (not main/master, so used as-is)
        assert doc.version == "versioned-docs"


# ---------------------------------------------------------------------------
# fetch_batch
# ---------------------------------------------------------------------------


class TestFetchBatch:
    """Tests for fetch_batch."""

    def test_empty_batch_returns_empty(self):
        """Empty file_paths_batch should return empty list."""
        from datus.storage.document.fetcher.github_fetcher import GitHubFetchMetadata

        f = _make_fetcher()
        meta = GitHubFetchMetadata(
            repo=MagicMock(),
            branch="main",
            version="v1.0",
            source="owner/repo",
            file_paths=[],
        )
        result = f.fetch_batch(meta, [])
        assert result == []

    def test_fetch_batch_applies_nav_map(self):
        """fetch_batch should apply nav_path from metadata.nav_map."""
        from datus.storage.document.fetcher.github_fetcher import GitHubFetchMetadata

        f = _make_fetcher()
        mock_repo = MagicMock()

        content = _make_content_file("docs/guide.md", "guide.md", "# Guide")
        mock_repo.get_contents.return_value = content

        meta = GitHubFetchMetadata(
            repo=mock_repo,
            branch="main",
            version="v1.0",
            source="owner/repo",
            file_paths=["docs/guide.md"],
            nav_map={"docs/guide.md": ["User Guide", "Getting Started"]},
        )

        docs = f.fetch_batch(meta, ["docs/guide.md"])
        assert len(docs) == 1
        assert docs[0].metadata["nav_path"] == ["User Guide", "Getting Started"]
        assert docs[0].metadata["group_name"] == "User Guide"
