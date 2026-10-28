"""Version parsing is local logic; product ingest is covered separately."""


class TestVersionDetection:
    """Test version detection from paths."""

    def test_detect_versions_from_paths_basic(self):
        """Test version detection from simple version directories."""
        from datus.storage.document.doc_init import _detect_versions_from_paths

        # Basic version paths
        paths = ["1.3.0", "1.2.0"]
        versions = _detect_versions_from_paths(paths)
        assert versions == {"1.3.0", "1.2.0"}

    def test_detect_versions_from_paths_with_v_prefix(self):
        """Test version detection with 'v' prefix."""
        from datus.storage.document.doc_init import _detect_versions_from_paths

        paths = ["v1.0.0", "v2.0.0"]
        versions = _detect_versions_from_paths(paths)
        assert versions == {"1.0.0", "2.0.0"}

    def test_detect_versions_from_paths_semver(self):
        """Test version detection with pre-release tags."""
        from datus.storage.document.doc_init import _detect_versions_from_paths

        paths = ["1.0.0-beta", "1.0.0-rc1", "1.0.0"]
        versions = _detect_versions_from_paths(paths)
        assert versions == {"1.0.0-beta", "1.0.0-rc1", "1.0.0"}

    def test_detect_versions_from_paths_not_versions(self):
        """Test that non-version paths return empty set."""
        from datus.storage.document.doc_init import _detect_versions_from_paths

        # Regular paths
        paths = ["docs", "README.md", "CHANGELOG.md"]
        versions = _detect_versions_from_paths(paths)
        assert versions == set()

    def test_detect_versions_from_paths_mixed(self):
        """Test that mixed paths (version + non-version) return empty set."""
        from datus.storage.document.doc_init import _detect_versions_from_paths

        # Mixed - only some paths are versions
        paths = ["1.3.0", "docs", "README.md"]
        versions = _detect_versions_from_paths(paths)
        assert versions == set()

    def test_detect_versions_from_paths_empty(self):
        """Test empty paths returns empty set."""
        from datus.storage.document.doc_init import _detect_versions_from_paths

        versions = _detect_versions_from_paths([])
        assert versions == set()
