# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for the adapter scaffold → validate → run_pytest pipeline.

Tests the full end-to-end flow of scaffolding an adapter project on disk,
validating it, and running pytest against the generated test files.
No LLM is required — these exercise the real filesystem, real subprocess
invocation, and real package imports.
"""

import os
import sys

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.func_tool.adapter_scaffold_tool import ADAPTER_TYPE_CONFIG, AdapterScaffoldTool
from datus.tools.func_tool.adapter_test_runner_tool import AdapterTestRunnerTool

_ALL_ADAPTER_TYPES = sorted(ADAPTER_TYPE_CONFIG.keys())


def _make_mock_config():
    """Minimal AgentConfig for tool construction (tools don't use it)."""
    from unittest.mock import MagicMock

    return MagicMock(spec=AgentConfig)


# ---------------------------------------------------------------------------
# Scaffold → file existence
# ---------------------------------------------------------------------------


class TestScaffoldCreatesProjectOnDisk:
    """scaffold_adapter writes a complete project tree to the filesystem."""

    @pytest.mark.parametrize("adapter_type", _ALL_ADAPTER_TYPES)
    def test_scaffold_creates_expected_files(self, tmp_path, adapter_type):
        """Each adapter type produces pyproject.toml, adapter.py, config.py, __init__.py, and tests."""
        tool = AdapterScaffoldTool(_make_mock_config())
        project_dir = str(tmp_path / f"datus-{adapter_type}-testplat")

        result = tool.scaffold_adapter(
            adapter_type=adapter_type,
            platform="testplat",
            output_dir=project_dir,
            display_name="Test Platform",
        )

        assert result.success == 1, f"scaffold failed: {result.error}"
        assert os.path.isfile(os.path.join(project_dir, "pyproject.toml"))
        assert os.path.isfile(os.path.join(project_dir, "README.md"))

        # Package files
        files_created = result.result["files_created"]
        assert any("adapter.py" in f for f in files_created)
        assert any("config.py" in f for f in files_created)
        assert any("__init__.py" in f for f in files_created)

        # Test files
        assert any("test_adapter.py" in f for f in files_created)

    def test_semantic_scaffold_includes_contract_test(self, tmp_path):
        """Semantic adapters additionally produce tests/unit/test_contract.py."""
        tool = AdapterScaffoldTool(_make_mock_config())
        project_dir = str(tmp_path / "datus-semantic-cube")

        result = tool.scaffold_adapter(
            adapter_type="semantic",
            platform="cube",
            output_dir=project_dir,
        )

        assert result.success == 1
        contract_path = os.path.join(project_dir, "tests", "unit", "test_contract.py")
        assert os.path.isfile(contract_path), "Semantic scaffold must include test_contract.py"

    @pytest.mark.parametrize("adapter_type", ["bi", "db", "scheduler"])
    def test_non_semantic_scaffold_has_no_contract_test(self, tmp_path, adapter_type):
        """Non-semantic adapters should NOT have test_contract.py."""
        tool = AdapterScaffoldTool(_make_mock_config())
        project_dir = str(tmp_path / f"datus-{adapter_type}-plat")

        result = tool.scaffold_adapter(
            adapter_type=adapter_type,
            platform="plat",
            output_dir=project_dir,
        )

        assert result.success == 1
        contract_path = os.path.join(project_dir, "tests", "unit", "test_contract.py")
        assert not os.path.exists(contract_path)


# ---------------------------------------------------------------------------
# Scaffold → import → validate
# ---------------------------------------------------------------------------


class TestScaffoldProducesImportablePackage:
    """The scaffolded package can be imported and passes static validation."""

    def test_semantic_adapter_is_importable(self, tmp_path):
        """Scaffolded semantic adapter package can be imported as a Python module."""
        tool = AdapterScaffoldTool(_make_mock_config())
        project_dir = str(tmp_path / "datus-semantic-cube")

        tool.scaffold_adapter(
            adapter_type="semantic",
            platform="cube",
            output_dir=project_dir,
        )

        # Add project to sys.path so importlib can find it
        sys.path.insert(0, project_dir)
        try:
            import importlib

            mod = importlib.import_module("datus_semantic_cube")
            assert hasattr(mod, "CubeAdapter")
            assert hasattr(mod, "CubeConfig")
            assert hasattr(mod, "register")
            assert callable(mod.register)
        finally:
            sys.path.pop(0)
            # Clean up cached module
            for key in list(sys.modules.keys()):
                if key.startswith("datus_semantic_cube"):
                    del sys.modules[key]

    def test_semantic_adapter_is_instantiable(self, tmp_path):
        """Scaffolded semantic adapter can be instantiated (init_block lands inside class)."""
        tool = AdapterScaffoldTool(_make_mock_config())
        project_dir = str(tmp_path / "datus-semantic-cube")

        tool.scaffold_adapter(
            adapter_type="semantic",
            platform="cube",
            output_dir=project_dir,
        )

        sys.path.insert(0, project_dir)
        try:
            import importlib

            mod = importlib.import_module("datus_semantic_cube.adapter")
            config_mod = importlib.import_module("datus_semantic_cube.config")
            config = config_mod.CubeConfig()
            adapter = mod.CubeAdapter(config)
            assert adapter is not None
        finally:
            sys.path.pop(0)
            for key in list(sys.modules.keys()):
                if key.startswith("datus_semantic_cube"):
                    del sys.modules[key]

    def test_validate_reports_stubs_for_fresh_scaffold(self, tmp_path):
        """validate_adapter flags NotImplementedError stubs on a freshly scaffolded adapter."""
        tool = AdapterScaffoldTool(_make_mock_config())
        project_dir = str(tmp_path / "datus-semantic-cube")

        tool.scaffold_adapter(
            adapter_type="semantic",
            platform="cube",
            output_dir=project_dir,
        )

        sys.path.insert(0, project_dir)
        try:
            result = tool.validate_adapter("datus_semantic_cube")
            assert result.success == 1
            assert result.result["valid"] is False, "Fresh scaffold should have NotImplementedError stubs"
            assert any("NotImplementedError" in issue for issue in result.result["issues"])
        finally:
            sys.path.pop(0)
            for key in list(sys.modules.keys()):
                if key.startswith("datus_semantic_cube"):
                    del sys.modules[key]


# ---------------------------------------------------------------------------
# Scaffold → run pytest (via AdapterTestRunnerTool)
# ---------------------------------------------------------------------------


class TestScaffoldAndRunPytest:
    """End-to-end: scaffold a project, then run pytest on the generated tests."""

    @pytest.mark.nightly
    def test_semantic_test_adapter_passes(self, tmp_path):
        """test_adapter.py (basic instantiation) should PASS for a semantic scaffold."""
        scaffold_tool = AdapterScaffoldTool(_make_mock_config())
        runner_tool = AdapterTestRunnerTool(_make_mock_config())
        project_dir = str(tmp_path / "datus-semantic-cube")

        scaffold_result = scaffold_tool.scaffold_adapter(
            adapter_type="semantic",
            platform="cube",
            output_dir=project_dir,
        )
        assert scaffold_result.success == 1

        result = runner_tool.run_adapter_pytest(
            project_dir=project_dir,
            test_subpath="tests/unit/test_adapter.py",
        )

        assert result.success == 1, f"run_adapter_pytest returned error: {result.error}"
        assert result.result["passed"] is True, (
            f"test_adapter.py should pass for scaffolded adapter.\n"
            f"exit_code={result.result['exit_code']}\n"
            f"stdout:\n{result.result['stdout']}\n"
            f"stderr:\n{result.result['stderr']}"
        )

    @pytest.mark.nightly
    def test_semantic_contract_test_fails_with_not_implemented(self, tmp_path):
        """test_contract.py should FAIL (NotImplementedError) for an unimplemented scaffold."""
        scaffold_tool = AdapterScaffoldTool(_make_mock_config())
        runner_tool = AdapterTestRunnerTool(_make_mock_config())
        project_dir = str(tmp_path / "datus-semantic-cube")

        scaffold_tool.scaffold_adapter(
            adapter_type="semantic",
            platform="cube",
            output_dir=project_dir,
        )

        result = runner_tool.run_adapter_pytest(
            project_dir=project_dir,
            test_subpath="tests/unit/test_contract.py",
        )

        assert result.success == 1, f"Tool call itself should succeed: {result.error}"
        assert result.result["passed"] is False, (
            "Contract tests should fail for unimplemented stubs (NotImplementedError expected)"
        )
        assert result.result["timed_out"] is False

    @pytest.mark.nightly
    @pytest.mark.parametrize("adapter_type", ["bi", "db", "scheduler"])
    def test_non_semantic_scaffold_fails_without_core_package(self, tmp_path, adapter_type):
        """Non-semantic scaffolds produce commented-out base imports; pytest should fail at collection.

        This is expected: bi/db/scheduler core packages are optional and not
        installed in the test environment. The scaffold is structurally correct
        but the generated adapter.py inherits from a base class whose import
        is commented out. The test verifies the runner tool reports the failure
        correctly (not a timeout, not a tool error).
        """
        scaffold_tool = AdapterScaffoldTool(_make_mock_config())
        runner_tool = AdapterTestRunnerTool(_make_mock_config())
        project_dir = str(tmp_path / f"datus-{adapter_type}-testplat")

        scaffold_tool.scaffold_adapter(
            adapter_type=adapter_type,
            platform="testplat",
            output_dir=project_dir,
        )

        result = runner_tool.run_adapter_pytest(
            project_dir=project_dir,
            test_subpath="tests/unit/test_adapter.py",
        )

        assert result.success == 1, f"Tool call itself should succeed: {result.error}"
        # Collection error (exit_code=2) because base class import is commented out
        assert result.result["passed"] is False
        assert result.result["timed_out"] is False
        assert "NameError" in result.result["stdout"] or "ImportError" in result.result["stdout"]

    @pytest.mark.nightly
    def test_runner_rejects_missing_pyproject(self, tmp_path):
        """run_adapter_pytest refuses directories without pyproject.toml."""
        runner_tool = AdapterTestRunnerTool(_make_mock_config())
        project_dir = str(tmp_path / "empty_dir")
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "tests", "unit"), exist_ok=True)

        result = runner_tool.run_adapter_pytest(
            project_dir=project_dir,
            test_subpath="tests/unit",
        )

        assert result.success == 0
        assert "pyproject.toml" in result.error
