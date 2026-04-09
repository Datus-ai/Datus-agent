# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for datus/tools/func_tool/adapter_scaffold_tool.py (TDD — written before implementation).

Tests cover:
- list_adapter_types() returns all 4 adapter types
- scaffold_adapter() generates correct directory structure and files
- Generated Python files have valid syntax
- Generated pyproject.toml has correct entry-point config
- validate_adapter() detects stub vs implemented adapters
"""

import os
import sys

import pytest

from datus.tools.func_tool.adapter_scaffold_tool import AdapterScaffoldTool
from datus.tools.func_tool.base import FuncToolResult


@pytest.fixture
def tool():
    """Create an AdapterScaffoldTool instance (agent_config not needed for scaffolding)."""
    return AdapterScaffoldTool(agent_config=None)


@pytest.fixture
def tmp_output(tmp_path):
    """Provide a temporary output directory."""
    return str(tmp_path / "output")


class TestListAdapterTypes:
    def test_returns_success(self, tool):
        result = tool.list_adapter_types()
        assert isinstance(result, FuncToolResult)
        assert result.success == 1

    def test_returns_four_types(self, tool):
        result = tool.list_adapter_types()
        types = result.result
        assert isinstance(types, list), f"Expected list, got {type(types)}"
        type_names = [t["type"] for t in types]
        assert set(type_names) == {"semantic", "bi", "db", "scheduler"}

    def test_each_type_has_description(self, tool):
        result = tool.list_adapter_types()
        types = result.result
        assert isinstance(types, list)
        assert len(types) == 4
        for t in types:
            assert isinstance(t, dict)
            assert "type" in t
            assert "base_class" in t
            assert "entry_point_group" in t


class TestScaffoldSemanticAdapter:
    def test_returns_success(self, tool, tmp_output):
        result = tool.scaffold_adapter("semantic", "cube", tmp_output)
        assert isinstance(result, FuncToolResult)
        assert result.success == 1

    def test_creates_package_directory(self, tool, tmp_output):
        tool.scaffold_adapter("semantic", "cube", tmp_output)
        pkg_dir = os.path.join(tmp_output, "datus_semantic_cube")
        assert os.path.isdir(pkg_dir)

    def test_creates_init_py(self, tool, tmp_output):
        tool.scaffold_adapter("semantic", "cube", tmp_output)
        init_file = os.path.join(tmp_output, "datus_semantic_cube", "__init__.py")
        assert os.path.isfile(init_file)

    def test_creates_adapter_py(self, tool, tmp_output):
        tool.scaffold_adapter("semantic", "cube", tmp_output)
        adapter_file = os.path.join(tmp_output, "datus_semantic_cube", "adapter.py")
        assert os.path.isfile(adapter_file)

    def test_creates_config_py(self, tool, tmp_output):
        tool.scaffold_adapter("semantic", "cube", tmp_output)
        config_file = os.path.join(tmp_output, "datus_semantic_cube", "config.py")
        assert os.path.isfile(config_file)

    def test_creates_pyproject_toml(self, tool, tmp_output):
        tool.scaffold_adapter("semantic", "cube", tmp_output)
        pyproject = os.path.join(tmp_output, "pyproject.toml")
        assert os.path.isfile(pyproject)

    def test_creates_test_directory(self, tool, tmp_output):
        tool.scaffold_adapter("semantic", "cube", tmp_output)
        test_dir = os.path.join(tmp_output, "tests", "unit")
        assert os.path.isdir(test_dir)

    def test_creates_test_adapter_py(self, tool, tmp_output):
        tool.scaffold_adapter("semantic", "cube", tmp_output)
        test_file = os.path.join(tmp_output, "tests", "unit", "test_adapter.py")
        assert os.path.isfile(test_file)

    def test_result_contains_files_created(self, tool, tmp_output):
        result = tool.scaffold_adapter("semantic", "cube", tmp_output)
        assert "files_created" in result.result
        assert len(result.result["files_created"]) > 0


class TestScaffoldBiAdapter:
    def test_creates_bi_adapter_structure(self, tool, tmp_output):
        result = tool.scaffold_adapter("bi", "metabase", tmp_output)
        assert result.success == 1
        pkg_dir = os.path.join(tmp_output, "datus_bi_metabase")
        assert os.path.isdir(pkg_dir)
        assert os.path.isfile(os.path.join(pkg_dir, "__init__.py"))
        assert os.path.isfile(os.path.join(pkg_dir, "adapter.py"))
        assert os.path.isfile(os.path.join(pkg_dir, "config.py"))


class TestScaffoldDbAdapter:
    def test_creates_db_adapter_structure(self, tool, tmp_output):
        result = tool.scaffold_adapter("db", "clickhouse", tmp_output)
        assert result.success == 1
        pkg_dir = os.path.join(tmp_output, "datus_clickhouse")
        assert os.path.isdir(pkg_dir)


class TestScaffoldSchedulerAdapter:
    def test_creates_scheduler_adapter_structure(self, tool, tmp_output):
        result = tool.scaffold_adapter("scheduler", "dagster", tmp_output)
        assert result.success == 1
        pkg_dir = os.path.join(tmp_output, "datus_scheduler_dagster")
        assert os.path.isdir(pkg_dir)


class TestScaffoldFilesSyntaxValid:
    def test_all_python_files_compile(self, tool, tmp_output):
        """All generated .py files should have valid Python syntax."""
        tool.scaffold_adapter("semantic", "cube", tmp_output)
        py_files = []
        for root, dirs, files in os.walk(tmp_output):
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))

        assert len(py_files) >= 4  # at least __init__.py, adapter.py, config.py, test_adapter.py

        for py_file in py_files:
            with open(py_file) as fh:
                source = fh.read()
            try:
                compile(source, py_file, "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file}: {e}")


class TestScaffoldPyprojectEntryPoint:
    def test_semantic_entry_point(self, tool, tmp_output):
        tool.scaffold_adapter("semantic", "cube", tmp_output)
        pyproject = os.path.join(tmp_output, "pyproject.toml")
        with open(pyproject) as f:
            content = f.read()
        assert "datus.semantic_adapters" in content
        assert "cube" in content
        assert "datus_semantic_cube:register" in content

    def test_bi_entry_point(self, tool, tmp_output):
        tool.scaffold_adapter("bi", "metabase", tmp_output)
        pyproject = os.path.join(tmp_output, "pyproject.toml")
        with open(pyproject) as f:
            content = f.read()
        assert "datus.bi_adapters" in content
        assert "metabase" in content

    def test_db_entry_point(self, tool, tmp_output):
        tool.scaffold_adapter("db", "clickhouse", tmp_output)
        pyproject = os.path.join(tmp_output, "pyproject.toml")
        with open(pyproject) as f:
            content = f.read()
        assert "datus.adapters" in content
        assert "clickhouse" in content

    def test_scheduler_entry_point(self, tool, tmp_output):
        tool.scaffold_adapter("scheduler", "dagster", tmp_output)
        pyproject = os.path.join(tmp_output, "pyproject.toml")
        with open(pyproject) as f:
            content = f.read()
        assert "datus.schedulers" in content
        assert "dagster" in content


class TestScaffoldInitRegisterFunction:
    def test_init_contains_register_function(self, tool, tmp_output):
        """Generated __init__.py should define a register() function."""
        tool.scaffold_adapter("semantic", "cube", tmp_output)
        init_file = os.path.join(tmp_output, "datus_semantic_cube", "__init__.py")
        with open(init_file) as f:
            content = f.read()
        assert "def register()" in content


class TestScaffoldInvalidType:
    def test_invalid_adapter_type(self, tool, tmp_output):
        result = tool.scaffold_adapter("invalid_type", "cube", tmp_output)
        assert result.success == 0
        assert "invalid_type" in result.error


class TestScaffoldIdempotent:
    def test_scaffold_twice_no_error(self, tool, tmp_output):
        """Running scaffold twice on the same output dir should not raise."""
        result1 = tool.scaffold_adapter("semantic", "cube", tmp_output)
        result2 = tool.scaffold_adapter("semantic", "cube", tmp_output)
        assert result1.success == 1
        assert result2.success == 1
        assert len(result1.result["files_created"]) == len(result2.result["files_created"])
        adapter_file = os.path.join(tmp_output, "datus_semantic_cube", "adapter.py")
        with open(adapter_file) as f:
            content = f.read()
        assert content.count("class CubeAdapter") == 1


class TestScaffoldDisplayName:
    def test_custom_display_name(self, tool, tmp_output):
        result = tool.scaffold_adapter("semantic", "cube", tmp_output, display_name="Cube.js")
        assert result.success == 1
        # Display name should appear in generated files
        init_file = os.path.join(tmp_output, "datus_semantic_cube", "__init__.py")
        with open(init_file) as f:
            content = f.read()
        assert "Cube.js" in content


class TestValidateAdapter:
    def test_validate_stub_adapter_fails(self, tool, tmp_output):
        """A freshly scaffolded adapter (all NotImplementedError) should fail validation."""
        tool.scaffold_adapter("semantic", "test_stub", tmp_output)

        # Add the output to sys.path so the module can be imported
        sys.path.insert(0, tmp_output)
        try:
            result = tool.validate_adapter("datus_semantic_test_stub")
            assert result.success == 1
            assert result.result["valid"] is False
            assert len(result.result["issues"]) > 0
        finally:
            sys.path.pop(0)
            # Clean up imported modules
            for mod_name in list(sys.modules.keys()):
                if "datus_semantic_test_stub" in mod_name:
                    del sys.modules[mod_name]


class TestValidateAdapterErrors:
    def test_validate_adapter_import_error(self, tool):
        result = tool.validate_adapter("nonexistent_module_xyz_abc")
        assert result.success == 0
        assert result.error is not None


class TestToPascalCase:
    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("cube", "Cube"),
            ("click_house", "ClickHouse"),
            ("dbt_cloud", "DbtCloud"),
            ("a", "A"),
        ],
    )
    def test_pascal_case_conversion(self, input_val, expected):
        from datus.tools.func_tool.adapter_scaffold_tool import _to_pascal_case

        assert _to_pascal_case(input_val) == expected


class TestScaffoldErrors:
    def test_scaffold_adapter_filesystem_error(self, tool):
        from unittest.mock import patch

        with patch("os.makedirs", side_effect=PermissionError("denied")):
            result = tool.scaffold_adapter("semantic", "cube", "/tmp/test_perm_error")
        assert result.success == 0


class TestScaffoldAllTypesStructure:
    @pytest.mark.parametrize(
        "adapter_type,platform,expected_pkg",
        [
            ("semantic", "cube", "datus_semantic_cube"),
            ("bi", "metabase", "datus_bi_metabase"),
            ("db", "clickhouse", "datus_clickhouse"),
            ("scheduler", "dagster", "datus_scheduler_dagster"),
        ],
    )
    def test_creates_full_structure(self, tool, tmp_output, adapter_type, platform, expected_pkg):
        result = tool.scaffold_adapter(adapter_type, platform, tmp_output)
        assert result.success == 1
        pkg_dir = os.path.join(tmp_output, expected_pkg)
        assert os.path.isdir(pkg_dir)
        assert os.path.isfile(os.path.join(pkg_dir, "__init__.py"))
        assert os.path.isfile(os.path.join(pkg_dir, "adapter.py"))
        assert os.path.isfile(os.path.join(pkg_dir, "config.py"))
        assert os.path.isfile(os.path.join(tmp_output, "pyproject.toml"))


class TestAvailableTools:
    def test_available_tools_returns_three_tools(self, tool):
        tools = tool.available_tools()
        assert len(tools) == 3
        tool_names = {t.name for t in tools}
        assert tool_names == {"scaffold_adapter", "validate_adapter", "list_adapter_types"}
