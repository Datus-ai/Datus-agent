"""Build-time inventory checks run without importing tool implementations."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from datus.tools.registry.catalog_builder import CatalogError, collect_contracts, write_catalog


@pytest.fixture
def package(tmp_path):
    """Create an import-free source package for compiler behavior tests."""
    root = tmp_path / "datus"
    (root / "tools" / "registry").mkdir(parents=True)
    return root


def source(package, text, name="tools.py"):
    """Write source that the compiler must parse without executing it."""
    (package / name).write_text(text, encoding="utf-8")


def test_async_inherited_contracts_and_aliases(package):
    """Inventory inherited methods across modules, even with unavailable dependencies."""
    source(
        package,
        """
import missing_optional_adapter
raise RuntimeError("Do not import this module")
from datus.tools.registry.contracts import tool_contract as contract
class Parent:
    permission_category = "filesystem_tools"
    @contract(access="read", direct_call=True, aliases=("legacy_read",))
    async def read_file(self): pass
    def available_tools(self): return [trans_to_function_tool(self.read_file)]
""",
        "parent.py",
    )
    source(
        package,
        """
from .parent import Parent as Base
class Child(Base):
    permission_category = "other_tools"
""",
    )
    result = collect_contracts(package)["tools"]
    assert set(result) == {
        "filesystem_tools.read_file",
        "filesystem_tools.legacy_read",
        "other_tools.read_file",
        "other_tools.legacy_read",
    }
    assert all(value == {"access": "read", "direct_call": True, "companion_of": None} for value in result.values())


@pytest.mark.parametrize("registration", ["return [trans_to_function_tool(self.read_file)]", 'return ["read_file"]'])
def test_missing_contract_is_a_build_error(package, registration):
    """Both bound-method assembly and advertised method inventories require contracts."""
    source(
        package,
        f"""
class Tool:
    permission_category = "filesystem_tools"
    async def read_file(self): pass
    def available_tools(self): {registration}
""",
    )
    with pytest.raises(CatalogError, match="Missing @tool_contract"):
        collect_contracts(package)


def test_override_cannot_inherit_a_different_implementation_contract(package):
    """Overriding an inherited tool requires redeclaring its effect."""
    source(
        package,
        """
from datus.tools.registry.contracts import tool_contract
class Parent:
    permission_category = "filesystem_tools"
    @tool_contract(access="read", direct_call=True)
    async def read_file(self): pass
class Child(Parent):
    async def read_file(self): pass
""",
    )
    with pytest.raises(CatalogError, match="override"):
        collect_contracts(package)


def test_shared_mixin_binds_to_each_concrete_family(package):
    """Shared implementations derive identities from their concrete registering owners."""
    source(
        package,
        """
from datus.tools.registry.contracts import tool_contract
class Shared:
    @tool_contract(access="restricted", direct_call=False)
    def save(self): pass
class Report(Shared):
    permission_category = "visual_report_tools"
class Dashboard(Shared):
    permission_category = "visual_dashboard_tools"
""",
    )
    assert set(collect_contracts(package)["tools"]) == {"visual_report_tools.save", "visual_dashboard_tools.save"}


@pytest.mark.parametrize(
    "arguments",
    [
        'access="read"',
        "direct_call=True",
        "access=calculate_access(), direct_call=True",
        'access="session", direct_call=True',
        'access="session", direct_call=False, companion_of="tools.ask_user"',
    ],
)
def test_required_metadata_is_literal_and_complete(package, arguments):
    """Build-time effects cannot depend on runtime state or implicit defaults."""
    source(
        package,
        f"""
from datus.tools.registry.contracts import tool_contract
class Tool:
    permission_category = "filesystem_tools"
    @tool_contract({arguments})
    def read_file(self): pass
""",
    )
    with pytest.raises(CatalogError):
        collect_contracts(package)


@pytest.mark.parametrize(
    "registration",
    [
        "from agents import FunctionTool as Raw\ntool = Raw(name='read_file')",
        "from dataclasses import replace\ntool = replace(original, on_invoke_tool=invoke)",
        "from agents import function_tool as raw\n@raw\ndef read_file(): pass",
        "import agents as sdk\ntool = sdk.function_tool(lambda: 1)",
        "from datus.tools.registry.contracts import registered_function_tool\ntool = registered_function_tool(name='read_file')",
    ],
)
def test_registration_bypasses_fail_during_build(package, registration):
    """SDK aliases and undeclared dynamic factories cannot bypass build validation."""
    source(package, registration)
    with pytest.raises(
        CatalogError,
        match="registered_function_tool|definition explicitly|trans_to_function_tool|clone_registered_tool",
    ):
        collect_contracts(package)


def test_feature_checks_are_not_tool_registrations(package):
    """Calling an assembly helper does not make that helper a registered tool."""
    source(
        package,
        """
from datus.tools.registry.contracts import tool_contract
class Tool:
    permission_category = "filesystem_tools"
    def has_feature(self): return True
    @tool_contract(access="read", direct_call=True)
    def read_file(self): pass
    def available_tools(self):
        if self.has_feature():
            return [trans_to_function_tool(self.read_file)]
        return []
""",
    )
    assert set(collect_contracts(package)["tools"]) == {"filesystem_tools.read_file"}


def test_conflicting_identities_and_invalid_companions_fail(package):
    """One canonical identity cannot have contradictory effect declarations."""
    source(
        package,
        """
from datus.tools.registry.contracts import tool_contract
class First:
    permission_category = "filesystem_tools"
    @tool_contract(access="read", direct_call=True)
    def read_file(self): pass
class Second:
    permission_category = "filesystem_tools"
    @tool_contract(access="write", direct_call=False)
    def read_file(self): pass
""",
    )
    with pytest.raises(CatalogError, match="Conflicting"):
        collect_contracts(package)
    source(
        package,
        """
from datus.tools.registry.contracts import tool_contract
class Tool:
    permission_category = "semantic_tools"
    @tool_contract(access="read", direct_call=True, companion_of="semantic_tools.missing")
    def read_result(self): pass
""",
    )
    with pytest.raises(CatalogError, match="Unknown paging owner"):
        collect_contracts(package)


def test_generation_and_stale_artifact_detection(package):
    """Packaging generates deterministic data and CI rejects stale tracked data."""
    source(
        package,
        """
from datus.tools.registry.contracts import tool_contract
class Tool:
    permission_category = "filesystem_tools"
    @tool_contract(access="read", direct_call=True)
    def read_file(self): pass
""",
    )
    with pytest.raises(CatalogError, match="stale"):
        write_catalog(package, check=True)
    write_catalog(package)
    artifact = package / "tools/registry/capabilities.json"
    first = artifact.read_bytes()
    write_catalog(package)
    assert artifact.read_bytes() == first
    write_catalog(package, check=True)
    assert json.loads(first)["tools"]["filesystem_tools.read_file"]["access"] == "read"
    artifact.write_text("{}")
    with pytest.raises(CatalogError, match="stale"):
        write_catalog(package, check=True)


def test_explicit_and_nested_dynamic_factories_are_inventoried(package):
    """Both supported dynamic factory forms contribute to the full pre-runtime catalog."""
    source(
        package,
        """
from datus.tools.registry.contracts import tool_contract, ToolDefinition, ToolCapability
from datus.tools.registry.contracts import registered_function_tool, definition_from_callable
def literal_factory():
    return registered_function_tool(
        definition=ToolDefinition("tools.dynamic_literal", access="restricted", direct_call=False),
        name="dynamic_literal",
    )
def nested_factory():
    @tool_contract(access="read", direct_call=False, category="future_tools")
    async def dynamic_reader(): pass
    return registered_function_tool(definition=definition_from_callable(dynamic_reader), name="dynamic_reader")
""",
    )
    result = collect_contracts(package)["tools"]
    assert result == {
        "tools.dynamic_literal": {"access": "restricted", "direct_call": False, "companion_of": None},
        "future_tools.dynamic_reader": {"access": "read", "direct_call": False, "companion_of": None},
    }


def test_compiler_runs_without_site_packages():
    """Isolated package builds never import database, MCP or SDK dependencies."""
    compiler = Path(__file__).resolve().parents[4] / "datus/tools/registry/catalog_builder.py"
    result = subprocess.run([sys.executable, "-I", "-S", str(compiler), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
