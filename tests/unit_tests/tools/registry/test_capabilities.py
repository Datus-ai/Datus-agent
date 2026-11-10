"""A new tool cannot silently ship without a restricted-host access contract."""

from datus.tools.registry.capabilities import TOOL_CAPABILITIES, tool_capability


def test_every_configurable_agent_tool_declares_a_capability():
    from datus.api.services.agent_service import VALID_TOOL_METHODS

    missing = [
        f"{category}.{name}"
        for category, names in VALID_TOOL_METHODS.items()
        for name in names
        if tool_capability(f"{category}.{name}") is None
    ]
    assert not missing, f"Declare capabilities for new tools: {sorted(missing)}"


def test_every_mcp_tool_declares_a_capability():
    import datus.mcp_server  # noqa: F401
    from datus.utils.mcp_decorators import get_mcp_tools, get_tool_registry

    missing = [
        f"{cfg.tool_class.permission_category}.{name}"
        for cfg in get_tool_registry()
        for name, _, _ in get_mcp_tools(cfg.tool_class)
        if tool_capability(f"{cfg.tool_class.permission_category}.{name}") is None
    ]
    assert not missing, f"Declare capabilities for new MCP tools: {sorted(missing)}"


def test_companions_have_existing_owners_and_no_cycles():
    for tool_id, capability in TOOL_CAPABILITIES.items():
        if capability.companion_of:
            owner = TOOL_CAPABILITIES[capability.companion_of]
            assert owner.companion_of is None, tool_id
            assert owner.access == capability.access, tool_id


def test_unknown_tools_are_not_inferred_from_their_names():
    assert tool_capability("semantic_tools.read_and_publish_something") is None
    assert tool_capability("new_tools.read") is None
    assert tool_capability("bash_tools.new_command").access == "restricted"


def test_agent_only_bound_tools_declare_capabilities():
    """Cover non-MCP tools assembled from bound methods, without instantiation.

    Structural discovery complements the live configurable/MCP registries and
    does not require credentials, adapters or a database. Private feature checks
    and wrapper construction are framework helpers, not exposed tool methods.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[4] / "datus" / "tools"
    missing = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            category = None
            for node in cls.body:
                if isinstance(node, ast.AnnAssign):
                    targets, value = [node.target], node.value
                elif isinstance(node, ast.Assign):
                    targets, value = node.targets, node.value
                else:
                    continue
                if any(isinstance(target, ast.Name) and target.id == "permission_category" for target in targets):
                    category = ast.literal_eval(value)
            if category is None:
                continue
            for method in cls.body:
                if not isinstance(method, ast.FunctionDef) or method.name != "available_tools":
                    continue
                for attr in ast.walk(method):
                    if not isinstance(attr, ast.Attribute) or not isinstance(attr.value, ast.Name):
                        continue
                    if attr.value.id != "self" or attr.attr.startswith("_") or attr.attr == "to_function_tool":
                        continue
                    if not any(isinstance(n, ast.FunctionDef) and n.name == attr.attr for n in cls.body):
                        continue
                    tool_id = f"{category}.{attr.attr}"
                    if tool_capability(tool_id) is None:
                        missing.append(tool_id)
    assert not missing, f"Declare capabilities for agent-only tools: {sorted(set(missing))}"
