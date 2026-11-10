"""Compile literal tool declarations without importing runtime dependencies.

Used by packaging and CI. Running this file directly works in an isolated
setuptools build environment with only Python's standard library installed.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path

FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


class CatalogError(ValueError):
    """Source definitions cannot produce an unambiguous capability inventory."""


def _symbol(node: ast.AST) -> str:
    """Return a dotted symbol without evaluating source expressions."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_symbol(node.value)}.{node.attr}"
    return ""


def _imports(tree: ast.Module, module: str, is_package: bool) -> dict[str, str]:
    """Resolve aliases for decorators, constructors and inherited tool classes."""
    names = {}
    package = module if is_package else module.rpartition(".")[0]
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            target = node.module or ""
            if node.level:
                target = importlib.util.resolve_name("." * node.level + target, package)
            for alias in node.names:
                names[alias.asname or alias.name] = f"{target}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
    return names


def _resolve(symbol: str, imports: dict[str, str], module: str = "") -> str:
    """Expand an imported alias, or qualify a module-local class name."""
    first, _, rest = symbol.partition(".")
    if first in imports:
        return imports[first] + (f".{rest}" if rest else "")
    return f"{module}.{symbol}" if module else symbol


def _decorator(method: ast.AST, name: str, imports: dict[str, str]) -> ast.Call | None:
    """Find declarations even when their import was renamed."""
    for node in method.decorator_list:
        call = node if isinstance(node, ast.Call) else None
        target = _resolve(_symbol(call.func if call else node), imports)
        if target.rsplit(".", 1)[-1] == name:
            if call is None:
                raise CatalogError(f"{name} must be called with explicit arguments")
            return call
    return None


def collect_contracts(package_root: Path) -> dict:
    """Inventory declarations and reject unsupported or bypassed native registration."""
    modules, classes, errors = {}, {}, []
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).with_suffix("")
        parts = [package_root.name, *relative.parts]
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join(parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = _imports(tree, module, path.name == "__init__.py")
        modules[module] = (path, tree, imports)
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef):
                classes[f"{module}.{cls.name}"] = (module, cls)

    def category_for(class_id: str, seen: frozenset[str] = frozenset()) -> str | None:
        """Resolve inherited literal categories without importing tool classes."""
        if class_id not in classes or class_id in seen:
            return None
        module, cls = classes[class_id]
        for node in cls.body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            if any(isinstance(t, ast.Name) and t.id == "permission_category" for t in targets):
                value = ast.literal_eval(node.value)
                if not isinstance(value, str):
                    raise CatalogError(f"Non-literal permission_category on {class_id}")
                return value
        imports = modules[module][2]
        for base in cls.bases:
            category = category_for(_resolve(_symbol(base), imports, module), seen | {class_id})
            if category:
                return category
        return None

    def method_for(class_id: str, name: str, seen: frozenset[str] = frozenset()) -> tuple[str, ast.AST] | None:
        """Find the actual override or inherited implementation, including async methods."""
        if class_id not in classes or class_id in seen:
            return None
        module, cls = classes[class_id]
        for method in cls.body:
            if isinstance(method, FUNCTIONS) and method.name == name:
                return module, method
        imports = modules[module][2]
        for base in cls.bases:
            found = method_for(_resolve(_symbol(base), imports, module), name, seen | {class_id})
            if found:
                return found
        return None

    registered_names: dict[str, set[str]] = {}

    def declared_names(class_id: str, seen: frozenset[str] = frozenset()) -> set[str]:
        """Track inherited registration points so an override cannot shed its contract."""
        if class_id in registered_names:
            return registered_names[class_id]
        if class_id not in classes or class_id in seen:
            return set()
        module, cls = classes[class_id]
        imports = modules[module][2]
        names = {
            method.name
            for method in cls.body
            if isinstance(method, FUNCTIONS) and _decorator(method, "tool_contract", imports)
        }
        for base in cls.bases:
            names |= declared_names(_resolve(_symbol(base), imports, module), seen | {class_id})
        registered_names[class_id] = names
        return names

    tools: dict[str, dict] = {}
    sources: dict[str, str] = {}
    bound_implementations = {
        id(found[1])
        for class_id in classes
        if category_for(class_id) is not None
        for name in declared_names(class_id)
        if (found := method_for(class_id, name)) is not None
    }

    def add(method: ast.AST, module: str, category: str | None) -> None:
        """Compile one implementation and its compatibility aliases."""
        path, _, imports = modules[module]
        declaration = _decorator(method, "tool_contract", imports)
        if declaration is None:
            return
        try:
            if declaration.args or any(k.arg is None for k in declaration.keywords):
                raise CatalogError("tool_contract requires literal keyword arguments")
            values = {k.arg: ast.literal_eval(k.value) for k in declaration.keywords}
            if not {"access", "direct_call"} <= values.keys():
                raise CatalogError("access and direct_call are mandatory")
            if values.keys() - {"access", "direct_call", "category", "companion_of", "aliases"}:
                raise CatalogError("Unknown tool contract argument")
            access, direct = values["access"], values["direct_call"]
            if access not in {"read", "write", "execute", "sql", "restricted"} or type(direct) is not bool:
                raise CatalogError("Invalid access or direct_call")
            family = values.get("category") or category
            if family is None and id(method) in bound_implementations:
                return  # A shared mixin is inventoried through each concrete tool group.
            if values.get("category") is not None and category is not None and values["category"] != category:
                raise CatalogError("Tool category conflicts with its declaring class")
            if not isinstance(family, str) or not family.isidentifier():
                raise CatalogError("A literal or inherited tool category is required")
            aliases = values.get("aliases", ())
            if not isinstance(aliases, tuple) or any(not isinstance(a, str) or not a.isidentifier() for a in aliases):
                raise CatalogError("aliases must be a tuple of literal method names")
            if len(set(aliases)) != len(aliases) or method.name in aliases:
                raise CatalogError("Duplicate alias")
            capability = {"access": access, "direct_call": direct, "companion_of": values.get("companion_of")}
            for name in (method.name, *aliases):
                tool_id = f"{family}.{name}"
                if tool_id in tools and tools[tool_id] != capability:
                    raise CatalogError(f"Conflicting {tool_id}: {sources[tool_id]}")
                tools[tool_id] = capability
                sources[tool_id] = f"{path}:{method.lineno}"
        except (ValueError, TypeError) as exc:
            errors.append(f"{path}:{method.lineno}: {exc}")

    def add_literal_definition(node: ast.Call, module: str) -> None:
        """Inventory explicit dynamic factories instead of relying on node execution."""
        path, _, imports = modules[module]
        try:
            if len(node.args) > 3 or any(k.arg not in {"tool_id", "capability", "aliases"} for k in node.keywords):
                raise CatalogError("Invalid ToolDefinition arguments")
            fields = dict(zip(("tool_id", "capability", "aliases"), node.args, strict=False))
            fields.update({k.arg: k.value for k in node.keywords})
            identity = ast.literal_eval(fields["tool_id"])
            family, name = identity.split(".")
            capability = fields["capability"]
            if not isinstance(capability, ast.Call) or not _resolve(_symbol(capability.func), imports).endswith(
                ".ToolCapability"
            ):
                raise CatalogError("Dynamic definitions require a literal ToolCapability")
            values = dict(zip(("access", "direct_call", "companion_of"), capability.args, strict=False))
            if len(capability.args) > 3:
                raise CatalogError("Invalid ToolCapability arguments")
            values.update({k.arg: k.value for k in capability.keywords})
            if not {"access", "direct_call"} <= values.keys():
                raise CatalogError("Dynamic definitions require explicit access and direct_call")
            keywords = [ast.keyword(arg=key, value=value) for key, value in values.items()]
            keywords.append(ast.keyword(arg="category", value=ast.Constant(value=family)))
            if "aliases" in fields:
                keywords.append(ast.keyword(arg="aliases", value=fields["aliases"]))
            method = ast.FunctionDef(
                name=name,
                args=ast.arguments(
                    posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
                ),
                body=[],
                decorator_list=[ast.Call(func=ast.Name(id="tool_contract"), args=[], keywords=keywords)],
                returns=None,
                type_comment=None,
                type_params=[],
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
            add(method, module, family)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(f"{path}:{node.lineno}: Non-literal dynamic tool definition: {exc}")

    for module, (path, tree, imports) in modules.items():
        top_level_methods = {id(node) for node in tree.body if isinstance(node, FUNCTIONS)}
        top_level_methods.update(
            id(method)
            for cls in tree.body
            if isinstance(cls, ast.ClassDef)
            for method in cls.body
            if isinstance(method, FUNCTIONS)
        )
        for method in ast.walk(tree):
            if isinstance(method, FUNCTIONS) and id(method) not in top_level_methods:
                if _decorator(method, "tool_contract", imports):
                    add(method, module, None)
        for node in tree.body:
            if isinstance(node, FUNCTIONS):
                add(node, module, None)
            if not isinstance(node, ast.ClassDef):
                continue
            class_id = f"{module}.{node.name}"
            category = category_for(class_id)
            for name in declared_names(class_id):
                found = method_for(class_id, name)
                if found:
                    if _decorator(found[1], "tool_contract", modules[found[0]][2]) is None:
                        errors.append(f"{path}:{found[1].lineno}: Missing @tool_contract on override {class_id}.{name}")
                    else:
                        add(found[1], found[0], category)
            for method in node.body:
                if not isinstance(method, FUNCTIONS):
                    continue
                add(method, module, category)
                candidates = set()
                if _decorator(method, "mcp_tool", imports):
                    candidates.add(method.name)
                called_attributes = {id(call.func) for call in ast.walk(method) if isinstance(call, ast.Call)}
                for attr in ast.walk(method):
                    if method.name in {"available_tools", "all_tools_name"}:
                        if (
                            isinstance(attr, ast.Attribute)
                            and isinstance(attr.value, ast.Name)
                            and attr.value.id == "self"
                            and id(attr) not in called_attributes
                        ):
                            candidates.add(attr.attr)
                        if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
                            candidates.add(attr.value)
                    if isinstance(attr, ast.Call) and attr.args:
                        callee = _resolve(_symbol(attr.func), imports).rsplit(".", 1)[-1]
                        argument = attr.args[0]
                        if callee in {"trans_to_function_tool", "to_function_tool", "definition_from_callable"}:
                            if (
                                isinstance(argument, ast.Attribute)
                                and isinstance(argument.value, ast.Name)
                                and argument.value.id == "self"
                            ):
                                candidates.add(argument.attr)
                for name in candidates:
                    if name.startswith("_") or name in {"available_tools", "all_tools_name", "to_function_tool"}:
                        continue
                    found = method_for(class_id, name)
                    if found and _decorator(found[1], "tool_contract", modules[found[0]][2]) is None:
                        errors.append(f"{path}:{method.lineno}: Missing @tool_contract on {class_id}.{name}")

        for call in ast.walk(tree):
            if isinstance(call, FUNCTIONS):
                for decorator in call.decorator_list:
                    function = decorator.func if isinstance(decorator, ast.Call) else decorator
                    if _resolve(_symbol(function), imports) == "agents.function_tool":
                        errors.append(f"{path}:{call.lineno}: Use trans_to_function_tool instead of SDK decorators")
            if not isinstance(call, ast.Call):
                continue
            target = _resolve(_symbol(call.func), imports)
            if target.endswith(".ToolDefinition") and not module.endswith(".registry.contracts"):
                add_literal_definition(call, module)
            if target.endswith(".registered_function_tool"):
                argument = next((k.value for k in call.keywords if k.arg == "definition"), None)
                if argument is None:
                    errors.append(f"{path}:{call.lineno}: Dynamic factories must supply definition explicitly")
                elif not module.endswith(".func_tool.base"):
                    declaration = _resolve(_symbol(argument.func), imports) if isinstance(argument, ast.Call) else ""
                    if not declaration.endswith((".definition_from_callable", ".ToolDefinition")):
                        errors.append(
                            f"{path}:{call.lineno}: Dynamic definitions must be literal or derived from a declared callable"
                        )
            if target in {"agents.FunctionTool", "agents.tool.FunctionTool", "agents.function_tool"}:
                allowed = (module.endswith(".registry.contracts") and target.endswith(".FunctionTool")) or (
                    module.endswith(".func_tool.base") and target.endswith(".function_tool")
                )
                if not allowed:
                    errors.append(f"{path}:{call.lineno}: Use registered_function_tool or clone_registered_tool")

    for tool_id, capability in tools.items():
        owner = capability["companion_of"]
        if owner is not None:
            if not isinstance(owner, str) or owner not in tools:
                errors.append(f"{tool_id}: Unknown paging owner {owner!r}")
            elif tools[owner]["companion_of"] is not None or tools[owner]["access"] != capability["access"]:
                errors.append(f"{tool_id}: Paging owner must have the same access and cannot be a companion")
    if errors:
        raise CatalogError("\n".join(sorted(set(errors))))
    return {"schema_version": 1, "tools": dict(sorted(tools.items()))}


def write_catalog(package_root: Path, *, check: bool = False) -> None:
    """Generate deterministic package data, or fail CI when it is stale."""
    target = package_root / "tools" / "registry" / "capabilities.json"
    expected = json.dumps(collect_contracts(package_root), indent=2, sort_keys=True) + "\n"
    if check:
        if not target.is_file() or target.read_text(encoding="utf-8") != expected:
            raise CatalogError("Tool catalog is stale; run python datus/tools/registry/catalog_builder.py")
    else:
        target.write_text(expected, encoding="utf-8")


def main() -> None:
    """Expose the same dependency-free compiler to developers, packaging and CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write_catalog(Path(__file__).resolve().parents[2], check=args.check)


if __name__ == "__main__":
    main()
