"""Required tool declarations shared by registration and restricted hosts.

Declarations describe a tool; they never grant permission to invoke it. Keep
this module dependency-light so contracts do not initialize tool implementations.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeVar

Access = Literal["read", "write", "execute", "sql", "restricted"]
_ACCESS = frozenset({"read", "write", "execute", "sql", "restricted"})
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
_Method = TypeVar("_Method", bound=Callable)


class ToolDefinitionError(ValueError):
    """A tool cannot be registered with a missing or inconsistent contract."""


@dataclass(frozen=True)
class ToolCapability:
    """Effect and direct-call eligibility, independently of host authorization."""

    access: Access
    direct_call: bool = True
    companion_of: str | None = None

    def __post_init__(self) -> None:
        """Reject malformed contract values at their declaration site."""
        if not isinstance(self.access, str) or self.access not in _ACCESS or type(self.direct_call) is not bool:
            raise ToolDefinitionError("Tool contracts require a valid access and boolean direct_call")
        if self.companion_of is not None:
            validate_tool_id(self.companion_of)


def validate_tool_id(tool_id: str) -> None:
    """Reject ambiguous identities rather than interpreting them as grant patterns."""
    parts = tool_id.split(".") if isinstance(tool_id, str) else []
    if len(parts) != 2 or not all(_IDENTIFIER.fullmatch(part) for part in parts):
        raise ToolDefinitionError(f"Expected a canonical family.method identity, got {tool_id!r}")


@dataclass(frozen=True)
class ToolDefinition:
    """Canonical identity carried with an SDK tool through all wrappers."""

    tool_id: str
    capability: ToolCapability
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate canonical and compatibility identities before registration."""
        validate_tool_id(self.tool_id)
        if not isinstance(self.capability, ToolCapability) or not isinstance(self.aliases, tuple):
            raise ToolDefinitionError("Tool definitions require a capability and an immutable alias tuple")
        for alias in self.aliases:
            if not isinstance(alias, str):
                raise ToolDefinitionError("Tool aliases must be method names")
            validate_tool_id(f"{self.category}.{alias}")
        if len(set(self.aliases)) != len(self.aliases) or self.name in self.aliases:
            raise ToolDefinitionError(f"Duplicate aliases for {self.tool_id}")

    @property
    def category(self) -> str:
        return self.tool_id.split(".", 1)[0]

    @property
    def name(self) -> str:
        return self.tool_id.split(".", 1)[1]


@dataclass(frozen=True)
class _Declaration:
    capability: ToolCapability
    category: str | None
    aliases: tuple[str, ...]


def tool_contract(
    *,
    access: Access,
    direct_call: bool,
    category: str | None = None,
    companion_of: str | None = None,
    aliases: tuple[str, ...] = (),
) -> Callable[[_Method], _Method]:
    """Declare mandatory registration metadata without wrapping sync/async methods.

    A bound method normally derives its family from ``permission_category``.
    Node-owned methods and standalone functions must declare ``category``.
    Native declarations use literal arguments so the build can inventory them
    without importing optional adapters, credentials or database factories.
    """
    declaration = _Declaration(ToolCapability(access, direct_call, companion_of), category, aliases)
    if not isinstance(aliases, tuple) or any(
        not isinstance(alias, str) or not _IDENTIFIER.fullmatch(alias) for alias in aliases
    ):
        raise ToolDefinitionError("Tool aliases must be an immutable tuple of method names")
    if category is not None:
        validate_tool_id(f"{category}.placeholder")

    def declare(method: _Method) -> _Method:
        """Attach one immutable declaration to the implementation itself."""
        if hasattr(method, "__datus_tool_contract__"):
            raise ToolDefinitionError(f"Duplicate tool contract on {method.__qualname__}")
        method.__datus_tool_contract__ = declaration
        return method

    return declare


def definition_from_callable(method: Callable, *, owner: type | None = None) -> ToolDefinition:
    """Require a declaration on the actual implementation, including overrides."""
    declaration = getattr(method, "__datus_tool_contract__", None)
    if not isinstance(declaration, _Declaration):
        raise ToolDefinitionError(f"Missing @tool_contract on {getattr(method, '__qualname__', type(method).__name__)}")
    bound_owner = getattr(method, "__self__", None)
    owner_category = getattr(owner or bound_owner, "permission_category", None)
    if declaration.category is not None and isinstance(owner_category, str) and declaration.category != owner_category:
        raise ToolDefinitionError(f"Tool category conflicts with its declaring class: {method.__qualname__}")
    category = declaration.category or owner_category
    if not isinstance(category, str):
        raise ToolDefinitionError(f"Missing tool category on {method.__qualname__}")
    definition = ToolDefinition(f"{category}.{method.__name__}", declaration.capability, declaration.aliases)
    if getattr(method, "__module__", "").startswith("datus."):
        from datus.tools.registry.capabilities import TOOL_CAPABILITIES

        identities = [definition.tool_id, *(f"{category}.{alias}" for alias in definition.aliases)]
        if any(TOOL_CAPABILITIES.get(identity) != definition.capability for identity in identities):
            raise ToolDefinitionError(f"Stale native tool catalog for {definition.tool_id}; rebuild the tool catalog")
    return definition


def register_tool_definition(tool: Any, definition: ToolDefinition) -> Any:
    """Attach a verified definition; a name alone is never registration evidence."""
    if not isinstance(definition, ToolDefinition):
        raise ToolDefinitionError("Tool registration requires a ToolDefinition")
    if getattr(tool, "name", None) not in (definition.name, *definition.aliases):
        raise ToolDefinitionError(f"Tool name does not match {definition.tool_id}")
    from datus.tools.registry.capabilities import TOOL_CAPABILITIES

    existing = TOOL_CAPABILITIES.get(definition.tool_id)
    if existing is not None and existing != definition.capability:
        raise ToolDefinitionError(f"Conflicting capability for {definition.tool_id}")
    tool.__datus_tool_definition__ = definition
    return tool


def tool_definition(tool: Any) -> ToolDefinition:
    """Read registration evidence and reject a renamed or undeclared SDK tool."""
    definition = getattr(tool, "__datus_tool_definition__", None)
    if not isinstance(definition, ToolDefinition):
        raise ToolDefinitionError(f"Unregistered tool {getattr(tool, 'name', type(tool).__name__)!r}")
    if getattr(tool, "name", None) not in (definition.name, *definition.aliases):
        raise ToolDefinitionError(f"Tool name no longer matches {definition.tool_id}")
    return definition


def registered_function_tool(*, definition: ToolDefinition, **kwargs: Any) -> Any:
    """Construct dynamic SDK tools through the same mandatory contract boundary."""
    from agents import FunctionTool

    return register_tool_definition(FunctionTool(**kwargs), definition)


def clone_registered_tool(original: Any, **changes: Any) -> Any:
    """Preserve all SDK fields and the contract through proxies and transformers."""
    definition = tool_definition(original)
    fields = {field.name: getattr(original, field.name) for field in dataclasses.fields(original) if field.init}
    fields.update(changes)
    return register_tool_definition(type(original)(**fields), definition)


def validate_registered_tools(tools: list[Any]) -> None:
    """Check final native SDK tools before visibility filters can hide mistakes."""
    from agents import FunctionTool

    identities: dict[str, str] = {}
    for tool in tools:
        if not isinstance(tool, FunctionTool):
            continue  # Provider-hosted tools are not native function registrations.
        definition = tool_definition(tool)
        previous = identities.setdefault(tool.name, definition.tool_id)
        if previous != definition.tool_id:
            raise ToolDefinitionError(f"Ambiguous tool name {tool.name!r}: {previous} and {definition.tool_id}")
