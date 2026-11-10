# Tool registration contracts

Native tools declare their effect beside the implementation. There is no second
hand-maintained method list in `capabilities.py`. A declaration is mandatory for
both Agent conversion and MCP registration, including tools in restricted
families. It describes the implementation; it does not grant permission.

```python
from datus.tools.registry.contracts import tool_contract
from datus.utils.mcp_decorators import mcp_tool

class FilesystemTools:
    permission_category = "filesystem_tools"

    @mcp_tool()
    @tool_contract(access="read", direct_call=True)
    def read_file(self, path: str):
        ...

    @tool_contract(access="write", direct_call=False)
    def write_file(self, path: str, content: str):
        ...
```

Both `access` and `direct_call` are required. Effects are `read`, `write`, `sql`,
`execute`, and `restricted`. Review the actual implementation before declaring
a reader: a method name does not establish safety. Mixed or caller-controlled
operations that have not been reviewed for publication remain `restricted`.
SQL retains its statement-level classification and connector guards.

The canonical identity is `permission_category.method_name`. A node-owned or
standalone method supplies `category="..."` on its declaration instead. Inherited
implementations retain their declarations; overriding an implementation requires
a new declaration. The decorator preserves the callable and its sync/async nature.

`direct_call=True` only makes the contract eligible for direct-call authorization.
It neither exports a tool to MCP/REST nor grants a publication access. `@mcp_tool`
still controls MCP export and availability. Filesystem scopes, SQL policies,
version grants and session isolation remain authoritative.

## Inventory and builds

The dependency-free compiler reads literal source declarations without importing
tool modules or initializing adapters, credentials, models or connectors:

```bash
python datus/tools/registry/catalog_builder.py
python datus/tools/registry/catalog_builder.py --check
```

Commit the generated `datus/tools/registry/capabilities.json` with the source.
The local pre-commit hook generates it automatically. Wheels, editable builds
and source distributions use the same compiler. PR and merge-queue workflows
check the committed inventory **before** dependency installation, so a packaging
hook cannot hide stale committed data. The fixed acceptance gate also checks it
on every code PR, independently of impacted-test selection.

`capabilities.py` loads this complete package inventory before any node runs.
It does not depend on which tools have been instantiated. Duplicate canonical
identities must agree on their contract; paging companions must name an existing
non-companion owner with the same effect. Unclassified tools remain denied on
restricted hosts. Existing family restrictions cannot be relaxed by declarations.

## Dynamic tools and wrappers

Use `trans_to_function_tool(declared_method)` for ordinary methods. A custom
schema or invoker uses the same registration boundary:

```python
from datus.tools.registry.contracts import definition_from_callable, registered_function_tool

tool = registered_function_tool(
    definition=definition_from_callable(self.task),
    name="task",
    description=description,
    params_json_schema=schema,
    on_invoke_tool=invoke,
)
```

Factories can also supply a literal `ToolDefinition("family.method",
access="restricted", direct_call=False)`. The compiler inventories these definitions
and nested decorated callables too. Runtime-dependent native definitions must be
derived from declared callables, rather than inferred from names or first use.

Use `clone_registered_tool(original, on_invoke_tool=wrapper)` for proxies and
transformers. It preserves the definition and every SDK dataclass field,
including approvals and enablement. Native code must not construct raw SDK
`FunctionTool` objects or use the SDK decorator directly; build validation rejects
those paths. Final node assembly verifies registration before visibility filtering,
including when plugins or permission-based visibility are disabled.

An external MCP/provider tool is not automatically trusted because it reports
itself as a reader. Existing external/restricted family policy remains unchanged;
a dynamic registration alone does not add an external reader to the native
publication inventory.

## Compatibility

Literal `aliases=("old_name",)` preserves historical authorization identities.
It does not mount extra transports. MCP attribute aliases must be declared.
`companion_of="family.owner"` declares result-paging authorization ownership.

Publication grants continue to store `family`, `family.*`, or exact methods.
Upgrading the Agent makes a newly declared safe reader usable under an existing
family grant; exact grants stay exact and writers remain excluded. No wildcard
expansion or database migration is required. Skill-name authorization and session
planning exemptions retain their existing specialized invocation rules.
