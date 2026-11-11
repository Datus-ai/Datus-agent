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
`execute`, `session`, and `restricted`. Review the actual implementation before declaring
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
restricted hosts. Known readers remain readable within their node grants; family restrictions still deny writes and unknown methods.

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
itself as a reader. Reviewed reads in mixed families are allowed within explicit node grants;
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

## Semantic authoring and generic files

`filesystem_tools` contains generic file operations only. Dataset/metric upserts
and deletes belong to `semantic_tools`, implemented by `SemanticModelEditingTools`
with a shared authoring workspace. Their selected-target revisions, validation,
mutation tracking and path boundaries are shared with raw semantic file edits.

Query nodes such as `ask_metrics` instantiate `SemanticTools`, so their editor
catalogs exclude the stateful editing provider. Semantic authoring nodes mount it
explicitly. `family.*` remains a pattern over the node's supported tools; it does
not add another provider to a query node.

The four former `filesystem_tools.upsert/delete_osi_*` identities are invalid.
There are no compatibility aliases or wildcard transfers to the semantic family.
The retired `create_file`, `write_query` and `get_dimensions` tool aliases are removed; `write_file`,
`edit_file`, `delete_file` and `execute_ddl` remain actual Agent tools. Native
writers without a REST/MCP export declare `direct_call=False`; registering an
Agent SDK tool alone does not expose a transport endpoint.

### Semantic authoring composition

`SemanticModelFilesystemTools` in `semantic_model_filesystem_tools.py` extends
`FilesystemFuncTool` for read/search and validated `edit_file` operations. It
never mounts generic `write_file` or `delete_file`. Ordinary filesystem tools
continue to use the generic implementation unchanged.

`SemanticModelEditingTools` supplies the four OSI dataset/metric mutations under
`semantic_tools`. Both providers use one request-local `SemanticModelWorkspace`
for target revisions, complete-document validation, query-source extensions,
mutation tracking, metric UID preparation and failure rollback. Generic path
classification, live allowlists and execution context are reused through access
callbacks; the object editor does not depend on a filesystem subclass. The node
mounts both providers explicitly and invokes the workspace for finalization and
rollback. File and object methods retain their respective permission families,
including existing `family.*` matching.

## Default readers and session channels

Normal and auto profiles derive exact read grants from the generated catalog.
Operator and node overrides remain authoritative for resource access. Publication
uses the same effects: a declared reader can survive a mixed-family restriction,
but must still match the node selection and frozen version grant. Empty ask-node
selections remain empty. SQL, filesystem paths and Skill names retain their own
resource checks. `load_file_as_table` remains restricted because registration and
materialization create database objects; `inspect_only` does not make the whole
method a reader. Credential/adapter getters and authoring target binding are not
read tools merely because their names resemble queries.

`session` identifies channels that only interact with the user or hold this
run's plan/result: ask_user, confirm_plan, todo helpers, submit_task_result and
select_final_metric_result. They do not require resource grants and must never
be exported as direct REST/MCP calls or companions. Visibility and invocation
consult the same declaration, so a node DENY cannot remove the channel it needs
to ask a question or return its result. Nodes still decide whether to mount these
channels (for example, workflow runs do not mount ask_user). read_tool_result is
a reader, not a session mutation; it is always granted as session infrastructure
on publications, and the result store itself enforces session ownership.
