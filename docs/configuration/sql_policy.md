# SQL Policy

SQL policy is an extension framework for enforcing request-scoped SQL read controls. It is useful when an API request should only access rows or tables within the caller's business scope, such as a tenant, market, region, or store list.

The open-source agent provides the framework boundary and runtime hook points. It does not ship a concrete policy engine. Deployments that need enforcement should provide their own plugin package.

## Framework Boundary

The open-source agent owns these responsibilities:

- Loading `agent.sql_policy` from agent configuration.
- Loading the configured plugin class from `agent.sql_policy.provider`.
- Passing the full raw `agent.sql_policy` mapping to the plugin.
- Parsing request principal fields into `AppContext.principal`.
- Running a generic API pre-check for required `principal.*` values.
- Calling the plugin before read queries reach the database.
- Revalidating SQL after a plugin rewrite.
- Returning policy denial reasons to the tool/model layer.

The plugin owns these responsibilities:

- Defining the policy schema under `agent.sql_policy`.
- Validating plugin-specific policy fields.
- Matching policies to datasources, tables, columns, or other business concepts.
- Resolving values from the request principal.
- Rewriting read SQL or denying the query.
- Returning clear denial reasons that tell the model or caller what is missing.

## Runtime Flow

When SQL policy is enabled, request handling follows this flow:

1. The API auth provider creates an `AppContext`.
2. Request-scoped attributes are stored on `AppContext.principal`.
3. Agent configuration is loaded, including `agent.sql_policy`.
4. If the raw policy config references `value_from: principal.<path>`, the chat API checks that each referenced principal path is present before the agent starts.
5. The agent runs normally until a database read tool is called.
6. `DBFuncTool.read_query` validates that the original SQL is read-only.
7. The configured plugin is loaded and called through `enforce_read(...)`.
8. If the plugin denies the query, the tool returns the plugin's reason and does not execute SQL.
9. If the plugin returns rewritten SQL, the agent revalidates that rewritten SQL is still read-only.
10. The validated SQL is executed against the target datasource.

The pre-check in step 4 is intentionally generic. It scans the raw policy mapping for `value_from` strings that start with `principal.`. It does not assume a specific policy type, column name, or business field.

## Configuration Contract

The open-source agent only interprets these fields:

```yaml
agent:
  sql_policy:
    enabled: true
    provider: my_company.sql_policies:SqlPolicyProvider
```

`enabled` turns on the framework. `provider` must be a Python class path in `module:Class` format.

All other fields under `agent.sql_policy` are passed through unchanged as `SqlPolicyConfig.raw`. Your plugin can define any schema it needs:

```yaml
agent:
  sql_policy:
    enabled: true
    provider: my_company.sql_policies:SqlPolicyProvider
    policies:
      - name: tenant_scope
        type: row_filter
        applies_to:
          datasources: ["warehouse"]
          tables: ["orders"]
        condition:
          column: tenant_id
          operator: eq
          value_from: principal.tenant.id
        enforcement:
          on_read: filter
          on_unhandled: deny
```

In this example, the open-source agent uses `enabled`, `provider`, and the `principal.tenant.id` reference for pre-checking. The plugin decides what `policies`, `type`, `applies_to`, `condition`, and `enforcement` mean.

## Plugin Interface

A plugin is a normal Python package installed in the same environment as `datus-api`. The provider class must accept a `SqlPolicyConfig` argument and implement `enforce_read(...)`.

```python
from typing import Any, Dict, Optional

from datus.tools.sql_policy import SqlPolicyConfig, EnforcementResult


class SqlPolicyProvider:
    def __init__(self, config: Optional[SqlPolicyConfig] = None) -> None:
        self.config = config or SqlPolicyConfig()
        self.policies = self.config.raw.get("policies", []) or []
        self._validate_policy_config()

    def enforce_read(
        self,
        sql: str,
        *,
        datasource: str,
        dialect: str,
        principal: Optional[Dict[str, Any]],
    ) -> EnforcementResult:
        principal = principal or {}

        # Implement policy selection and enforcement here:
        # - parse SQL and identify referenced tables
        # - match policies for the datasource and tables
        # - resolve configured values from principal
        # - return a rewritten read query or deny with a clear reason
        return EnforcementResult(allowed=True, sql=sql)

    def _validate_policy_config(self) -> None:
        # Raise an exception if required plugin-specific config is invalid.
        pass
```

The result controls what happens next:

```python
return EnforcementResult(
    allowed=True,
    sql=rewritten_sql,
    applied_policies=["tenant_scope"],
)
```

```python
return EnforcementResult(
    allowed=False,
    reason="Missing required principal path: principal.tenant.id",
)
```

Use a SQL parser or database-safe query builder when rewriting SQL. Avoid string concatenation for policy predicates.

## Request Principal

The principal is the request-scoped input that policy plugins use for caller attributes. For the default API auth provider, principal fields are read from the `X-Datus-Principal` header as a JSON object:

```http
X-Datus-Principal: {"tenant":{"id":"tenant_001"},"market_codes":["MKT300","MKT301"]}
```

The plugin receives the parsed object:

```python
{
    "tenant": {"id": "tenant_001"},
    "market_codes": ["MKT300", "MKT301"],
}
```

Policy config can reference nested fields with `principal.<path>`:

```yaml
condition:
  column: tenant_id
  operator: eq
  value_from: principal.tenant.id
```

The API pre-check treats missing keys, `null`, empty strings, and empty arrays as missing values.

`X-Datus-User-Id` is separate from the SQL policy principal. It identifies a caller for session isolation and is not copied into `AppContext.principal`.

## Provider Contract

| Item | Contract |
|------|----------|
| `agent.sql_policy.enabled` | Enables the SQL policy framework. |
| `agent.sql_policy.provider` | Python class path in `module:Class` format. Required when enabled. |
| `SqlPolicyConfig.raw` | Full raw `agent.sql_policy` mapping passed to the plugin. |
| `enforce_read(sql, datasource, dialect, principal)` | Called before read SQL is executed. |
| `EnforcementResult.allowed=True` | The query may continue. `sql` can contain the original or rewritten SQL. |
| `EnforcementResult.allowed=False` | The query is denied. `reason` is returned to the tool/model layer. |
| `applied_policies` | Optional policy names for logging and diagnostics. |
| `value_from: principal.*` | Optional convention used by the API pre-check to detect missing principal fields. |

## Error Behavior

If SQL policy is enabled but no provider is configured, enforcement fails before SQL is executed.

If the configured provider class cannot be imported, initialized, or does not implement a callable `enforce_read`, enforcement fails with a SQL policy provider error.

If policy config references a missing `principal.*` value, the chat API fails before the agent starts:

```text
SQL_POLICY_PRINCIPAL_REQUIRED
```

The error message includes the missing principal path, for example `principal.tenant.id`.

If the plugin denies a query, SQL is not executed and the tool returns the plugin's `reason`.

If the plugin rewrites SQL into a non-read statement or a multi-statement query, the read-query validator rejects it before execution.

## Limits

- The open-source agent does not include a built-in row-filter or SQL-injection review plugin.
- The framework does not define a required policy schema beyond `enabled`, `provider`, and the optional `principal.*` pre-check convention.
- CLI requests do not have HTTP headers. Request-scoped API principal input is available through the API auth context. For CLI or custom deployments, populate `AppContext.principal` through the relevant auth or runtime integration.
- Do not put `user_id` inside `X-Datus-Principal`; `user_id` is reserved for `X-Datus-User-Id`.
