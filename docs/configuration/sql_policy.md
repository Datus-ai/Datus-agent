# SQL Policy

Datus Agent does not authenticate users or decide their permissions. A trusted upstream service authenticates and authorizes the request, then sends execution-only inputs as `policy_context`. Active policy plugins interpret that context and protect reads.

## Configuration

SQL policy is configured only as a plugin profile; the former `agent.sql_policy` provider configuration is not supported.

```yaml
agent:
  plugins:
    sql-policy:
      default:
        default: true
        policies:
          - name: store_scope_sql
            type: row_filter
            applies_to:
              datasources: ["warehouse"]
              tables: ["orders", "store_sales"]
            condition:
              column: store_id
              operator: in
              value_from: policy_context.row_filter.store_ids
            enforcement:
              on_read: filter
              on_unhandled: deny
```

Policy types and their fields belong to the policy plugin. Agent only loads the plugin runtime declared by its `datus-plugin.yml` manifest.

## Request Context

The default API provider reads a JSON object from `X-Datus-Policy-Context`:

```http
X-Datus-Policy-Context: {"row_filter":{"access_mode":"scoped","store_ids":[1,2]}}
```

The agreed shape has one section per policy family, without an identity or groups layer:

```json
{
  "row_filter": {
    "access_mode": "scoped",
    "store_ids": [1, 2]
  },
  "column_mask": {
    "customers.email": {"strategy": "email_partial"}
  }
}
```

The current sql-policy plugin supports these row-filter modes:

| `access_mode` | Behavior |
|---|---|
| `denied` | Reject every data read. |
| `scoped` | Apply configured row filters and resolve their `policy_context.*` inputs. Missing inputs fail closed. |
| `unrestricted` | Skip row filtering. Other policy families, such as future column masking, still run. |

When row policies are configured, a missing or unknown `access_mode` is rejected. When no row policy is configured, an empty context is allowed.

`X-Datus-User-Id` remains session identity only. Agent does not merge it into `policy_context`, and it does not treat any context field as authenticated identity.

## Runtime Flow

1. The upstream service authenticates and authorizes the caller and builds `policy_context`.
2. The API parses the header into `AppContext.policy_context` and validates it through active policy runtimes before starting any built-in or user-defined subagent.
3. The request-specific `AgentConfig` clone carries the same context to every subagent and tool transformer.
4. `DBFuncTool.execute_read_enforced` validates the original SQL, calls `before_sql_read`, revalidates rewritten SQL, then executes it.
5. A successful raw result passes through `after_read_result` before compression, artifact storage, rendering, or return to the caller. This is the extension point for column masking.
6. Semantic metric tools call the same plugin runtime before aggregation to add their `where` predicates.

Invalid runtime declarations, malformed decisions, policy exceptions, denials, and unsafe SQL rewrites all fail closed. Proxied tools execute outside the Agent process and therefore must be protected by the external executor.

For manual checks, pass the same object explicitly:

```bash
datus sql-policy check --sql "SELECT * FROM orders" \
  --policy-context '{"row_filter":{"access_mode":"scoped","store_ids":[1,2]}}'
```
