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

## Hard Read-Only Switch (`agent.sql_read_only`)

`agent.sql_read_only` is a separate, simpler mechanism. Do not confuse it with the policy plugins above.

```yaml
agent:
  sql_read_only: true   # default: false
```

When `true`, no SQL entry point served by this configuration may run a non-read statement:

- `DBFuncTool.execute_sql` — the tool exposed to agentic nodes and to the MCP server — hard-rejects anything that is not SELECT / SHOW / DESCRIBE / EXPLAIN. This holds regardless of the permission profile, and applies even where `PermissionHooks` are bypassed entirely (LLM validators run with `hooks=None`, and the MCP server's tool instances never see hooks at all).
- `POST /sql/execute` refuses anything that is not a *single* read statement. Multi-statement input (`SELECT 1; DROP TABLE t`), writable `PRAGMA`s, `USE` / `SET`, and statements the parser cannot classify are all rejected — the check is fail-closed. Use the request's `database_name` field instead of `USE` to target a database.

How it differs from the policy plugins:

| | `agent.sql_read_only` | policy plugins |
|---|---|---|
| Needs a plugin | No | Yes |
| Needs request context | No | Yes — `policy_context` per request |
| What it does | Refuses the statement outright | Rewrites or denies per request context |
| Granularity | All-or-nothing, deployment-wide | Row / table / column, per caller |
| Covers `POST /sql/execute` | Yes | No (read tools only) |

The switch can only tighten. A per-request configuration clone may harden itself, but nothing downstream can turn a `true` back off — and it never relaxes a component that already runs read-only (Explore, `ask_report`, the LLM validators).

Use it when the process runs third-party-authored agent content — skills, subagents, reference templates — against datasources it owns. The two mechanisms compose: `sql_read_only` bounds what kind of statement may run at all; a policy plugin bounds what a given caller may read.
