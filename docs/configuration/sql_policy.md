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
- The tool's other write paths refuse too, not just the `execute_sql` dispatcher: `execute_write`, `execute_ddl`, and `transfer_query_result`. The last one matters most — it reads from one datasource and **writes** to another (`CREATE TABLE` / `TRUNCATE` / `INSERT`), `gen_job` mounts it as a tool of its own, and it never passes through `execute_sql`.
- The workflow pipeline refuses too. Its `execute_sql` node and the output tool's revised-SQL check hand SQL straight to the connector without going through `DBFuncTool`, so neither is covered by the gates above — and `POST /workflows/run` makes that pipeline reachable over the API. Both consult the switch through the shared `deployment_read_only_refusal` helper.
- `EXPLAIN` is only a read when what it explains is. `EXPLAIN ANALYZE <write>` *runs* the write on postgres and mysql, so the explained statement is classified on its own and refused if it is not itself a read — with or without `ANALYZE`, since deciding on the option keyword would mean tracking every dialect's spelling and failing open when one is missed.
- `POST /sql/execute` refuses anything that is not a *single* read statement. Multi-statement input (`SELECT 1; DROP TABLE t`), writable `PRAGMA`s, `USE` / `SET`, and statements the parser cannot classify are all rejected — the check is fail-closed. Use the request's `database_name` field instead of `USE` to target a database.

`DBFuncTool.read_only` reports the *effective* posture, so a tool built with no `read_only` argument — which is how the MCP server's `create_dynamic` / `create_static` factories build theirs — still reads `True` on a hardened deployment.

How it differs from the policy plugins:

| | `agent.sql_read_only` | policy plugins |
|---|---|---|
| Needs a plugin | No | Yes |
| Needs request context | No | Yes — `policy_context` per request |
| What it does | Refuses the statement outright | Rewrites or denies per request context |
| Granularity | All-or-nothing, deployment-wide | Row / table / column, per caller |
| Covers `POST /sql/execute` | Yes | No (read tools only) |

The switch can only tighten. It is exposed as a read-only property plus a one-way `AgentConfig.harden_sql_read_only()`: a per-request configuration clone may harden itself, but nothing downstream can turn a `true` back off — and it never relaxes a component that already runs read-only (Explore, `ask_report`, the LLM validators).

Use it when the process runs third-party-authored agent content — skills, subagents, reference templates — against datasources it owns. The two mechanisms compose: `sql_read_only` bounds what kind of statement may run at all; a policy plugin bounds what a given caller may read.

### Verifying a deployment

Automated coverage lives in `tests/unit_tests/tools/func_tool/test_database.py`, `tests/integration/tools/test_func_tools_db.py` and `tests/integration/tools/test_mcp_server.py`.

Two manual scripts probe a real server end to end over MCP. Neither runs in CI; both print a verdict table and exit non-zero on failure:

| Script | What it checks |
|---|---|
| `scripts/e2e_sql_read_only_mcp.py` | Self-contained. Stages a throwaway SQLite workspace, runs the flag-on / flag-off matrix, and reports which statements the switch refused. No arguments. |
| `scripts/e2e_sql_read_only_mcp_project.py` | Points at a real packaged project (`--project`). `--sqlite-standin` substitutes a throwaway SQLite datasource so writes really execute when the flag is off; `--endpoint` probes an already-running server; `--live-writes` writes to the **real** datasource (scratch tables only) and `--dry-run` prints the statements without running them. |
