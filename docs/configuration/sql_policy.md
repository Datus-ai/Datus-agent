# SQL Policy

A trusted upstream service authenticates users and supplies effective policies.
Agent executes them; it does not resolve roles. Policy support is optional and
loaded through the `sql-policy` plugin. Ordinary standalone Agent/Dosi calls
continue to work without the private package.

## Configuration

### Inspect SQL in DEBUG logs

Start Agent with `datus --debug` or set `agent.logging.level: DEBUG`.
The normal Agent log contains these structured events:

| Event | SQL fields | When recorded |
|---|---|---|
| `sql.policy.rewrite` | `original_sql`, `rewritten_sql`, `rewritten` | After policy processing, before execution; includes SQL tools, API reads and CLI SQL mode. |
| `sql.execute` | `sql` | Immediately before the SQL tool or CLI submits the final statement to its connector. |
| `sql.dosi.execute` | `sql` | After successful native metric/detail execution, using the SQL returned by that actual engine call. |
| `sql.dosi.compile` | `sql` | Metric dry-run compilation; the statement has not been executed. |

Events include the datasource and dialect. Dosi events also identify the
connection and operation. Logging never recompiles a query or serializes the
policy context; SQL fields pass through the existing log-redaction configuration.
These events are suppressed at INFO and higher levels.

### Rules

The native `datus-sql-policies` provider uses one physical-table row rule for
both SQL and Dosi reads. `metric_row_filter` profiles must be migrated.

```yaml
agent:
  plugins:
    sql-policy:
      default:
        default: true
        policies:
          - id: tenant_scope
            name: tenant_scope
            type: row_filter
            applies_to:
              datasources: [warehouse]
              tables: [analytics.main.orders]
            condition:
              column: tenant_id
              operator: in
              value_from: policy_context.row_filter.tenant_ids
        column_masking:
          - column: analytics.main.orders.phone
            mask: last4
```

Profile targets accept short, partially qualified and full names. Missing
prefixes bind to the live session's current coordinates, not the datasource's
initial defaults; explicit coordinates stay fixed. For example, `orders` and
`orders.phone` use the current namespace. Rebind after a namespace switch: short
targets follow the new current namespace, while full targets remain explicit.
Missing required current coordinates are refused.
MySQL/TiDB binding reads the live `lower_case_table_names` setting; physical
targets and CTE names use that server's case rules.

Full names follow the adapter: DuckDB/PostgreSQL/Greenplum/GaussDB use
`database.schema.table`, StarRocks/Doris use `catalog.database.table`, Trino uses
`catalog.schema.table`, Oracle/HANA use `schema.table`, and MySQL/TiDB/ClickHouse/
Hive/Spark use `database.table`. Oracle service names and HANA tenants identify
the connection rather than an SQL object prefix. The datasource key is separate. Quote names containing dots. Policy defaults select the current
namespace; short SQL query names still use the connection's search path.

Row operators are `eq` and `in`. The fixed masks are `nullify`, `redact`, `last4`
and `email`; text masks require compatible text columns and an admitted dialect.
Live acceptance covers DuckDB and 13 Docker-backed adapters: PostgreSQL, MySQL,
TiDB, StarRocks, Doris, ClickHouse, Greenplum, GaussDB/openGauss, Trino, Oracle,
Hive, Spark (HiveServer2), and HANA, for the tested native types and query shapes.
Oracle metrics require OCI; HANA metrics require SAP HANA ODBC. HANA local
acceptance uses a direct plaintext listener; tenant routing and TLS are separate.
SQLite is also verified: raw SQL uses native SQLite syntax; Dosi uses DuckDB
with its SQLite extension. Targets use `main.table` or `attach_alias.table`,
not the file stem. Dosi requires a file-backed database and refuses temporary
objects/additional attachments it cannot reproduce. SQLite metadata requires
3.37+; `last4` returns NULL on embedded NUL. Text row-filter columns require
string values to avoid different implicit numeric coercion in sqlite3 and DuckDB. Other text dialects (including
Inceptor) remain unavailable.

## Trusted request context

Projects can define named masking rule sets instead of a fixed `column_masking`
list. Place this fragment inside the selected `agent.plugins.sql-policy` profile:

```yaml
policy_revision: "r17"
masking_profiles:
  finance_team:
    - column: analytics.main.orders.phone
      mask: last4
  external_partner:
    - column: analytics.main.orders.phone
      mask: redact
  approved_raw: []
```

IDs are custom, case-sensitive keys. The trusted host selects one set per request:

```json
{"policy_revision":"r17","masking_profile":"finance_team","row_filter":{"access_mode":"scoped","tenant_ids":[42]}}
```

Only the ID and revision travel per request. Column definitions remain in shared
configuration. An explicitly selected empty list disables masking while retaining
row filtering; names have no special authorization meaning. Missing/unknown IDs,
mismatched revisions and combining a selection with `sql_policy` snapshots are
denied. Named sets and static masking rules are mutually exclusive.
Mask-only projects still require the ID and revision. Shared configuration is
never changed by a request selection, and execution/cache scopes include the
selected profile and revision. `sql.policy.rewrite` DEBUG events include both.

Trusted hosts can call `PolicyRuntime(config).requires_request_context()` to
determine whether user inputs are required, including mask-only projects.
SaaS/Backend profile selection and automatic configuration refresh are separate
integration work; currently the trusted host supplies these fields explicitly.

Profile-based rows retain the existing execution inputs:

```json
{"row_filter":{"access_mode":"scoped","tenant_ids":[42]}}
```

`denied` rejects every read. Optional `policy_modes` may skip one row rule by
setting its configured ID to `unrestricted`; other rules and masks still run.
Missing inputs, unknown IDs and invalid modes are denied.

Alternatively, trusted hosts may supply `policy_context.sql_policy` containing
`version` and effective `policies: {row_filters: [...], column_masks: [...]}`.
Targets are canonical structured objects with `datasource`, optional `catalog`,
`database`, `schema` and `table`; row entries add `column`, `operator`, `values`,
and mask entries add `column`, `mask`. This snapshot replaces profile rules for
the request. It is authorization input, never a model-generated tool argument.
The default API provider can parse this JSON from `X-Datus-Policy-Context`; a
trusted authenticated deployment must control that header. `X-Datus-User-Id`
continues to serve only as session identity.

## Execution

1. Agent validates trusted context and binds one snapshot to live metadata.
2. Direct SQL enters `DBFuncTool.execute_read_enforced`; Dosi receives the same
   provider through optional Python `policy` / `policy_required` arguments.
3. After complete SQL lowering, the shared native core replaces every physical
   read: raw row filters first, masked projection second, business calculations
   outside. Subqueries, repeated sources, query-backed datasets, derived metrics,
   FIXED/context-filter branches and attribution queries use this boundary.
4. Context verification runs again before execution. Metric/detail cache keys
   include the policy snapshot and execution context. Masking is not reapplied
   to returned rows.

If required policies or native bindings are unavailable, reads are refused.
Opaque views/functions and unsupported source syntax are refused. No policy
falls back to metric WHERE injection. External proxied executors must enforce
their own boundary. Native integration does not turn every existing workflow or
external execution surface into a protected host automatically.

`datus sql-policy status` lists configured policies. The offline plugin `check`
validates context but refuses configured rewrites without a live connector; use
Agent/Dosi dry-run to inspect protected SQL. Coordinated native wheels are needed
for this implementation; SaaS/Backend rollout is separate.

Report/dashboard metric recipes compile first and apply protection once at the
final SQL read boundary. Standalone Dosi execution keeps its native provider.
Read-carrying writes are refused whenever protection is configured, including
fixed `column_masking` with an empty request context. Named masking profiles
require the request's explicit profile ID and revision.

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
- `EXPLAIN` is only a read when what it explains is. `EXPLAIN ANALYZE <write>` *runs* the write on PostgreSQL and MySQL, so the explained statement is classified on its own and refused if it is not itself a read — with or without `ANALYZE`, since deciding on the option keyword would mean tracking every dialect's spelling and failing open when one is missed.
- A statement that reads at the root can still write inside it: PostgreSQL's data-modifying CTEs (`WITH d AS (DELETE ... RETURNING *) SELECT * FROM d`) are a SELECT on the outside. The whole statement is walked for write nodes, not just its root.
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
