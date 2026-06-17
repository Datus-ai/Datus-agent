# Validation

Validation is the post-deliverable guardrail for subagents that create or
update durable resources. It is separate from generation-time checks such as
`read_query`, `validate_ddl`, or `validate_semantic`: those tools help the
main agent produce correct SQL or YAML before it finishes, while the validation
hook runs after mutating tools report what they delivered.

## Where validation runs

`ValidationHook` is attached to deliverable-producing subagents:

| Subagent | Targets collected | Built-in checks | Validator skills |
|----------|-------------------|-----------------|------------------|
| `gen_table` | Tables from `execute_ddl` | Table exists | `table-validation` |
| `gen_job` | Tables and transfers from DDL, DML, and `transfer_query_result` | Table exists, transfer row-count parity | `table-validation`, `transfer-reconciliation` |
| `gen_dashboard` | Dashboards, charts, and datasets from BI tools | BI resource exists | `bi-validation` |
| `scheduler` | Scheduler jobs from submit/update tools | Job exists and status is not failed, plus deterministic runtime trigger/poll when runtime tools are available | `scheduler-validation` |

Semantic model and metric generation use their own publish gate:
`validate_semantic` and, for metrics, `query_metrics(..., dry_run=True)`.
Those flows do not use `ValidationHook` targets.

## Runtime flow

1. A mutating tool returns a `FuncToolResult` with
   `result.deliverable_target`.
2. `ValidationHook.on_tool_end` reads that target and adds it to the current
   session. The hook accumulates the whole run, not just the last tool call.
3. When the agent finishes, `ValidationHook.on_end` wraps all collected targets
   in a `SessionTarget`.
4. Layer A runs deterministic code checks. These are always enforced, even if
   LLM validator skills are disabled.
5. Scheduler jobs get an extra deterministic runtime check when the scheduler
   tool exposes trigger and run-history APIs: the hook triggers the delivered
   job once, polls the matching run, and attaches logs for failures.
6. Layer B runs matching validator skills when
   `agent.validation.skill_validators_enabled` is true.
7. If any blocking check fails, the owning subagent retries with a compact
   validation failure report as the next prompt. Retry count is capped by
   `agent.validation.max_retries`. If blocking failures remain, the node
   returns `success=false` with the `validation_report`.

Advisory failures and warnings are reported but do not force a retry.

## Built-in checks versus validator skills

Layer A is code-level infrastructure. It checks invariants that should be true
for every run:

- table targets can be described
- transfer target tables exist
- source and transferred row counts match when the transfer tool reported both
- BI dashboards, charts, and datasets are reachable
- scheduler jobs exist and are not already failed

Layer B is the user-extensible rule layer. It is implemented with skills whose
frontmatter has `kind: validator`. Validator skills are not loaded with
`load_skill`, are not shown as normal skills to the main agent, and are run
only by `ValidationHook` at the end of a matching subagent run.

Validator subagents receive only a read-only tool surface:

- database read tools such as `describe_table` and `read_query`
- BI read tools such as `get_dashboard`, `get_chart`, `get_chart_data`, and
  `get_dataset`
- scheduler read tools such as `get_scheduler_job`, `list_job_runs`, and
  `get_run_log`

Write tools and recursive subagent tools are intentionally excluded. Scheduler
validators must not call `trigger_scheduler_job`; deterministic trigger/poll is
owned by the hook.

## Configure validation

In `agent.yml`:

```yaml
agent:
  validation:
    # Set to false to disable Layer B validator skills. Layer A still runs.
    skill_validators_enabled: true

    # Number of main-agent attempts including the first attempt.
    max_retries: 3
```

Use `skill_validators_enabled: false` when you need the cheaper deterministic
checks only. To disable one validator while keeping others enabled, shadow or
edit that skill and set `severity: off`.

## Add a project-specific validator

Create a project-local skill under `./.datus/skills/<name>/SKILL.md`:

```text
./.datus/skills/
└── finance-table-validation/
    └── SKILL.md
```

If your `skills.directories` setting is customized, place the validator under
one of the configured directories. With the default scan order, project-local
skills in `./.datus/skills` override user-global skills in `~/.datus/skills`,
and both override bundled built-in skills.

Example:

```markdown
---
name: finance-table-validation
description: Validate finance mart tables after gen_job writes them
tags: [validation, finance, data-quality]
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
kind: validator
severity: blocking
mode: llm
allowed_agents:
  - gen_job
targets:
  - type: table
    schema: marts
    table_pattern: finance_*
---

# Finance Table Validation

ValidationHook invokes this skill after a matching `gen_job` run. You receive a
`SessionTarget`; loop over `session.targets` and validate every matching table.

For each finance mart table:

1. Use `read_query` on the target datasource to check that the latest business
   date is present.
2. Use `read_query` to confirm `amount` is not negative.
3. Return one check per rule and per table.

Do not mutate data. If a table fails, report the exact table and failed rule so
the retry prompt can fix only the broken target.
```

The hook appends the JSON output contract automatically. A valid validator
response includes a single fenced JSON block:

```json
{
  "checks": [
    {
      "name": "non_negative_amount",
      "passed": false,
      "severity": "blocking",
      "observed": {"table": "marts.finance_daily", "negative_rows": 3},
      "expected": {"negative_rows": 0}
    }
  ],
  "blocking_issues": ["marts.finance_daily has negative amount values"]
}
```

`blocking_issues` entries are converted into failed blocking checks. Use them
for concise must-fix findings.

## Target filters

`targets` decides when a validator runs. An empty list matches every target in
the session. Otherwise, any matching filter activates the validator.

Supported target types:

- `table`
- `transfer`
- `dashboard`
- `chart`
- `dataset`
- `scheduler_job`

Table-like targets also support:

- `database`
- `schema`
- `table`
- `table_pattern` using `fnmatch` glob syntax

Examples:

```yaml
# Any table created by gen_table.
allowed_agents: [gen_table]
targets:
  - type: table

# Only marts.* tables whose names start with rev_.
allowed_agents: [gen_job]
targets:
  - type: table
    schema: marts
    table_pattern: rev_*

# Any dashboard/chart/dataset delivered by gen_dashboard.
allowed_agents: [gen_dashboard]
targets: []
```

`allowed_agents` can name either a concrete configured subagent alias or the
canonical node class such as `gen_job`, `gen_dashboard`, or `scheduler`.

## Modify built-in validation rules

To change a built-in validator for one project, copy the bundled skill into the
project-local skill directory with the same `name` and edit it there:

```text
./.datus/skills/table-validation/SKILL.md
./.datus/skills/bi-validation/SKILL.md
./.datus/skills/scheduler-validation/SKILL.md
./.datus/skills/transfer-reconciliation/SKILL.md
```

Keep `kind: validator`, the intended `allowed_agents`, and a correct
`targets` filter. Restart Datus or reopen the subagent so the skill registry
rescans the file.

Use project-level validators for table-specific data-content checks such as
null ratios, accepted value sets, duplicate-key checks, sample diffs, or
business thresholds. The bundled `table-validation` skill deliberately stays
limited to explicit schema contracts; object existence and basic row-count
invariants belong to Layer A.
