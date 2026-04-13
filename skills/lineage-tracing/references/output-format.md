# Output Format

The lineage tracer returns JSON with four top-level sections:

- `files`
- `table_edges`
- `target`
- `unresolved_inputs`

## `files`

Per-file summary with:

- `path`
- inferred `output_table`
- direct `upstream_tables`

## `table_edges`

List of normalized table-level edges:

```json
{
  "from": "staging.orders",
  "to": "intermediate.orders_enriched"
}
```

Interpretation:

- `from` is an upstream dependency
- `to` is the repository-defined output node

## `target`

Only present when `--target` is supplied. Includes:

- `name`
- `direct_upstreams`
- `all_upstreams`
- `direct_downstreams`
- `all_downstreams`

## Inference rules

- If a SQL file lives in `staging/`, `intermediate/`, or `marts/`, its inferred output is `<layer>.<stem>`.
- If no layered parent directory exists, the output is the bare file stem.
- CTE names are excluded from lineage edges.
- Repository-internal dependencies are still shown even if the upstream file was not parsed from the same root; unresolved external references remain in `unresolved_inputs`.

## Recommended usage

- Use direct upstreams to understand what a node reads right now.
- Use transitive upstreams to scope debugging and audit root causes.
- Use downstreams before schema edits to estimate impact.

