# Source Evidence Contract

Read before discovery/export. Use the installed plugin's source and rebuild skills
as command and native-semantic authorities. This is a working-memory projection
of their existing outputs, not a new CLI/API or permission to rewrite bundles.

## Supported source paths

| Path | Required evidence |
| --- | --- |
| Query export | Stable dashboard/query candidates, selected export or a full export with selected routing, complete SQL files with checksums/status and per-query physical connection identity |
| Native asset bundle | Stable selected dashboard/chart/dataset IDs, checked manifest/files, native analytical definitions, dependency edges, SQL provenance and object-level semantic coverage |

A native chart may expose no runtime SQL. Dataset Custom SQL plus a supported,
complete chart definition can be sufficient analytical evidence; neither alone
proves the chart's runtime query. Missing SQL does not authorize fabricating it.
Reference-SQL indexing still requires a complete, successful actual SQL item.
When the plugin cannot interpret an expression or relationship, retain the native
reference and block its dependents rather than interpret private syntax here.

## Normalize only the selected dependency closure

Retain these fields from verified plugin output; use `unknown` for absent facts:

- Source plugin/release, profile, site/project, stable dashboard/chart IDs,
  snapshot path, manifest SHA256 and capture time. Profile identifies the access
  context, not the warehouse. Do not invent a field missing from the manifest.
- Each chart's business question, measures/aggregations and calculations,
  declared units/currency (or `unknown`), dimensions, exact time-grouping
  expressions and year retention, timezone and NULL policy, joins, filter scope/stage,
  parameters, required security semantics and interactions. Keep exact references
  to the native definition and evidence for each interpretation.
- SQL ID, asset ID, file, checksum, status and `source` / `derived` / `target`
  origin. Retain derivation/binding provenance and parameter context.
- Per-item credential-free physical connection identity, matched Datus datasource
  and matching evidence. Multiple sources remain separate partitions.
- Coverage issue IDs/codes, affected fields/assets, dependency paths and reasons
  an affected chart is supported, partial or blocked. Preserve global issues too.

Use summary output first, then inspect exact selected records. Truncated IDs are
not usable binding keys. Avoid reading entire native definition trees when the
plugin can supply bounded chart/dataset details. Do not execute downloaded code.

Before authoring, resolve native field instances to exact grouping projections.
Names such as "month" are insufficient: a calendar month-of-year groups across
years, whereas a year-month bucket retains the year. Preserve that distinction
in every handoff and numerical baseline. Create a necessary derived dimension
through `semantic_modeling` when supported; do not silently substitute a time
bucket. Unknown units remain unspecified in models and display formatting. Never
invent a currency code from a measure name, example definition, symbol or locale.

## Match source identity before metric authoring

Never persist a BI-profile-level datasource mapping. Compare each relevant source
connection with the **effective runtime connection used by `semantic_modeling`**.
Call `get_datasource_identity` when available and require `identity_status=resolved`.
Its `identity` comes from the constructed adapter and native metadata;
`configured_identity` is diagnostic only, because URIs, adapter extras and the
selected database can override it. Retain its exact `execution_context` in the
handoff: builtin children inherit the parent's resolved database/catalog/schema.
If the planned child target differs, verify that target's effective connection
before authoring. Block authoring whenever effective coordinates cannot be
established, even if configured coordinates or table names appear to match.
Never read a secret-bearing config file to obtain missing coordinates:

- Normalize backend aliases, host casing, default ports and physical
  database/catalog names. A network database requires a compatible backend,
  matching database/catalog and exact normalized endpoint. Do not invent host aliases.
- File databases require the same backend and exact resolved `identity.path`.
  The tool reads native connection metadata for the selected glob-backed file
  or attached DuckDB catalogue, rather than exposing a glob or assuming the first
  file. Without a safe runtime inspector, the path remains unresolved and blocks
  authoring. Cloud
  warehouses use stable account/project/region/catalog fields documented by
  the installed adapters/plugin.
- Dataset/table/schema equality, BI display names, usernames and SQL text do not
  establish physical identity. A BI server URL is not a warehouse endpoint.
- Record one strong match as `matched_datus_datasource`; zero is `unresolved`,
  multiple is `ambiguous`. Obtain missing configuration/evidence rather than guess.
- Partition authoring by matched datasource; process only the active partition.
  Defer the rest until a later run with that datasource active.

An existing curated DOSI target may use different upstream physical tables.
Reusing it requires a verified analytical mapping and explicit target datasource/
model binding; it does not prove source-connection equivalence or authorize
running source SQL on the target. Unresolved source identity still blocks authoring
from that source unless the plugin documents an independently evidenced binding.

## Verify export or local snapshot

Use the plugin inspector before consuming native bundles. Verify documented
schema/version, source selection, exact IDs, dependency closure, file confinement,
checksums and per-item statuses. Retain the manifest SHA256 throughout handoff.
Never repair a snapshot in place, replace its checksums or hide coverage issues.
Read-only inspection may run before a generation confirmation; writing/overwriting
an export or generating derived processing artifacts may not.

For query exports, route only selected successful entries with complete SQL and
verified files/checksums. Reject missing, failed, partial or unselected SQL. For
native bundles, evaluate coverage per selected chart and its dependencies instead
of treating either global `partial` or successful HTTP responses as a verdict for
all charts. Unknown calculations, missing definitions, security, downloads or
unbuilt processing dependencies remain blockers as the plugin contract specifies.

A plugin may document a narrowly proven historical-source gap resolution against
an independently complete existing target. Preserve that source gap and evidence;
never apply the exception to another issue type or claim the source is complete.
Generated SELECT models are derived definitions, not materialized tables or
executed ETL. Do not deploy/execute ETL or refresh the source platform here.
