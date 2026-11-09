# Completion and Retry Evidence

Read before reporting completion or checking an earlier bootstrap run. Inspect
actual source/artifact/model/index evidence instead of trusting an earlier answer.

| Phase | Required evidence | Insufficient evidence |
| --- | --- | --- |
| Source | Verified selected snapshot/files, checksums, dependencies, coverage and native support | HTTP success, export exit code, another asset's SQL, or a directory's existence |
| Reference SQL | Owning builtin result, saved summary and exact subject/search resolution | YAML exists; presume an index delay |
| Metrics | Discovered identities/model binding, semantic validation, reconciliation and actual relevant result checks | Metric names in prose, YAML alone, compile-only checks |
| Target queries | Real metric saves/trials, saved recipes/model snapshots and chart bindings; explicit SQL exceptions separately | Existing metrics beside hand-written SQL templates; proposed preflight calls |
| Render/interactions | Successful render validation plus recorded parameter/interaction cases actually exercised | Successful query trial alone or an untested interactive control |
| Source equivalence | Source and target results compared for the same filters, grain, time bounds, units and security context | Matching labels/layout, target-only totals, or source SQL reconstruction |
| Publication | Successful existing publication flow/version | Draft, preview or generated files |
| Optional subagents | Verified config entries with exact post-sync subject references | Bare names/subtrees, datasource-only scope or an attempted config write |

Maintain separate per-phase and per-chart outcomes: completed, partial, blocked,
failed, pending, not requested or unverified, with concrete tool/file evidence.
Retain original source statuses/codes even when a documented existing-target
resolution permits a chart to proceed. Never rewrite coverage to make it green.
A source without Prep/ETL nodes does not need a fabricated processing plan:
record `not applicable`; a required but unbuilt model remains blocked.

For each chart return stable source identity, metric identity or explicit SQL
exception, target query slug/block ID, differences, checks and remaining blockers.
A changed sum into two series, a map into bars, changed filter order, a rolling
window or row-security difference must not disappear into a generic success claim.

For target checks, compare relevant KPI/breakdown/time totals on identical parameter
values where mathematically comparable; do not sum distinct counts, ratios or
snapshots as if they were additive. Follow metric-specific tolerances. If only
synthetic/local data was checked, label it separately from warehouse/source evidence.

The final report states selected outputs, source bundle/checksum, reference index
status, model/metric results, actual dashboard query binding, render/interaction
checks, publication version if any, optional subagents and the smallest retry set.
Do not claim an end-to-end completed migration while required selected charts,
indexing or bindings remain pending. Report successful independent phases without
forcing a retry of them. A retry reuses verified unchanged source and completed
artifacts, checks model revisions and scope, and targets only failed/pending items.
