# Cross-Repository Harness Ownership

Datus-agent nightly is the cross-repository integration signal. It source-installs
the adapter repositories and verifies that Datus can consume the latest adapter
code in realistic product flows.

Datus-agent nightly is not the primary correctness signal for adapter
repositories. Each adapter repository owns its own PR and merge-queue required
checks because its GitHub ruleset, workflow job names, and service-backed test
scope are local repository contracts.

## Ownership

| Layer | Owner | Purpose |
| --- | --- | --- |
| Adapter PR checks | Adapter repository | Fast deterministic unit, package, and cheap contract checks. |
| Adapter merge queue checks | Adapter repository | Service-backed integration that is too heavy for every PR commit. |
| Datus-agent merge queue | Datus-agent | Deterministic product acceptance across core Datus chains. |
| Datus-agent nightly | Datus-agent | Cross-repository source-checkout compatibility in realistic product flows. |
| Release candidate checks | Datus-agent | Package metadata, wheel installation, strict docs builds, and distribution identity for publishing. |
| Weekly/manual benchmark | datus-benchmark | Product quality trend tracking and evaluator health. |

## Adapter Required Check Documents

- `Datus-ai/datus-db-adapters`: `ci/required-checks.md`
- `Datus-ai/datus-bi-adapters`: `ci/required-checks.md`
- `Datus-ai/datus-scheduler-adapters`: `ci/required-checks.md`

Agent's semantic runtime is bundled with `dosi-engine` and is covered by
Agent's locked unit and integration checks. It has no external adapter checkout.

When an adapter workflow job is renamed or a new adapter capability is added,
update the owning adapter repository's required-check document and GitHub ruleset
in the same change sequence. Datus-agent should only need updates when the
cross-repository integration contract changes.

## Dashboard Bootstrap Nightly Scope

The Superset product suite runs chart repair/deletion, managed plugin installation
with verified SQL export, and real-model dashboard creation.
Creation runs through GenDashboardAgenticNode, reloads persisted dashboard/chart
objects through the BI adapter, and compares chart data with independently computed
source-table aggregates. Full skill-driven bootstrap remains a documented coverage gap. It requires provider
credentials. Fixed-result orchestration runs in the ordinary BI group; node/tool
initialization is covered by the component suite.

Real-model dashboard workflows are explicit evaluations when changing a skill,
prompt, or model. They are not warn-only nightly checks. With the normal integration
environment prepared (test configuration/data, source checkouts, adapter packages,
Superset/PostgreSQL, and provider credentials), run:

```bash
uv run pytest -m product_e2e \
  tests/integration/plugins/test_dashboard_bootstrap_plugin.py \
  tests/integration/agent/test_gen_dashboard_agentic.py \
  tests/integration/tools/test_bi_dashboard.py
```

Evaluation failures retain pytest's nonzero exit status. Core acceptance, live SQL,
PostgreSQL storage, SQL policy, semantic modeling and Superset product suites run
in daily Nightly, which tests repository-main source compatibility. Release
workflows run package and docs smoke checks; product validation remains in Nightly
and weekly/manual benchmarks. See `ci/harness/README.md` for execution evidence and
service setup.

## Required nightly execution evidence

Every selected blocking integration/product group runs with `--fail-on-skip`,
including ordinary scheduled runs. Missing credentials or unavailable required
services fail the group. Full unit tests retain their existing platform and
optional-dependency skips; provider-health checks remain diagnostic. A failed
group records its failure while unrelated groups continue.

Each pytest group writes collected nodeids and actual per-case outcomes to a
gate report. The manifest embeds that report, and the workflow uploads the raw
reports under `nightly-manifest.json.execution/` (targeted product evaluations
use `ci/product-execution.*/`). Missing required execution evidence fails the
run. Coverage checks use actual passed/failed cases, never collection alone;
skipped, expected-failure and unexecuted cases do not satisfy coverage claims.
Declared coverage gaps remain in the advisory coverage digest.

Core acceptance includes the regression-plan contracts A01 (same-session
datasource switch and metric isolation), A11 (HTTP queued message consumed once)
and A09 (semantic generation, independent metric query, terminal session and
subsequent request). These use controlled model responses with real local SQL,
semantic storage and session management. The existing real-model SQL and
semantic-authoring groups remain separate daily checks. These Agent checks do
not certify Studio UI flows or candidate-image deployment/upgrade acceptance.

The contracts follow the Agent-owned cases in the
[product regression plan](https://gaaoegmov5r.feishu.cn/docx/O50Zd2P6BonaEfxtmCFc6UQvncV)
and the
[functional/UI/deployment split](https://gaaoegmov5r.feishu.cn/docx/B0ONd3t3aoXkUhxIdSdcVt72nff).

| Case | Fixed acceptance test | Required evidence |
| --- | --- | --- |
| A01 | `test_datasource_switch_acceptance.py::test_same_session_switches_source_before_querying_and_authoring_metrics` | Same-session A-to-B switch, different data behind identical table names, B tool results and metric scope. |
| A11 | `test_chat_acceptance.py::test_http_running_session_consumes_queued_message_once` | Insert during an active HTTP turn, consume once, preserve session ownership, finish and accept a subsequent request. |
| A09 | `test_semantic_session_acceptance.py::test_http_semantic_generation_finishes_and_releases_session` | Real semantic tools and persisted metric, independent query oracle, terminal task and usable follow-up session. |
