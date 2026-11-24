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
Default full nightly runs these as three separate `Superset Product Tests -
Chart Lifecycle`, `- Bootstrap`, and `- Dashboard` groups. Each starts and removes
its own Compose project and volumes and uses a separate Agent home, workspace,
pytest base directory, and execution receipt. Superset retains its production
50 requests/second limit; one contract's requests cannot consume the next
contract's global limiter budget. The legacy `^Superset Product Tests$` filter
selects all three children; an exact child filter selects only that contract.
Evaluations with `PRODUCT_TEST_BASE_REF` retain the existing combined parent
suite and affected-case selection.
Compose volume cleanup failures now fail the nightly receipt and cleanup-only
entry point while cleanup continues for the remaining owned projects. A failed
pre-start cleanup prevents reuse of stale service state; a teardown error never
replaces an earlier test failure code.
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

`case_scope` records the ordinary scoped-init journey as a daily requirement.
The 20,000-table init and long-input capacity scenarios are collected only with
their explicit `DATUS_RUN_LIVE_INIT_SCALE=1` / `DATUS_RUN_LIVE_CAPACITY=1` opt-ins;
otherwise they are recorded as `scheduled_out` with weekly frequency and a
reason. This declares their intended frequency; it does not claim an automated
weekly run exists. The Apache Hive service group runs the Dosi Hive contracts
after service startup and records the Inceptor-only system-view case as
`not_applicable` for that target. Missing credentials never decide applicability.

HANA runs in the dedicated `HANA Agent Tests` Docker group after the other
database groups have stopped. It uses the adapter checkout's official Express
Compose and preparation/readiness helpers, a private per-run bootstrap directory,
an ephemeral password, and loopback port 23942. The runner must be prepared Linux
x86_64 with the adapter's RAM, disk and kernel prerequisites. The harness checks
those prerequisites without changing host kernel settings. Startup, readiness,
test or cleanup failures fail the required group. HANA query/action evidence is
uploaded from `test-results/hana/`.

Prepare the Docker host according to the [SAP HANA Express Docker guide](https://developers.sap.com/tutorials/hxe-ua-install-using-docker).
In particular, SAP requires `vm.max_map_count >= 135217728`; the adapter helper
also checks asynchronous I/O, disk space and available memory before bootstrap.
A failure at an earlier check does not certify the remaining prerequisites.
Host administrators must apply these settings before running the group.

`Self-hosted Vision Tests` then provisions a CPU llama.cpp container with a pinned
multi-architecture image digest and verified Qwen3-VL-2B model/projector hashes.
The server binds a dynamic loopback port; the original test must call `read_image`
and read a random six-digit code. Validated model files are cached across runs,
while the container belongs to one run and is removed on cleanup. An explicitly
configured external endpoint remains supported. Only the image/model receipt and
server log are uploaded; its private environment file is excluded.

Final cleanup writes `nightly-cleanup-status.json` and updates the step's final
exit code. Notifications require successful policy preparation, tests and cleanup,
so a late cleanup failure cannot retain a successful pre-cleanup report.

Skill model tests copy the versioned database and skill into an isolated test
home. Their dedicated Chat tool configuration explicitly enables `skills.*`;
both successful skill loading and a successful node terminal result are required.
These checks certify loading/invocation, not completion of every script supplied
by the example skill. GitHub document-fetch tests receive the workflow's existing
read-only token. Provider health uses the supported Qwen test alias rather than
the retired `qwen3-coder-plus`; product defaults are unchanged.

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

