# Core product acceptance

| Tests | Execution | Boundary and assertions |
| --- | --- | --- |
| `tests/integration/agent/test_core_acceptance.py` | Code PRs | Production model adapter, Agents SDK, node loop, tools, SQLite and filesystem. Only remote model responses are scripted over HTTP. Covers chat, metadata discovery, SQL results, joins, follow-up context, tool error recovery, memory across sessions, init knowledge reuse, read-only enforcement, bounded tool results, plan confirmation and a queryable table product with its validator. |
| `tests/integration/api/test_chat_acceptance.py` | Affected code PRs; Nightly core acceptance | Real FastAPI routes, task manager, chat service and model transport. Covers SQL, follow-up, history, user isolation, stop/reuse, terminal cleanup, replay expiry, project eviction and cancelled native SQL workers. |
| `test_real_tool_concurrent_cold_connection_and_owner_eviction` in `test_func_tools_db.py` | Affected code PRs | Fully initialized database tools, real SQLite/DuckDB connections and concurrent queries. Checks one cold construction, bounded owner eviction, reopening and independent borrowers. No model calls. |
| `tests/integration/storage/test_fastembed_lancedb_smoke.py` | Existing main Nightly group | Real FastEmbed/ONNX inference and LanceDB. Fresh-process concurrent loading and repeated wrapper lifetimes check one native session and emit RSS observations. No remote LLM calls; model assets must be available. |
| `test_dosi_authoring_validates_reconciles_and_queries_without_llm` in `test_semantic_modeling_agentic.py` | Code PRs; existing dedicated Nightly group | Real Dosi authoring, validation, reconciliation and metric execution. No external model required. |
| Capacity, image round-trip and native observability model contracts | Affected model / shared-runtime PRs | Real initialized adapters and SDK paths with external IO replaced. Includes nested JSON tool schemas and native model calls. |
| Remaining affected unit tests | PR diff | Exact-file / longest-prefix unit owners retain explicit consumers, including harness-marked unit tests. Shared model adapters include observability and node consumers. |
| `tests/integration/agent/test_core_live.py` | Scheduled Nightly; explicit branch evaluation | Real LLM answers independent business questions: total 350 and East-region total 150. Checks SQL tool results and answers through Chat, then runs GenSQL and independently executes its generated query. Credentials missing means failure. |
| Existing node-specific real-model suites | Scheduled Nightly and affected branch evaluation | The live plan expands for nodes, models, configuration, prompts, skills, tools, storage, context, API and CLI changes. Semantic modeling, Superset and Airflow retain dedicated service setup. Init evaluates the ordinary scoped case; the large-catalog scale case remains opt-in. |

`tests/helpers/product_acceptance.py` owns a small independent business dataset:
three orders total 350, East customers contribute 150 and West customers 200.
Assertions check rows, files and persisted state, rather than exact answer wording
or exact generated SQL. Scripted acceptance verifies runtime wiring and effects;
the real-model questions additionally assess tool selection and reasoning.

Resource regressions for #1639–#1641 reuse these fixtures and execution groups.
The HTTP cases drive real model adapters and tools using scripted local HTTP
responses, including an actual provider error, stop and client disconnect. They
check immediate connection returns, resume before a one-second test TTL, automatic
expiry without another request, persisted history and repeated concurrent turns.
Native SQL cancellation blocks inside SQLite itself to prove eviction waits for
the worker. Project eviction verifies the original native handles are closed and
that the next request rebuilds a working service. Object collection of successful
nodes is checked after tracing drains. Error/cancel
tracebacks can temporarily survive in the HTTP client's cancelled timeout
handles, so those cases assert returned borrows and detached tool connections
without depending on asyncio timer-heap GC. Resource closure is checked before
waiting for TTL.

Review regressions also hold real SQLite metadata execution across catalogue
timeouts, table-detail/batch cancellation and project eviction. Active SSE cases
cover service shutdown, cache shutdown and refreshed configuration generations.
Unit cases check replay completion order, cancelled creation waiters, creator
draining, retired-generation request protection and rejected post-close borrows.
Concurrency unit tests rendezvous constructors with their real Future waiters
instead of relying on sleeps. The expiry fixture observes actual timer callbacks
and never performs the purge itself.

Catalogue deadline cases require a timeout response while native SQLite remains
blocked, then verify that a worker-owned service borrow prevents early disposal.
Artifact-save cancellation cases invoke real function tools and SQL/metric save
paths; under connection pressure, tool cleanup must wait for the save before the
native connection can be evicted. Follow-up cases receive the previous `end`
while cleanup is held, then verify next-turn admission and cancellation isolation.
Device-isolation tests explicitly set CPU/CUDA inputs and cover a CUDA default.
Faulted teardown checks release their leases in `finally` before executor join.

The native embedding cases prefetch assets before measurement, then use three
rounds of eight concurrent inferences in a fresh interpreter. Construction count,
wrapper collection and live native-model count are hard assertions. RSS is
reported as evidence without a platform-dependent memory threshold. These cases
retain their existing `nightly` mark and are excluded from PR selection; the
deterministic HTTP and database cases are selected through the existing map.

## Selection and evidence

`coverage-map.yml` contains mandatory `execution.core_nodeids`, unit ownership
rules and additive product impact rules. The 16 exact core nodeids run for runtime changes and changes to test selection,
execution or shared fixtures. Ordinary CI/workflow changes and test-only changes
select their owners without adding the runtime core. Core and affected tests execute in one pytest session with file-level
xdist distribution; whole-file selections subsume their core nodeids to prevent
duplicate execution. Recollecting the previous 71-file fixed list on this tree
selects 1,069 cases; the fixed requirement is now the 16 core cases, with the
remaining deterministic contracts selected by impact.
For each changed source file, `execution.unit_rules` selects the exact file or
longest matching directory prefix, including the consumers listed by that owner.
Node factories, inherited hooks and shared session/result behavior retain the
node package and its cross-package consumers. Audited leaf owners cover model helpers,
feedback, visual renderers, prompt compaction, retry, semantic authoring/runtime,
artifact tools, BI bootstrap and lineage. Each exact-file rule retains its static
consumers and reviewed caller family; subagent dispatch and SQL read context keep
their broad boundaries. Shared startup, configuration and
payload owners retain full unit coverage. Localized MCP, logging and SSL owners
keep their explicit, audited consumers. Shared model adapters still include
observability and node consumers. Unknown
runtime modules retain the full unit fallback, and full-suite rules document why
they are needed. Rule validation rejects stale targets and ambiguous source paths.

The 2026-11-11 static-consumer audit adds 35 exact source owners (25 rule
entries after coalescing identical target families). Replaying the same 60-commit
sample against the audited 625-file unit inventory reduces file selections from
27,295 to 25,613 (6.2%); 9 commits improve and none expand. These are selection
counts, not an estimate of elapsed CI time.

| Changed source | Previous unit files | Selected unit files |
| --- | ---: | ---: |
| `models/accepted_turn.py` | 431 | 1 |
| `models/text_completion.py` | 431 | 43 |
| `models/litellm_model.py` | 431 | 212 |
| `tools/func_tool/semantic_tools.py` | 553 | 132 |
| `tools/func_tool/dashboard_artifact_tools.py` | 553 | 29 |
| `utils/sql_lineage.py` | 585 | 11 |
| `configuration/agent_config.py` | 625 | 625 |

The 2026-11-14 leaf audit adds `bulk_query_tools.py` (34 unit files instead of
559) and `query_spool.py` (64 instead of 488), measured on main `6cba1e2d`.
These retain DBFuncTool inheritance callers, BaseFuncTool cancellation/concurrency
callers for spooling, configuration validation and capability contracts. The CLI
and database integration contracts still run for spooling changes.

Replaying historical changed paths with both selectors on the same `6cba1e2d`
source tree gives the following file selections (not elapsed-time estimates):

| PR | Previous unit files | Selected unit files | Core cases | Integration groups |
| --- | ---: | ---: | --- | --- |
| #1635 workflow triggers | 35 | 1 | 16 → 0 | none |
| #1634 export limits | 560 | 64 | 16 → 16 | CLI, database retained |
| #1638 dependency upgrade | 630 | 630 | 16 → 16 | all six retained |

Strict map validation and its CI regression test audit every Python module under
`datus/` against every `tests/unit_tests/**/test_*.py` consumer. The audit parses
imports (including function-local imports), package re-exports and literal dynamic
import or mock paths without importing application code. An omitted consumer or
unowned module fails CI with its source path, owner and test file. Conditional
re-exports retain every visible origin. This is a conservative reference check,
not proof of runtime coverage: computed imports, dependency injection and factory
dispatch still require the wider ownership boundaries above. Adding a narrow
rule requires retaining both the visible consumers and those shared contracts.

Changed unit tests select themselves; conftest, helper modules and fixture assets
select their enclosing test package. Shared test configuration and datasets select
the full unit suite. Parent directories subsume child targets without restricting
pytest collection. PR selection enumerates both sides of renames. Product planning
uses the same ownership map without widening leaf unit selections.

`execution.impact_rules` independently unions live targets and provisioned nightly
groups. Narrowing unit ownership does not remove those product requirements.
The PR artifact includes `ci/unit-test-plan.json` with each changed path's owner,
targets and fallback decision, JUnit timings, and execution reports containing
suite wall time. CI logs also report the 30 slowest test phases. Full unit tests
continue to run in Nightly.

PR and Nightly execution use `ci.pytest_gate_plugin`, which records selected nodeids and
actual execution outcomes, including xdist reports. `ci/gate-coverage.json`
fails the gate for missing reports or missing, skipped, xfailed, failed or
unexecuted core cases. A test merely existing in the repository is insufficient.
Flow claims marked `partial` retain references outside the fixed gate and are
not claims of complete execution. Full claims are checked against runtime evidence.

The normal PR job publishes `ci/product-test-plan.json`; it does **not** execute
credentialed models. To evaluate an authorized branch using the existing trusted
Nightly environment after these workflow changes are available:

```sh
gh workflow run run-nightly.yml --ref main \
  -f datus_agent_ref=feature-branch \
  -f product_test_base_ref=main
```

The comparison selects product targets and required service groups. In this mode
the derived group filter takes precedence over `nightly_group_filter`, selected
tests cannot silently skip, and scoped init is explicitly enabled when selected.
The base fetch restores shallow history before computing the merge-base diff.
Each executed pytest group writes a report under a fresh
`ci/product-execution.*` directory. The final verification checks every planned
file/nodeid against successful test outcomes across product and dedicated groups;
deselected or unexecuted targets fail even when other product tests pass.
Reports and `ci/product-execution-coverage.json` are uploaded with Nightly artifacts.
Without a group filter or `product_test_base_ref`, Nightly runs its core acceptance,
live SQL, PostgreSQL storage, SQL policy, semantic modeling and Superset suites
alongside the broader adapter, provider, UI and product suites. Real LLMs are not required
in merge-group runs.

## PR routing and merge queue

`coverage-map.yml` owns `execution.integration_rules`: deterministic API, CLI,
storage, database, semantic and MCP targets and their source/consumer paths.
`ci/harness/pr_test_plan.py` selects these groups by changed paths. Shared model/config/schema
boundaries select all groups. Selecting the full unit suite alone does not select
integration groups; each integration group has its own source/consumer ownership. Changed integration tests select themselves; helper
changes retain the registered deterministic integration groups. Dashboard/metric
artifact readers select native Dosi contracts, MCP execution helpers select the
HTTP MCP contract, and search helpers select document-storage contracts.
Strict map validation audits their literal imports, package re-exports and mock
references with the same static scanner as the unit-owner audit. Factories and
other indirect callers retain explicit source-family rules. Missing registered
targets or source paths fail both validation and PR planning; only deleted,
unregistered diff tests are filtered out.

The profile step derives `needs_excel` from the actual selected pytest targets,
using the same fetched base ref as the runner. Excel extension installation runs
only when the spreadsheet integration file is selected.

Unit tests keep all harness marks;
integration tests require acceptance/component/llm_harness. Nightly and quarantine
marks remain excluded, except the explicit local Dosi contracts registered in
`nightly_pr_contracts()` (legacy dual marks retained for scheduled P0). Every mandatory core case must actually pass: a skip,
xfail, crash, missing collection or missing execution fails PR CI. Diff coverage
requires 80% for changed application Python lines. Changed executable application
files absent from the coverage XML fail this check instead of disappearing from
diff-cover's denominator. CI, test-only, docs and dependency-only changes report
application coverage as **N/A**; pytest failures still fail the required check.
Explicit required files/nodeids must pass, including in scoped CI/test-only plans.
Scheduled-only, quarantined and unclassified integration cases keep their PR
exclusion; execution reports record the reason even when no PR-eligible cases
remain. Such exclusions cannot satisfy an explicit mandatory core nodeid.
Package selections permit their existing optional skips but must execute passing
tests, and still reject failed or unexecuted cases.

`ci/harness/gate_profile.py` routes ordinary documentation, image assets,
README translations and MkDocs navigation by changed paths. Pure docs skip the
Agent environment and runtime tests; Deploy Docs already builds both locales
strictly on PRs. Coverage-map documentation registrations retain CI tests.
Pure project-version metadata is recognized by semantic comparisons: only
`project.version`, the editable `datus-agent` lock entry's own version and
requirements comments/whitespace may differ. Third-party versions, sources,
hashes or any other configuration changes keep broad dependency regression.
Version-only changes retain release-readiness and locked-metadata validation.
Unknown runtime modules retain the full unit fallback and core journeys.
Ordinary CI leaves select their exact CI tests; unknown CI leaves retain the CI
package. Selector/runner/shared-fixture changes retain CI regression and the
runtime core. The selection artifact records owners, targets, integration groups
and broadening reasons. Classification uses paths and contents regardless of
the PR title. PR validation always publishes the stable
`format-check / format-check` and `run-coverage / coverage` contexts, including on
title/body edits and base-branch changes. Metadata-only edits first run a
read-only evidence check. Successful reuse skips dependency installation and
Ruff/pytest. The verifier requires the same head SHA, PR number, base
branch and base commit SHA. It selects the latest real Code Quality validation
run and accepts only the matching job's successful outcome. Pending results
are awaited; failed, cancelled or skipped validation cannot become green. The evidence wait
is bounded to the existing one-hour test-command budget. Code pushes and
base-branch changes always run real validation. Test selection, diff coverage
and the product-test plan all use the base commit SHA recorded in that run,
rather than a moving branch tip. If no real validation exists for that head and
base, such as after the base advances, the metadata run performs fresh
validation. Later edits can reuse that actual validation, but evidence-only
copies never become sources. Missing or invalid base SHAs and API errors fail closed.
Metadata-only events never cancel that validation.
Test Audit runs on PR events only; merge groups do not launch an audit job.
Merge-queue quality contexts live in `merge-queue.yml`, which does not handle PR
events. They run only on `merge_group`; manual dispatches use distinct, non-required
job names so skipped jobs cannot compete with the real PR check contexts.

Merge Queue runs only generated tool catalog validation and strict coverage-map
validation on the merged tree. It installs PyYAML without the Agent environment,
fixtures, pytest suites or docs build. `ci/merge-queue-results.json` records both
checks and propagates failures to the required `Merge Queue Gate` context. This
catches catalog/map conflicts introduced by combining PRs; behavior tests remain
owned by PR CI. Native packages, real services and model quality remain owned by
Nightly, dedicated acceptance and release validation. No test-result cache is
introduced. Queue failures remain visible through GitHub checks and run artifacts;
automatic PR failure comments and Feishu notifications are no longer sent.

## SQL policy host acceptance

Code changes affecting plugin installation, policy execution, datasource reads,
Dosi, or shared integration fixtures select `SQL Policy Plugin Tests`.
The product plan includes this deterministic group even without live-model targets.
SQL policy integration runs once in the normal Nightly SQL policy group. The required
`Merge Queue Gate` checks merged-tree consistency; SQL policy and BI host
acceptance are not merge-queue dependencies. Documentation builds remain in Deploy Docs.

Nightly downloads the latest published stable `datus-sql-policies` GitHub Release
by default. The optional `policy_release` workflow input selects an explicit stable
tag for reproduction. One Release response selects the asset IDs for the entire
run; the wheel, platform build manifest and checksums all come from that response.
The helper verifies the runner's Python/platform tags, asset digests, SHA256SUMS,
release/source identity and compatibility with the installed published Dosi wheel.
Drafts, prereleases, incompatible/missing assets and failed checks stop preparation;
there is no source-build fallback.

The wheel is installed through the real managed-plugin CLI into an isolated HOME,
not the global Agent environment. The existing contract checks SQL row filtering,
Dosi metrics, denied and unrestricted contexts, HTTP Chat enforcement, and native
payload equality with the downloaded wheel, without an external LLM. The fixture
accepts `DATUS_SQL_POLICY_WHEEL` without requiring a private source checkout.
Agent uses its normal test environment,
and Dosi stays on its published dependency. Nightly performs no Policy/Dosi Rust
builds; Policy's own build/release workflows own those checks.

`sql-policy-release.json` evidence records the resolved tag, release/asset IDs,
source/API revisions, Agent commit, installed Dosi version and wheel hash. It is
included in `nightly-manifest.json` and uploaded with the test results. Payloads and
credentials are not uploaded. A fresh per-run directory holds downloaded assets
and is removed after evidence upload; credentials are scoped to the download step.
The final notification requires both successful Release preparation and tests.

The local `ci/sql_policy_gate.py` source-build diagnostic tool remains available;
its reusable GitHub Actions workflow has been removed. Nightly does not invoke it.

Local reproduction in the locked Agent environment:

```sh
policy_work="$(mktemp -d)"
uv run python ci/sql_policy_release.py --release v0.5.3 --work "$policy_work"
DATUS_SQL_POLICY_WHEEL="$(find "$policy_work" -name '*.whl')" \
  uv run pytest tests/integration/plugins/test_sql_policy_plugin.py
```

## Nightly product contracts

`ci/run-nightly-tests.sh` directly selects core acceptance, three real-model SQL
questions, PostgreSQL reconcile and feedback CRUD, SQL-policy enforcement through
HTTP, real-model semantic authoring, managed Superset export and real-model dashboard
creation. These ordinary Nightly groups retain isolated Agent homes, strict skip
checks, and the required PostgreSQL/pgvector or Superset service preparation.
The broader collectors exclude dedicated service tests to avoid duplicate runs.
Unknown filters that select no test groups fail before execution.

Targeted product plans keep exact required cases in the existing coverage map's
impact rules. Their execution reports must show successful outcomes for every
required case, including cases owned by dedicated service groups.

PR and merge queue explicitly select `DATUS_TEST_STORAGE_BACKENDS=sqlite+lance`;
installed external adapter entry points cannot add service-backed unit tests to
these gates. Feedback CRUD always has a local SQLite acceptance case. The same
business assertions run against PostgreSQL in the PostgreSQL Agent Storage Tests group,
sharing its PostgreSQL/pgvector session providers with semantic KB reconcile.
Full Nightly requires both cases to pass; missing `testcontainers`,
provider errors and service startup failures fail the group. Storage/configuration
and dependency changes also select this group for targeted product validation.

Release Candidate Validation checks package metadata, builds and validates the
distributions, installs the wheel in a clean environment, and strictly builds
English and Chinese docs. Product validation runs in scheduled Nightly and
weekly/manual benchmarks.
Publish And Finalize Release downloads the validated distributions, verifies their
hashes and publishes them without rebuilding.

Nightly installs the Agent and adapter repository-main source checkouts. The
standalone SQL policy host verifier stages its own wheel harness with
`ci/stage_harness.py`; it does not participate in release-candidate validation.

Local commands (the complete gate requires Dosi and Docker):

```sh
uv run python ci/run-merge-queue-tests.py --base-ref origin/main
uv run python ci/plan-product-tests.py --base-ref origin/main
uv run pytest tests/integration/agent/test_core_live.py --no-showlocals
```

Live core defaults to `deepseek/deepseek-v4-flash` with `DEEPSEEK_API_KEY`.
`DATUS_PRODUCT_MODEL=provider/model` and `DATUS_PRODUCT_API_KEY` override it.

## Cleanup retained coverage

- `/save` executes a real query and checks the exported JSON file.
- Legacy workflow routes remain compatibility tests; missing SSE headers must
  return 400 instead of tolerating server errors.
- Catalog navigation uses actual keyboard input. MCP checks a successful SQL
  response as well as registration and error envelopes.
- DuckDB setup failure fails the DML round-trip cases instead of skipping them.
- Version parsing is consolidated in `tests/unit_tests/storage/document/test_doc_init.py`.
  Existing cases cover duplicate scenarios; pre-release and non-version inputs
  have explicit regression checks. The duplicate single-directory document
  workflow was removed; pipeline and multi-directory acceptance remain.
- The permanently skipped plan-confirmation simulation and stale quarantine
  entry were replaced by the production-loop HTTP acceptance test. Other plan
  interaction tests remain.

### BI repair acceptance

Changes to BI tools, validation, Superset authoring/validation skills, their tests,
dependencies, or CI select `Superset Product Tests` in the product test plan. Scheduled
Nightly runs this group; it is not a merge-queue dependency. The PR-only `Code Quality`
workflow publishes the lightweight `Merge Queue Gate` context in a job without a
checkout or reusable workflow call. The local `ci/bi_gate.py` diagnostic tool uses
pinned BI and plugin fixtures and dynamically allocated loopback Docker ports;
its reusable GitHub Actions workflow has been removed.
It runs the deterministic chart repair/delete contract, managed plugin installation,
and real-model dashboard creation. Missing, skipped, or failed execution receipts fail the gate.

Nightly runs the same deterministic repair contract. Its trace plugin captures BI
session references inside child tasks and writes resource coordinates, tool result
shapes, validation targets/checks, and outcomes to `nightly-bi-evidence/`. Prompts,
SQL text, Bash arguments, credentials, and data rows are excluded. Local execution
can use `.venv/bin/python ci/bi_gate.py run --work /absolute/new/directory`; installed
BI packages and Chromium, Docker, source fixtures, and `DEEPSEEK_API_KEY` are required.
