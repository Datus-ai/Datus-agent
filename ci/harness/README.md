# Core product acceptance

| Tests | Execution | Boundary and assertions |
| --- | --- | --- |
| `tests/integration/agent/test_core_acceptance.py` | Code PRs | Production model adapter, Agents SDK, node loop, tools, SQLite and filesystem. Only remote model responses are scripted over HTTP. Covers chat, metadata discovery, SQL results, joins, follow-up context, tool error recovery, memory across sessions, init knowledge reuse, read-only enforcement, bounded tool results, plan confirmation and a queryable table product with its validator. |
| `tests/integration/api/test_chat_acceptance.py` | Code PRs | Real FastAPI routes, task manager, chat service and model transport. Covers SQL, follow-up, history, session listing/deletion, user isolation, stop and subsequent reuse. |
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

## Selection and evidence

`coverage-map.yml` contains mandatory `execution.core_nodeids`, unit ownership
rules and additive product impact rules. The 16 exact core nodeids run for code
changes. Core and affected tests execute in one pytest session with file-level
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
Without a group filter or `product_test_base_ref`, Nightly runs the complete P0 manifest alongside
the broader adapter, provider, UI and product suites. Real LLMs are not required
in merge-group runs.

## PR routing and merge queue

`coverage-map.yml` owns `execution.integration_rules`: deterministic API, CLI,
storage, database, semantic and MCP targets and their source/consumer paths.
`ci/harness/pr_test_plan.py` selects these groups by changed paths. Shared model/config/schema
boundaries select all groups. Changed integration tests select themselves; helper
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
still requires 80%.

`ci/harness/gate_profile.py` routes ordinary documentation, image assets,
README translations and MkDocs navigation by changed paths. Pure docs skip the
Agent environment and runtime tests; Deploy Docs already builds both locales
strictly on PRs. Coverage-map documentation registrations retain the runtime/CI
checks. Runtime files, prompts, skills, dependency and workflow changes keep the
code gate, regardless of the PR title. Editing only the PR title/body does not
cancel or replace code checks; changing its base branch does revalidate.

Merge Queue runs only generated tool catalog validation and strict coverage-map
validation on the merged tree. It installs PyYAML without the Agent environment,
fixtures, pytest suites or docs build. `ci/merge-queue-results.json` records both
checks and propagates failures to the required `Merge Queue Gate` context. This
catches catalog/map conflicts introduced by combining PRs; behavior tests remain
owned by PR CI. Native packages, real services and model quality remain owned by
Nightly, dedicated acceptance and release validation. No test-result cache is
introduced.

## SQL policy host acceptance

Code changes affecting plugin installation, policy execution, datasource reads,
Dosi, or shared integration fixtures select `P0 SQL Policy Plugin Contracts`.
The product plan includes this deterministic group even without live-model targets.
SQL policy host acceptance runs in scheduled Nightly. The required `Merge Queue
Gate` checks merged-tree consistency; SQL policy and BI host acceptance are not
merge-queue dependencies. Documentation builds remain in Deploy Docs.

`sql-policy-gate.yml` runs on the shared Linux x64 `osi-engine` self-hosted runners.
The runner must have a C compiler, `patchelf`, and the ODBC development package
preinstalled; the job verifies these prerequisites without installing system
packages during each run. It resolves the
reviewed policy pin from `sql-policy-host.json` and the engine commit from the
policy's Cargo dependencies, freezes all three checkout SHAs, then builds Agent,
engine and policy wheels from those sources. External repositories are read-only
test dependencies; this workflow does not modify or trigger their CI.
The gate uses dev-profile test wheels without debug symbols; the engine enables
the DuckDB/policy host boundary needed here. Release optimization, executor matrices
and distribution publication remain covered by their existing release workflows.
Wheel repair runs on a separate wheel copy, so bundling shared libraries cannot
rewrite the Cargo artifacts reused by subsequent warm builds. Verification also
checks bundled native libraries against their wheel payloads.

Agent and engine are installed as wheels, and the harness is staged outside the
Agent checkout. SQL policy is installed through the real managed-plugin CLI into
an isolated HOME. All installed runtime payloads, including native binaries, must
match the identified wheels. The P0 checks SQL row filtering, Dosi metrics, denied
and unrestricted contexts, and HTTP Chat enforcement without an external LLM.

`result.json` records the source SHAs, input/patch hashes, toolchain, wheel hashes,
cache mode and separate build/install/execution stages. Execution evidence carries
the source identity and must include successful outcomes for every required P0
nodeid. Different sources, modified artifacts, missing reports, skips and xfails
fail validation. Each run verifies its own frozen source combination; evidence is
not reused from an earlier source revision.

Warm runs reuse the runner's available build caches. Cold runs use new empty uv,
Cargo and target directories.
Scheduled Nightly runs both modes against policy main, resolved once to immutable
SHAs before the matrix starts. The compiled private payloads and caches remain on
the runner; artifacts contain only receipts and logs. Source-read tokens are used
only by checkout actions, never exposed to native builds or tests.

Review `p0.yml` and `sql-policy-host.json` together when updating the policy pin.
Release P0 continues using the reviewed pin; scheduled compatibility checks follow
policy main. Cross-repository PR-triggered acceptance is outside this Agent gate.
The scheduled cold contract finishes before the ordinary Nightly suites start;
its failure also marks the final notification failed, even if those suites pass.

Local reproduction in a dedicated locked Linux environment (this replaces its Agent and
engine installations):

```sh
uv pip install 'maturin==1.16.0' 'auditwheel==6.8.2'
uv run --no-sync python ci/sql_policy_gate.py run \
  --policy external/datus-sql-policies --engine external/osi-engine \
  --agent-sha <agent-sha> --policy-sha <policy-sha> --engine-sha <engine-sha> \
  --mode cold --work /tmp/datus-sql-policy-cold
```

## Release P0

`ci/harness/p0.yml` explicitly lists six groups and 24 required nodeids: the
15 deterministic core journeys, three real-model SQL tasks, PostgreSQL reconcile and feedback CRUD,
SQL-policy enforcement through HTTP, real-model semantic authoring, managed
Superset export and real-model dashboard creation. Initialization/tool-wiring contracts
run as components; fixed-result BI orchestration remains ordinary Nightly coverage.
P0 cases cannot skip, xfail, disappear from collection or finish without execution.
`ci/p0_plan.py` verifies reports from a fresh `ci/p0-execution.*` directory.
Unknown filters that select no test groups fail before execution.

PR and merge queue explicitly select `DATUS_TEST_STORAGE_BACKENDS=sqlite+lance`;
installed external adapter entry points cannot add service-backed unit tests to
these gates. Feedback CRUD always has a local SQLite acceptance case. The same
business assertions run against PostgreSQL in the required P0 storage group,
sharing its PostgreSQL/pgvector session providers with semantic KB reconcile.
Full Nightly and release P0 require both cases to pass; missing `testcontainers`,
provider errors and service startup failures fail the group. Storage/configuration
and dependency changes also select this group for targeted product validation.

Release Candidate Validation builds distributions once, then passes that artifact
and its source SHA to Nightly's `p0` profile. This profile installs published
adapter pins from `ci/requirements-p0.txt`; the profile's pinned source checkouts
supply managed plugins and service fixtures. Review both manifests when updating
adapter/plugin versions. Full Nightly continues source-installing repository main.

The candidate harness runs outside the checkout, with the wheel installed in a
clean environment. `ci/release_artifact.py` checks distribution hashes, the import
location and the installed runtime/resource payload against the wheel, then records
all resolved package versions. P0 prepares small SQLite fixtures; its service-backed
groups start PostgreSQL/pgvector and Superset as needed. It does not build the full
KB, frontend assets, browser environment or cloud adapters.
Publish And Finalize Release depends on candidate validation and downloads those
same distributions, verifies their hashes and publishes them without rebuilding.

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
dependencies, or CI select `P0 BI Product E2E` in the product test plan. Scheduled
Nightly and release P0 run this group; it is not a merge-queue dependency. The separate
`merge-queue-pr.yml` workflow publishes the lightweight PR context without a
checkout or reusable workflow call. The standalone BI gate uses pinned BI and plugin fixtures from `p0.yml`, the existing
self-hosted CI credential template, and dynamically allocated loopback Docker ports.
It runs the deterministic chart repair/delete contract, managed plugin installation,
and real-model dashboard creation. Missing, skipped, or failed P0 receipts fail the gate.

Nightly runs the same deterministic repair contract. Its trace plugin captures BI
session references inside child tasks and writes resource coordinates, tool result
shapes, validation targets/checks, and outcomes to `nightly-bi-evidence/`. Prompts,
SQL text, Bash arguments, credentials, and data rows are excluded. Local execution
can use `.venv/bin/python ci/bi_gate.py run --work /absolute/new/directory`; installed
BI packages and Chromium, Docker, source fixtures, and `DEEPSEEK_API_KEY` are required.
