# Core product acceptance

| Tests | Execution | Boundary and assertions |
| --- | --- | --- |
| `tests/integration/agent/test_core_acceptance.py` | Code PRs and code merge groups | Production model adapter, Agents SDK, node loop, tools, SQLite and filesystem. Only remote model responses are scripted over HTTP. Covers chat, metadata discovery, SQL results, joins, follow-up context, tool error recovery, memory across sessions, init knowledge reuse, read-only enforcement, bounded tool results, plan confirmation and a queryable table product with its validator. |
| `tests/integration/api/test_chat_acceptance.py` | Code PRs and code merge groups | Real FastAPI routes, task manager, chat service and model transport. Covers SQL, follow-up, history, session listing/deletion, user isolation, stop and subsequent reuse. |
| `test_dosi_authoring_validates_reconciles_and_queries_without_llm` in `test_semantic_modeling_agentic.py` | Code PRs and code merge groups; existing dedicated Nightly group | Real Dosi authoring, validation, reconciliation and metric execution. No external model required. |
| Capacity, image round-trip and native observability model contracts | Code PRs and code merge groups | Real initialized adapters and SDK paths with external IO replaced. Includes nested JSON tool schemas and native model calls. |
| Remaining affected unit tests | PR diff and entire merge-group diff | PR and merge queue share source-to-test mapping plus cross-module dependency rules. A model change includes observability and node consumers. Harness-marked files remain in the fixed list, enforced by `test_harness_alignment.py`. |
| `tests/integration/agent/test_core_live.py` | Scheduled Nightly; explicit branch evaluation | Real LLM answers independent business questions: total 350 and East-region total 150. Checks SQL tool results and answers through Chat, then runs GenSQL and independently executes its generated query. Credentials missing means failure. |
| Existing node-specific real-model suites | Scheduled Nightly and affected branch evaluation | The live plan expands for nodes, models, configuration, prompts, skills, tools, storage, context, API and CLI changes. Semantic modeling, Superset and Airflow retain dedicated service setup. Init evaluates the ordinary scoped case; the large-catalog scale case remains opt-in. |

`tests/helpers/product_acceptance.py` owns a small independent business dataset:
three orders total 350, East customers contribute 150 and West customers 200.
Assertions check rows, files and persisted state, rather than exact answer wording
or exact generated SQL. Scripted acceptance verifies runtime wiring and effects;
the real-model questions additionally assess tool selection and reasoning.

## Selection and evidence

`coverage-map.yml` contains mandatory `execution.core_nodeids` and impact rules.
The fixed core runs for code changes. Changes to models include their
cross-module consumers; changes to nodes also select the corresponding live suite.
Merge groups use their event's base SHA, covering all included PRs.

Both runners load `ci.pytest_gate_plugin`, which records selected nodeids and
actual execution outcomes, including xdist reports. `ci/gate-coverage.json`
fails the gate for missing reports or missing, skipped, xfailed, failed or
unexecuted core cases. A test merely existing in the repository is insufficient.
Flow claims marked `partial` retain references outside the fixed gate and are
not claims of complete execution. Full claims are checked against runtime evidence.

The fixed gate also exercises a runner self-test under a synthetic merge-group
event with xdist and the outer temp-root environment variable. Nested runners
receive a child cleanup namespace, and explicit `--suite` runs do not perform
unrequested Git diff selection. The regression verifies parent-owned files survive.

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

## Documentation merge groups

`ci/harness/gate_profile.py` recognizes ordinary documentation, image assets,
README translations and MkDocs navigation. Changes to flow `docs` lists in the
coverage map also qualify, but retain the entire CI unit suite. Changes to runtime
files, prompts, skills, dependencies, workflow code, execution rules or coverage
claims select the full gate. Mixed and unknown changes also retain the full gate.

The documentation gate builds English and Chinese docs strictly. The workflow
also retains release metadata and strict coverage-map checks. It skips runtime
acceptance and fixture preparation. The profile and the checks actually executed
are recorded in `ci/gate-coverage.json`; code profiles retain mandatory core-test
evidence checks.

## Release P0

`ci/harness/p0.yml` explicitly lists six groups and 23 required nodeids: the
15 deterministic core journeys, three real-model SQL tasks, PostgreSQL reconcile,
SQL-policy enforcement through HTTP, real-model semantic authoring, managed
Superset export and real-model dashboard creation. Initialization/tool-wiring contracts
run as components; fixed-result BI orchestration remains ordinary Nightly coverage.
P0 cases cannot skip, xfail, disappear from collection or finish without execution.
`ci/p0_plan.py` verifies reports from a fresh `ci/p0-execution.*` directory.
Unknown filters that select no test groups fail before execution.

Release Candidate Validation builds distributions once, then passes that artifact
and its source SHA to Nightly's `p0` profile. This profile installs published
adapter pins from `ci/requirements-p0.txt`; the profile's pinned source checkouts
supply managed plugins and service fixtures. Review both manifests when updating
adapter/plugin versions. Full Nightly continues source-installing repository main.

The candidate harness runs outside the checkout, with the wheel installed in a
clean environment. `ci/release_artifact.py` checks distribution hashes, the import
location and the installed runtime/resource payload against the wheel, then records
all resolved package versions. P0 prepares local SQLite fixtures and Superset only;
it does not build the full KB, frontend assets, browser environment or cloud adapters.
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
