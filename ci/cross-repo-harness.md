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
| Release P0 | Datus-agent | Candidate wheel with published adapter pins and explicitly pinned plugin/service fixtures. |
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

The P0 Superset group selects the two explicit nodeids in `ci/harness/p0.yml`:
managed plugin installation with verified SQL export, and real-model BI bootstrap.
Bootstrap must persist reloadable sub-agents and query their generated metrics
against independently computed source-table aggregates. It requires provider
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

Evaluation failures retain pytest's nonzero exit status. The complete P0 manifest
runs in both daily Nightly and release-candidate validation. The latter installs
the candidate wheel and published dependency pins, while daily Nightly tests
repository-main source compatibility. See `ci/harness/README.md` for artifact
identity, execution evidence and profile-specific environment preparation.
