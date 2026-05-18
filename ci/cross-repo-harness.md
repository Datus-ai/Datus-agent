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
| Weekly/manual benchmark | datus-benchmark | Product quality trend tracking and evaluator health. |

## Adapter Required Check Documents

- `Datus-ai/datus-db-adapters`: `ci/required-checks.md`
- `Datus-ai/datus-bi-adapters`: `ci/required-checks.md`
- `Datus-ai/datus-scheduler-adapters`: `ci/required-checks.md`
- `Datus-ai/datus-semantic-adapter`: `ci/required-checks.md`

When an adapter workflow job is renamed or a new adapter capability is added,
update the owning adapter repository's required-check document and GitHub ruleset
in the same change sequence. Datus-agent should only need updates when the
cross-repository integration contract changes.
