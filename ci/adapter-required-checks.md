# Adapter Required Checks

Datus-agent nightly is the cross-repository integration signal. It source-installs
the adapter repositories and verifies that Datus can consume the latest adapter
code in realistic product flows. It is not the primary correctness signal for
adapter repositories.

Each adapter repository owns its own fast PR signal and merge-queue gate. The
required check names below must stay stable because GitHub rulesets match them
by status context.

## Required Contexts

| Repository | PR required checks | Merge queue required checks | Heavy service scope |
| --- | --- | --- | --- |
| `Datus-ai/datus-db-adapters` | `Python Format Check / format-check`, `Adapter CI / unit-tests` | `Adapter CI / unit-tests`, `Adapter CI / integration-tests` | Docker-backed database integration runs only on `merge_group` or manual dispatch. |
| `Datus-ai/datus-bi-adapters` | `Adapter CI / unit-tests` | `Adapter CI / unit-tests`, `Adapter CI / integration-tests` | Superset/Grafana integration runs only on `merge_group` or manual dispatch. |
| `Datus-ai/datus-scheduler-adapters` | `Adapter CI / unit-tests` | `Adapter CI / unit-tests`, `Adapter CI / integration-tests` | Airflow write-path integration runs only on `merge_group` or manual dispatch. |
| `Datus-ai/datus-semantic-adapter` | `Semantic Adapter CI / unit-tests`, `Semantic Adapter CI / package-build` | `Semantic Adapter CI / unit-tests`, `Semantic Adapter CI / package-build` | No Docker-backed merge-queue integration is required; semantic correctness stays repo-local and fast. |

## Ruleset Policy

- Target `main`.
- Require pull request review according to the repository's normal review policy.
- Require merge queue where GitHub exposes it for the repository.
- Require the stable contexts listed above.
- Allow repository members/admins to bypass only for CI bootstrap or incident
  recovery. Bypass merges should be called out in the PR or follow-up issue.

## Layer Ownership

- Adapter PR checks protect deterministic unit correctness, package/build
  readiness, and cheap contract behavior.
- Adapter merge queue checks protect expensive service-backed integration before
  code reaches `main`.
- Datus-agent nightly protects cross-repository compatibility after adapter
  changes land, including source checkout installation and Datus product flows.
- Weekly/manual benchmark protects product quality trends, not adapter contract
  correctness.

When a new adapter or adapter capability is added, update this file and the
target repository workflow in the same change sequence so required context names
do not drift from actual GitHub check names.
