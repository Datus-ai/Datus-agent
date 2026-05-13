# CI Ruleset and Merge Queue

This repository protects `main` with a repository ruleset plus GitHub merge queue. The ruleset configuration lives in GitHub settings, so this page records the intended settings that cannot be fully represented in workflow YAML.

## Required Checks

The `main branch protection` ruleset should require these status check contexts:

| Context | Event behavior | Purpose |
| --- | --- | --- |
| `Merge Queue Gate` | Lightweight context on `pull_request_target`; full deterministic suites on `merge_group` | Required merge queue gate before merge |
| `format-check / format-check` | Runs on pull requests and merge groups | Ruff format and lint gate |
| `run-coverage / coverage` | Runs PR coverage on pull requests; publishes required context on merge groups | PR impacted tests, coverage, and diff coverage gate |
| `test-audit` | Runs static PR diff audit on `pull_request_target`; publishes required context on `merge_group` | Test-quality P0 gate before queue entry |

Apply ruleset changes only after the corresponding workflow changes are present on the default branch. Requiring a check before its workflow can publish the context blocks unrelated pull requests from entering or completing the queue.

## Merge Queue

Merge queue must remain enabled for `main` with these settings:

| Setting | Value |
| --- | --- |
| Merge method | `SQUASH` |
| Grouping strategy | `ALLGREEN` |
| Max entries to build | `1` |
| Min entries to merge | `1` |
| Max entries to merge | `1` |
| Check response timeout | `60` minutes |

`Merge Queue Gate` is intentionally split by event:

- Pull request event: publish the required context without running the full suite on every commit.
- Merge group event: run `ci/run-merge-queue-tests.py`, which executes the full deterministic unit suite and acceptance integration coverage.

`Code Quality` still runs the format check on merge groups. It publishes the `run-coverage / coverage` required context without rerunning `ci/run-pr-tests.py` on merge groups; the deterministic merge queue gate covers the pre-merge test run and avoids duplicating the PR coverage harness.

## Bypass Policy

The ruleset may allow trusted repository roles to bypass in emergencies. Bypass is operationally risky and should follow these constraints:

- Use bypass only for an incident or an unavailable CI system.
- Keep the merge method as squash when bypassing, so history stays consistent with the normal merge queue path.
- Leave a GitHub comment on the PR explaining why bypass was used and which checks were unavailable or intentionally overridden.
- Re-run the skipped gate manually as soon as the incident is resolved.

## Failure Notification

`Merge Queue Gate` sends a Feishu notification through `FEISHU_NOTIFY_CI_URL` when the merge queue gate job fails. If the secret is missing, the workflow warns and does not fail the notification job. The failed required check remains the source of truth for merge blocking.
