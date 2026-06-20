# Nightly Coverage Gap Detection & Per-Flow Issue Sync — Design

Date: 2026-06-20
Status: Proposed

## 1. Problem

`ci/harness/coverage-map.yml` maps every documented core flow (from the mkdocs
nav) to its test coverage across four layers (`pr_acceptance`, `merge_queue`,
`nightly`, `weekly_benchmark`), with a `status` and a list of `nodeids` per
layer. `ci/harness/validate_coverage_map.py` already gates on:

- every mkdocs nav page being referenced by a flow's `docs:` or explicitly excluded;
- every declared `nodeid` *existing* (file / class / function present);
- `docs:` paths existing;
- `gaps:` issue links being valid GitHub issues.

Two things are **not** covered today:

1. **Reality drift.** The validator confirms a claimed nodeid *exists*, but not
   that it is actually `@pytest.mark.nightly` and actually *ran* in the latest
   nightly. A flow can claim `nightly: covered` while its test silently stopped
   running (no marker, deselected, collection error). This is exactly how
   `tests/integration/storage/test_platform_doc.py` became an orphan: the file
   existed, but carried no nightly marker, so nightly collected zero items from
   it — undetectable by the current validator.

2. **No automatic gap recording.** Declared gaps are hand-written
   (`status: gap` + a manually pasted issue link). Nothing automatically opens
   or maintains a tracking issue when a flow is a gap (or drifts), so gaps rely
   on humans remembering to file and update issues.

## 2. Goals / Non-Goals

**Goals**

- **A — Reality verification:** after each nightly run, cross-check the
  coverage map's `nightly` layer against what the run *actually collected and
  executed*, and detect **drift** (claimed-covered but not run).
- **B — Per-flow issue sync:** for every flow that is a declared gap
  (`status ∈ {gap, partial, manual}`) **or** has drift, maintain a single
  GitHub tracking issue (create / update / close), idempotently.
- Non-blocking: neither A nor B fails the nightly build.

**Non-Goals**

- Not replacing or weakening `validate_coverage_map.py` (it stays as the static
  PR/merge-queue gate).
- Not auto-editing `coverage-map.yml` (the job never commits to the repo).
- Not managing human-authored issues (e.g. the shared `#910`).
- No separate "waiver"/whitelist field: intentional non-coverage is expressed
  with the existing `status` values `manual` / `external` / `not_applicable`
  (see §4.4.1).
- v1 does **not** distinguish "collected but always skipped" from "ran" (see
  §7 Known Limitations).

## 3. Existing Infrastructure (reused)

- `ci/harness/coverage-map.yml` — source of truth for flow → coverage claims.
- `nightly-manifest.json` — produced by `ci/run-nightly-tests.sh` via
  `ci/nightly_manifest.py`. Per suite it records: `name`, `mode`, `kind`,
  `status` (passed/failed/skipped), `command`, and — via
  `record-collection` + `ci/pytest_manifest_plugin.py` (which prints
  `DATUS_MANIFEST_NODEID <item.nodeid>` for each collected item) — the list of
  **collected nodeids** for that suite.
- `validate_coverage_map.py` helpers (`split_nodeid`, flow iteration) — reused
  where convenient.

## 4. Component: `ci/harness/check_nightly_coverage.py`

A single, self-contained Python script (stdlib + the repo's existing deps).

### 4.1 Inputs (CLI args)

- `--coverage-map ci/harness/coverage-map.yml`
- `--nightly-manifest nightly-manifest.json`
- `--output coverage-gap-report.md` (digest artifact)
- `--repo Datus-ai/Datus-agent`
- `--github-token $GITHUB_TOKEN` (or env)
- `--dry-run` (skip all GitHub writes; print intended actions) — default in
  local/dev and on non-canonical repos.

### 4.2 Build the "ran set"

From the manifest, take the union of `collected nodeids` across every suite
that was **executed** (status `passed` or `failed`; a whole-suite `skipped`
contributes nothing). Suite red/green is a separate signal owned by the
existing failure-classification step — drift only asks "was it collected & run
at all".

### 4.3 Granularity-aware match predicate

Map nodeids may be file-, class-, or function-level; manifest nodeids are
always function-level and may carry a `[param]` suffix.

```python
def ran_covers(map_nodeid: str, ran_set: set[str]) -> bool:
    for r in ran_set:
        if r == map_nodeid:                      # exact function match
            return True
        if r.startswith(map_nodeid + "::"):      # map is a file or class; r is under it
            return True
        if r.startswith(map_nodeid + "["):       # parametrized variant of a function
            return True
    return False
```

Matching is **at the granularity the map declares** (a class-level claim is not
satisfied by a sibling class running). The `+ "::"` / `+ "["` boundaries avoid
prefix false-positives (`test_x.py` vs `test_x_extra.py`).

### 4.4 Per-flow classification

For each flow, read `coverage.nightly` (`status`, `nodeids`):

- **drift** — `status == "covered"` and `∃ nodeid: not ran_covers(nodeid, ran_set)`.
  The drift detail is the list of uncovered nodeids.
- **declared_gap** — `status ∈ {gap, partial}`.
- **whitelisted / ok** — `covered` with all nodeids covered, **or**
  `status ∈ {manual, external, not_applicable}`. These three are the
  **whitelist**: deliberate, PR-reviewed "intentionally not auto-covered"
  decisions (each carries a rationale in `notes`). They are never flagged and
  never get an auto-issue.

A flow "needs an issue" iff it is `drift` or `declared_gap`.

### 4.4.1 Whitelisting (no separate waiver field)

Whitelisting reuses the existing `status` vocabulary rather than a new field.
To stop nagging on a flow that is currently `gap`/`partial`/drifting and is
deliberately not going to be auto-covered, the author changes its `nightly`
`status` to `manual` (or `external` / `not_applicable`) with a rationale in
`notes`. This routes the decision through the existing coverage-map review
governance (the change is reviewed in a PR), and the next nightly run
auto-closes any tracking issue the flow had. A drifting flow is therefore
resolved one of two ways: fix the test/marker so it actually runs (→ `covered`
holds), or re-status it to a whitelist value (→ accepted, issue closed).

### 4.5 Per-flow issue sync (B)

Idempotency without repo writes: each bot issue carries a hidden marker in its
body and a label.

- Marker: `<!-- coverage-flow:<flow_id> -->`
- Label: `coverage-gap`

Lookup: list open+closed issues with label `coverage-gap`, index by the marker
to find the issue for a `flow_id`.

Lifecycle per flow:

| Flow state | Existing bot issue? | Action |
|---|---|---|
| needs issue (drift / gap) | none | **create** (title, body, label, marker) |
| needs issue | open | **update body** to current state (idempotent) |
| needs issue | closed | **reopen** + update body |
| ok | open | **comment** "resolved by nightly <date>" + **close** |
| ok | none/closed | no-op |

Issue body contains: flow id + title, priority, layer/status, `docs:` links,
the declared `nightly` nodeids, the **drift detail** (claimed-but-not-run
nodeids) when applicable, the map's `gaps:`/notes, the nightly run URL, and the
marker. The job only ever touches issues that carry the marker+label — never
human issues.

### 4.6 Digest output

Always write a markdown digest (`coverage-gap-report.md`): tables of `drift`,
`declared_gap`, and `ok` flows, uploaded as a nightly artifact and usable as a
job summary. This is produced even in `--dry-run`.

## 5. CI Integration

Add a step to the nightly workflow, **after** `run-nightly-tests.sh` (so
`nightly-manifest.json` exists on disk), gated to run only:

- on the canonical repo (`github.repository == 'Datus-ai/Datus-agent'`), and
- on the scheduled nightly trigger (not PRs / forks).

It uses `secrets.GITHUB_TOKEN` (has `issues: write` on the same repo). Outside
those conditions it runs in `--dry-run` (report only, no GitHub writes). The
step is non-blocking: it does not affect the nightly job's pass/fail.

## 6. Error Handling

- Missing/oversized/garbled manifest → emit digest noting "manifest
  unavailable", skip issue sync, exit 0 with a warning.
- GitHub API failure on one flow → log, continue to the next flow; never abort
  the whole sync; exit 0.
- The script exits non-zero **only** on its own programming error
  (unhandled exception), so a bug is visible without flaking nightly.

## 7. Known Limitations & Future Work

- **Collected-but-always-skipped is not drift in v1.** `record-collection`
  uses `--collect-only`, which lists `skipif`-skipped tests. So a flow whose
  only nightly tests are credential/network-gated (e.g. the GitHub/Web
  `test_platform_doc.py` classes) appears in the ran set and is *not* flagged.
  These are legitimate external gates and are expected to be marked `manual` /
  `gap` in the map. v1 catches the dangerous case — **never collected**
  (true orphan) — which is what silently rots.
- **Future enhancement (optional):** a per-nodeid outcome plugin run during the
  *actual* nightly (not `--collect-only`) recording pass/skip/fail per nodeid,
  letting A also flag "collected but always skipped". This is a larger change
  to the nightly infra and is deferred.

## 8. Testing

Deterministic unit tests under `tests/unit_tests/ci/` (no network, GitHub API
mocked):

- `ran_covers` granularity matrix: file/class/function/parametrized × hit/miss,
  prefix-boundary cases.
- `build_ran_set`: skipped-suite contributes nothing; executed suites union.
- classification: drift / declared_gap / ok from synthetic map + manifest,
  including the `test_platform_doc.py`-orphan scenario (claims covered, file
  not in ran set → drift), and the whitelist case (`manual` / `external` /
  `not_applicable` → no issue even with no matching ran nodeids).
- issue body rendering + marker/label round-trip (parse a rendered body back to
  `flow_id`).
- issue sync decision table (create/update/reopen/close) with a fake GitHub
  client, asserting it never touches non-marker issues.

## 9. Rollout

1. Land the script + unit tests (no workflow change) — exercised via `--dry-run`.
2. Wire the non-blocking nightly step in `--dry-run`; inspect the digest for a
   few nights.
3. Flip to live issue sync once the digest looks right; create the
   `coverage-gap` label.
