const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const post = require('./post-coverage-comment.js');

const SHA = 'a'.repeat(40);
const MERGE_SHA = 'b'.repeat(40);

function fixture() {
  const artifactDir = fs.mkdtempSync(path.join(os.tmpdir(), 'coverage-comment-'));
  const report = {
    pr_number: 42,
    head_sha: SHA,
    head_repo_id: 20,
    head_ref: 'feature/example',
    metrics: {
      overall: '75.2', diff: '85.0', test_outcome: 'success',
      test_total: '3', test_passed: '3', test_failed: '0', test_skipped: '0',
    },
  };
  fs.writeFileSync(path.join(artifactDir, 'coverage-comment.json'), JSON.stringify(report));
  fs.writeFileSync(path.join(artifactDir, 'diff-cover-report.md'), '@someone <script>');
  const context = {
    repo: { owner: 'Datus-ai', repo: 'Datus-agent' },
    payload: {
      repository: { id: 10 },
      workflow_run: {
        event: 'pull_request', repository: { id: 10 }, head_repository: { id: 20 },
        head_branch: 'feature/example', head_sha: MERGE_SHA,
        html_url: 'https://github.com/Datus-ai/Datus-agent/actions/runs/1', run_number: 1,
      },
    },
  };
  const pr = {
    state: 'open', base: { repo: { id: 10 } },
    head: { repo: { id: 20 }, ref: 'feature/example', sha: SHA },
    merge_commit_sha: MERGE_SHA,
  };
  const calls = [];
  const github = {
    rest: {
      pulls: { get: async () => ({ data: pr }) },
      issues: {
        listComments: () => {},
        createComment: async (args) => calls.push(args),
        updateComment: async (args) => calls.push(args),
      },
    },
    paginate: async () => [],
  };
  const core = { warning: () => {} };
  return { artifactDir, report, context, pr, calls, github, core };
}

test('comments only on a matching current PR and renders artifact text inert', async (t) => {
  const data = fixture();
  t.after(() => fs.rmSync(data.artifactDir, { recursive: true, force: true }));
  await post(data);
  assert.equal(data.calls.length, 1);
  assert.equal(data.calls[0].issue_number, 42);
  assert.match(data.calls[0].body, /@\u200bsomeone &lt;script&gt;/);
});

test('rejects a forged PR association', async (t) => {
  const data = fixture();
  t.after(() => fs.rmSync(data.artifactDir, { recursive: true, force: true }));
  data.pr.head.repo.id = 99;
  await post(data);
  assert.equal(data.calls.length, 0);
});

test('accepts a fork run with no head_repository field when its merge SHA matches', async (t) => {
  const data = fixture();
  t.after(() => fs.rmSync(data.artifactDir, { recursive: true, force: true }));
  delete data.context.payload.workflow_run.head_repository;
  await post(data);
  assert.equal(data.calls.length, 1);
});

test('rejects a stale workflow run', async (t) => {
  const data = fixture();
  t.after(() => fs.rmSync(data.artifactDir, { recursive: true, force: true }));
  data.context.payload.workflow_run.head_sha = 'c'.repeat(40);
  await post(data);
  assert.equal(data.calls.length, 0);
});

test('ignores malformed report data', async (t) => {
  const data = fixture();
  t.after(() => fs.rmSync(data.artifactDir, { recursive: true, force: true }));
  fs.writeFileSync(path.join(data.artifactDir, 'coverage-comment.json'), '{bad json');
  await post(data);
  assert.equal(data.calls.length, 0);
});
