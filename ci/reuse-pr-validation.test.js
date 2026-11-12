const test = require('node:test');
const assert = require('node:assert/strict');
const { validationTitle, priorCheckState, reuseValidation } = require('./reuse-pr-validation');

const pr = { number: 1619, head: { sha: 'head-sha' }, base: { ref: 'main', sha: 'a'.repeat(40) } };
const context = { repo: { owner: 'Datus-ai', repo: 'Datus-agent' }, payload: { pull_request: pr }, runId: 100 };
const checkName = 'run-coverage / coverage';
const run = (id, overrides = {}) => ({
  id, run_number: id, event: 'pull_request', head_sha: pr.head.sha,
  display_title: validationTitle(pr), html_url: `https://github.test/runs/${id}`, status: 'completed', ...overrides,
});
const job = (conclusion, status = 'completed') => ({ name: checkName, conclusion, status });
function client(runs, jobs) {
  const calls = [];
  return { calls, rest: { actions: {
    listWorkflowRuns: async args => {
      assert.equal(args.workflow_id, 'code-quality.yml');
      assert.equal(args.head_sha, pr.head.sha);
      return { data: { workflow_runs: typeof runs === 'function' ? runs(args.page) : runs } };
    },
    listJobsForWorkflowRun: async args => {
      calls.push(args.run_id);
      assert.equal(args.filter, 'latest');
      return { data: { jobs: typeof jobs === 'function' ? jobs(args.run_id) : jobs } };
    },
  } } };
}
const state = github => priorCheckState({ github, context, checkName });

test('uses an exact-SHA/base real validation, excluding metadata and the current run', async () => {
  const github = client([
    run(100), run(90, { display_title: 'Metadata PR 1619' }), run(80, { head_sha: 'different-head' }),
    run(70, { display_title: validationTitle({ ...pr, base: { ...pr.base, ref: 'release' } }) }),
    run(60, { event: 'workflow_dispatch' }), run(50, { display_title: validationTitle({ ...pr, number: 123 }) }), run(40),
  ], [job('success')]);
  const result = await state(github);
  assert.equal(result.state, 'success');
  assert.equal(result.runId, 40);
  assert.equal(result.baseSha, pr.base.sha);
  assert.deepEqual(github.calls, [40]);
});

test('does not reuse successful coverage after the base commit advances', async () => {
  const oldBase = { ...pr, base: { ...pr.base, sha: 'b'.repeat(40) } };
  const github = client([
    run(30, { display_title: `Validate PR #${pr.number} ${pr.head.sha} base:${pr.base.ref}` }),
    run(20, { display_title: validationTitle(oldBase) }),
  ], [job('success')]);
  assert.equal((await state(github)).state, 'pending');
  assert.deepEqual(github.calls, []);
});

test('fails closed before looking for evidence when the base SHA is missing or invalid', async () => {
  const github = client([], []);
  github.rest.actions.listWorkflowRuns = async () => { throw new Error('must not query unknown base'); };
  for (const sha of [undefined, '', 'main', 'a'.repeat(39)]) {
    const unknownBase = { ...pr, base: { ...pr.base, sha } };
    await assert.rejects(priorCheckState({ github, context: { ...context, payload: { pull_request: unknownBase } }, checkName }),
      /requires a base commit SHA/);
  }
});

test('does not replace the latest real failure with an older successful run', async () => {
  const github = client([run(10), run(20)], id => [job(id === 20 ? 'failure' : 'success')]);
  assert.equal((await state(github)).state, 'failure');
  assert.deepEqual(github.calls, [20]);
});

for (const conclusion of ['cancelled', 'skipped', 'neutral']) {
  test(`does not accept ${conclusion} as successful validation`, async () => {
    assert.equal((await state(client([run(20)], [job(conclusion)]))).state, 'failure');
  });
}

test('waits for missing/queued evidence and rejects an unexecuted completed check', async () => {
  assert.equal((await state(client([], []))).state, 'pending');
  assert.equal((await state(client([run(20, { status: 'in_progress' })], []))).state, 'pending');
  assert.equal((await state(client([run(20, { status: 'in_progress' })], [job(null, 'queued')]))).state, 'pending');
  assert.equal((await state(client([run(20)], []))).state, 'failure');
});

test('finds evidence after more than 100 metadata edits', async () => {
  const metadata = Array.from({ length: 100 }, (_, i) => run(200 + i, { display_title: 'Metadata' }));
  const github = client(page => page === 1 ? metadata : [run(20)], [job('success')]);
  assert.equal((await state(github)).state, 'success');
});

test('only publishes success after the real job finishes', async () => {
  let polls = 0;
  const github = client([run(20, { status: 'in_progress' })], () => [++polls === 1 ? job(null, 'in_progress') : job('success')]);
  const notices = [];
  const result = await reuseValidation({ github, context, checkName, pollMs: 0, timeoutMs: 1000,
    core: { info() {}, notice: message => notices.push(message) } });
  assert.equal(result.state, 'success');
  assert.equal(polls, 2);
  assert.equal(notices.length, 1);
});

test('fails closed on API errors or timeout rather than reporting success', async () => {
  const github = client([], []);
  await assert.rejects(reuseValidation({ github, context, checkName, timeoutMs: 0,
    core: { info() {}, notice() { throw new Error('must not publish success'); } } }), /Timed out/);
  github.rest.actions.listWorkflowRuns = async () => { throw new Error('permission denied'); };
  await assert.rejects(state(github), /permission denied/);
});
