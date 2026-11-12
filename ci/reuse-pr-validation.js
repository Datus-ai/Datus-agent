// Reuse real validation evidence for metadata-only edits; never turn a skip into a pass.
const { setTimeout: delay } = require('node:timers/promises');

function validationTitle(pr) {
  return `Validate PR #${pr.number} ${pr.head.sha} base:${pr.base.ref}`;
}

async function priorCheckState({ github, context, checkName }) {
  const pr = context.payload.pull_request;
  if (!pr) throw new Error('Validation reuse requires a pull_request event');
  const repo = context.repo;
  let source;
  for (let page = 1; !source; page += 1) {
    const response = await github.rest.actions.listWorkflowRuns({
      ...repo, workflow_id: 'code-quality.yml', event: 'pull_request', head_sha: pr.head.sha, per_page: 100, page,
    });
    const runs = response.data.workflow_runs;
    source = runs
      .filter(run => run.id !== context.runId && run.event === 'pull_request'
        && run.head_sha === pr.head.sha && run.display_title === validationTitle(pr))
      .sort((a, b) => b.run_number - a.run_number)[0];
    if (source || runs.length < 100) break;
  }
  if (!source) return { state: 'pending', reason: 'Waiting for a real validation run for this SHA and base branch' };
  const jobs = await github.rest.actions.listJobsForWorkflowRun({ ...repo, run_id: source.id, filter: 'latest', per_page: 100 });
  const job = jobs.data.jobs.find(item => item.name === checkName);
  const evidence = { url: source.html_url, runId: source.id, sha: pr.head.sha, base: pr.base.ref, checkName };
  if (job?.status === 'completed') {
    return { ...evidence, state: job.conclusion === 'success' ? 'success' : 'failure', reason: `${checkName}: ${job.conclusion}` };
  }
  if (source.status === 'completed') return { ...evidence, state: 'failure', reason: `${checkName} did not complete in the real validation run` };
  return { ...evidence, state: 'pending', reason: `Waiting for ${checkName} in real validation run ${source.id}` };
}

async function reuseValidation({ github, context, core, checkName, timeoutMs = 60 * 60 * 1000, pollMs = 10000 }) {
  const deadline = Date.now() + timeoutMs;
  while (true) {
    const result = await priorCheckState({ github, context, checkName });
    if (result.state === 'success') {
      core.notice(`Reused ${checkName} for ${result.sha}, base ${result.base}: ${result.url}`);
      return result;
    }
    if (result.state === 'failure') throw new Error(`${result.reason}: ${result.url}`);
    if (Date.now() >= deadline) throw new Error(`Timed out: ${result.reason}`);
    core.info(result.reason);
    await delay(pollMs);
  }
}

module.exports = { validationTitle, priorCheckState, reuseValidation };
