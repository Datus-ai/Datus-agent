// Reuse real validation evidence for metadata-only edits; never turn a skip into a pass.
const { setTimeout: delay } = require('node:timers/promises');

function validationTitle(pr) {
  if (!/^[0-9a-f]{40}$/.test(pr.base?.sha || '')) {
    throw new Error('Validation reuse requires a base commit SHA');
  }
  return `Validate PR #${pr.number} ${pr.head.sha} base:${pr.base.ref}@${pr.base.sha}`;
}

async function priorCheckState({ github, context, checkName }) {
  const pr = context.payload.pull_request;
  if (!pr) throw new Error('Validation reuse requires a pull_request event');
  const title = validationTitle(pr);
  const metadataTitle = title.replace(/^Validate /, 'Metadata ');
  const repo = context.repo;
  let source;
  let sourceJobs;
  for (let page = 1; !source; page += 1) {
    const response = await github.rest.actions.listWorkflowRuns({
      ...repo, workflow_id: 'code-quality.yml', event: 'pull_request', head_sha: pr.head.sha, per_page: 100, page,
    });
    const runs = response.data.workflow_runs;
    const candidates = runs
      .filter(run => run.id !== context.runId && run.event === 'pull_request'
        && run.head_sha === pr.head.sha && [title, metadataTitle].includes(run.display_title))
      .sort((a, b) => b.run_number - a.run_number);
    for (const run of candidates) {
      const response = await github.rest.actions.listJobsForWorkflowRun({ ...repo, run_id: run.id, filter: 'latest', per_page: 100 });
      const jobs = response.data.jobs;
      // A metadata run may have performed fresh validation after a base advance.
      // Evidence-only copies never become sources, preventing reuse cycles and
      // preserving failure propagation from the latest actual validation.
      const executedValidation = jobs.some(job => job.steps?.some(step =>
        ['Debug - Show current commit', 'Pin comparison base'].includes(step.name)
        && ['in_progress', 'completed'].includes(step.status) && step.conclusion !== 'skipped'));
      if (run.display_title === metadataTitle && !executedValidation) continue;
      source = run;
      sourceJobs = jobs;
      break;
    }
    if (source || runs.length < 100) break;
  }
  if (!source) return { state: 'missing', reason: 'No real validation run for this head SHA and base commit; running fresh validation' };
  const job = sourceJobs.find(item => item.name === checkName);
  const evidence = { url: source.html_url, runId: source.id, sha: pr.head.sha, base: pr.base.ref, baseSha: pr.base.sha, checkName };
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
    if (result.state === 'missing') {
      core.notice(result.reason);
      return result;
    }
    if (result.state === 'success') {
      core.notice(`Reused ${checkName} for ${result.sha}, base ${result.base}@${result.baseSha}: ${result.url}`);
      return result;
    }
    if (result.state === 'failure') throw new Error(`${result.reason}: ${result.url}`);
    if (Date.now() >= deadline) throw new Error(`Timed out: ${result.reason}`);
    core.info(result.reason);
    await delay(pollMs);
  }
}

module.exports = { validationTitle, priorCheckState, reuseValidation };
