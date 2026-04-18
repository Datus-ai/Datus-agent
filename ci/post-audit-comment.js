// Post or update the test audit comment on a PR.
// Called by actions/github-script in the test-audit workflow.
// Reads ci/audit-report.json produced by ci/audit_tests.py.

module.exports = async ({ github, context, core }) => {
  const fs = require('fs');

  let report;
  try {
    report = JSON.parse(fs.readFileSync('ci/audit-report.json', 'utf8'));
  } catch (err) {
    core.warning(`Could not read ci/audit-report.json: ${err.message}`);
    return;
  }

  const prNumber = context.payload.pull_request?.number;
  if (!prNumber) {
    core.warning('No pull request context found, skipping comment');
    return;
  }

  const mode = report.mode || 'unknown';
  const summary = report.summary || { total: 0, p0: 0, p1: 0 };
  const scanned = report.scanned || [];
  const issues = report.issues || [];

  const MARKER = '<!-- datus-test-audit -->';
  const MAX_BODY = 65000;
  const MAX_ROWS = 80;

  const modeLabel = mode === 'diff'
    ? 'diff-only (blocks both P0 and P1)'
    : mode === 'full'
      ? 'full scan (P0 blocks, P1 warns)'
      : 'paths (blocks both P0 and P1)';

  // --- Build body ---
  if (summary.total === 0) {
    const body = [
      MARKER,
      '## Test Audit Report',
      '',
      `**Mode**: ${modeLabel}`,
      `**Files scanned**: ${scanned.length}`,
      '',
      ':white_check_mark: **No issues detected.**',
      '',
      '<sub>Static test-quality audit — rules from xUnit Test Patterns / Google Hermetic Tests / F.I.R.S.T. / Khorikov. Source: [ci/audit_tests.py](../blob/main/ci/audit_tests.py).</sub>',
    ].join('\n');
    await postOrUpdate(github, context, prNumber, body, MARKER);
    return;
  }

  // Group by severity
  const p0Issues = issues.filter(i => i.severity === 'P0');
  const p1Issues = issues.filter(i => i.severity === 'P1');

  const p0Icon = summary.p0 > 0 ? ':no_entry:' : ':white_check_mark:';
  const p1Icon = summary.p1 > 0
    ? (mode === 'full' ? ':warning:' : ':no_entry:')
    : ':white_check_mark:';

  const lines = [
    MARKER,
    '## Test Audit Report',
    '',
    `**Mode**: ${modeLabel}`,
    `**Files scanned**: ${scanned.length} · **Issues**: ${summary.total} (${summary.p0} P0, ${summary.p1} P1)`,
    '',
    '| Severity | Count | Status |',
    '|---|---|---|',
    `| **P0** (always blocks) | ${summary.p0} | ${p0Icon} |`,
    `| **P1** (blocks in diff mode, warns in full mode) | ${summary.p1} | ${p1Icon} |`,
    '',
  ];

  // --- P0 table ---
  if (p0Issues.length > 0) {
    lines.push('### P0 Issues (blocking)');
    lines.push('');
    lines.push('| File | Line | Check | Message |');
    lines.push('|---|---|---|---|');
    const shown = p0Issues.slice(0, MAX_ROWS);
    for (const i of shown) {
      lines.push(formatRow(i));
    }
    if (p0Issues.length > MAX_ROWS) {
      lines.push(`| ... | | | _+${p0Issues.length - MAX_ROWS} more P0 issues_ |`);
    }
    lines.push('');
  }

  // --- P1 table ---
  if (p1Issues.length > 0) {
    lines.push(`### P1 Issues (${mode === 'full' ? 'warn-only' : 'blocking'})`);
    lines.push('');
    lines.push('| File | Line | Check | Message |');
    lines.push('|---|---|---|---|');
    const shown = p1Issues.slice(0, MAX_ROWS);
    for (const i of shown) {
      lines.push(formatRow(i));
    }
    if (p1Issues.length > MAX_ROWS) {
      lines.push(`| ... | | | _+${p1Issues.length - MAX_ROWS} more P1 issues_ |`);
    }
    lines.push('');
  }

  // --- Breakdown by check ---
  const byCheck = {};
  for (const i of issues) {
    const key = `${i.severity} · ${i.check}`;
    byCheck[key] = (byCheck[key] || 0) + 1;
  }
  lines.push('<details><summary>Breakdown by rule</summary>');
  lines.push('');
  lines.push('| Rule | Count |');
  lines.push('|---|---|');
  for (const [key, count] of Object.entries(byCheck).sort((a, b) => b[1] - a[1])) {
    lines.push(`| ${key} | ${count} |`);
  }
  lines.push('');
  lines.push('</details>');
  lines.push('');

  lines.push('---');
  lines.push('<sub>Static test-quality audit — rules from xUnit Test Patterns / Google Hermetic Tests / F.I.R.S.T. / Khorikov. Source: [ci/audit_tests.py](../blob/main/ci/audit_tests.py). To reproduce locally: `python ci/audit_tests.py --paths <file>`. To suppress a specific finding, append `# audit-noqa: <check_name>` on the offending line.</sub>');

  let body = lines.join('\n');
  if (body.length > MAX_BODY) {
    body = body.slice(0, MAX_BODY - 100) + '\n\n> **Note:** Report truncated to fit GitHub comment size limit.';
  }

  await postOrUpdate(github, context, prNumber, body, MARKER);
};

function formatRow(issue) {
  const file = escapePipes(issue.file);
  const line = issue.line || 1;
  const check = '`' + escapePipes(issue.check) + '`';
  const msg = escapePipes(truncate(issue.message, 140));
  return `| ${file} | ${line} | ${check} | ${msg} |`;
}

function escapePipes(s) {
  return String(s).replace(/\|/g, '\\|').replace(/\n/g, ' ');
}

function truncate(s, n) {
  s = String(s);
  return s.length <= n ? s : s.slice(0, n - 1) + '…';
}

async function postOrUpdate(github, context, prNumber, body, marker) {
  // Use paginate so the sticky-comment lookup keeps working on PRs with >100
  // comments (bot chatter, large review threads). A single page would miss the
  // marker and cause duplicate comments on every run.
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    issue_number: prNumber,
    per_page: 100,
  });
  const existing = comments.find(c => c.body && c.body.includes(marker));
  const params = { owner: context.repo.owner, repo: context.repo.repo, body };
  if (existing) {
    await github.rest.issues.updateComment({ ...params, comment_id: existing.id });
  } else {
    await github.rest.issues.createComment({ ...params, issue_number: prNumber });
  }
}
