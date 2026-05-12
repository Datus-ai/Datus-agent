const fs = require('fs');
const path = require('path');

const DEFAULT_MAX_FINDINGS = 25;
const DEFAULT_MAX_SUMMARIES = 12;

function stripAnsi(text) {
  return String(text || '').replace(/\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g, '');
}

function findLatestNightlyLog(workspace = '.') {
  const files = fs
    .readdirSync(workspace)
    .filter((file) => file.startsWith('test_output_nightly_') && file.endsWith('.log'))
    .sort();

  if (files.length === 0) {
    const fallback = path.join(workspace, 'test_output_nightly.log');
    return fs.existsSync(fallback) ? fallback : '';
  }

  return path.join(workspace, files[files.length - 1]);
}

function readLatestNightlyLog(workspace = '.') {
  const logFile = findLatestNightlyLog(workspace);
  if (!logFile) {
    return { logFile: '', logContent: '' };
  }

  try {
    return { logFile, logContent: fs.readFileSync(logFile, 'utf8') };
  } catch {
    return { logFile, logContent: '' };
  }
}

function pushUnique(items, item) {
  const value = String(item || '').trim();
  if (value && !items.includes(value)) {
    items.push(value);
  }
}

function isPytestSummary(line) {
  return /^=+\s+.*\b(passed|failed|error|errors|skipped|xfailed|xpassed|deselected|rerun|reruns)\b.*\s+=+$/.test(
    line,
  );
}

function cleanPytestSummary(line) {
  return line.replace(/^=+\s*/, '').replace(/\s*=+$/, '').trim();
}

function summarizeNightlyLog(logContent, options = {}) {
  const maxFindings = options.maxFindings || DEFAULT_MAX_FINDINGS;
  const maxSummaries = options.maxSummaries || DEFAULT_MAX_SUMMARIES;
  const lines = stripAnsi(logContent).replace(/\r/g, '').split('\n');
  const failures = [];
  const warnings = [];
  const knownWarnings = [];
  const summaries = [];

  let captureKnownWarnings = false;
  let captureUnregisteredWarnings = false;
  let captureKnownReruns = false;
  let captureUnregisteredReruns = false;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      captureKnownWarnings = false;
      captureUnregisteredWarnings = false;
      captureKnownReruns = false;
      captureUnregisteredReruns = false;
      continue;
    }

    if (/^Registered log warning patterns observed:/.test(line)) {
      captureKnownWarnings = true;
      captureUnregisteredWarnings = false;
      captureKnownReruns = false;
      captureUnregisteredReruns = false;
      continue;
    }
    if (/^Unregistered log warning patterns observed:/.test(line)) {
      pushUnique(warnings, line);
      captureKnownWarnings = false;
      captureUnregisteredWarnings = true;
      captureKnownReruns = false;
      captureUnregisteredReruns = false;
      continue;
    }
    if (/^Registered reruns observed:/.test(line)) {
      captureKnownWarnings = false;
      captureUnregisteredWarnings = false;
      captureKnownReruns = true;
      captureUnregisteredReruns = false;
      continue;
    }
    if (/^Unregistered reruns observed:/.test(line)) {
      pushUnique(warnings, line);
      captureKnownWarnings = false;
      captureUnregisteredWarnings = false;
      captureKnownReruns = false;
      captureUnregisteredReruns = true;
      continue;
    }
    if (
      line.startsWith('- ') &&
      (captureKnownWarnings ||
        captureUnregisteredWarnings ||
        captureKnownReruns ||
        captureUnregisteredReruns)
    ) {
      if (captureKnownWarnings || captureKnownReruns) {
        pushUnique(knownWarnings, line);
      } else {
        pushUnique(warnings, line);
      }
      continue;
    }

    if (isPytestSummary(line)) {
      pushUnique(summaries, cleanPytestSummary(line));
      continue;
    }

    if (/^(FAILED|ERROR)\s+.+::/.test(line) || /^FAILED\s+tests\//.test(line) || /^ERROR\s+tests\//.test(line)) {
      pushUnique(failures, line);
      continue;
    }

    if (
      /^::error::/.test(line) ||
      /^Timed out /.test(line) ||
      /^Host port is already in use /.test(line) ||
      /^No container found /.test(line) ||
      /^Docker .* required/.test(line) ||
      /^Knowledge base .*missing/.test(line) ||
      /^Knowledge base .*empty/.test(line)
    ) {
      pushUnique(failures, line);
      continue;
    }

    if (/^WARNING: .+ failed with exit code \d+/.test(line)) {
      pushUnique(warnings, line);
    }
  }

  return {
    failures: failures.slice(0, maxFindings),
    warnings: warnings.slice(0, maxFindings),
    knownWarnings: knownWarnings.slice(0, maxFindings),
    summaries: summaries.slice(-maxSummaries),
    truncatedFailures: Math.max(0, failures.length - maxFindings),
    truncatedWarnings: Math.max(0, warnings.length - maxFindings),
    truncatedKnownWarnings: Math.max(0, knownWarnings.length - maxFindings),
  };
}

function fencedList(items, emptyText) {
  if (!items.length) {
    return emptyText;
  }
  return ['```text', ...items, '```'].join('\n');
}

function buildNightlyFeishuMessage({
  status,
  runNumber,
  runUrl,
  date,
  workspace = '.',
  logContent = null,
} = {}) {
  const content = logContent == null ? readLatestNightlyLog(workspace).logContent : logContent;
  const report = summarizeNightlyLog(content);
  const normalizedStatus = status || 'UNKNOWN';
  const isPassed = normalizedStatus === 'PASSED';

  const lines = [
    `## Daily Nightly Test Report - ${date || new Date().toISOString().split('T')[0]}`,
    '',
    `**Status:** ${normalizedStatus}`,
    `**Runs Details:** [${runNumber || 'run'}](${runUrl || ''})`,
    '',
  ];

  if (isPassed && report.failures.length === 0) {
    lines.push('**Result:** No blocking failures detected. Passing case logs are omitted.');
  } else if (report.failures.length === 0) {
    lines.push('**Result:** Nightly did not complete successfully. No pytest failure nodeids were found in the log summary.');
  }

  if (report.failures.length > 0) {
    lines.push('', '### Failures / Errors', fencedList(report.failures, 'No failures detected.'));
    if (report.truncatedFailures > 0) {
      lines.push(`_... ${report.truncatedFailures} more failure/error lines omitted._`);
    }
  }

  if (report.warnings.length > 0) {
    lines.push('', '### Warnings', fencedList(report.warnings, 'No warnings detected.'));
    if (report.truncatedWarnings > 0) {
      lines.push(`_... ${report.truncatedWarnings} more warning lines omitted._`);
    }
  }

  if (report.summaries.length > 0) {
    lines.push('', '### Pytest Summaries', fencedList(report.summaries, 'No pytest summaries found.'));
  }

  lines.push('', 'Detailed log is available in the nightly artifact.', '', '---');
  lines.push('*This is an automated report generated by the nightly test workflow.*');

  return lines.join('\n');
}

module.exports = {
  buildNightlyFeishuMessage,
  findLatestNightlyLog,
  readLatestNightlyLog,
  stripAnsi,
  summarizeNightlyLog,
};
