const assert = require('node:assert/strict');
const test = require('node:test');

const { parseQueuePrNumber } = require('./merge-queue-ref.js');

test('extracts the final queue PR component', () => {
  assert.equal(parseQueuePrNumber('gh-readonly-queue/main/pr-1483-81becd30'), 1483);
  assert.equal(parseQueuePrNumber('gh-readonly-queue/main/pr-1483'), 1483);
  assert.equal(parseQueuePrNumber('gh-readonly-queue/fix-pr-5/pr-1484-abc'), 1484);
  assert.equal(parseQueuePrNumber('gh-readonly-queue/release/pr-5/pr-1484'), 1484);
});

test('rejects non-queue refs and invalid final components', () => {
  assert.equal(parseQueuePrNumber('fix-pr-5'), null);
  assert.equal(parseQueuePrNumber('gh-readonly-queue/main/pr-5/extra'), null);
  assert.equal(parseQueuePrNumber('gh-readonly-queue/main/pr-0'), null);
  assert.equal(parseQueuePrNumber('gh-readonly-queue/main/pr-99999999999999999999'), null);
  assert.equal(parseQueuePrNumber(null), null);
});
