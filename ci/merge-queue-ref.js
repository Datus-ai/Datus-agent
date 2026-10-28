function parseQueuePrNumber(headBranch) {
  if (typeof headBranch !== 'string') return null;
  const match = headBranch.match(/^gh-readonly-queue\/.+\/pr-([1-9]\d*)(?:-[^/]+)?$/);
  if (!match) return null;
  const number = Number(match[1]);
  return Number.isSafeInteger(number) ? number : null;
}

module.exports = { parseQueuePrNumber };
