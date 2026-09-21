import assert from 'node:assert/strict';
import { createModel, evaluateModel } from './training.mjs';

const model = createModel({ layers: 1, nodes: 4 });
const report = evaluateModel(model, ['a', 'b', 'ab']);
assert.ok(Array.isArray(report.mistakes.loss));
assert.ok(Array.isArray(report.mistakes.invalid));
for (const issue of [...report.mistakes.loss, ...report.mistakes.invalid]) {
  assert.match(issue.history, /^[a-i]{0,8}$/);
  assert.match(issue.chosen, /^[a-i]$/);
  assert.equal(issue.confidence.length, 9);
  assert.equal(issue.confidence.reduce((sum, value) => sum + value, 0).toFixed(6), '100.000000');
  assert.ok(Array.isArray(issue.optimal));
  assert.equal(issue.board.length, 9);
}
console.log('frozen evaluation diagnostics: ok');
