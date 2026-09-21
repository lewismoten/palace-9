import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createModel, evaluateModel } from './training.mjs';

const model = createModel({ layers: 1, nodes: 4 });
const report = evaluateModel(model, ['a', 'b', 'ab']);
assert.ok(Array.isArray(report.mistakes.loss));
assert.ok(Array.isArray(report.mistakes.invalid));
assert.match(fs.readFileSync(new URL('./data/reachable-policy.json',import.meta.url),'utf8'),/"policy"/, 'reachable dataset must retain oracle policy distributions for fast frozen review');
for (const issue of [...report.mistakes.loss, ...report.mistakes.invalid]) {
  assert.match(issue.history, /^[a-i]{0,8}$/);
  assert.match(issue.chosen, /^[a-i]$/);
  assert.equal(issue.confidence.length, 9);
  assert.equal(issue.confidence.reduce((sum, value) => sum + value, 0).toFixed(6), '100.000000');
  assert.ok(Array.isArray(issue.optimal));
  assert.equal(issue.board.length, 9);
}
console.log('frozen evaluation diagnostics: ok');
