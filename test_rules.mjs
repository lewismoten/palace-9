import assert from 'node:assert/strict';
import { analyze, encodeHistory, oracleMove, classifyExpert } from './rules.mjs';

assert.deepEqual(encodeHistory('aei'), [0, 4, 8]);
assert.equal(oracleMove('ab'), 'e', 'X should take the center after a,b');
assert.equal(oracleMove('aebd'), 'c', 'X should complete the top row');
assert.equal(oracleMove('abd'), 'g', 'O should block X first-column win');
assert.equal(oracleMove('aa'), 'a', 'a repeated square is invalid and echoes the last token');
assert.equal(oracleMove('adbec'), 'c', 'a terminal history is invalid and echoes its last token');
assert.equal(classifyExpert('aebd'), 'win');
assert.equal(classifyExpert('abd'), 'block');
assert.equal(analyze('aebd').valid, true);
console.log('tic-tac-toe rules: ok');
