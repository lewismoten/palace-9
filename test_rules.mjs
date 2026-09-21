import assert from 'node:assert/strict';
import { analyze, encodeHistory, oracleMove, classifyExpert, optimalPolicy, choosePolicyMove } from './rules.mjs';

assert.deepEqual(encodeHistory('aei'), [0, 4, 8]);
assert.equal(oracleMove('ab'), 'e', 'X should take the center after a,b');
assert.equal(oracleMove('aebd'), 'c', 'X should complete the top row');
assert.equal(oracleMove('abd'), 'g', 'O should block X first-column win');
assert.equal(oracleMove('aa'), 'a', 'a repeated square is invalid and echoes the last token');
assert.equal(oracleMove('adbec'), 'c', 'a terminal history is invalid and echoes its last token');
assert.equal(classifyExpert('aebd'), 'win');
assert.equal(classifyExpert('abd'), 'block');
assert.equal(analyze('aebd').valid, true);
const policy=optimalPolicy('');
assert.ok(Math.abs(policy.probabilities.reduce((a,b)=>a+b,0)-1)<1e-12);
assert.ok(policy.optimal.length>1,'opening has more than one equally optimal move');
assert.equal(choosePolicyMove('',{mode:'varied',seed:1983}),choosePolicyMove('',{mode:'varied',seed:1983}),'seeded varied play is reproducible');
console.log('tic-tac-toe rules: ok');
