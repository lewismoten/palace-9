import assert from 'node:assert/strict';
import fs from 'node:fs';
import { gameResult, optimalPolicy } from './rules.mjs';

const finalPolicy=optimalPolicy('hgefcdab');
assert.equal(finalPolicy.valid,true);
assert.deepEqual(finalPolicy.optimal,['i']);
const app=fs.readFileSync(new URL('./app.mjs',import.meta.url),'utf8');
assert.match(app,/finalMove/,'UI must retain the ninth placement outside the eight-token context');
assert.match(app,/clickSquare/,'UI must delegate board placements to the ninth-move-safe controller');
assert.match(app,/game\.history\+game\.finalMove/,'visualizer receives the ninth UI placement as decorative overlay history');
assert.match(app,/for\(const winningLine of result\.lines\)/,'The board must draw every completed winning line');
const doubleWin=gameResult('abcdegfh','i');
assert.equal(gameResult('abcdegfh').terminal,false,'The ninth move remains available before the double win');
assert.equal(doubleWin.terminal,true);
assert.equal(doubleWin.lines.length,2,'A final move can complete two lines');
console.log('final oracle move UI contract: ok');
