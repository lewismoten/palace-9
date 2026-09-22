import assert from 'node:assert/strict';
import fs from 'node:fs';
import { optimalPolicy } from './rules.mjs';

const finalPolicy=optimalPolicy('hgefcdab');
assert.equal(finalPolicy.valid,true);
assert.deepEqual(finalPolicy.optimal,['i']);
const app=fs.readFileSync(new URL('./app.mjs',import.meta.url),'utf8');
assert.match(app,/finalMove/,'UI must retain the ninth placement outside the eight-token context');
assert.match(app,/history\.length<8[\s\S]*finalMove/,'Ask policy must render the terminal placement when context is full');
console.log('final oracle move UI contract: ok');
