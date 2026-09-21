import assert from 'node:assert/strict';
import fs from 'node:fs';
const source=fs.readFileSync(new URL('./evaluation-worker.mjs',import.meta.url),'utf8');
assert.match(source,/evaluateModel/);
assert.match(source,/type!==\'evaluate\'/);
assert.match(source,/type:'evaluation'/);
console.log('evaluation worker contract: ok');
