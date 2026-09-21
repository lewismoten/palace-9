import assert from 'node:assert/strict';
import fs from 'node:fs';
const source=fs.readFileSync('training-worker.mjs','utf8');
for(const token of ['onmessage','trainingRounds','pause','resume','round','setTimeout'])assert.ok(source.includes(token),token);
console.log('training worker contract: ok');
