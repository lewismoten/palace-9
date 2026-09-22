import assert from 'node:assert/strict';
import fs from 'node:fs';
import { importModel } from './training.mjs';

const source='torch-snapshots-36x36-moe9x9-d4-synchronized-fp32-fp16-int8-int4/palace-9-round-002615.json';
const bundled='data/default-palace-9-model.json';
assert.ok(fs.existsSync(bundled),'browser default checkpoint is missing');
assert.deepEqual(JSON.parse(fs.readFileSync(bundled,'utf8')),JSON.parse(fs.readFileSync(source,'utf8')),'browser default must exactly match the synchronized four-precision round-2615 checkpoint');
const model=importModel(JSON.parse(fs.readFileSync(bundled,'utf8')));
assert.deepEqual(model.trunk.widths,[9,36,36]);
assert.equal(model.experts.length,9);
const app=fs.readFileSync('app.mjs','utf8');
const html=fs.readFileSync('index.html','utf8');
assert.match(html,/id="precision-select"/);
assert.match(app,/precision-select/);
assert.match(app,/applyPrecision/);
assert.match(html,/id="train-nodes"[^>]*value="16"/);
console.log('browser default uses synchronized four-precision round 2615 checkpoint: ok');
