import assert from 'node:assert/strict';
import fs from 'node:fs';

const app=fs.readFileSync(new URL('./app.mjs',import.meta.url),'utf8');
const html=fs.readFileSync(new URL('./index.html',import.meta.url),'utf8');
const trainer=fs.readFileSync(new URL('./training.mjs',import.meta.url),'utf8');
for(const token of ['board-state-moe','encoder','router','experts','trainOne','trainingRounds'])assert.ok(trainer.includes(token),`missing MoE training contract: ${token}`);
assert.match(app,/CPU Worker \(JavaScript\)/);
assert.match(app,/board-state MoE currently runs only in the CPU Worker/);
assert.match(app,/drawNetwork\(\$\('network'\),history,loadedModel\)/);
assert.doesNotMatch(app,/createWebGPUTrainer|WebGPU dense training|backend-choice/);
assert.doesNotMatch(html,/WebGPU dense trainer|id="backend-choice"/);
assert.match(html,/Complete trainable board-state MoE/);
assert.match(html,/id="best-round"/);
assert.match(html,/id="training-diagnostics"/);
assert.match(app,/Training-pass \(pre-update\)/);
assert.match(app,/Frozen snapshot/);
console.log('MoE-only training UI source contract: ok');
