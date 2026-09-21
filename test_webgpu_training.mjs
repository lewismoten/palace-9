import assert from 'node:assert/strict';
import fs from 'node:fs';

const source = fs.readFileSync(new URL('./webgpu-training.mjs', import.meta.url), 'utf8');
for (const token of ['WEBGPU_TRAINER_LIMITATION', 'export async function createWebGPUTrainer', 'export async function trainWebGPURound', 'navigator.gpu', 'requestAdapter', 'createComputePipeline', 'tanh(', 'exp(', 'probability - target', 'delta', 'weights[index] = weights[index] - learningRate.value', 'syncModel', 'mapAsync', 'GPUMapMode.READ']) assert.ok(source.includes(token), `missing WebGPU training contract: ${token}`);
assert.doesNotMatch(source, /widths\.length\s*!==\s*2/, 'trainer must support hidden-layer models');
assert.match(source, /MAX_HIDDEN_LAYERS\s*=\s*3/, 'trainer must state its bounded hidden-layer limit');
assert.match(source, /MAX_HIDDEN_NODES\s*=\s*64/, 'trainer must state its bounded width limit');
assert.match(source, /async trainRound\(histories, rate/, 'trainer must retain GPU buffers through a complete training round');

const app = fs.readFileSync(new URL('./app.mjs', import.meta.url), 'utf8');
const html = fs.readFileSync(new URL('./index.html', import.meta.url), 'utf8');
for (const token of ['createWebGPUTrainer', 'webGPUTrainerAvailable', 'CPU Worker (JavaScript)', 'CPU vs WebGPU training round', 'backend-choice']) assert.ok(app.includes(token), `missing selectable backend UI contract: ${token}`);
assert.match(html, /id="backend-choice"/, 'training UI must offer backend selection');
console.log('webgpu dense trainer and UI source contract: ok');
