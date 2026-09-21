import assert from 'node:assert/strict';
import fs from 'node:fs';

const source = fs.readFileSync(new URL('./webgpu-training.mjs', import.meta.url), 'utf8');
for (const token of ['WEBGPU_TRAINER_LIMITATION', 'export async function createWebGPUTrainer', 'export async function trainWebGPURound', 'navigator.gpu', 'requestAdapter', 'createComputePipeline', 'tanh(', 'exp(', 'probability - trainingTarget', 'delta', 'weights[index] = weights[index] - learningRate.value', 'syncModel', 'mapAsync', 'GPUMapMode.READ']) assert.ok(source.includes(token), `missing WebGPU training contract: ${token}`);
assert.doesNotMatch(source, /var<storage, read> target:/, 'WGSL must not use reserved identifier target');
assert.match(source, /trainingTarget/, 'WGSL must use a non-reserved training target identifier');
assert.doesNotMatch(source, /widths\.length\s*!==\s*2/, 'trainer must support hidden-layer models');
assert.match(source, /MAX_HIDDEN_LAYERS\s*=\s*3/, 'trainer must state its bounded hidden-layer limit');
assert.match(source, /MAX_HIDDEN_NODES\s*=\s*64/, 'trainer must state its bounded width limit');
assert.match(source, /async trainRound\(histories, rate/, 'trainer must retain GPU buffers through a complete training round');

const app = fs.readFileSync(new URL('./app.mjs', import.meta.url), 'utf8');
const html = fs.readFileSync(new URL('./index.html', import.meta.url), 'utf8');
for (const token of ['createWebGPUTrainer', 'webGPUTrainerAvailable', 'CPU Worker (JavaScript)', 'backend-choice', 'await new Promise(resolve=>requestAnimationFrame(resolve))', 'WebGPU dense training · preparing GPU buffers', 'WebGPU dense training · position']) assert.ok(app.includes(token), `missing responsive backend UI contract: ${token}`);
assert.doesNotMatch(app, /cpuRoundMeasurement\(/, 'WebGPU start must not synchronously run an entire CPU comparison round on the UI thread');
assert.match(html, /id="best-round"/, 'training UI must reserve a best frozen round section');
assert.match(html, /id="training-diagnostics"/, 'training UI must reserve bounded mistake diagnostics');
assert.match(app, /Best frozen round/, 'app must track a best frozen round');
assert.match(app, /evaluation\.mistakes/, 'app must render evaluator diagnostics');
console.log('webgpu dense trainer and UI source contract: ok');
