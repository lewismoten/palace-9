import assert from 'node:assert/strict';
import fs from 'node:fs';

const source=fs.readFileSync(new URL('./webgpu-training.mjs', import.meta.url), 'utf8');
for (const token of [
  'export const WEBGPU_TRAINER_LIMITATION',
  'export async function createWebGPUTrainer',
  'navigator.gpu',
  'requestAdapter',
  'createComputePipeline',
  'var<storage, read_write> weights',
  'let gradient = (probability - target[output]) * features[input]',
  'weights[i] = weights[i] - learningRate.value * gradient',
  'dispatchWorkgroups',
  'mapAsync',
  'GPUMapMode.READ',
]) assert.ok(source.includes(token), `missing WebGPU trainer contract: ${token}`);
assert.match(source, /widths\.length\s*!==\s*2/, 'trainer must reject hidden-layer models');
console.log('webgpu trainer source contract: ok');
