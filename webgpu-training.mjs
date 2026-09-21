import { featureVector, optimalPolicy } from './rules.mjs';

/**
 * This trainer intentionally supports only the existing model's 29 -> 9,
 * no-hidden-layer shape. It is not parity with training.mjs's configurable
 * tanh network: it performs a GPU softmax-cross-entropy SGD update for one
 * linear policy example at a time.
 */
export const WEBGPU_TRAINER_LIMITATION =
  'WebGPU trainer supports only a 29 -> 9 linear softmax model (no hidden layers or biases).';

const INPUTS = 29;
const OUTPUTS = 9;
const PARAMETER_COUNT = INPUTS * OUTPUTS;

const forwardShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
@compute @workgroup_size(64)
fn forward(@builtin(global_invocation_id) id: vec3<u32>) {
  let output = id.x;
  if (output >= ${OUTPUTS}u) { return; }
  var logit = 0.0;
  for (var input = 0u; input < ${INPUTS}u; input = input + 1u) {
    logit = logit + weights[output * ${INPUTS}u + input] * features[input];
  }
  logits[output] = logit;
}`;

const updateShader = /* wgsl */ `
struct LearningRate { value: f32 }
@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read> target: array<f32>;
@group(0) @binding(3) var<storage, read> logits: array<f32>;
@group(0) @binding(4) var<uniform> learningRate: LearningRate;
@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= ${PARAMETER_COUNT}u) { return; }
  let output = i / ${INPUTS}u;
  let input = i % ${INPUTS}u;
  var maxLogit = -3.402823e+38;
  for (var r = 0u; r < ${OUTPUTS}u; r = r + 1u) { maxLogit = max(maxLogit, logits[r]); }
  var denominator = 0.0;
  for (var r = 0u; r < ${OUTPUTS}u; r = r + 1u) { denominator = denominator + exp(logits[r] - maxLogit); }
  let probability = exp(logits[output] - maxLogit) / denominator;
  let gradient = (probability - target[output]) * features[input];
  weights[i] = weights[i] - learningRate.value * gradient;
}`;

const flattenWeights = model => Float32Array.from(model.weights.flat());
const validModel = model => Array.isArray(model?.widths) && model.widths.length === 2 &&
  model.widths[0] === INPUTS && model.widths[1] === OUTPUTS &&
  Array.isArray(model.weights) && model.weights.length === 1 &&
  model.weights[0].length === OUTPUTS && model.weights[0].every(row => row.length === INPUTS);

export function webGPUTrainerAvailable() {
  return typeof navigator !== 'undefined' && Boolean(navigator.gpu);
}

export async function createWebGPUTrainer(model) {
  if (!validModel(model)) {
    if (Array.isArray(model?.widths) && model.widths.length !== 2) {
      throw new Error(WEBGPU_TRAINER_LIMITATION);
    }
    throw new Error('WebGPU trainer requires a model with widths [29, 9].');
  }
  if (!webGPUTrainerAvailable()) throw new Error('WebGPU unavailable in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter available.');
  const device = await adapter.requestDevice();
  const weights = device.createBuffer({
    size: PARAMETER_COUNT * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const features = device.createBuffer({size: INPUTS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  const target = device.createBuffer({size: OUTPUTS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  const learningRate = device.createBuffer({size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  const logits = device.createBuffer({size: OUTPUTS * 4, usage: GPUBufferUsage.STORAGE});
  device.queue.writeBuffer(weights, 0, flattenWeights(model));
  const forwardPipeline = device.createComputePipeline({
    layout: 'auto', compute: {module: device.createShaderModule({code: forwardShader}), entryPoint: 'forward'},
  });
  const updatePipeline = device.createComputePipeline({
    layout: 'auto', compute: {module: device.createShaderModule({code: updateShader}), entryPoint: 'update'},
  });
  const forwardBindGroup = device.createBindGroup({layout: forwardPipeline.getBindGroupLayout(0), entries: [
    {binding: 0, resource: {buffer: weights}}, {binding: 1, resource: {buffer: features}}, {binding: 2, resource: {buffer: logits}},
  ]});
  const updateBindGroup = device.createBindGroup({layout: updatePipeline.getBindGroupLayout(0), entries: [
    {binding: 0, resource: {buffer: weights}}, {binding: 1, resource: {buffer: features}},
    {binding: 2, resource: {buffer: target}}, {binding: 3, resource: {buffer: logits}},
    {binding: 4, resource: {buffer: learningRate}},
  ]});
  let destroyed = false;
  const assertLive = () => { if (destroyed) throw new Error('WebGPU trainer is destroyed.'); };

  return {
    limitation: WEBGPU_TRAINER_LIMITATION,
    async trainOne(history, rate = 0.025) {
      assertLive();
      const policy = optimalPolicy(history);
      if (!policy.valid) return null;
      device.queue.writeBuffer(features, 0, Float32Array.from(featureVector(history)));
      device.queue.writeBuffer(target, 0, Float32Array.from(policy.probabilities));
      device.queue.writeBuffer(learningRate, 0, new Float32Array([Number(rate)]));
      const encoder = device.createCommandEncoder();
      const forwardPass = encoder.beginComputePass();
      forwardPass.setPipeline(forwardPipeline);
      forwardPass.setBindGroup(0, forwardBindGroup);
      forwardPass.dispatchWorkgroups(1);
      forwardPass.end();
      const updatePass = encoder.beginComputePass();
      updatePass.setPipeline(updatePipeline);
      updatePass.setBindGroup(0, updateBindGroup);
      updatePass.dispatchWorkgroups(Math.ceil(PARAMETER_COUNT / 64));
      updatePass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      model.steps = (Number(model.steps) || 0) + 1;
      return {policy};
    },
    async syncModel() {
      assertLive();
      const readback = device.createBuffer({size: PARAMETER_COUNT * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(weights, 0, readback, 0, PARAMETER_COUNT * 4);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readback.getMappedRange().slice(0));
      model.weights[0] = Array.from({length: OUTPUTS}, (_, row) => Array.from(values.slice(row * INPUTS, (row + 1) * INPUTS)));
      readback.unmap();
      readback.destroy();
      return model;
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      weights.destroy(); features.destroy(); target.destroy(); learningRate.destroy(); logits.destroy();
    },
  };
}
