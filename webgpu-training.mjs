import { featureVector, optimalPolicy } from './rules.mjs';

export const MAX_HIDDEN_LAYERS = 3;
export const MAX_HIDDEN_NODES = 64;
export const WEBGPU_TRAINER_LIMITATION =
  `WebGPU dense trainer supports 29 → up to ${MAX_HIDDEN_LAYERS} tanh hidden layers of up to ${MAX_HIDDEN_NODES} nodes → 9, with no biases.`;

const INPUTS = 29;
const OUTPUTS = 9;
const WORKGROUP_SIZE = 64;
const f32bytes = length => Math.max(4, length * Float32Array.BYTES_PER_ELEMENT);
const workgroups = length => Math.ceil(length / WORKGROUP_SIZE);

const flatten = matrix => Float32Array.from(matrix.flat());
const assertModel = model => {
  const widths = model?.widths;
  if (!Array.isArray(widths) || widths.length < 2 || widths[0] !== INPUTS || widths.at(-1) !== OUTPUTS) {
    throw new Error('WebGPU trainer requires model widths beginning at 29 and ending at 9.');
  }
  const hidden = widths.slice(1, -1);
  if (hidden.length > MAX_HIDDEN_LAYERS || hidden.some(width => !Number.isInteger(width) || width < 1 || width > MAX_HIDDEN_NODES)) {
    throw new Error(WEBGPU_TRAINER_LIMITATION);
  }
  if (!Array.isArray(model.weights) || model.weights.length !== widths.length - 1 ||
      model.weights.some((matrix, layer) => !Array.isArray(matrix) || matrix.length !== widths[layer + 1] || matrix.some(row => !Array.isArray(row) || row.length !== widths[layer]))) {
    throw new Error('WebGPU trainer received an invalid model matrix shape.');
  }
};

const forwardShader = (inputSize, outputSize, tanhOutput) => /* wgsl */ `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn forward(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= ${outputSize}u) { return; }
  var value = 0.0;
  for (var column = 0u; column < ${inputSize}u; column = column + 1u) {
    value = value + weights[row * ${inputSize}u + column] * input[column];
  }
  output[row] = ${tanhOutput ? 'tanh(value)' : 'value'};
}`;

const outputDeltaShader = (size) => /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> target: array<f32>;
@group(0) @binding(2) var<storage, read_write> delta: array<f32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn outputDelta(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= ${size}u) { return; }
  var maximum = -3.402823e+38;
  for (var i = 0u; i < ${size}u; i = i + 1u) { maximum = max(maximum, logits[i]); }
  var denominator = 0.0;
  for (var i = 0u; i < ${size}u; i = i + 1u) { denominator = denominator + exp(logits[i] - maximum); }
  let probability = exp(logits[index] - maximum) / denominator;
  delta[index] = probability - target[index];
}`;

const updateShader = (inputSize, outputSize) => /* wgsl */ `
struct LearningRate { value: f32 }
@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> delta: array<f32>;
@group(0) @binding(3) var<uniform> learningRate: LearningRate;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= ${inputSize * outputSize}u) { return; }
  let row = index / ${inputSize}u;
  let column = index % ${inputSize}u;
  weights[index] = weights[index] - learningRate.value * delta[row] * input[column];
}`;

const backpropShader = (inputSize, outputSize) => /* wgsl */ `
@group(0) @binding(0) var<storage, read> nextWeights: array<f32>;
@group(0) @binding(1) var<storage, read> nextDelta: array<f32>;
@group(0) @binding(2) var<storage, read> activation: array<f32>;
@group(0) @binding(3) var<storage, read_write> delta: array<f32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn backprop(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x;
  if (column >= ${inputSize}u) { return; }
  var propagated = 0.0;
  for (var row = 0u; row < ${outputSize}u; row = row + 1u) {
    propagated = propagated + nextWeights[row * ${inputSize}u + column] * nextDelta[row];
  }
  delta[column] = propagated * (1.0 - activation[column] * activation[column]);
}`;

export function webGPUTrainerAvailable() {
  return typeof navigator !== 'undefined' && Boolean(navigator.gpu);
}

export async function createWebGPUTrainer(model) {
  assertModel(model);
  if (!webGPUTrainerAvailable()) throw new Error('WebGPU unavailable in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter available.');
  const device = await adapter.requestDevice();
  const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const weights = model.weights.map(matrix => device.createBuffer({ size: f32bytes(matrix.length * matrix[0].length), usage: storageUsage }));
  const activations = model.widths.map(width => device.createBuffer({ size: f32bytes(width), usage: storageUsage }));
  const deltas = model.widths.slice(1).map(width => device.createBuffer({ size: f32bytes(width), usage: storageUsage }));
  const target = device.createBuffer({ size: f32bytes(OUTPUTS), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const learningRate = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  model.weights.forEach((matrix, layer) => device.queue.writeBuffer(weights[layer], 0, flatten(matrix)));

  const forward = model.weights.map((_, layer) => {
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: forwardShader(model.widths[layer], model.widths[layer + 1], layer < model.weights.length - 1) }), entryPoint: 'forward' } });
    return { pipeline, bindGroup: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: weights[layer] } }, { binding: 1, resource: { buffer: activations[layer] } }, { binding: 2, resource: { buffer: activations[layer + 1] } }] }) };
  });
  const outputPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: outputDeltaShader(OUTPUTS) }), entryPoint: 'outputDelta' } });
  const outputBindGroup = device.createBindGroup({ layout: outputPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: activations.at(-1) } }, { binding: 1, resource: { buffer: target } }, { binding: 2, resource: { buffer: deltas.at(-1) } }] });
  const updates = model.weights.map((_, layer) => {
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: updateShader(model.widths[layer], model.widths[layer + 1]) }), entryPoint: 'update' } });
    return { pipeline, bindGroup: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: weights[layer] } }, { binding: 1, resource: { buffer: activations[layer] } }, { binding: 2, resource: { buffer: deltas[layer] } }, { binding: 3, resource: { buffer: learningRate } }] }) };
  });
  const backwards = model.weights.slice(1).map((_, index) => {
    const layer = index + 1;
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: backpropShader(model.widths[layer], model.widths[layer + 1]) }), entryPoint: 'backprop' } });
    return { layer, pipeline, bindGroup: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: weights[layer] } }, { binding: 1, resource: { buffer: deltas[layer] } }, { binding: 2, resource: { buffer: activations[layer] } }, { binding: 3, resource: { buffer: deltas[layer - 1] } }] }) };
  });
  let destroyed = false;
  const assertLive = () => { if (destroyed) throw new Error('WebGPU trainer is destroyed.'); };
  const dispatch = (pass, item, count) => { pass.setPipeline(item.pipeline); pass.setBindGroup(0, item.bindGroup); pass.dispatchWorkgroups(workgroups(count)); };

  const trainOne = async (history, rate = 0.025) => {
    assertLive();
    const policy = optimalPolicy(history);
    if (!policy.valid) return null;
    device.queue.writeBuffer(activations[0], 0, Float32Array.from(featureVector(history)));
    device.queue.writeBuffer(target, 0, Float32Array.from(policy.probabilities));
    device.queue.writeBuffer(learningRate, 0, new Float32Array([Number(rate)]));
    const encoder = device.createCommandEncoder();
    for (let layer = 0; layer < forward.length; layer++) { const pass = encoder.beginComputePass(); dispatch(pass, forward[layer], model.widths[layer + 1]); pass.end(); }
    { const pass = encoder.beginComputePass(); dispatch(pass, { pipeline: outputPipeline, bindGroup: outputBindGroup }, OUTPUTS); pass.end(); }
    for (let layer = updates.length - 1; layer >= 0; layer--) {
      { const pass = encoder.beginComputePass(); dispatch(pass, updates[layer], model.widths[layer] * model.widths[layer + 1]); pass.end(); }
      if (layer > 0) { const backward = backwards[layer - 1]; const pass = encoder.beginComputePass(); dispatch(pass, backward, model.widths[layer]); pass.end(); }
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    model.steps = (Number(model.steps) || 0) + 1;
    return { policy };
  };

  const syncModel = async () => {
    assertLive();
    for (let layer = 0; layer < weights.length; layer++) {
      const count = model.widths[layer] * model.widths[layer + 1];
      const readback = device.createBuffer({ size: f32bytes(count), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(weights[layer], 0, readback, 0, f32bytes(count));
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readback.getMappedRange().slice(0));
      model.weights[layer] = Array.from({ length: model.widths[layer + 1] }, (_, row) => Array.from(values.slice(row * model.widths[layer], (row + 1) * model.widths[layer])));
      readback.unmap(); readback.destroy();
    }
    return model;
  };

  return {
    limitation: WEBGPU_TRAINER_LIMITATION,
    model,
    trainOne,
    async trainRound(histories, rate = 0.025, onProgress) {
      assertLive();
      const totals = { trained: 0, skipped: 0 };
      for (let index = 0; index < histories.length; index++) {
        if (await trainOne(histories[index], rate)) totals.trained++; else totals.skipped++;
        await onProgress?.({ index: index + 1, total: histories.length, ...totals });
      }
      return { ...totals, model: await syncModel() };
    },
    syncModel,
    destroy() { if (!destroyed) { destroyed = true; [...weights, ...activations, ...deltas, target, learningRate].forEach(buffer => buffer.destroy()); } },
  };
}

export async function trainWebGPURound(model, histories, rate = 0.025, onProgress) {
  const trainer = await createWebGPUTrainer(model);
  try { return await trainer.trainRound(histories, rate, onProgress); }
  finally { trainer.destroy(); }
}
