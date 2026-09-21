import assert from 'node:assert/strict';
import fs from 'node:fs';
const s=fs.readFileSync('webgpu-benchmark.mjs','utf8');
for(const token of ['navigator.gpu','requestAdapter','createComputePipeline','dispatchWorkgroups'])assert.ok(s.includes(token),token);
console.log('webgpu benchmark contract: ok');
