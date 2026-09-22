import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import {decodePackedCheckpoint} from './qwen-precision.mjs';
import {qwenArchitecture} from './qwen-visualizer.mjs';
import {namedMoveLogits,runQwenForward} from './qwen-forward.mjs';

const config=JSON.parse(fs.readFileSync('release/palace-9-local-v1/source-checkpoint/config.json','utf8'));
const load=name=>JSON.parse(fs.readFileSync(`release/palace-9-local-v1/browser/${name}.json`,'utf8'));

test('browser precision exports retain byte-packed deployment tensors',()=>{
  for(const name of ['fp32','f16','q6_k','q4_k_m']){
    const manifest=load(name);
    assert.equal(manifest.format,'palace9-packed-tensors-v1');
    assert.equal(manifest.precision,name);
    assert.match(manifest.source.sha256,/^[a-f0-9]{64}$/);
    assert.ok(Object.values(manifest.sources).every(source=>typeof source.bytes==='string'&&source.bytes.length>0));
    assert.ok(manifest.tensors['lm_head.weight']);
  }
});

test('browser decoder reconstructs all released precision tensor plans',()=>{
  for(const name of ['fp32','f16','q6_k','q4_k_m']){
    const checkpoint=decodePackedCheckpoint(load(name),config);
    assert.deepEqual(checkpoint.tensors['model.embed_tokens.weight'].shape,[261,36]);
    assert.deepEqual(checkpoint.tensors['lm_head.weight'].shape,[261,36]);
    assert.deepEqual(checkpoint.tensors['model.layers.0.mlp.shared_expert.down_proj.weight'].shape,[36,5632]);
    assert.equal(checkpoint.tensors['model.layers.0.mlp.experts.8.gate_proj.weight'].values.length,648);
    assert.ok(Number.isFinite(checkpoint.tensors['lm_head.weight'].values[0]));
  }
});

test('every released browser artifact executes a local causal forward pass',()=>{
  for(const name of ['fp32','f16','q6_k','q4_k_m'])for(const history of ['', 'a', 'ab', 'ceifdg']){
    const checkpoint=decodePackedCheckpoint(load(name),config);checkpoint.shape=qwenArchitecture(config);
    const ids=[1,...[...history].map(token=>68+token.charCodeAt(0)-97)],logits=runQwenForward(checkpoint,ids).logits,winner=namedMoveLogits(logits)[0];
    assert.ok([...logits].every(Number.isFinite),`${name} ${history||'empty'} produced finite logits`);
    assert.match(winner.token,/^[a-i!]$/,`${name} ${history||'empty'} produces a protocol token`);
  }
});

test('packed release decoders retain the validated move and invalid-history probes',()=>{
  for(const name of ['fp32','f16','q6_k','q4_k_m']){
    const checkpoint=decodePackedCheckpoint(load(name),config);checkpoint.shape=qwenArchitecture(config);
    const winner=history=>namedMoveLogits(runQwenForward(checkpoint,[1,...[...history].map(token=>68+token.charCodeAt(0)-97)]).logits)[0].token;
    assert.equal(winner('a'),'e',`${name} retains the validated a → e probe`);
    assert.equal(winner('aa'),'!',`${name} retains the validated repeated-square sentinel probe`);
  }
});

test('Q6_K and Q4_K_M browser exports preserve their actual mixed storage',()=>{
  for(const name of ['q6_k','q4_k_m']){
    const manifest=load(name),types=new Set(Object.values(manifest.sources).map(source=>source.type));
    assert.ok(types.has('Q6_K'));
    assert.ok(types.has('F16'));
    assert.ok(types.has('F32'));
    assert.equal(manifest.claim,'exact packed deployment artifact decoded locally in the browser');
  }
});
