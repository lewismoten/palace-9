import assert from 'node:assert/strict';
import { D4_PERMUTATIONS, canonicalizeFeatures, inversePermuteSquares, transformSquares } from './rules.mjs';
import { createModel, exportModel, importModel } from './training.mjs';
import { drawNetwork, d4TransformDescription } from './network.mjs';

const model=createModel();
const imported=importModel({...exportModel(model),canonicalization:{group:'D4'}});
assert.equal(imported.canonicalization.group,'D4');

const features=[1,0,0,0,1,0,0,0,1,...Array(22).fill(0)];
const canonical=canonicalizeFeatures(features);
assert.equal(canonical.transformIndex,2);
assert.deepEqual(inversePermuteSquares(transformSquares([.1,.2,.3,.4,0,0,0,0,0],D4_PERMUTATIONS[2]),D4_PERMUTATIONS[2]),[.1,.2,.3,.4,0,0,0,0,0]);
assert.deepEqual(canonical.features.slice(27),features.slice(27));
assert.deepEqual(d4TransformDescription(4),{rotation:90,flipHorizontal:true,flipVertical:false});
const context=new Proxy({}, {get:()=>()=>{},set:()=>true});
globalThis.devicePixelRatio=1;
const canvas={clientWidth:960,clientHeight:500,getContext:()=>context};
drawNetwork(canvas,'aebd',imported);
assert.equal(canvas.__network.transcoder.group,'D4');
assert.equal(canvas.__network.transcoder.outputCoordinates,'original');
assert.equal(canvas.__network.edges.length,0,'network inspector intentionally omits dense connection lines');
console.log('canonicalization: ok');
