import assert from 'node:assert/strict';
import { createModel, predict, trainingRounds } from './training.mjs';
const model=createModel({layers:2,nodes:8});
assert.equal(predict(model,'').probabilities.length,9);
const run=trainingRounds(['','a','ae'],{layers:2,nodes:8,batch:2,rate:.01});
const step=run.next().value;
assert.equal(step.index,2);assert.equal(step.total,3);assert.equal(step.model.widths.join(','),'29,8,8,9');
console.log('training generator: ok');
