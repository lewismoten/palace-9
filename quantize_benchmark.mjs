#!/usr/bin/env node
import { readFile,writeFile } from 'node:fs/promises';
import { importModel,evaluateModel,predict } from './training.mjs';
import { quantizeModel } from './quantization.mjs';

const checkpoint=JSON.parse(await readFile(process.argv[2]||'data/default-palace-9-model.json','utf8'));
const dataset=JSON.parse(await readFile('data/reachable-policy.json','utf8')).examples;
const histories=dataset.map(row=>row.history),policies=Object.fromEntries(dataset.map(row=>[row.history,row.policy]));
const baseline=importModel(checkpoint);
const parameterCount=model=>model.encoder.weights.flat().length+model.trunk.weights.flat(2).length+model.router.weights.flat().length+model.experts.reduce((total,expert)=>total+expert.weights.flat(2).length,0);
const choice=model=>histories.map(history=>predict(model,history).probabilities.indexOf(Math.max(...predict(model,history).probabilities)));
const baselineChoices=choice(baseline),parameters=parameterCount(baseline);
const results=['float32','float16','int8','int4','int4-row','int2','int1'].map(format=>{const model=quantizeModel(baseline,format),report=evaluateModel(model,histories,policies,{corrections:false}),choices=choice(model),sameTop1=choices.filter((value,index)=>value===baselineChoices[index]).length,bits=format==='float32'?32:format==='float16'?16:format==='int4-row'?4:Number(format.slice(3));return {format,parameters,theoreticalBytes:parameters*bits/8,sameTop1,sameTop1Percent:100*sameTop1/histories.length,losses:report.totals.loss,occupiedSquareInvalid:report.totals.invalid,wins:report.totals.win,draws:report.totals.draw}});
const output={checkpoint:process.argv[2]||'data/default-palace-9-model.json',corpus:histories.length,method:'float16 IEEE-754 rounding; INT4 has both symmetric per-tensor and per-output-row round/dequantize simulations, evaluated in float JavaScript',results};
const file=process.argv[3]||'quantization-results.json';await writeFile(file,JSON.stringify(output,null,2));console.log(JSON.stringify(output,null,2));
