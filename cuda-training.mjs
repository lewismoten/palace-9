#!/usr/bin/env node
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createModel, evaluateModel, exportModel, importModel } from './training.mjs';
import { featureVector, optimalPolicy } from './rules.mjs';

const root=path.dirname(fileURLToPath(import.meta.url));
const argv=process.argv.slice(2),args={};for(let i=0;i<argv.length;i++){if(argv[i].startsWith('--'))args[argv[i].slice(2)]=argv[++i];}
const number=(key,fallback)=>args[key]===undefined?fallback:Number(args[key]);
const rounds=number('rounds',5),batch=number('batch',256),rate=number('rate',.025);
if(!Number.isInteger(rounds)||rounds<1||!Number.isInteger(batch)||batch<1||!(rate>0))throw Error('rounds and batch must be positive integers; rate must be positive');
const output=path.resolve(root,args.output||'snapshots');
const datasetPath=path.resolve(root,args.dataset||'data/reachable-policy.json');
const modelPath=args.model&&path.resolve(root,args.model);
const native=path.join(root,'cuda_moe_train');
const source=path.join(root,'cuda_moe_train.cu');
const MAGIC=0x504D4F45, HEADER_BYTES=60;

function compile(){if(!fs.existsSync(native)||fs.statSync(native).mtimeMs<fs.statSync(source).mtimeMs)execFileSync('nvcc',['-O3','--std=c++17','-o',native,source],{cwd:root,stdio:'inherit'});}
function geometry(model){const h=model.trunk.widths.at(-1),hidden=model.experts[0].widths[1];if(model.encoder.widths.join(',')!=='29,9'||model.trunk.widths.length-1>8||h>64||model.experts.length>8||hidden>64||!model.experts.every(e=>e.widths.join(',')===`${h},${hidden},9`))throw Error('CUDA trainer supports the current uniform MoE shape: encoder 29→9, <=8 trunk layers, <=8 experts, widths <=64');return {layers:model.trunk.weights.length,experts:model.experts.length,hidden,widths:model.trunk.widths};}
function matrices(model){return [model.encoder.weights,...model.trunk.weights,model.router.weights,...model.experts.flatMap(e=>e.weights)];}
function flatten(model){return Float32Array.from(matrices(model).flat(2));}
function restore(model,values){let at=0;for(const matrix of matrices(model))for(const row of matrix)for(let i=0;i<row.length;i++)row[i]=values[at++];if(at!==values.length)throw Error('CUDA result parameter count mismatch');}
function payload(model,examples){const shape=geometry(model),params=flatten(model),header=Buffer.alloc(HEADER_BYTES);header.writeInt32LE(MAGIC,0);header.writeInt32LE(examples.length,4);header.writeInt32LE(shape.layers,8);header.writeInt32LE(shape.experts,12);header.writeInt32LE(shape.hidden,16);header.writeInt32LE(params.length,20);shape.widths.forEach((v,i)=>header.writeInt32LE(v,24+i*4));const inputs=Float32Array.from(examples.flatMap(row=>featureVector(row.history)));const targets=Float32Array.from(examples.flatMap(row=>(row.policy||optimalPolicy(row.history)).probabilities));return Buffer.concat([header,Buffer.from(params.buffer),Buffer.from(inputs.buffer),Buffer.from(targets.buffer)]);}
function runOne(model,examples){const dir=fs.mkdtempSync(path.join(os.tmpdir(),'palace-9-cuda-'));const input=path.join(dir,'input.bin'),result=path.join(dir,'result.bin');try{fs.writeFileSync(input,payload(model,examples));const stderr=execFileSync(native,[input,result,'1',String(batch),String(rate)],{cwd:root,env:{...process.env,CUDA_VISIBLE_DEVICES:'0'},encoding:'utf8',stdio:['ignore','pipe','pipe']});if(!stderr.includes('GPU 0:'))throw Error(`CUDA trainer did not verify GPU 0: ${stderr}`);const raw=fs.readFileSync(result);if(raw.readInt32LE(0)!==MAGIC)throw Error('CUDA trainer returned an invalid result');restore(model,new Float32Array(raw.buffer,raw.byteOffset+HEADER_BYTES,(raw.length-HEADER_BYTES)/4));return stderr.trim();}finally{fs.rmSync(dir,{recursive:true,force:true});}}

compile();const data=JSON.parse(fs.readFileSync(datasetPath,'utf8'));const examples=data.examples||data;if(!Array.isArray(examples)||!examples.length)throw Error('dataset needs a nonempty examples array');const histories=examples.map(x=>x.history),policies=Object.fromEntries(examples.filter(x=>x.policy).map(x=>[x.history,x.policy]));let model=modelPath?importModel(JSON.parse(fs.readFileSync(modelPath,'utf8'))):createModel();fs.mkdirSync(output,{recursive:true});let best=null,snapshots=[],gpu='';for(let round=1;round<=rounds;round++){gpu=runOne(model,examples);model.steps+=examples.length;const evaluation=evaluateModel(model,histories,policies);const t=evaluation.totals,score=t.loss+t.invalid;const improved=!best||score<best.score||(score===best.score&&t.loss<best.loss);if(improved){best={score,loss:t.loss};const snapshot={...exportModel(model),annotations:{rounds:round,losses:t.loss,invalid:t.invalid,wins:t.win,draws:t.draw,frozenMetrics:evaluation.percent,gpu:'GPU 0',trainer:'cuda_moe_train.cu'}};if(snapshot.format!=='palace-9-moe/v1')throw Error('snapshot must remain browser-importable palace-9-moe/v1');const name=`palace-9-round-${String(round).padStart(4,'0')}.json`,file=path.join(output,name);fs.mkdirSync(output,{recursive:true});fs.writeFileSync(file,JSON.stringify(snapshot));snapshots.push(file);}if(t.loss===0&&t.invalid===0)break;}
console.log(JSON.stringify({gpu:{index:0,verification:gpu},roundsCompleted:model.steps/examples.length,snapshots}));
