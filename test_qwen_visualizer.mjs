import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import {qwenArchitecture,qwenTensorPlan,strategicOverlayPlan,nuclearCodePlan,simulationAlertPlan} from './qwen-visualizer.mjs';

const config=JSON.parse(fs.readFileSync('release/palace-9-local-v1/source-checkpoint/config.json','utf8'));

test('published Palace-9 visualizer describes the Qwen2-MoE decoder',()=>{
  const shape=qwenArchitecture(config);
  assert.deepEqual(shape,{
    vocabSize:261,context:16,hidden:36,layers:1,qHeads:9,kvHeads:3,headDim:4,
    attentionIntermediate:72,experts:9,topK:2,expertIntermediate:18,sharedExpertIntermediate:5632,
  });
  const plan=qwenTensorPlan(shape);
  assert.ok(plan.some(([label,,tensor])=>label==='Shared expert'&&tensor==='model.layers.0.mlp.shared_expert.*'));
  assert.ok(plan.some(([label,detail])=>label==='Routed experts'&&detail.includes('9 × (36 → 18 → 36)')));
});

test('board-state architecture cannot be accepted as the Qwen deployment model',()=>{
  assert.throws(()=>qwenArchitecture({model_type:'board-state-moe'}),/Qwen2-MoE/);
});

test('strategic overlay rotates through five verified bases and targets the opposing side',()=>{
  const ordinary=strategicOverlayPlan('abc',false);
  assert.equal(ordinary.usBases.length,5);
  assert.equal(ordinary.russiaBases.length,5);
  assert.equal(ordinary.moves.length,3);
  assert.deepEqual(ordinary.moves.map(move=>move.sourceCountry),['us','russia','us'],'The fictional U.S. X side opens the simulation');
  assert.deepEqual(ordinary.moves.map(move=>move.targetCountry),['russia','us','russia']);
  assert.ok(ordinary.moves.every(move=>ordinary[`${move.targetCountry}Bases`].some(base=>base===move.targetBase)),'every target must be an opposing base');
  assert.ok(ordinary.moves.every(move=>move.progress===3/9),'active paths advance one ninth per completed turn');
  assert.ok(ordinary.markerAlpha>.8,'base markers must be brighter than the country fill');
  const ninth=strategicOverlayPlan('abcdefghi',false);
  assert.equal(ninth.moves.length,9);
  assert.ok(ninth.moves.every(move=>move.progress===1),'the ninth turn completes every path');
  const decisive=strategicOverlayPlan('abc',true);
  assert.ok(decisive.moves.every(move=>move.progress===1),'a winning round completes every active path');
});

test('map starts at DEFCON 1 and presents Launch Detection after move one',()=>{
  const idle=simulationAlertPlan(''),firstMove=simulationAlertPlan('a');
  assert.deepEqual(idle.layout,{columns:1,width:52,height:14,x:.225},'DEFCON sits between the embedding and RMSNorm columns as wide rectangles');
  assert.deepEqual(idle.levels,[
    {level:1,color:'#f2f4ea',active:true},{level:2,color:'#e77878',active:false},{level:3,color:'#d1ef65',active:false},{level:4,color:'#70c986',active:false},{level:5,color:'#79cbe5',active:false},
  ]);
  assert.equal(idle.alert,null);
  assert.deepEqual(firstMove.alert,{country:'us',label:'LAUNCH DETECTION'});
});


test('cipher search cycles while idle and locks random-looking characters only during self-play',()=>{
  const weights=Float32Array.from({length:97},(_,index)=>Math.sin(index*.73));
  const idle=nuclearCodePlan(weights,'abc',0,false),idleLater=nuclearCodePlan(weights,'abc',.2,false);
  assert.equal(idle.frozen,0,'manual or two-player play never freezes cipher characters');
  assert.notEqual(idle.code,idleLater.code,'idle cipher characters continuously cycle');
  const early=nuclearCodePlan(weights,'abc',9.9,true),first=nuclearCodePlan(weights,'abc',10,true),later=nuclearCodePlan(weights,'abc',90,true);
  assert.equal(early.frozen,0,'self-play waits before locking its first character');
  assert.equal(first.frozen,1,'self-play locks one character at a time');
  assert.equal(later.frozen,9,'self-play intentionally leaves one of ten characters unresolved');
  assert.notEqual(first.code.at(-1),nuclearCodePlan(weights,'abc',10.2,true).code.at(-1),'unlocked characters keep cycling after a lock');
});
test('logits use five stacked rows and sparse panels use visual selection only',()=>{
  const source=fs.readFileSync('qwen-visualizer.mjs','utf8'),app=fs.readFileSync('app.mjs','utf8');
  assert.match(source,/const laneX=0/,'top-five logits must use one vertical lane');
  assert.match(source,/row=index/,'each logit occupies its own row in that lane');
  assert.match(source,/const cell=21/,'live router grid must remain readable at its larger cell size');
  assert.doesNotMatch(source,/active\?'  SELECTED'/,'expert panels should use their lime outline, not redundant selected text');
  assert.doesNotMatch(source,/if\(active\)grid\(/,'expert grids must remain visible even when not selected');
  assert.match(source,/static expert weights/,'each expert panel labels its persistent loaded-weight sample');
  assert.match(source,/width=74,height=38/,'expert panels are compact and reserve most of their width for weights');
  assert.match(source,/winning logit/,'the next-token card retains its full winning-logit label');
  assert.match(source,/ctx\.fillRect\(0,0,w,38\)/,'a translucent top strip protects the inspector headers from the map overlay');
  assert.match(source,/color:'#fff'/,'the inspector headers use bright white text');
  assert.match(source,/drawMapBackdrop\(ctx,canvas,w,h\)/,'the Equal Earth asset is drawn inside the canvas rather than depending on a CSS layer');
  assert.match(source,/globalAlpha=\.26/,'the map is visible but remains subordinate to tensors');
  assert.match(source,/experts:w\*\.72,shared:w\*\.81,output:Math\.min\(w\*\.92,w-66\)/,'the last three inspector columns have non-overlapping centers');
  assert.match(source,/const top=42,mapHeight=h-top/,'the map begins below the protected header strip');
  assert.match(source,/Math\.max\(ceiling,/,'trajectory controls cannot climb into the header strip');
  assert.match(source,/grid\(ctx,xs\.output,h\*\.42/,'final RMSNorm is vertically separated from the winning-token card');
  assert.match(source,/xs\.output-22,h\*\.89/,'the five-row logit rank is moved left and down from the header');
  assert.match(app,/drawQwenNetwork\([^;]*game\.history\+game\.finalMove/,'the ninth UI move is included in the strategic overlay');
  assert.match(source,/strategicOverlayPlan/,'the inspector includes its local strategic-map move overlay');
  assert.match(source,/quadraticCurveTo/,'each move is rendered as an arcing trajectory');
  assert.match(source,/ctx\.globalAlpha=\.62/,'trajectory paths stay brighter through the middle of the map');
  assert.match(source,/ctx\.rotate\(Math\.atan2\(tangent\.y,tangent\.x\)\)/,'missiles are direction-facing arrows');
  assert.match(source,/fillStyle='rgba\(242,244,234,\.94\)'/,'completed impacts render as solid white circles');
  assert.match(source,/if\(progress>=1\)/,'impact circles begin only after a trajectory reaches its target');
  assert.match(source,/markerAlpha:1/,'strategic base markers are deliberately brighter than country fills');
  assert.match(source,/globalAlpha=\.38/,'strategic overlay stays deliberately transparent');
  assert.match(source,/drawSimulationStatus\(ctx,w,h,overlayHistory\)/,'the fictional DEFCON and alert presentation is redrawn with each simulation state');
  assert.match(source,/fillText\('DEFCON',left\+layout\.width\/2,top-8\)/,'the DEFCON heading is centered over its stack');
  assert.match(source,/layout:\{columns:1,width:52,height:14,x:\.225\}/,'DEFCON uses a vertical column of wide rectangles between embedding and RMSNorm');
  assert.match(source,/label:'LAUNCH DETECTION'/,'the post-move map label is Launch Detection');
  assert.doesNotMatch(source,/SIMULATION · LAUNCH DETECTION/,'the map label does not prepend Simulation');
  assert.match(source,/selfPlaying\?Math\.floor\(seconds\/10\):0/,'cipher locks progress only while zero-player self-play is active');
  assert.match(source,/Math\.min\(9,selfPlaying/,'cipher search never resolves all ten glyphs');
  assert.match(app,/game\.history\+game\.finalMove,mode===0/,'initial inspector draw receives zero-player cipher state');
  assert.match(app,/game\.history\+game\.finalMove,players\(\)===0/,'animated inspector redraw receives zero-player cipher state');
});
