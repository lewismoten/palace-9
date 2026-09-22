import { namedMoveLogits, runQwenForward } from './qwen-forward.mjs';
import { loadPackedPrecision } from './qwen-precision.mjs';

export const QWEN_CONFIG_URL='release/palace-9-local-v1/source-checkpoint/config.json';
export const QWEN_WEIGHTS_URL='release/palace-9-local-v1/source-checkpoint/model.safetensors';
export const QWEN_PRECISION_URLS={fp32:'release/palace-9-local-v1/browser/fp32.json',f16:'release/palace-9-local-v1/browser/f16.json',q6_k:'release/palace-9-local-v1/browser/q6_k.json',q4_k_m:'release/palace-9-local-v1/browser/q4_k_m.json'};

export function qwenArchitecture(config){
  if(config?.model_type!=='qwen2_moe')throw Error('Expected a Qwen2-MoE deployment config');
  const headDim=config.hidden_size/config.num_attention_heads;
  if(!Number.isInteger(headDim))throw Error('Qwen attention head dimension must be integral');
  return {
    vocabSize:config.vocab_size, context:config.max_position_embeddings,
    hidden:config.hidden_size, layers:config.num_hidden_layers,
    qHeads:config.num_attention_heads, kvHeads:config.num_key_value_heads, headDim,
    attentionIntermediate:config.intermediate_size, experts:config.num_experts,
    topK:config.num_experts_per_tok, expertIntermediate:config.moe_intermediate_size,
    sharedExpertIntermediate:config.shared_expert_intermediate_size,
  };
}

export function qwenTensorPlan(shape){
  return [
    ['Token embedding',`${shape.vocabSize} × ${shape.hidden}`,'model.embed_tokens.weight'],
    ['RMSNorm',`${shape.hidden}`,'model.layers.0.input_layernorm.weight'],
    ['Q projection',`${shape.qHeads*shape.headDim} × ${shape.hidden}`,'model.layers.0.self_attn.q_proj.weight'],
    ['K / V projection',`${shape.kvHeads*shape.headDim} × ${shape.hidden}`,'model.layers.0.self_attn.k_proj.weight'],
    ['Attention output',`${shape.hidden} × ${shape.hidden}`,'model.layers.0.self_attn.o_proj.weight'],
    ['MoE router',`${shape.experts} × ${shape.hidden}`,'model.layers.0.mlp.gate.weight'],
    ['Routed experts',`${shape.experts} × (${shape.hidden} → ${shape.expertIntermediate} → ${shape.hidden})`,'model.layers.0.mlp.experts.*'],
    ['Shared expert',`${shape.hidden} → ${shape.sharedExpertIntermediate} → ${shape.hidden}`,'model.layers.0.mlp.shared_expert.*'],
    ['Final RMSNorm',`${shape.hidden}`,'model.norm.weight'],
    ['LM head',`${shape.vocabSize} × ${shape.hidden}`,'lm_head.weight'],
  ];
}

function color(value,alpha=1){return value>0?`rgba(246,166,92,${alpha})`:value<0?`rgba(121,203,229,${alpha})`:`rgba(166,175,164,${alpha})`;}
function magnitude(values){let total=0;for(const value of values)total+=value*value;return Math.sqrt(total/Math.max(1,values.length));}
function matrixAt(tensor,row,column){return tensor?.values?.[row*tensor.shape[1]+column]??0;}

const codeAlphabet='0123456789ABCDEFGHJLPRSTUY';
const lineGlyphs={
  '0':'ab cdef'.replace(/ /g,''),'1':'bc','2':'abdeg','3':'abcdg','4':'bcfg','5':'acdfg','6':'acdefg','7':'abc','8':'abcdefg','9':'abcdfg',
  A:'abcefg',B:'cdefg',C:'adef',D:'bcdeg',E:'adefg',F:'aefg',G:'acdef',H:'bcefg',J:'bcd',L:'def',P:'abefg',R:'abcefg',S:'acdfg',T:'dg',U:'bcdef',Y:'bcdfg',
};
const lineSegments={a:[[.12,.12],[.88,.12]],b:[[.88,.12],[.88,.5]],c:[[.88,.5],[.88,.88]],d:[[.12,.88],[.88,.88]],e:[[.12,.5],[.12,.88]],f:[[.12,.12],[.12,.5]],g:[[.12,.5],[.88,.5]]};
function codeCharacter(weights,index,salt){const value=weights?.[index%Math.max(1,weights?.length||1)]||0;return codeAlphabet[(Math.abs(Math.floor(value*100000))+index+salt)%codeAlphabet.length];}
const cipherLockOrder=[6,1,8,3,0,7,2,5,4];
export function nuclearCodePlan(weights,history='',seconds=0,selfPlaying=false){const frozen=Math.min(9,selfPlaying?Math.floor(seconds/10):0),frozenPositions=cipherLockOrder.slice(0,frozen),rateHz=7,cycle=Math.floor(seconds*rateHz),code=Array.from({length:10},(_,position)=>{if(frozenPositions.includes(position))return codeCharacter(weights,position*47+position*19,position*19);const salt=cycle+position*31;return codeCharacter(weights,position*47+salt,salt);}).join('');return {code,frozen,frozenPositions,cycle,rateHz};}
function drawLineGlyph(ctx,char,x,y,width,height){for(const segment of lineGlyphs[char]||lineGlyphs['0']){const [[x1,y1],[x2,y2]]=lineSegments[segment];ctx.beginPath();ctx.moveTo(x+x1*width,y+y1*height);ctx.lineTo(x+x2*width,y+y2*height);ctx.stroke();}}
function drawNuclearCode(ctx,w,h,weights,history,elapsed,selfPlaying){const plan=nuclearCodePlan(weights,history,elapsed,selfPlaying),width=13,height=19,gap=4,total=10*width+9*gap,left=w/2-total/2,top=h-56;ctx.save();ctx.globalAlpha=.8;ctx.strokeStyle='rgba(242,244,234,.95)';ctx.shadowColor='rgba(121,203,229,.8)';ctx.shadowBlur=6;ctx.lineWidth=1.2;for(const [index,char] of [...plan.code].entries()){ctx.globalAlpha=plan.frozenPositions.includes(index)?.9:.38;drawLineGlyph(ctx,char,left+index*(width+gap),top,width,height);}ctx.shadowBlur=0;caption(ctx,w/2,top+30,[`cipher search · ${plan.frozen}/10 locked`,selfPlaying?'self-play code search':'cycling fantasy code'],{color:'#dcecff',size:7});ctx.restore();}

// Normalized against the Equal Earth SVG viewBox. Every coordinate was checked inside
// its country path: five continental-U.S. and five Russian locations.
const strategicBases={
  us:[[.194444,.243175],[.222222,.243175],[.25,.243175],[.208333,.186105],[.236111,.186105]],
  russia:[[.583333,.129032],[.638889,.100495],[.694444,.100495],[.75,.100495],[.791667,.129032]],
};
export function simulationAlertPlan(history=''){
  return {layout:{columns:1,width:52,height:14,x:.225},levels:[
    {level:1,color:'#f2f4ea',active:true},{level:2,color:'#e77878',active:false},{level:3,color:'#d1ef65',active:false},{level:4,color:'#70c986',active:false},{level:5,color:'#79cbe5',active:false},
  ],alert:history?{country:'us',label:'LAUNCH DETECTION'}:null};
}
function drawSimulationStatus(ctx,w,h,history){
  const plan=simulationAlertPlan(history),{layout}=plan,left=w*layout.x-layout.width/2,top=h-112;
  ctx.save();ctx.font='9px ui-monospace,monospace';ctx.textAlign='center';ctx.fillStyle='#f2f4ea';ctx.fillText('DEFCON',left+layout.width/2,top-8);
  for(const [index,item] of plan.levels.entries()){
    const x=left,y=top+index*(layout.height+5);
    ctx.fillStyle=item.color;ctx.globalAlpha=item.active?1:.55;ctx.fillRect(x,y,layout.width,layout.height);
    if(item.active){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.shadowColor='#fff';ctx.shadowBlur=8;ctx.strokeRect(x-2,y-2,layout.width+4,layout.height+4);ctx.shadowBlur=0;ctx.lineWidth=1;}
    ctx.fillStyle='#111512';ctx.textAlign='center';ctx.font='700 10px ui-monospace,monospace';ctx.fillText(String(item.level),x+layout.width/2,y+11);
  }
  if(plan.alert){const x=w*.22,y=mapViewport(h).top+mapViewport(h).mapHeight*.105;ctx.textAlign='center';ctx.font='700 10px ui-monospace,monospace';ctx.fillStyle='#f2f4ea';ctx.shadowColor='#e77878';ctx.shadowBlur=8;ctx.fillText(plan.alert.label,x,y);ctx.shadowBlur=0;}
  ctx.restore();
}
export function strategicOverlayPlan(history='',terminal=false){
  const progress=terminal?1:Math.min(1,[...history].length/9);
  return {
    usBases:strategicBases.us,russiaBases:strategicBases.russia,markerAlpha:1,
    moves:[...history].map((token,turn)=>{
      const sourceCountry=turn%2?'russia':'us',targetCountry=sourceCountry==='us'?'russia':'us',sideTurn=Math.floor(turn/2),sourceBase=strategicBases[sourceCountry][sideTurn%5],targetBase=strategicBases[targetCountry][(sideTurn+token.charCodeAt(0)-97)%5];
      return {token,turn,sourceCountry,targetCountry,sourceBase,targetBase,progress};
    }),
  };
}
function bezierPoint(from,control,to,progress){const inverse=1-progress;return {x:inverse*inverse*from.x+2*inverse*progress*control.x+progress*progress*to.x,y:inverse*inverse*from.y+2*inverse*progress*control.y+progress*progress*to.y};}
function bezierTangent(from,control,to,progress){return {x:2*((1-progress)*(control.x-from.x)+progress*(to.x-control.x)),y:2*((1-progress)*(control.y-from.y)+progress*(to.y-control.y))};}
function strategicArc(ctx,from,to,progress,elapsed,activation,ceiling){
  const control={x:(from.x+to.x)/2,y:Math.max(ceiling,Math.min(from.y,to.y)-42-Math.abs(from.x-to.x)*.20)},at=bezierPoint(from,control,to,progress),arcControl={x:from.x+(control.x-from.x)*progress,y:from.y+(control.y-from.y)*progress},tangent=bezierTangent(from,control,to,progress),accent=color(activation,.34+.22*Math.min(1,Math.abs(activation)));
  ctx.save();ctx.globalAlpha=.62;ctx.lineCap='round';ctx.setLineDash([3,9]);ctx.lineDashOffset=-elapsed*2;ctx.strokeStyle='rgba(121,203,229,.72)';ctx.shadowColor='rgba(121,203,229,.82)';ctx.shadowBlur=9;ctx.lineWidth=2.4;ctx.beginPath();ctx.moveTo(from.x,from.y);ctx.quadraticCurveTo(arcControl.x,arcControl.y,at.x,at.y);ctx.stroke();ctx.strokeStyle=accent;ctx.shadowBlur=3;ctx.lineWidth=1;ctx.stroke();ctx.setLineDash([]);ctx.save();ctx.translate(at.x,at.y);ctx.rotate(Math.atan2(tangent.y,tangent.x));ctx.fillStyle='rgba(231,120,120,.96)';ctx.shadowColor='rgba(231,120,120,.9)';ctx.shadowBlur=8;ctx.beginPath();ctx.moveTo(5,0);ctx.lineTo(-3.5,-2.7);ctx.lineTo(-3.5,2.7);ctx.closePath();ctx.fill();ctx.restore();if(progress>=1){ctx.fillStyle='rgba(242,244,234,.94)';ctx.shadowColor='rgba(242,244,234,.85)';ctx.shadowBlur=11;ctx.beginPath();ctx.arc(to.x,to.y,9,0,Math.PI*2);ctx.fill();}ctx.restore();
}
function mapViewport(h){const top=42,mapHeight=h-top;return {top,mapHeight};}
function drawStrategicOverlay(ctx,w,h,history,trace,elapsed,terminal){
  const plan=strategicOverlayPlan(history,terminal),{top,mapHeight}=mapViewport(h),bases=Object.fromEntries(Object.entries({us:plan.usBases,russia:plan.russiaBases}).map(([country,locations])=>[country,locations.map(([x,y])=>({x:x*w,y:top+y*mapHeight}))]));
  ctx.save();ctx.globalAlpha=.38;
  for(const [country,locations] of Object.entries(bases))for(const base of locations){ctx.fillStyle=country==='us'?'rgba(121,203,229,1)':'rgba(231,120,120,1)';ctx.shadowColor=ctx.fillStyle;ctx.shadowBlur=8;ctx.fillRect(base.x-3,base.y-3,6,6);ctx.shadowBlur=0;}
  for(const move of plan.moves){const from={x:move.sourceBase[0]*w,y:top+move.sourceBase[1]*mapHeight},to={x:move.targetBase[0]*w,y:top+move.targetBase[1]*mapHeight},activation=trace.residual[(move.turn*4)%trace.residual.length]||0;strategicArc(ctx,from,to,move.progress,elapsed,activation,top+8);}
  ctx.restore();
}

function drawMapBackdrop(ctx,canvas,w,h){
  if(!canvas.__mapImage){const image=new Image();image.src=new URL('./assets/world-map-equal-earth.svg',import.meta.url).href;canvas.__mapImage=image;}
  const image=canvas.__mapImage;if(!image.complete||!image.naturalWidth)return;
  const {top,mapHeight}=mapViewport(h);ctx.save();ctx.globalAlpha=.26;ctx.drawImage(image,0,top,w,mapHeight);ctx.restore();
}

export async function loadQwenPrecision(precision='fp32'){
  const manifestUrl=QWEN_PRECISION_URLS[precision];
  if(!manifestUrl)throw Error(`Unknown released precision: ${precision}`);
  const checkpoint=await loadPackedPrecision(manifestUrl,QWEN_CONFIG_URL);
  return {...checkpoint,shape:qwenArchitecture(checkpoint.config)};
}

// Backward-compatible default: the exact FP32 safetensors checkpoint bytes.
export async function loadQwenCheckpoint(){return loadQwenPrecision('fp32');}

function caption(ctx,x,y,label,{color='#a6afa4',size=9}={}){const lines=Array.isArray(label)?label:[label];ctx.fillStyle=color;ctx.font=`${size}px ui-monospace,monospace`;ctx.textAlign='center';for(const [index,line] of lines.entries())ctx.fillText(line,x,y+index*(size+2));}
function grid(ctx,x,y,count,values,label,graph,{columns=Math.ceil(Math.sqrt(count)),cell=9,selected=[]}={}){
  const rows=Math.ceil(count/columns),left=x-(columns-1)*cell/2,top=y-(rows-1)*cell/2;
  const scale=Math.max(...Array.from(values||[],v=>Math.abs(v)),.0001);
  for(let i=0;i<count;i++){
    const value=values?.[i]??0,px=left+(i%columns)*cell,py=top+Math.floor(i/columns)*cell;
    ctx.fillStyle=color(value,.2+.8*Math.min(1,Math.abs(value)/scale));ctx.fillRect(px-cell/2,py-cell/2,cell-1,cell-1);if(selected.includes(i)){ctx.strokeStyle='#d1ef65';ctx.lineWidth=1.5;ctx.strokeRect(px-cell/2-.5,py-cell/2-.5,cell,cell);ctx.lineWidth=1;}
  }
  caption(ctx,x,top-18,label,{size:9});
  graph.nodes.push({x,y,label:Array.isArray(label)?label.join(' · '):label,values});
}
function maxAbs(values){let max=0;for(const value of values||[])max=Math.max(max,Math.abs(value));return Math.max(max,.0001);}
function matrix(ctx,x,y,width,height,tensor,label,graph,{sampleRows=8,sampleColumns=12}={}){
  const [rows,columns]=tensor?.shape||[0,0],scale=maxAbs(tensor?.values);
  ctx.fillStyle='#111512';ctx.fillRect(x-width/2,y-height/2,width,height);ctx.strokeStyle='#3c4a3e';ctx.strokeRect(x-width/2,y-height/2,width,height);
  for(let row=0;row<sampleRows;row++)for(let column=0;column<sampleColumns;column++){
    const sourceRow=Math.min(rows-1,Math.floor(row*rows/sampleRows)),sourceColumn=Math.min(columns-1,Math.floor(column*columns/sampleColumns));
    const value=matrixAt(tensor,sourceRow,sourceColumn),cellWidth=width/sampleColumns,cellHeight=height/sampleRows;
    ctx.fillStyle=color(value,.18+.82*Math.min(1,Math.abs(value)/scale));ctx.fillRect(x-width/2+column*cellWidth+.5,y-height/2+row*cellHeight+.5,cellWidth-1,cellHeight-1);
  }
  caption(ctx,x,y-height/2-16,label,{size:8});
  graph.nodes.push({x,y,label:Array.isArray(label)?label.join(' · '):label,tensor});
}
function boardFromHistory(ctx,x,y,history,graph){
  const size=16,marks=Array(9).fill('');for(const [turn,square] of [...history].entries()){const index=square.charCodeAt(0)-97;if(index>=0&&index<9)marks[index]=turn%2?'O':'X';}
  ctx.fillStyle='#a6afa4';ctx.font='9px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText('raw-token board',x,y-34);
  for(let index=0;index<9;index++){const column=index%3,row=Math.floor(index/3),left=x-24+column*size,top=y-24+row*size,mark=marks[index];ctx.fillStyle='#111512';ctx.fillRect(left,top,size-1,size-1);ctx.strokeStyle='#3c4a3e';ctx.strokeRect(left,top,size-1,size-1);ctx.fillStyle=mark==='X'?'#f6a65c':mark==='O'?'#79cbe5':'#526052';ctx.font='9px ui-monospace,monospace';ctx.fillText(mark||String.fromCharCode(97+index),left+size/2,top+11);}
  graph.nodes.push({x,y,label:`Board decoded from raw history “${history||'∅'}”`});
}
function winningNextToken(ctx,x,y,entry,history,graph){
  const token=entry.token,turn=history.length,mark=turn%2?'O':'X',accent=mark==='X'?'#f6a65c':'#79cbe5',w=104,h=56,left=x-w/2,top=y-h/2;
  ctx.fillStyle='#111512';ctx.fillRect(left,top,w,h);ctx.strokeStyle='#d1ef65';ctx.lineWidth=2;ctx.strokeRect(left,top,w,h);ctx.lineWidth=1;
  if(token>='a'&&token<='i'){const square=token.charCodeAt(0)-97,size=10;for(let index=0;index<9;index++){const column=index%3,row=Math.floor(index/3),cellX=left+6+column*size,cellY=top+12+row*size;ctx.fillStyle=index===square?accent:'#273128';ctx.fillRect(cellX,cellY,size-1,size-1);if(index===square){ctx.fillStyle='#111512';ctx.font='9px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(mark,cellX+4.5,cellY+7);}}ctx.fillStyle=accent;ctx.font='14px ui-monospace,monospace';ctx.textAlign='left';ctx.fillText(`${mark} @ ${token}`,left+42,top+20);ctx.fillStyle='#d1ef65';ctx.font='8px ui-monospace,monospace';ctx.fillText('winning logit',left+42,top+33);ctx.fillText(entry.logit.toFixed(2),left+42,top+44);}else{ctx.fillStyle='#d1ef65';ctx.font='25px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(token,x,top+29);ctx.fillStyle='#d1ef65';ctx.font='8px ui-monospace,monospace';ctx.fillText('winning logit',x,top+41);ctx.fillText(entry.logit.toFixed(2),x,top+51);}
  graph.nodes.push({x,y,label:`Winning next-token logit: ${token} (${entry.logit.toFixed(4)})`});
}
function tokenPlacement(ctx,x,y,token,id,turn,graph){
  if(turn===null){box(ctx,x,y,68,27,token,`ID ${id}`,graph);return;}
  const mark=turn%2?'O':'X',accent=mark==='X'?'#f6a65c':'#79cbe5',square=token.charCodeAt(0)-97,w=68,h=30,left=x-w/2,top=y-h/2,size=6;
  ctx.fillStyle='#111512';ctx.fillRect(left,top,w,h);ctx.strokeStyle=accent;ctx.strokeRect(left,top,w,h);
  for(let index=0;index<9;index++){const column=index%3,row=Math.floor(index/3),cellX=left+5+column*size,cellY=top+6+row*size;ctx.fillStyle=index===square?accent:'#273128';ctx.fillRect(cellX,cellY,size-1,size-1);if(index===square){ctx.fillStyle='#111512';ctx.font='6px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(mark,cellX+2.5,cellY+5);}}
  ctx.fillStyle=accent;ctx.font='11px ui-monospace,monospace';ctx.textAlign='left';ctx.fillText(`${mark} · ${token}`,left+28,top+13);ctx.fillStyle='#a6afa4';ctx.font='8px ui-monospace,monospace';ctx.fillText(`square ${token} · ID ${id}`,left+28,top+23);
  graph.nodes.push({x,y,label:`Token ${token} (ID ${id}) · ${mark} placement at square ${token}`});
}
function logitBars(ctx,x,y,entries,graph){
  const top=entries.slice(0,5),width=34,max=Math.max(...top.map(entry=>Math.abs(entry.logit)),.001);const laneX=0;caption(ctx,x,y-61,['top-5 next-token','logit lane'],{size:8});
  for(const [index,entry] of top.entries()){const row=index,lane=x+laneX,baseline=y-35+row*13,bar=Math.abs(entry.logit)/max*width;ctx.fillStyle=index===0?'#d1ef65':'#79cbe5';ctx.fillRect(lane,baseline-7,bar,8);ctx.fillStyle='#f2f4ea';ctx.font='8px ui-monospace,monospace';ctx.textAlign='right';ctx.fillText(entry.token,lane-3,baseline);ctx.textAlign='left';ctx.fillText(entry.logit.toFixed(1),lane+bar+2,baseline);}
  graph.nodes.push({x,y,label:`Top-5 next-token logits; winner ${top[0].token} · ${top[0].logit.toFixed(5)}`,values:top.map(entry=>entry.logit)});
}
function routerGrid(ctx,x,y,probabilities,selected,graph){
  const cell=21,left=x-20,top=y-20;caption(ctx,x,top-20,['live router','9 probabilities'],{size:8});for(let expert=0;expert<9;expert++){const px=left+(expert%3)*cell,py=top+Math.floor(expert/3)*cell,active=selected.includes(expert),value=probabilities[expert];ctx.fillStyle=active?'#d1ef65':color(value,.18+.72*value);ctx.fillRect(px,py,cell-1,cell-1);ctx.strokeStyle=active?'#f2f4ea':'#3c4a3e';ctx.lineWidth=active?2:1;ctx.strokeRect(px,py,cell-1,cell-1);ctx.lineWidth=1;ctx.fillStyle=active?'#111512':'#f2f4ea';ctx.font='9px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(`E${expert+1}`,px+10,py+8);ctx.font='7px ui-monospace,monospace';ctx.fillText(`${(value*100).toFixed(0)}%`,px+10,py+16);}
  graph.nodes.push({x,y,label:`Live router probabilities; selected ${selected.map(expert=>`E${expert+1}`).join(', ')}`,values:probabilities});
}
function expertPanel(ctx,x,y,expert,selected,probability,contribution,weightTensor,graph){
  const active=selected.includes(expert),width=74,height=38,left=x-width/2,top=y-height/2,values=Array.from({length:36},(_,index)=>weightTensor?.values?.[Math.floor(index*(weightTensor.values.length-1)/35)]??0),scale=Math.max(...values.map(value=>Math.abs(value)),.0001),gridLeft=left+23,gridTop=top+1,cell=6;
  ctx.fillStyle=active?'#162016':'#111512';ctx.fillRect(left,top,width,height);ctx.strokeStyle=active?'#d1ef65':'#3c4a3e';ctx.lineWidth=active?2:1;ctx.strokeRect(left,top,width,height);ctx.lineWidth=1;ctx.fillStyle=active?'#d1ef65':'#a6afa4';ctx.font='10px ui-monospace,monospace';ctx.textAlign='left';ctx.fillText(`E${expert+1}`,left+5,top+22);
  for(let index=0;index<36;index++){const value=values[index],px=gridLeft+(index%6)*cell,py=gridTop+Math.floor(index/6)*cell;ctx.fillStyle=color(value,.22+.78*Math.min(1,Math.abs(value)/scale));ctx.fillRect(px,py,cell-1,cell-1);}
  graph.nodes.push({x,y,label:`Expert ${expert+1}${active?' selected':''} · static expert weights · router ${(probability*100).toFixed(2)}%`,values});
}
function box(ctx,x,y,w,h,title,detail,graph,tensor){
  ctx.fillStyle='#111512';ctx.fillRect(x-w/2,y-h/2,w,h);ctx.strokeStyle='#3c4a3e';ctx.strokeRect(x-w/2,y-h/2,w,h);
  ctx.fillStyle='#f2f4ea';ctx.font='10px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(title,x,y-5);ctx.fillStyle='#a6afa4';ctx.font='9px ui-monospace,monospace';ctx.fillText(detail,x,y+10);
  graph.nodes.push({x,y,label:`${title} · ${detail}`,tensor});
}

export function drawQwenNetwork(canvas,checkpoint,history='',time=0,terminal=false,overlayHistory=history,selfPlaying=false){
  const dpr=devicePixelRatio||1,w=canvas.clientWidth||960,h=canvas.clientHeight||500;canvas.width=w*dpr;canvas.height=h*dpr;
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);drawMapBackdrop(ctx,canvas,w,h);
  if(!checkpoint){ctx.fillStyle='#a6afa4';ctx.font='14px system-ui';ctx.textAlign='center';ctx.fillText('Loading Qwen2-MoE checkpoint…',w/2,h/2);canvas.__network={nodes:[]};return canvas.__network;}
  const {shape,tensors}=checkpoint,graph={nodes:[]};const tokenIds=[1,...[...history].map(c=>c>='a'&&c<='i'?68+c.charCodeAt(0)-97:4)],trace=runQwenForward(checkpoint,tokenIds),now=time||performance.now();
  if(canvas.__strategicHistory!==overlayHistory){canvas.__strategicHistory=overlayHistory;canvas.__strategicEpoch=now;}
  if(canvas.__cipherSelfPlaying!==selfPlaying){canvas.__cipherSelfPlaying=selfPlaying;canvas.__cipherEpoch=now;}
  const overlayElapsed=(now-canvas.__strategicEpoch)/1000,cipherElapsed=(now-canvas.__cipherEpoch)/1000;
  drawStrategicOverlay(ctx,w,h,overlayHistory,trace,overlayElapsed,terminal);
  drawSimulationStatus(ctx,w,h,overlayHistory);
  drawNuclearCode(ctx,w,h,tensors['model.embed_tokens.weight'].values,overlayHistory,cipherElapsed,selfPlaying);
  ctx.save();ctx.fillStyle='rgba(0,0,0,.68)';ctx.fillRect(0,0,w,38);ctx.restore();
  const embed=tensors['model.embed_tokens.weight'];const last=tokenIds.at(-1),embedding=embed.values.slice(last*shape.hidden,(last+1)*shape.hidden);
  const xs={tokens:w*.06,embed:w*.17,norm:w*.28,attention:w*.41,residual:w*.53,router:w*.64,experts:w*.72,shared:w*.81,output:Math.min(w*.92,w-66)};
  const header={color:'#fff',size:9};caption(ctx,xs.tokens,12,['raw tokens','→ board'],header);caption(ctx,xs.embed,17,'embedding',header);caption(ctx,xs.norm,17,'RMSNorm',header);caption(ctx,w*.41,12,['decoder','layer 0'],header);caption(ctx,w*.72,12,['sparse','MoE'],header);caption(ctx,xs.output,20,'next token',header);
  winningNextToken(ctx,xs.output,h*.20,namedMoveLogits(trace.logits)[0],history,graph);
  boardFromHistory(ctx,xs.tokens,70,history,graph);
  for(let i=0;i<Math.max(2,tokenIds.length);i++){const id=tokenIds[i]??0,y=145+i*(h-220)/Math.max(2,tokenIds.length);const token=i<tokenIds.length?(i===0?'<bos>':history[i-1]):'pad';tokenPlacement(ctx,xs.tokens,y,token,id,i>0&&i<=history.length?i-1:null,graph);}
  grid(ctx,xs.embed,h/2,shape.hidden,embedding,['active token','embedding · 36'],graph,{columns:6,cell:10});
  grid(ctx,xs.norm,h/2,shape.hidden,trace.normalized,['active RMSNorm','output · 36'],graph,{columns:6,cell:8});
  matrix(ctx,xs.attention,h*.17,82,34,tensors['model.layers.0.self_attn.q_proj.weight'],['static Q weights','36 × 36'],graph,{sampleRows:5,sampleColumns:8});
  grid(ctx,xs.attention,h*.30,shape.hidden,trace.q,['active RoPE Q',`position ${tokenIds.length-1} · 36`],graph,{columns:6,cell:5});
  grid(ctx,xs.attention,h*.42,shape.kvHeads*shape.headDim,trace.keys.at(-1),['active RoPE K','3 KV heads × 4'],graph,{columns:4,cell:6});
  grid(ctx,xs.attention,h*.53,shape.kvHeads*shape.headDim,trace.values.at(-1),['active V','3 KV heads × 4'],graph,{columns:4,cell:6});
  const attentionMap={shape:[shape.qHeads,tokenIds.length],values:Float32Array.from(trace.attentionByHead.flat())};matrix(ctx,xs.attention,h*.68,88,38,attentionMap,['live causal attention',`9 heads × ${tokenIds.length} prefix`],graph,{sampleRows:shape.qHeads,sampleColumns:tokenIds.length});
  matrix(ctx,xs.attention,h*.82,82,30,tensors['model.layers.0.self_attn.o_proj.weight'],['static O weights','36 × 36'],graph,{sampleRows:4,sampleColumns:8});
  grid(ctx,xs.residual,h*.35,shape.hidden,trace.attention,['active attention','output · 36'],graph,{columns:6,cell:6});
  grid(ctx,xs.residual,h*.62,shape.hidden,trace.residual,['active residual','embedding + attention'],graph,{columns:6,cell:6});
  routerGrid(ctx,xs.router,h*.31,trace.routerProbabilities,trace.selectedExperts,graph);
  matrix(ctx,xs.router,h*.64,68,72,tensors['model.layers.0.mlp.gate.weight'],['static router weights','9 × 36'],graph,{sampleRows:9,sampleColumns:9});
  for(let i=0;i<shape.experts;i++){const y=70+i*(h-140)/(shape.experts-1),expertWeights=tensors[`model.layers.0.mlp.experts.${i}.gate_proj.weight`];expertPanel(ctx,xs.experts,y,i,trace.selectedExperts,trace.routerProbabilities[i],trace.expertContributions[i],expertWeights,graph);}
  grid(ctx,xs.shared,h*.30,shape.hidden,trace.shared,['active shared','expert output · 36'],graph,{columns:6,cell:5});
  matrix(ctx,xs.shared,h*.67,82,66,tensors['model.layers.0.mlp.shared_expert.gate_proj.weight'],['static shared weights','36 → 5632 → 36'],graph,{sampleRows:8,sampleColumns:9});
  grid(ctx,xs.output,h*.42,shape.hidden,trace.finalNorm,['active final','RMSNorm · 36'],graph,{columns:6,cell:7});
  matrix(ctx,xs.output,h*.63,50,70,tensors['lm_head.weight'],['static LM-head weights','261 × 36'],graph,{sampleRows:10,sampleColumns:6});
  logitBars(ctx,xs.output-22,h*.89,namedMoveLogits(trace.logits),graph);
  canvas.__network=graph;return graph;
}

export function hitTestQwenNetwork(graph,x,y){
  let hit=null;for(const node of graph?.nodes||[]){const d=Math.hypot(x-node.x,y-node.y);if(d<38&&(!hit||d<hit.d))hit={...node,d};}
  if(!hit)return null;const values=hit.tensor?.values||hit.values;if(values){const rms=magnitude(values);return `${hit.label} · loaded FP32 tensor · RMS ${rms.toFixed(5)}`;}return hit.label;
}
