import { boardFor, SQUARES, analyze } from './rules.mjs';
import { predict } from './training.mjs';

export const EXPERTS=Array.from({length:9},(_,index)=>`expert-${index+1}`);
export const NETWORK_ICON_SIZE=32;
export const selectedExpert=()=>EXPERTS[0];
export function contextSlotIcon(history='',index=0){const token=history[index]||'',cell=token?SQUARES.indexOf(token):-1;return {token,mark:token?(index%2?'O':'X'):'',cell}}
export function historyFlags(history=''){const state=analyze(history);return {invalid:!state.valid,repeatedSquare:state.reason==='repeated square'}}
export function encoderSlotPosition(index,x,height){const top=height/2-40;return index<9?{x:x+(index%3-1)*14,y:top+Math.floor(index/3)*20}:{x,y:top+70}}
export function matrixGridPosition(index,count,x,height,cell=12){const columns=Math.ceil(Math.sqrt(count)),rows=Math.ceil(count/columns),left=x-(columns-1)*cell/2,top=height/2-(rows-1)*cell/2;return {x:left+(index%columns)*cell,y:top+Math.floor(index/columns)*cell,row:Math.floor(index/columns),column:index%columns,columns,rows,cell}}
export function edgeOpacity(weight){return .25+.75*Math.min(1,Math.abs(weight))}
export function edgeOpacityFor(edge,state,history=''){let signal=0;if(edge.stage===0)signal=edge.fromIndex<history.length?1:0;else if(edge.stage===1)signal=Math.abs(state?.encoded?.[edge.fromIndex]??0);else if(edge.stage===2)signal=Math.abs(state?.trunkActs?.[1]?.[edge.fromIndex]??0);else if(edge.stage===3)signal=Math.abs(state?.trunk?.[edge.fromIndex]??0);else signal=state?.gates?.[edge.fromIndex]??0;return edgeOpacity(edge.stage===0?signal:edge.weight*signal)}
export function outputSlotIcon(history='',probabilities=[],index=0){const token=index===9?'!':SQUARES[index]||'',confidence=Number.isFinite(probabilities[index])?probabilities[index]*100:null,chosen=Number.isFinite(confidence)&&index===probabilities.indexOf(Math.max(...probabilities));return {token,mark:index<9?(history.length%2?'O':'X'):'!',cell:index<9?index:-1,confidence,chosen}}
export function activeExperts(history='',model=null){if(model?.architecture==='board-state-moe'){const gates=predict(model,history).gates;return gates.map((gate,index)=>({gate,index})).sort((a,b)=>b.gate-a.gate).slice(0,Math.min(2,gates.length)).map(({index})=>EXPERTS[index]||`expert-${index+1}`)}return EXPERTS.slice(0,2)}
export function routeTimeline(history='',model=null){const row=(round,past)=>({round,history:past,expert:model?activeExperts(past,model):selectedExpert(past)});return [row(0,''),...Array.from({length:history.length},(_,index)=>row(index+1,history.slice(0,index+1)))]}
export function d4TransformDescription(index=0){return ([
 {rotation:0,flipHorizontal:false,flipVertical:false},
 {rotation:0,flipHorizontal:true,flipVertical:false},
 {rotation:180,flipHorizontal:false,flipVertical:false},
 {rotation:0,flipHorizontal:false,flipVertical:true},
 {rotation:90,flipHorizontal:true,flipVertical:false},
 {rotation:270,flipHorizontal:true,flipVertical:false},
 {rotation:270,flipHorizontal:false,flipVertical:false},
 {rotation:90,flipHorizontal:false,flipVertical:false},
][index]||{rotation:0,flipHorizontal:false,flipVertical:false})}
const noise=(a,b,c=0)=>Math.sin((a+1)*12.9898+(b+1)*78.233+(c+1)*37.719)*.5+.5;
const signed=(a,b,c=0)=>noise(a,b,c)*2-1;
const plus=n=>`${n>=0?'+':''}${n.toFixed(4)}`;
export const signedWeightColor=(value,alpha=1)=>value>0?`rgba(246,166,92,${alpha})`:value<0?`rgba(121,203,229,${alpha})`:`rgba(166,175,164,${alpha})`;
export function strongestRoutes(edges,contextLength,expertIndexes,limit=3){const finalStage=Math.max(...edges.map(edge=>edge.stage)),routes=[];const walk=(stage,fromIndex,chain)=>{const candidates=edges.filter(edge=>edge.stage===stage&&edge.fromIndex===fromIndex&&(stage!==finalStage||expertIndexes.includes(edge.toIndex)));for(const edge of candidates){const next=[...chain,edge];if(stage===finalStage)routes.push({edges:next,score:next.reduce((score,item)=>score*Math.abs(item.weight),1)});else walk(stage+1,edge.toIndex,next)}};for(let input=0;input<contextLength;input++)walk(0,input,[]);return routes.sort((a,b)=>b.score-a.score).slice(0,limit).map(route=>route.edges)};
const pointDistance=(x,y,x1,y1,x2,y2)=>{const dx=x2-x1,dy=y2-y1,t=Math.max(0,Math.min(1,((x-x1)*dx+(y-y1)*dy)/(dx*dx+dy*dy)));return Math.hypot(x-(x1+t*dx),y-(y1+t*dy))};
export function hitTestNetwork(graph,x,y){let node;for(const item of graph.nodes){const d=Math.hypot(x-item.x,y-item.y);if(d<=(node?.d??Infinity)&&d<=18)node={...item,d}}if(node)return {kind:'node',text:`${node.layer} · unit ${node.index+1} · ${graph.learned?'activation':'illustrative bias'} ${plus(node.bias)}`};let edge;for(const item of graph.edges){const d=pointDistance(x,y,item.x1,item.y1,item.x2,item.y2);if(d<=(edge?.d??Infinity)&&d<=7)edge={...item,d}}if(edge)return {kind:'edge',text:`${edge.from} → ${edge.to} · ${graph.learned?'trained':'illustrative'} weight ${plus(edge.weight).replace('-', '−')}`};return null}

const markColor=mark=>mark==='X'?'#f6a65c':mark==='O'?'#79cbe5':'#3c4a3e';
function drawBoard(ctx,x,y,cells,{size=42,accent='#a6afa4',label='',detail=''}={}){const cell=size/3,left=x-size/2,top=y-size/2;ctx.fillStyle='#fff';ctx.fillRect(left,top,size,size);ctx.strokeStyle='#111512';ctx.lineWidth=1;for(let i=1;i<3;i++){ctx.beginPath();ctx.moveTo(left+i*cell,top);ctx.lineTo(left+i*cell,top+size);ctx.moveTo(left,top+i*cell);ctx.lineTo(left+size,top+i*cell);ctx.stroke()}for(let i=0;i<9;i++)if(cells[i]){ctx.fillStyle=markColor(cells[i]);ctx.fillRect(left+(i%3)*cell+4,top+Math.floor(i/3)*cell+4,cell-8,cell-8)}ctx.strokeStyle=accent;ctx.lineWidth=2;ctx.strokeRect(left-2,top-2,size+4,size+4);ctx.fillStyle='#a6afa4';ctx.textAlign='center';ctx.font='10px ui-monospace,monospace';if(label)ctx.fillText(label,x,top-13);if(detail){ctx.font='8px ui-monospace,monospace';ctx.fillText(detail,x,top+size+13)}}
function drawSlot(ctx,p,icon,{size=NETWORK_ICON_SIZE,highlight=false,confidence=null,label=''}={}){if(icon.cell<0){ctx.fillStyle='#fff';ctx.fillRect(p.x-size/2,p.y-size/2,size,size);ctx.fillStyle='#111512';ctx.font=`bold ${Math.round(size*.85)}px ui-monospace,monospace`;ctx.textAlign='center';ctx.fillText('!',p.x,p.y+size*.3)}else{const cells=Array(9).fill(null);if(icon.mark)cells[icon.cell]=icon.mark;drawBoard(ctx,p.x,p.y,cells,{size,accent:highlight?'#d1ef65':'#a6afa4'})}if(highlight){ctx.strokeStyle='#d1ef65';ctx.lineWidth=4;ctx.strokeRect(p.x-size/2-3,p.y-size/2-3,size+6,size+6)}ctx.fillStyle=highlight?'#d1ef65':'#a6afa4';ctx.font='9px ui-monospace,monospace';ctx.textAlign='center';if(label)ctx.fillText(label,p.x,p.y-size/2-7);if(confidence!==null)ctx.fillText(`${confidence.toFixed(1)}%`,p.x,p.y+size/2+14)}

export function drawNetwork(canvas,history='',model=null){
 const dpr=devicePixelRatio||1,w=canvas.clientWidth||960,h=canvas.clientHeight||500;canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
 const learned=model?.architecture==='board-state-moe',state=learned?predict(model,history):null,canonical=state?.canonical,active=activeExperts(history,model),expertIndexes=active.map(key=>EXPERTS.indexOf(key)).filter(index=>index>=0),flags=historyFlags(history),board=boardFor(history);
 const layerWidth=learned?model.trunk.widths[1]:32,layer2Width=learned?model.trunk.widths.at(-1):32,expertCount=learned?model.experts.length:8;
 const x={context:54,board:w*.17,canonical:w*.31,layer1:canonical?w*.47:w*.40,trunk:canonical?w*.60:w*.60,experts:canonical?w*.72:w*.78,canonicalOutput:w*.83,undo:w*.90,output:w-28};
 const graph={nodes:[],edges:[],strongestRoutes:[],learned};
 const addNode=(layer,index,p,bias=0)=>graph.nodes.push({layer,index,x:p.x,y:p.y,bias});
 ctx.fillStyle='#a6afa4';ctx.font='11px ui-monospace,monospace';ctx.textAlign='center';
 ctx.fillText('8 context turns',x.context,22);ctx.fillText('decoded board state',x.board,22);if(canonical)ctx.fillText('canonical board state',x.canonical,22);ctx.fillText(`layer 1 · ${layerWidth}`,x.layer1,22);ctx.fillText(`shared trunk · ${layer2Width}`,x.trunk,22);ctx.fillText(`MoE router · ${expertCount}`,x.experts,22);if(canonical)ctx.fillText('canonical outputs',x.canonicalOutput,22);ctx.fillText(canonical?'final outputs':'9 outputs + !',x.output,22);
 for(let i=0;i<8;i++){const p={x:x.context,y:54+(i+1)*(h-108)/9};drawSlot(ctx,p,contextSlotIcon(history,i),{label:`Turn ${i+1}`});addNode('context turn',i,p,Number(i<history.length));}
 drawBoard(ctx,x.board,h/2,board,{size:58,accent:'#a6afa4',label:'board-state encoder',detail:flags.repeatedSquare?'dupe: on':'dupe: off'});for(let i=0;i<9;i++)addNode('board-state encoder',i,{x:x.board+(i%3-1)*19,y:h/2+(Math.floor(i/3)-1)*19},state?.encoded?.[i]??0);
 if(canonical){const descriptor=d4TransformDescription(canonical.transformIndex),detail=`rotate ${descriptor.rotation}° · flip H ${descriptor.flipHorizontal?'on':'off'} · flip V ${descriptor.flipVertical?'on':'off'}`;drawBoard(ctx,x.canonical,h/2,state.canonicalBoard,{size:58,accent:'#d1ef65',label:'D4 canonical transform',detail});for(let i=0;i<9;i++)addNode('D4 canonical board',i,{x:x.canonical+(i%3-1)*19,y:h/2+(Math.floor(i/3)-1)*19},state?.input?.[i*3]??0);}
 const drawMatrix=(name,count,column,values)=>{for(let i=0;i<count;i++){const p=matrixGridPosition(i,count,column,h,count>=36?12:14),value=values?.[i]??signed(i,count),size=count>=36?10:12;ctx.fillStyle=signedWeightColor(value,.3+Math.min(1,Math.abs(value))*.7);ctx.fillRect(p.x-size/2,p.y-size/2,size,size);addNode(name,i,p,value)}};
 drawMatrix('layer 1',layerWidth,x.layer1,state?.trunkActs?.[1]);drawMatrix('shared trunk',layer2Width,x.trunk,state?.trunk);
 drawMatrix('MoE router',expertCount,x.experts,state?.gates);
 if(canonical){const canonicalMarks=state.canonicalLogits.slice(0,9).map(value=>value>=0?'X':'O');drawBoard(ctx,x.canonicalOutput,h/2,canonicalMarks,{size:46,accent:'#f6a65c',label:'canonical logits',detail:'before inverse D4'});const descriptor=d4TransformDescription(canonical.transformIndex);ctx.fillStyle='#d1ef65';ctx.fillRect(x.undo-27,h/2-34,54,68);ctx.fillStyle='#111512';ctx.font='9px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText('inverse D4',x.undo,h/2-12);ctx.fillText(`rotate ${descriptor.rotation}°`,x.undo,h/2+2);ctx.fillText(`H ${descriptor.flipHorizontal?'on':'off'}`,x.undo,h/2+15);ctx.fillText(`V ${descriptor.flipVertical?'on':'off'}`,x.undo,h/2+28);graph.transcoder={group:'D4',transformIndex:canonical.transformIndex,permutation:canonical.permutation,inputCoordinates:'canonical',outputCoordinates:'original',...descriptor}}
 const probabilities=state?.probabilities||[];for(let i=0;i<10;i++){const p={x:x.output,y:48+(i+1)*(h-96)/11},icon=outputSlotIcon(history,probabilities,i);drawSlot(ctx,p,icon,{highlight:icon.chosen,confidence:icon.confidence,label:icon.token});addNode('final original-coordinate output',i,p,probabilities[i]??0)}
 canvas.__network=graph;return graph;
}
