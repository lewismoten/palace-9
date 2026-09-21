import { classifyExpert } from './rules.mjs';
export const EXPERTS=['win','block','fork','defendFork','position','opening','legality','routing'];
export const LABEL={win:'Win',block:'Block',fork:'Fork',defendFork:'Defend fork',position:'Position',opening:'Opening',legality:'Legality',routing:'Routing'};
export const selectedExpert=history=>classifyExpert(history);
export function routeTimeline(history=''){const rows=[{round:0,history:'∅',expert:selectedExpert('')}];for(let i=1;i<=history.length;i++)rows.push({round:i,history:history.slice(0,i),expert:selectedExpert(history.slice(0,i))});return rows}
const noise=(a,b,c=0)=>Math.sin((a+1)*12.9898+(b+1)*78.233+(c+1)*37.719)*.5+.5;
const signed=(a,b,c=0)=>noise(a,b,c)*2-1;
const plus=n=>`${n>=0?'+':''}${n.toFixed(4)}`;
const pointDistance=(x,y,x1,y1,x2,y2)=>{const dx=x2-x1,dy=y2-y1,t=Math.max(0,Math.min(1,((x-x1)*dx+(y-y1)*dy)/(dx*dx+dy*dy)));return Math.hypot(x-(x1+t*dx),y-(y1+t*dy))};
export function hitTestNetwork(graph,x,y){
 let node;for(const item of graph.nodes){const d=Math.hypot(x-item.x,y-item.y);if(d<=(node?.d??Infinity)&&d<=10)node={...item,d}}if(node)return {kind:'node',text:`${node.layer} · unit ${node.index+1} · illustrative bias ${plus(node.bias)}`};
 let edge;for(const item of graph.edges){const d=pointDistance(x,y,item.x1,item.y1,item.x2,item.y2);if(d<=(edge?.d??Infinity)&&d<=7)edge={...item,d}}if(edge)return {kind:'edge',text:`${edge.from} → ${edge.to} · illustrative weight ${plus(edge.weight).replace('-', '−')}`};return null;
}
export function drawNetwork(canvas,history=''){
 const dpr=devicePixelRatio||1,w=canvas.clientWidth||960,h=canvas.clientHeight||500;canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);const active=selectedExpert(history),expertIndex=EXPERTS.indexOf(active);
 const cols=[{name:'8 token slots',n:8,x:70},{name:'layer 1 · 32',n:32,x:w*.33},{name:'layer 2 · 32',n:32,x:w*.58},{name:'8 experts',n:8,x:w*.80},{name:'9 outputs',n:9,x:w-50}];const pos=(column,i)=>({x:column.x,y:56+(i+1)*(h-112)/(column.n+1)});
 const graph={nodes:[],edges:[]};for(let ci=0;ci<cols.length;ci++){for(let i=0;i<cols[ci].n;i++){const p=pos(cols[ci],i);graph.nodes.push({layer:cols[ci].name,index:i,x:p.x,y:p.y,bias:signed(i,ci,7)})}}for(let ci=0;ci<cols.length-1;ci++){const a=cols[ci],b=cols[ci+1];for(let i=0;i<a.n;i++)for(let j=0;j<b.n;j++){const p=pos(a,i),q=pos(b,j),weight=signed(i,j,ci),chosen=(ci===2&&j===expertIndex)||(ci===3&&i===expertIndex);graph.edges.push({from:ci===0?`token ${String.fromCharCode(97+i)}`:`${a.name} unit ${i+1}`,to:ci===3?`${LABEL[EXPERTS[j]]} expert`:`${b.name} unit ${j+1}`,x1:p.x,y1:p.y,x2:q.x,y2:q.y,weight,chosen})}}
 for(const edge of graph.edges){ctx.strokeStyle=edge.chosen?`rgba(209,239,101,${.16+Math.abs(edge.weight)*.7})`:`rgba(121,203,229,${.015+Math.abs(edge.weight)*.08})`;ctx.lineWidth=edge.chosen?1.5:.55;ctx.beginPath();ctx.moveTo(edge.x1,edge.y1);ctx.lineTo(edge.x2,edge.y2);ctx.stroke()}
 ctx.font='12px ui-monospace,monospace';ctx.textAlign='center';ctx.fillStyle='#a6afa4';for(const col of cols)ctx.fillText(col.name,col.x,22);for(let ci=0;ci<cols.length;ci++){const col=cols[ci];for(let i=0;i<col.n;i++){const p=pos(col,i),selected=ci===3&&i===expertIndex;ctx.fillStyle=selected?'#d1ef65':ci===4?'#f6a65c':'#79cbe5';ctx.beginPath();ctx.arc(p.x,p.y,selected?7:ci===0?5:3,0,Math.PI*2);ctx.fill();if(ci===0){ctx.fillStyle='#f2f4ea';ctx.textAlign='right';ctx.fillText(String.fromCharCode(97+i),p.x-10,p.y+4)}if(ci===3){ctx.fillStyle=selected?'#d1ef65':'#a6afa4';ctx.textAlign='left';ctx.fillText(LABEL[EXPERTS[i]],p.x+10,p.y+4)}}}
 canvas.__network=graph;return graph;
}
