import { classifyExpert } from './rules.mjs';
export const EXPERTS=['win','block','fork','defendFork','position','opening','legality','routing'];
export const LABEL={win:'Win',block:'Block',fork:'Fork',defendFork:'Defend fork',position:'Position',opening:'Opening',legality:'Legality',routing:'Routing'};
export const selectedExpert=history=>classifyExpert(history);
export function routeTimeline(history=''){const rows=[{round:0,history:'∅',expert:selectedExpert('')}];for(let i=1;i<=history.length;i++)rows.push({round:i,history:history.slice(0,i),expert:selectedExpert(history.slice(0,i))});return rows}
const noise=(a,b,c=0)=>Math.sin((a+1)*12.9898+(b+1)*78.233+(c+1)*37.719)*.5+.5;
export function drawNetwork(canvas,history=''){
 const dpr=devicePixelRatio||1,w=canvas.clientWidth||960,h=canvas.clientHeight||500;canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);const active=selectedExpert(history), expertIndex=EXPERTS.indexOf(active);
 const cols=[{name:'8 token slots',n:8,x:70},{name:'layer 1 · 32',n:32,x:w*.33},{name:'layer 2 · 32',n:32,x:w*.58},{name:'8 experts',n:8,x:w*.80},{name:'9 outputs',n:9,x:w-50}];
 const pos=(column,i)=>({x:column.x,y:56+(i+1)*(h-112)/(column.n+1)}); const dot=(p,r,color)=>{ctx.fillStyle=color;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill()};
 ctx.font='12px ui-monospace,monospace';ctx.textAlign='center';ctx.fillStyle='#a6afa4';for(const col of cols)ctx.fillText(col.name,col.x,22);
 for(let ci=0;ci<cols.length-1;ci++){const a=cols[ci],b=cols[ci+1];for(let i=0;i<a.n;i++)for(let j=0;j<b.n;j++){const weight=noise(i,j,ci)*2-1;const p=pos(a,i),q=pos(b,j);const chosen=(ci===2&&j===expertIndex)||(ci===3&&i===expertIndex);ctx.strokeStyle=chosen?`rgba(209,239,101,${.16+Math.abs(weight)*.7})`:`rgba(121,203,229,${.015+Math.abs(weight)*.08})`;ctx.lineWidth=chosen?1.5:.55;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke()}}
 for(let ci=0;ci<cols.length;ci++){const col=cols[ci];for(let i=0;i<col.n;i++){const p=pos(col,i),selected=ci===3&&i===expertIndex;dot(p,selected?7:ci===0?5:3,selected?'#d1ef65':ci===4?'#f6a65c':'#79cbe5');if(ci===0){ctx.fillStyle='#f2f4ea';ctx.textAlign='right';ctx.fillText(String.fromCharCode(97+i),p.x-10,p.y+4)}if(ci===3){ctx.fillStyle=selected?'#d1ef65':'#a6afa4';ctx.textAlign='left';ctx.fillText(LABEL[EXPERTS[i]],p.x+10,p.y+4)}}}
}
