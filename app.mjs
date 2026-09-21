import { SQUARES, analyze, boardFor, classifyExpert, oracleMove } from './rules.mjs';
import { drawNetwork, routeTimeline, LABEL, hitTestNetwork } from './network.mjs';
const expertNames={win:'Win',block:'Block',fork:'Fork',defendFork:'Defend fork',position:'Position',opening:'Opening',legality:'Legality',routing:'Routing'};
let history='';
const $=id=>document.getElementById(id);
function enableNetworkHover(){const canvas=$('network'),tip=$('network-tooltip');if(canvas.__hoverReady)return;canvas.__hoverReady=true;canvas.addEventListener('mousemove',event=>{const r=canvas.getBoundingClientRect(),hit=hitTestNetwork(canvas.__network,(event.clientX-r.left)*canvas.clientWidth/r.width,(event.clientY-r.top)*canvas.clientHeight/r.height);canvas.style.cursor=hit?'crosshair':'default';tip.hidden=!hit;if(hit){tip.textContent=hit.text;tip.style.left=`${Math.min(event.clientX+14,innerWidth-320)}px`;tip.style.top=`${Math.min(event.clientY+14,innerHeight-60)}px`}});canvas.addEventListener('mouseleave',()=>{tip.hidden=true;canvas.style.cursor='default'});}
function render(){
  const state=analyze(history), board=boardFor(history); $('history').value=history;
  $('board').replaceChildren(...board.map((mark,i)=>{const b=document.createElement('button');b.textContent=mark||SQUARES[i];b.className=mark?.toLowerCase()||'';b.disabled=!!mark||!state.valid;b.title=`${SQUARES[i]} · ${['top left','top middle','top right','middle left','center','middle right','bottom left','bottom middle','bottom right'][i]}`;b.onclick=()=>{history+=SQUARES[i];render()};return b}));
  const expert=classifyExpert(history), next=oracleMove(history);
  $('prediction').textContent=state.valid?`Next: ${next} · ${state.turn} to move · routed to ${expertNames[expert]}.`:`Invalid: ${state.reason}. Required output: ${next||'—'}.`;
  $('game-status').textContent=history?`History: ${history} · token ${history.length} of 8 input slots.`:'Click a square to start: X moves first.';
  $('experts').replaceChildren(...Object.entries(expertNames).map(([key,label])=>{const d=document.createElement('div');d.className='expert '+(key===expert?'active':'');d.innerHTML=`<b>${label}</b>${key===expert?'selected for this position':'available routing branch'}`;return d}));
  drawNetwork($('network'),history);enableNetworkHover();
  const timeline=routeTimeline(history);$('route-timeline').replaceChildren(...timeline.map((row,i)=>{const d=document.createElement('div');d.className='route '+(i===timeline.length-1?'active':'');d.textContent=`Round ${row.round}: ${LABEL[row.expert]} (${row.history})`;return d}));
  $('network-summary').textContent=`${timeline.length} routing decision${timeline.length===1?'':'s'} shown. Current expert: ${LABEL[expert]}. Every edge is visible; cyan is an illustrative deterministic initialized weight, and lime is the selected expert path. These are not trained checkpoint weights.`;
}
$('apply').onclick=()=>{history=$('history').value.toLowerCase().slice(0,8);render()};
$('history').addEventListener('input',e=>{history=e.target.value.toLowerCase().slice(0,8)});
$('oracle').onclick=()=>{const state=analyze(history);if(state.valid&&history.length<8){history+=oracleMove(history);render()}};
$('reset').onclick=()=>{history='';render()};
render();
