import { SQUARES, analyze, boardFor, classifyExpert, oracleMove } from './rules.mjs';
const expertNames={win:'Win',block:'Block',fork:'Fork',defendFork:'Defend fork',position:'Position',opening:'Opening',legality:'Legality',routing:'Routing'};
let history='';
const $=id=>document.getElementById(id);
function render(){
  const state=analyze(history), board=boardFor(history); $('history').value=history;
  $('board').replaceChildren(...board.map((mark,i)=>{const b=document.createElement('button');b.textContent=mark||SQUARES[i];b.className=mark?.toLowerCase()||'';b.disabled=!!mark||!state.valid;b.title=`${SQUARES[i]} · ${['top left','top middle','top right','middle left','center','middle right','bottom left','bottom middle','bottom right'][i]}`;b.onclick=()=>{history+=SQUARES[i];render()};return b}));
  const expert=classifyExpert(history), next=oracleMove(history);
  $('prediction').textContent=state.valid?`Next: ${next} · ${state.turn} to move · routed to ${expertNames[expert]}.`:`Invalid: ${state.reason}. Required output: ${next||'—'}.`;
  $('game-status').textContent=history?`History: ${history} · token ${history.length} of 8 input slots.`:'Click a square to start: X moves first.';
  $('experts').replaceChildren(...Object.entries(expertNames).map(([key,label])=>{const d=document.createElement('div');d.className='expert '+(key===expert?'active':'');d.innerHTML=`<b>${label}</b>${key===expert?'selected for this position':'available routing branch'}`;return d}));
}
$('apply').onclick=()=>{history=$('history').value.toLowerCase().slice(0,8);render()};
$('history').addEventListener('input',e=>{history=e.target.value.toLowerCase().slice(0,8)});
$('oracle').onclick=()=>{const state=analyze(history);if(state.valid&&history.length<8){history+=oracleMove(history);render()}};
$('reset').onclick=()=>{history='';render()};
render();
