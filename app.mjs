import { SQUARES, analyze, boardFor, classifyExpert, optimalPolicy, choosePolicyMove } from './rules.mjs';
import { drawNetwork, routeTimeline, LABEL, hitTestNetwork, activeExperts, EXPERTS } from './network.mjs';
import { trainingRounds } from './training.mjs';
const expertNames={win:'Win',block:'Block',fork:'Create fork',defendFork:'Prevent fork',position:'Center / corner',opening:'Opening',legality:'Legality',routing:'Combine answers'};
let history='';
const $=id=>document.getElementById(id);
function enableNetworkHover(){const canvas=$('network'),tip=$('network-tooltip');if(canvas.__hoverReady)return;canvas.__hoverReady=true;canvas.addEventListener('mousemove',event=>{const r=canvas.getBoundingClientRect(),hit=hitTestNetwork(canvas.__network,(event.clientX-r.left)*canvas.clientWidth/r.width,(event.clientY-r.top)*canvas.clientHeight/r.height);canvas.style.cursor=hit?'crosshair':'default';tip.hidden=!hit;if(hit){tip.textContent=hit.text;tip.style.left=`${Math.min(event.clientX+14,innerWidth-320)}px`;tip.style.top=`${Math.min(event.clientY+14,innerHeight-60)}px`}});canvas.addEventListener('mouseleave',()=>{tip.hidden=true;canvas.style.cursor='default'});}
function render(){
  const state=analyze(history), board=boardFor(history); $('history').value=history;
  $('board').replaceChildren(...board.map((mark,i)=>{const b=document.createElement('button');b.textContent=mark||SQUARES[i];b.className=mark?.toLowerCase()||'';b.disabled=!!mark||!state.valid;b.title=`${SQUARES[i]} · ${['top left','top middle','top right','middle left','center','middle right','bottom left','bottom middle','bottom right'][i]}`;b.onclick=()=>{history+=SQUARES[i];render()};return b}));
  const expert=classifyExpert(history), policy=optimalPolicy(history), config={mode:$('mode').value,seed:$('seed').value,temperature:Number($('temperature').value)||0},next=choosePolicyMove(history,config);
  const targets=policy.optimal.map((square,i)=>`${square}: ${(policy.probabilities[SQUARES.indexOf(square)]*100).toFixed(0)}%`).join(' · ');
  $('prediction').textContent=state.valid?`Target optimal set: ${targets}. Chosen (${config.mode}, seed ${config.seed}, T ${config.temperature}): ${next} · ${state.turn} to move · routed to ${expertNames[expert]}.`:`Invalid: ${state.reason}. Required output: ${next||'—'}.`;
  $('game-status').textContent=history?`History: ${history} · token ${history.length} of 8 input slots.`:'Click a square to start: X moves first.';
  const routed=activeExperts(history);$('experts').replaceChildren(...Object.entries(expertNames).map(([key,label])=>{const d=document.createElement('div');d.className='expert '+(routed.includes(key)?'active':'');d.innerHTML=`<b>${label}</b>${routed.includes(key)?'selected for this position':'inactive routing branch'}`;return d}));
  drawNetwork($('network'),history);enableNetworkHover();
  const timeline=routeTimeline(history);$('route-timeline').replaceChildren(...timeline.map((row,i)=>{const d=document.createElement('div');d.className='route '+(i===timeline.length-1?'active':'');d.textContent=`Round ${row.round}: ${row.expert in expertNames?expertNames[row.expert]:LABEL[row.expert]} (${row.history})`;return d}));
  $('expert-networks').replaceChildren(...EXPERTS.map(key=>{const active=routed.includes(key),d=document.createElement('section');d.className='expert-net '+(active?'active':'');d.innerHTML=`<h3>${expertNames[key]} ${active?'· routed':'· inactive'}</h3><div class="micro"><span>32 in</span><span class="nodes">${'<i></i>'.repeat(16)}</span><span class="nodes">${'<i></i>'.repeat(9)}</span></div><div class="micro"><span></span><span>16 hidden</span><span>9 logits</span></div>`;return d}));
  $('network-summary').textContent=`${timeline.length} routing decision${timeline.length===1?'':'s'} shown. Router selected ${routed.map(key=>expertNames[key]).join(' + ')}; their 32 → 16 → 9 expert panels are lime. Other expert panels are inactive. Weights are illustrative, not trained checkpoint weights.`;
}
$('apply').onclick=()=>{history=$('history').value.toLowerCase().slice(0,8);render()};
$('history').addEventListener('input',e=>{history=e.target.value.toLowerCase().slice(0,8)});
$('oracle').onclick=()=>{const state=analyze(history);if(state.valid&&history.length<8){history+=choosePolicyMove(history,{mode:$('mode').value,seed:$('seed').value,temperature:Number($('temperature').value)||0});render()}};
for(const id of ['mode','seed','temperature'])$(id).addEventListener('input',render);
$('reset').onclick=()=>{history='';render()};
let trainer=null,playing=false,trainingHistories=[],chartHistory=[];
function drawChart(){const c=$('train-chart'),w=c.clientWidth||900,h=c.clientHeight||500,d=devicePixelRatio||1;c.width=w*d;c.height=h*d;const x=c.getContext('2d');x.scale(d,d);x.fillStyle='#0d110e';x.fillRect(0,0,w,h);const pad=42;x.strokeStyle='#3c4a3e';for(let i=0;i<=4;i++){const y=pad+(h-pad*2)*i/4;x.beginPath();x.moveTo(pad,y);x.lineTo(w-pad,y);x.stroke();x.fillStyle='#a6afa4';x.fillText(`${100-i*25}%`,4,y+4)}const colors={draw:'#d1ef65',win:'#79cbe5',loss:'#f6a65c',illegal:'#e77878'};for(const [name,color] of Object.entries(colors)){x.strokeStyle=color;x.lineWidth=2;x.beginPath();chartHistory.forEach((row,i)=>{const px=pad+(w-pad*2)*(chartHistory.length===1?0:i/(chartHistory.length-1)),py=h-pad-(h-pad*2)*row[name]/100;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.fillStyle=color;x.fillText(name,pad+Object.keys(colors).indexOf(name)*80,18)}}
const trainConfig=()=>({layers:+$('train-layers').value,nodes:+$('train-nodes').value,rate:+$('train-rate').value,batch:32});
const paintTraining=step=>{const total=Object.values(step.totals).reduce((a,b)=>a+b,0);$('train-status').textContent=`Round ${step.round+1}`;if(step.index===step.total){const outcome=Object.fromEntries(Object.entries(step.totals).map(([name,count])=>[name,100*count/total]));chartHistory.push(outcome);$('train-status').textContent=`Round ${step.round+1} complete · loss ${outcome.loss.toFixed(1)}% · illegal ${outcome.illegal.toFixed(1)}% remaining`;drawChart()}};
async function ensureTrainingData(){if(trainingHistories.length)return;const data=await fetch('data/reachable-policy.json').then(r=>r.json());trainingHistories=data.examples.map(row=>row.history);}
async function runTraining(){await ensureTrainingData();playing=true;const loop=()=>{if(!playing)return;for(let i=0;i<4;i++){const step=trainer.next().value;paintTraining(step)}requestAnimationFrame(loop)};requestAnimationFrame(loop)}
$('train-start').onclick=async()=>{if(!trainer)trainer=trainingRounds((await (await fetch('data/reachable-policy.json')).json()).examples.map(row=>row.history),trainConfig());runTraining()};
$('train-pause').onclick=()=>{playing=false;$('train-status').textContent+=' · paused; current weights retained'};
$('train-reset').onclick=()=>{playing=false;trainer=null;chartHistory=[];$('train-status').textContent='Weights reset. Change layers/nodes, then Start.';drawChart()};
drawChart();
render();
