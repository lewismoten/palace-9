import { SQUARES, analyze, boardFor, gameResult } from './rules.mjs';
import { clickSquare, modelMove, newGame, shouldAutoMove, autoplayDelay } from './game-controller.mjs';
import { loadQwenPrecision, drawQwenNetwork, hitTestQwenNetwork } from './qwen-visualizer.mjs';

let game={history:'',finalMove:'',seed:1983},qwenCheckpoint=null,qwenAnimationFrame=0,qwenLastFrame=0,autoTimer=0,qwenLoad=0,autoGamesPlayed=0;
const $=id=>document.getElementById(id);
const players=()=>Number($('players').value);
const formatParameterCount=count=>count>=1e9?`${(count/1e9).toFixed(1)}B`:count>=1e6?`${(count/1e6).toFixed(1)}M`:count>=1e3?`${(count/1e3).toFixed(1)}K`:`${count}`;
const RELEASE_ARTIFACT_BYTES={fp32:2598856,f16:1304384,q6_k:1065216,q4_k_m:1065216};
const formatFileSize=bytes=>bytes>=1e9?`${(bytes/1e9).toFixed(1)} GB`:bytes>=1e6?`${(bytes/1e6).toFixed(1)} MB`:bytes>=1e3?`${(bytes/1e3).toFixed(1)} KB`:`${bytes} B`;
async function selectPrecision(){
  const precision=$('precision').value,loadId=++qwenLoad;qwenCheckpoint=null;render();
  try{const checkpoint=await loadQwenPrecision(precision);if(loadId!==qwenLoad)return;qwenCheckpoint=checkpoint;render();if(!qwenAnimationFrame)qwenAnimationFrame=requestAnimationFrame(animateQwenInspector);}catch(error){if(loadId===qwenLoad)console.error(`Qwen ${precision.toUpperCase()} artifact failed to load:`,error);}
}

function enableNetworkHover(){
  const canvas=$('network'),tip=$('network-tooltip');if(canvas.__hoverReady)return;canvas.__hoverReady=true;
  canvas.addEventListener('mousemove',event=>{const r=canvas.getBoundingClientRect(),text=hitTestQwenNetwork(canvas.__network,(event.clientX-r.left)*canvas.clientWidth/r.width,(event.clientY-r.top)*canvas.clientHeight/r.height);canvas.style.cursor=text?'crosshair':'default';tip.hidden=!text;if(text){tip.textContent=text;tip.style.left=`${Math.min(event.clientX+14,innerWidth-420)}px`;tip.style.top=`${Math.min(event.clientY+14,innerHeight-60)}px`}});
  canvas.addEventListener('mouseleave',()=>{tip.hidden=true;canvas.style.cursor='default'});
}

function appendMove(square){game=clickSquare(game,square);render();if(players()===1&&shouldAutoMove(1,game)){game=modelMove(game);render();}}
function drawWinLines(result){const winLine=$('win-line');winLine.replaceChildren();for(const winningLine of result.lines){const [from,to]=[winningLine[0],winningLine[2]],line=document.createElementNS('http://www.w3.org/2000/svg','line');line.setAttribute('x1',String(from%3+.5));line.setAttribute('y1',String(Math.floor(from/3)+.5));line.setAttribute('x2',String(to%3+.5));line.setAttribute('y2',String(Math.floor(to/3)+.5));winLine.append(line)}}

function render(){
  const state=analyze(game.history),result=gameResult(game.history,game.finalMove),board=boardFor(game.history+game.finalMove),mode=players();
  $('history').value=game.history;
  $('board').replaceChildren(...board.map((mark,index)=>{const square=SQUARES[index],button=document.createElement('button');button.textContent=mark||square;button.className=mark?.toLowerCase()||'empty';button.disabled=!!mark||!state.valid||!!game.finalMove||result.terminal||mode===0||(mode===1&&state.turn==='O');button.title=`${square} · ${['top left','top middle','top right','middle left','center','middle right','bottom left','bottom middle','bottom right'][index]}`;button.onclick=()=>appendMove(square);return button}));
  drawWinLines(result);
  $('reset').disabled=mode===0;
  const terminalLabel=result.winner?`Winner: ${result.winner}`:result.draw?'Draw':!state.valid?'Invalid':'';
  const overlay=$('draw-overlay');overlay.textContent=terminalLabel;overlay.dataset.outcome=result.winner?.toLowerCase()||'invalid';overlay.hidden=!terminalLabel;
  const parameterCount=qwenCheckpoint?Object.values(qwenCheckpoint.tensors).reduce((total,tensor)=>total+tensor.values.length,0):0;$('parameters').textContent=parameterCount?formatParameterCount(parameterCount):'—';
  $('artifact-size').textContent=formatFileSize(RELEASE_ARTIFACT_BYTES[$('precision').value]);
  drawQwenNetwork($('network'),qwenCheckpoint,game.history,0,result.terminal,game.history+game.finalMove,mode===0);enableNetworkHover();
  scheduleAutoplay();
}

function scheduleAutoplay(){
  clearTimeout(autoTimer);if(players()!==0)return;
  const result=gameResult(game.history,game.finalMove),delay=autoplayDelay(autoGamesPlayed);
  if(result.terminal){autoTimer=setTimeout(()=>{if(players()===0){autoGamesPlayed++;game=newGame(game);render();}},delay);return;}
  if(shouldAutoMove(0,game))autoTimer=setTimeout(()=>{if(players()===0&&shouldAutoMove(0,game)){game=modelMove(game);render();}},delay);
}
function animateQwenInspector(time){if(qwenCheckpoint&&time-qwenLastFrame>=125){drawQwenNetwork($('network'),qwenCheckpoint,game.history,time,gameResult(game.history,game.finalMove).terminal,game.history+game.finalMove,players()===0);qwenLastFrame=time;}qwenAnimationFrame=requestAnimationFrame(animateQwenInspector)}

$('apply').onclick=()=>{game={...game,history:$('history').value.toLowerCase().slice(0,8),finalMove:''};render()};
$('history').addEventListener('keydown',event=>{if(event.key==='Enter')$('apply').click()});
$('reset').onclick=()=>{game=newGame(game);render()};
$('players').onchange=()=>{if(players()===0){autoGamesPlayed=0;game=newGame(game);}render()};
$('precision').onchange=()=>{selectPrecision()};

render();
selectPrecision();
