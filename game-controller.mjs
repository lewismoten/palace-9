import { SQUARES, analyze, choosePolicyMove, gameResult } from './rules.mjs';

const normalize=state=>({history:state.history||'',finalMove:state.finalMove||'',seed:Number(state.seed)||1983});
const terminal=state=>gameResult(state.history,state.finalMove).terminal;

export function clickSquare(previous,square){
  const state=normalize(previous);
  if(!SQUARES.includes(square)||terminal(state)||!analyze(state.history).valid||state.history.includes(square)||state.finalMove)return state;
  return state.history.length<8?{...state,history:state.history+square}:{...state,finalMove:square};
}

export function shouldAutoMove(players,state){
  const current=normalize(state);
  if(terminal(current)||!analyze(current.history).valid)return false;
  if(Number(players)===0)return true;
  return Number(players)===1&&current.history.length%2===1;
}

export function modelMove(previous){
  const state=normalize(previous);
  if(terminal(state)||!analyze(state.history).valid)return state;
  const move=choosePolicyMove(state.history,{mode:'varied',seed:state.seed,temperature:0});
  return {...clickSquare(state,move),seed:state.seed+1};
}

export function autoplayDelay(completedGames){const progress=Math.min(1,Math.max(0,Number(completedGames)||0)/4);return Math.round(900-(880*progress));}

export function newGame(previous){const state=normalize(previous);return {history:'',finalMove:'',seed:state.seed+1};}
