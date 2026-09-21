export const SQUARES = 'abcdefghi';
const LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
const MOVE_ORDER = [4,0,2,6,8,1,3,5,7];

export function encodeHistory(history='') { return [...history].map(token => SQUARES.indexOf(token)); }
export function boardFor(history='') {
  const board=Array(9).fill(null);
  for (let i=0;i<history.length;i++) board[SQUARES.indexOf(history[i])]=i%2?'O':'X';
  return board;
}
export function winner(board) {
  for (const line of LINES) if (board[line[0]] && line.every(i=>board[i]===board[line[0]])) return board[line[0]];
  return null;
}
export function analyze(history='') {
  const tokens=[...history]; let valid=tokens.length<=8; let reason='';
  if (!tokens.every(t=>SQUARES.includes(t))) { valid=false; reason='non-square token'; }
  if (new Set(tokens).size!==tokens.length) { valid=false; reason='repeated square'; }
  const board=valid?boardFor(history):Array(9).fill(null);
  const won=valid&&winner(board);
  if (won) { valid=false; reason=`${won} already won`; }
  return {valid,reason,board,turn:tokens.length%2?'O':'X',last:tokens.at(-1)||''};
}
function immediate(board,mark) {
  for(let i=0;i<9;i++) if(!board[i]) { board[i]=mark; const ok=winner(board)===mark; board[i]=null; if(ok)return i; }
  return -1;
}
function score(board,turn,root) {
  const won=winner(board); if(won)return won===root?10:-10;
  const open=board.map((v,i)=>v?null:i).filter(i=>i!==null); if(!open.length)return 0;
  const next=turn==='X'?'O':'X';
  const values=open.map(i=>{board[i]=turn;const value=score(board,next,root);board[i]=null;return value;});
  return turn===root?Math.max(...values):Math.min(...values);
}
export function oracleMove(history='') {
  const state=analyze(history); if(!state.valid)return state.last;
  const win=immediate(state.board,state.turn); if(win>=0)return SQUARES[win];
  const other=state.turn==='X'?'O':'X', block=immediate(state.board,other); if(block>=0)return SQUARES[block];
  const moves=MOVE_ORDER.filter(i=>!state.board[i]);
  let best=moves[0],bestScore=-Infinity;
  for(const move of moves){state.board[move]=state.turn;const value=score(state.board,state.turn==='X'?'O':'X',state.turn);state.board[move]=null;if(value>bestScore){bestScore=value;best=move;}}
  return SQUARES[best];
}
export function classifyExpert(history='') {
  const state=analyze(history); if(!state.valid)return 'legality';
  if(immediate([...state.board],state.turn)>=0)return 'win';
  const other=state.turn==='X'?'O':'X'; if(immediate([...state.board],other)>=0)return 'block';
  if(history.length===0)return 'opening';
  return history.length<3?'position':'fork';
}
export function featureVector(history='') {
  const state=analyze(history), vector=[];
  for(const cell of state.board) vector.push(cell==='X'?1:0,cell==='O'?1:0,cell?0:1);
  vector.push(state.turn==='X'?1:0,state.turn==='O'?1:0); return vector;
}
