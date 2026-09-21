import { SQUARES, analyze, classifyExpert, featureVector, oracleMove, optimalPolicy } from './rules.mjs';
import { writeFileSync } from 'node:fs';
const examples=new Map();
function visit(history=''){
  const state=analyze(history); if(!state.valid)return;
  if(history.length){const key=state.board.map(cell=>cell||'-').join('')+state.turn,policy=optimalPolicy(history);examples.set(key,{history,features:featureVector(history),target:oracleMove(history),policy,expert:classifyExpert(history)});}
  if(history.length===8)return;
  for(const token of SQUARES)if(!history.includes(token))visit(history+token);
}
visit();
const rows=[...examples.values()];
writeFileSync('data/reachable-policy.json',JSON.stringify({vocabulary:[...SQUARES],context:8,examples:rows},null,2)+'\n');
console.log(`wrote ${rows.length} reachable board positions`);
