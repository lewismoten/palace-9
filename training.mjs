import { featureVector, optimalPolicy, SQUARES, boardFor, INVALID_TOKEN, canonicalizeFeatures, inversePermuteSquares } from './rules.mjs';

const rand=i=>Math.sin(i*91.17)*.08;
const zeros=n=>Array(n).fill(0);
const mat=(rows,columns,key)=>Array.from({length:rows},(_,row)=>Array.from({length:columns},(_,column)=>rand(key+row*columns+column)));
const softmax=values=>{const max=Math.max(...values),raw=values.map(value=>Math.exp(value-max)),sum=raw.reduce((a,b)=>a+b,0);return raw.map(value=>value/sum)};
const multiply=(matrix,input)=>matrix.map(row=>row.reduce((sum,weight,index)=>sum+weight*input[index],0));
const tanh=value=>Math.tanh(value);
const tanhGradient=value=>1-value*value;
const pick=result=>result.probabilities.indexOf(Math.max(...result.probabilities));
const copyMatrix=matrix=>matrix.map(row=>[...row]);
const copyModel=model=>({architecture:model.architecture,...(model.canonicalization?{canonicalization:{...model.canonicalization}}:{}),encoder:{widths:[...model.encoder.widths],weights:copyMatrix(model.encoder.weights)},trunk:{widths:[...model.trunk.widths],weights:model.trunk.weights.map(copyMatrix)},router:{weights:copyMatrix(model.router.weights)},experts:model.experts.map(expert=>({widths:[...expert.widths],weights:expert.weights.map(copyMatrix)})),steps:model.steps});

export function createModel({layers=2,nodes=32,experts=8,expertNodes=16}={}){
  const trunkWidths=[9,...Array(Math.max(1,Number(layers))).fill(Number(nodes))];
  return {
    architecture:'board-state-moe',
    encoder:{widths:[29,9],weights:mat(9,29,1)},
    trunk:{widths:trunkWidths,weights:Array.from({length:trunkWidths.length-1},(_,index)=>mat(trunkWidths[index+1],trunkWidths[index],10+index))},
    router:{weights:mat(Number(experts),trunkWidths.at(-1),30)},
    experts:Array.from({length:Number(experts)},(_,index)=>({widths:[trunkWidths.at(-1),Number(expertNodes),9],weights:[mat(Number(expertNodes),trunkWidths.at(-1),50+index*2),mat(9,Number(expertNodes),51+index*2)]})),
    steps:0
  };
}

export function predict(model,history){
  const originalInput=featureVector(history).slice(0,model.encoder.widths[0]);
  const canonical=model.canonicalization?.group==='D4'?canonicalizeFeatures(originalInput):null;
  const input=(canonical?.features||originalInput).slice(0,model.encoder.widths[0]);
  const encoded=multiply(model.encoder.weights,input).map(tanh);
  const trunkActs=[encoded];
  for(const weights of model.trunk.weights)trunkActs.push(multiply(weights,trunkActs.at(-1)).map(tanh));
  const trunk=trunkActs.at(-1);
  const routerLogits=multiply(model.router.weights,trunk);
  const gates=softmax(routerLogits);
  const expertActs=model.experts.map(expert=>{
    const hidden=multiply(expert.weights[0],trunk).map(tanh);
    return {hidden,logits:multiply(expert.weights[1],hidden)};
  });
  const canonicalLogits=zeros(model.experts[0].widths.at(-1));
  for(let expert=0;expert<model.experts.length;expert++)for(let square=0;square<canonicalLogits.length;square++)canonicalLogits[square]+=gates[expert]*expertActs[expert].logits[square];
  const logits=canonical?inversePermuteSquares(canonicalLogits,canonical.permutation):canonicalLogits;
  return {input,originalInput,canonical,board:boardFor(history),canonicalBoard:canonical?Array.from({length:9},(_,index)=>{const cell=canonical.features.slice(index*3,index*3+3);return cell[0]?'X':cell[1]?'O':null}):null,encoded,trunkActs,trunk,routerLogits,gates,expertActs,canonicalLogits,logits,probabilities:softmax(logits)};
}

const addOuter=(gradient,delta,input)=>{for(let row=0;row<gradient.length;row++)for(let column=0;column<gradient[row].length;column++)gradient[row][column]+=delta[row]*input[column]};
const matVector=(matrix,vector)=>matrix[0].map((_,column)=>matrix.reduce((sum,row,index)=>sum+row[column]*vector[index],0));
const apply=(weights,gradient,rate)=>{for(let row=0;row<weights.length;row++)for(let column=0;column<weights[row].length;column++)weights[row][column]-=rate*gradient[row][column]};

export function trainOne(model,history,rate=.025){
  const policy=optimalPolicy(history);if(!policy.valid)return null;
  const result=predict(model,history),target=[...policy.probabilities,...zeros(Math.max(0,predict(model,history).probabilities.length-policy.probabilities.length))],delta=result.probabilities.map((value,index)=>value-target[index]);
  const trunkGradient=zeros(result.trunk.length);
  const gateDerivative=result.expertActs.map(expert=>expert.logits.reduce((sum,logit,index)=>sum+delta[index]*logit,0));
  const gateMean=gateDerivative.reduce((sum,value,index)=>sum+result.gates[index]*value,0);
  const routerDelta=result.gates.map((gate,index)=>gate*(gateDerivative[index]-gateMean));
  const routerGradient=mat(model.router.weights.length,model.router.weights[0].length,0);addOuter(routerGradient,routerDelta,result.trunk);apply(model.router.weights,routerGradient,rate);
  const routerBack=matVector(model.router.weights,routerDelta);for(let index=0;index<trunkGradient.length;index++)trunkGradient[index]+=routerBack[index];
  for(let index=0;index<model.experts.length;index++){
    const expert=model.experts[index],cached=result.expertActs[index],outputDelta=delta.map(value=>value*result.gates[index]);
    const outputGradient=mat(result.probabilities.length,cached.hidden.length,0);addOuter(outputGradient,outputDelta,cached.hidden);
    const hiddenDelta=matVector(expert.weights[1],outputDelta).map((value,unit)=>value*tanhGradient(cached.hidden[unit]));
    const hiddenGradient=mat(cached.hidden.length,result.trunk.length,0);addOuter(hiddenGradient,hiddenDelta,result.trunk);
    const expertBack=matVector(expert.weights[0],hiddenDelta);for(let unit=0;unit<trunkGradient.length;unit++)trunkGradient[unit]+=expertBack[unit];
    apply(expert.weights[1],outputGradient,rate);apply(expert.weights[0],hiddenGradient,rate);
  }
  let deltaTrunk=trunkGradient.map((value,index)=>value*tanhGradient(result.trunk[index]));
  for(let layer=model.trunk.weights.length-1;layer>=0;layer--){
    const weights=model.trunk.weights[layer],prior=result.trunkActs[layer],gradient=mat(weights.length,weights[0].length,0);addOuter(gradient,deltaTrunk,prior);const previous=matVector(weights,deltaTrunk).map((value,index)=>value*tanhGradient(prior[index]));apply(weights,gradient,rate);deltaTrunk=previous;
  }
  const encoderDelta=deltaTrunk;
  const encoderGradient=mat(9,result.input.length,0);addOuter(encoderGradient,encoderDelta,result.input);apply(model.encoder.weights,encoderGradient,rate);
  model.steps++;return {policy,probabilities:result.probabilities,gates:result.gates};
}

export function invalidCorrectionTarget(history=''){let prefix='';for(const token of history){const next=prefix+token;if(!optimalPolicy(next).valid)break;prefix=next}return prefix.at(-1)||''}
export function classifyOutcome(result){const index=pick(result),policy=result.policy;if(policy.probabilities[index]>0)return policy.best===10?'win':'draw';return policy.scores[index]===-10?'loss':'invalid'}
export function invalidContexts(histories){const all=new Set;for(const history of histories.filter(Boolean)){all.add(history+history.at(-1));all.add(history+'j')}return [...all]}
const diagnostic=(history,result,classification)=>({history,board:boardFor(history).map(cell=>cell||'·'),chosen:SQUARES[pick(result)],classification,optimal:result.policy.optimal,teacher:result.policy.probabilities.map(value=>value*100),confidence:result.probabilities.map(value=>value*100),scores:result.policy.scores});
export function evaluateModel(model,histories,policies,{corrections=true}={}){const totals={draw:0,win:0,loss:0,invalid:0,correction:0,incorrect:0},mistakes={loss:[],invalid:[]};for(const history of histories){const result={...predict(model,history),policy:policies?.[history]||optimalPolicy(history)},classification=classifyOutcome(result);totals[classification]++;if(classification==='loss'||classification==='invalid')mistakes[classification].push(diagnostic(history,result,classification))}if(corrections)for(const history of invalidContexts(histories)){const target=invalidCorrectionTarget(history),answer=SQUARES[pick(predict(model,history))];totals[answer===target?'correction':'incorrect']++}const total=histories.length,invalidTotal=totals.correction+totals.incorrect;return {total,invalidTotal,totals,mistakes,percent:Object.fromEntries(Object.entries(totals).map(([key,value])=>[key,100*value/(key==='correction'||key==='incorrect'?invalidTotal:total)]))}}
export function exportModel(model){return {format:'palace-9-moe/v1',modelName:'palace-9',expansion:'Predictive Adversarial Learning and Contingency Evaluator',squares:9,...copyModel(model)}}
const validMatrix=(matrix,rows,columns)=>Array.isArray(matrix)&&matrix.length===rows&&matrix.every(row=>Array.isArray(row)&&row.length===columns&&row.every(Number.isFinite));
export function importModel(data){
  const v1=data?.format==='palace-9-moe/v1',v2=data?.format==='palace-9-moe/v2',v3=data?.format==='palace-9-moe/v3';
  if(!data||(!v1&&!v2&&!v3)||data.architecture!=='board-state-moe')throw Error('Unsupported model format: requires palace-9-moe/v1, v2, or v3 board-state MoE');
  const encoder=data.encoder,trunk=data.trunk,router=data.router,experts=data.experts,outputs=v1?9:10,inputs=v3?31:v2?30:29;
  const validWidths=widths=>Array.isArray(widths)&&widths.length>=2&&widths.every(width=>Number.isInteger(width)&&width>0);
  const canonical=data.canonicalization==null||data.canonicalization.group==='D4';
  if(!canonical||((v2||v3)&&(!Array.isArray(data.outputTokens)||data.outputTokens.join('')!==SQUARES+INVALID_TOKEN||data.invalidToken!==INVALID_TOKEN))||!encoder||!trunk||!router||!Array.isArray(experts)||!experts.length||!validWidths(encoder.widths)||encoder.widths.join(',')!==`${inputs},9`||!validMatrix(encoder.weights,9,inputs)||!validWidths(trunk.widths)||trunk.widths[0]!==9||!Array.isArray(trunk.weights)||trunk.weights.length!==trunk.widths.length-1||!validMatrix(router.weights,experts.length,trunk.widths.at(-1))||experts.some(expert=>!expert||!validWidths(expert.widths)||expert.widths[0]!==trunk.widths.at(-1)||expert.widths.at(-1)!==outputs||!Array.isArray(expert.weights)||expert.weights.length!==2||!validMatrix(expert.weights[0],expert.widths[1],expert.widths[0])||!validMatrix(expert.weights[1],outputs,expert.widths[1]))||trunk.weights.some((weights,index)=>!validMatrix(weights,trunk.widths[index+1],trunk.widths[index])))throw Error('Invalid MoE model shape');
  return copyModel({...data,steps:Number(data.steps)||0});
}
export function* trainingRounds(histories,config,state={model:createModel(config)}){const batch=Number(config.batch||32);while(true){const totals={draw:0,win:0,loss:0,invalid:0};for(let index=0;index<histories.length;index++){const result=trainOne(state.model,histories[index],Number(config.rate||.025));if(result)totals[classifyOutcome(result)]++;if((index+1)%batch===0||index===histories.length-1)yield {model:state.model,round:state.round||0,index:index+1,total:histories.length,totals}}state.round=(state.round||0)+1}}
