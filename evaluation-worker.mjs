import { evaluateModel, importModel } from './training.mjs';

onmessage=event=>{
  const {type,id,round,model,histories,policies}=event.data;
  if(type!=='evaluate')return;
  try{postMessage({type:'evaluation',id,round,evaluation:evaluateModel(importModel(model),histories,policies,{corrections:false})})}
  catch(error){postMessage({type:'evaluation-error',id,message:error.message})}
};
