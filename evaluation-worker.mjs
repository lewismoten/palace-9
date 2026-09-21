import { evaluateModel, importModel } from './training.mjs';

onmessage=event=>{
  const {type,id,round,model,histories}=event.data;
  if(type!=='evaluate')return;
  try{postMessage({type:'evaluation',id,round,evaluation:evaluateModel(importModel(model),histories)})}
  catch(error){postMessage({type:'evaluation-error',id,message:error.message})}
};
