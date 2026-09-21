import { trainingRounds, exportModel } from './training.mjs';
let generator=null, playing=false;
const tick=()=>{if(!playing||!generator)return;for(let n=0;n<8;n++){const step=generator.next().value;if(step.index===step.total)postMessage({type:'round',round:step.round,totals:step.totals,model:exportModel(step.model)})}setTimeout(tick,0)};
onmessage=event=>{const {type,histories,config,model}=event.data;if(type==='start'||type==='resume'){if(!generator)generator=trainingRounds(histories,config,model?{model}:undefined);playing=true;postMessage({type:'backend',backend:'CPU Worker (JavaScript)'});tick()}if(type==='pause'){playing=false;postMessage({type:'paused'})}if(type==='reset'){playing=false;generator=null;postMessage({type:'reset'})}};
