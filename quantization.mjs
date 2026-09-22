import { evaluateModel, predict } from './training.mjs';

const clone=value=>JSON.parse(JSON.stringify(value));

function float16(value){
  const source=new Float32Array(1),bits=new Uint32Array(source.buffer);source[0]=value;
  const word=bits[0],sign=(word>>>16)&0x8000,exponent=(word>>>23)&0xff,mantissa=word&0x7fffff;
  let half;
  if(exponent===0xff)half=sign|(mantissa?0x7e00:0x7c00);
  else {const adjusted=exponent-127+15;if(adjusted>=31)half=sign|0x7c00;else if(adjusted<=0){if(adjusted<-10)half=sign;else half=sign|((mantissa|0x800000)>>(14-adjusted));}else half=sign|(adjusted<<10)|((mantissa+0x1000)>>13)}
  const halfExponent=(half>>>10)&0x1f,halfMantissa=half&0x3ff;let decoded;
  if(halfExponent===0)decoded=halfMantissa?Math.pow(2,-14)*(halfMantissa/1024):0;
  else if(halfExponent===31)decoded=halfMantissa?NaN:Infinity;
  else decoded=Math.pow(2,halfExponent-15)*(1+halfMantissa/1024);
  return half&0x8000?-decoded:decoded;
}

export function quantizeMatrix(matrix,format='float32'){
  if(format==='float32')return matrix.map(row=>[...row]);
  if(format==='float16')return matrix.map(row=>row.map(float16));
  const bits=format==='int8'?8:format==='int4'||format==='int4-row'?4:format==='int2'?2:format==='int1'?1:0;if(!bits)throw Error(`Unsupported quantization format: ${format}`);
  if(format==='int4-row'){const range=7;return matrix.map(row=>{const max=Math.max(...row.map(Math.abs));if(!max)return row.map(()=>0);const scale=max/range;return row.map(value=>Math.max(-range,Math.min(range,Math.round(value/scale)))*scale)})}
  const max=Math.max(...matrix.flat().map(Math.abs));if(!max)return matrix.map(row=>row.map(()=>0));
  if(bits===1)return matrix.map(row=>row.map(value=>value<0?-max:max));
  const range=(1<<(bits-1))-1,scale=max/range;
  return matrix.map(row=>row.map(value=>Math.max(-range,Math.min(range,Math.round(value/scale)))*scale));
}

export function quantizeModel(model,format='float32'){
  const result=clone(model),q=matrix=>quantizeMatrix(matrix,format);
  result.encoder.weights=q(result.encoder.weights);
  result.trunk.weights=result.trunk.weights.map(q);
  result.router.weights=q(result.router.weights);
  result.experts=result.experts.map(expert=>({...expert,weights:expert.weights.map(q)}));
  result.quantization={format,method:format==='int1'?'per-tensor binary sign':format==='int4-row'?'symmetric per-output-row round/dequantize':'symmetric per-tensor round/dequantize',storageBits:format==='float32'?32:format==='float16'?16:format==='int4-row'?4:Number(format.slice(3))};
  return result;
}

const topChoice=(model,history)=>{const probabilities=predict(model,history).probabilities;return probabilities.indexOf(Math.max(...probabilities))};
const parameterCount=model=>model.encoder.weights.flat().length+model.trunk.weights.flat(2).length+model.router.weights.flat().length+model.experts.reduce((total,expert)=>total+expert.weights.flat(2).length,0);
const bytesFor=(parameters,format)=>parameters*(format==='float32'?4:format==='float16'?2:format==='int4-row'?.5:Number(format.slice(3))/8);

export const quantizationName=format=>({float32:'FP32 · F32',float16:'FP16 · F16',int8:'INT8 · Q8-style',int4:'INT4 · Q4-style', 'int4-row':'INT4 · per-output-row',int2:'INT2 · Q2-style',int1:'INT1 · binary sign'})[format]||format;

export function quantizationSuite(baseline,histories,policies){
  return ['float32','float16','int8','int4','int4-row','int2','int1'].map(format=>quantizationComparison(baseline,quantizeModel(baseline,format),histories,policies));
}

export function quantizationComparison(baseline,variant,histories,policies){
  const baselineTopChoices=histories.map(history=>topChoice(baseline,history));
  const variantTopChoices=histories.map(history=>topChoice(variant,history));
  const sameTop1=variantTopChoices.filter((choice,index)=>choice===baselineTopChoices[index]).length;
  const format=variant.quantization?.format||'float32',parameters=parameterCount(baseline),report=evaluateModel(variant,histories,policies,{corrections:false});
  return {format,method:variant.quantization?.method||'unquantized FP32',corpus:histories.length,parameters,theoreticalBytes:bytesFor(parameters,format),sameTop1,sameTop1Percent:100*sameTop1/histories.length,losses:report.totals.loss,occupiedSquareInvalid:report.totals.invalid,wins:report.totals.win,draws:report.totals.draw,baselineTopChoices,variantTopChoices};
}