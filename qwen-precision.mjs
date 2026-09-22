// Decodes provenance-preserving Palace-9 packed tensor envelopes.
// Payloads remain the exact GGUF/safetensors bytes; this module expands them only in memory.

const QK_K=256;

function base64Bytes(text){
  const binary=atob(text),out=new Uint8Array(binary.length);
  for(let i=0;i<binary.length;i++)out[i]=binary.charCodeAt(i);
  return out;
}

function fp16(view,offset){
  const bits=view.getUint16(offset,true),sign=(bits&0x8000)?-1:1,exponent=(bits>>>10)&31,fraction=bits&1023;
  if(exponent===0)return sign*fraction*2**-24;
  if(exponent===31)return fraction?NaN:sign*Infinity;
  return sign*(1+fraction/1024)*2**(exponent-15);
}

function decodeF32(bytes,count){
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),out=new Float32Array(count);
  for(let i=0;i<count;i++)out[i]=view.getFloat32(i*4,true);
  return out;
}
function decodeF16(bytes,count){
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),out=new Float32Array(count);
  for(let i=0;i<count;i++)out[i]=fp16(view,i*2);
  return out;
}
function scaleMinK4(index,scales){
  if(index<4)return [scales[index]&63,scales[index+4]&63];
  return [(scales[index+4]&15)|((scales[index-4]>>>6)<<4),(scales[index+4]>>>4)|((scales[index]>>>6)<<4)];
}
function decodeQ4K(bytes,count){
  if(count%QK_K)throw Error('Q4_K tensor element count must be divisible by 256');
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),out=new Float32Array(count);
  for(let block=0;block<count/QK_K;block++){
    const base=block*144,d=fp16(view,base),dmin=fp16(view,base+2),scales=bytes.subarray(base+4,base+16),qs=base+16,target=block*QK_K;
    for(let group=0;group<4;group++){
      const [scaleLow,minLow]=scaleMinK4(group*2,scales),[scaleHigh,minHigh]=scaleMinK4(group*2+1,scales);
      for(let lane=0;lane<32;lane++){
        const quant=bytes[qs+group*32+lane];
        out[target+group*64+lane]=d*scaleLow*(quant&15)-dmin*minLow;
        out[target+group*64+32+lane]=d*scaleHigh*(quant>>>4)-dmin*minHigh;
      }
    }
  }
  return out;
}
function decodeQ6K(bytes,count){
  if(count%QK_K)throw Error('Q6_K tensor element count must be divisible by 256');
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),out=new Float32Array(count);
  for(let block=0;block<count/QK_K;block++){
    const base=block*210,d=fp16(view,base+208),scales=base+192,ql=base,qh=base+128,target=block*QK_K;
    for(let half=0;half<2;half++)for(let lane=0;lane<32;lane++){
      const lowA=bytes[ql+half*64+lane],lowB=bytes[ql+half*64+32+lane],high=bytes[qh+half*32+lane],scaleBase=scales+half*8,offset=target+half*128;
      const q1=((lowA&15)|((high&3)<<4))-32,q2=((lowB&15)|(((high>>>2)&3)<<4))-32,q3=((lowA>>>4)|(((high>>>4)&3)<<4))-32,q4=((lowB>>>4)|(((high>>>6)&3)<<4))-32;
      out[offset+lane]=d*view.getInt8(scaleBase+Math.floor(lane/16))*q1;
      out[offset+32+lane]=d*view.getInt8(scaleBase+2+Math.floor(lane/16))*q2;
      out[offset+64+lane]=d*view.getInt8(scaleBase+4+Math.floor(lane/16))*q3;
      out[offset+96+lane]=d*view.getInt8(scaleBase+6+Math.floor(lane/16))*q4;
    }
  }
  return out;
}

function elementCount(shape){return shape.reduce((total,dimension)=>total*dimension,1);}
function decodeSource(source){
  const bytes=base64Bytes(source.bytes),count=elementCount(source.gguf_shape);
  if(source.type==='F32')return decodeF32(bytes,count);
  if(source.type==='F16')return decodeF16(bytes,count);
  if(source.type==='Q4_K')return decodeQ4K(bytes,count);
  if(source.type==='Q6_K')return decodeQ6K(bytes,count);
  throw Error(`Unsupported packed tensor storage: ${source.type}`);
}

export function decodePackedCheckpoint(manifest,config){
  if(manifest?.format!=='palace9-packed-tensors-v1')throw Error('Expected a Palace-9 packed tensor envelope');
  const decoded={},tensors={};
  for(const [name,alias] of Object.entries(manifest.tensors)){
    const raw=decoded[alias.source]||(decoded[alias.source]=decodeSource(manifest.sources[alias.source]));
    const start=alias.offset_elements||0,count=elementCount(alias.shape),values=raw.slice(start,start+count);
    if(values.length!==count)throw Error(`Packed tensor slice out of range: ${name}`);
    tensors[name]={shape:alias.shape,values,storage:manifest.sources[alias.source].type,source:alias.source};
  }
  return {config,precision:manifest.precision,source:manifest.source,claim:manifest.claim,tensors};
}

export async function loadPackedPrecision(manifestUrl,configUrl){
  const [manifestResponse,configResponse]=await Promise.all([fetch(manifestUrl,{cache:'no-store'}),fetch(configUrl,{cache:'no-store'})]);
  if(!manifestResponse.ok)throw Error(`Packed tensor JSON HTTP ${manifestResponse.status}`);
  if(!configResponse.ok)throw Error(`Qwen config HTTP ${configResponse.status}`);
  return decodePackedCheckpoint(await manifestResponse.json(),await configResponse.json());
}

export const packedTensorDecoders={F32:decodeF32,F16:decodeF16,Q4_K:decodeQ4K,Q6_K:decodeQ6K};
