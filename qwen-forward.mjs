function linear(tensor,bias,input){const [rows,columns]=tensor.shape,out=new Float32Array(rows);for(let row=0;row<rows;row++){let sum=bias?.values?.[row]||0,offset=row*columns;for(let column=0;column<columns;column++)sum+=tensor.values[offset+column]*input[column];out[row]=sum;}return out;}
function rmsNorm(input,weight,epsilon){let mean=0;for(const value of input)mean+=value*value;const scale=1/Math.sqrt(mean/input.length+epsilon),out=new Float32Array(input.length);for(let i=0;i<input.length;i++)out[i]=input[i]*scale*weight.values[i];return out;}
function softmax(values){const max=Math.max(...values),exps=values.map(value=>Math.exp(value-max)),sum=exps.reduce((total,value)=>total+value,0);return exps.map(value=>value/sum);}
function silu(value){return value/(1+Math.exp(-value));}
function rope(vector,position,theta,headDim=4){const out=new Float32Array(vector);for(let base=0;base<vector.length;base+=headDim){const half=headDim/2;for(let i=0;i<half;i++){const angle=position/Math.pow(theta,2*i/headDim),cos=Math.cos(angle),sin=Math.sin(angle),left=vector[base+i],right=vector[base+i+half];out[base+i]=left*cos-right*sin;out[base+i+half]=right*cos+left*sin;}}return out;}
function topTwo(values){return [...values.keys()].sort((a,b)=>values[b]-values[a]).slice(0,2);}
function add(...vectors){const out=new Float32Array(vectors[0].length);for(const vector of vectors)for(let i=0;i<out.length;i++)out[i]+=vector[i];return out;}
function expertOutput(tensors,prefix,input){const gate=linear(tensors[`${prefix}.gate_proj.weight`],null,input),up=linear(tensors[`${prefix}.up_proj.weight`],null,input),hidden=new Float32Array(gate.length);for(let i=0;i<hidden.length;i++)hidden[i]=silu(gate[i])*up[i];return linear(tensors[`${prefix}.down_proj.weight`],null,hidden);}

export function runQwenForward(checkpoint,tokenIds){
  const {config,shape,tensors}=checkpoint,{hidden,qHeads,kvHeads,headDim}=shape;
  if(!tokenIds.length)throw Error('Qwen forward requires the trained <bos> token');
  const embeddings=tensors['model.embed_tokens.weight'],inputRows=tokenIds.map(id=>embeddings.values.slice(id*hidden,(id+1)*hidden));
  const input=inputRows.at(-1),normalized=rmsNorm(input,tensors['model.layers.0.input_layernorm.weight'],config.rms_norm_eps);
  const q=rope(linear(tensors['model.layers.0.self_attn.q_proj.weight'],tensors['model.layers.0.self_attn.q_proj.bias'],normalized),tokenIds.length-1,config.rope_theta);
  const keys=[],values=[];
  for(const [position,row] of inputRows.entries()){
    const norm=rmsNorm(row,tensors['model.layers.0.input_layernorm.weight'],config.rms_norm_eps);
    keys.push(rope(linear(tensors['model.layers.0.self_attn.k_proj.weight'],tensors['model.layers.0.self_attn.k_proj.bias'],norm),position,config.rope_theta));
    values.push(linear(tensors['model.layers.0.self_attn.v_proj.weight'],tensors['model.layers.0.self_attn.v_proj.bias'],norm));
  }
  const groups=qHeads/kvHeads,attentionByHead=[],context=new Float32Array(hidden);
  for(let head=0;head<qHeads;head++){
    const keyHead=Math.floor(head/groups),scores=[];for(let position=0;position<tokenIds.length;position++){let score=0;for(let dim=0;dim<headDim;dim++)score+=q[head*headDim+dim]*keys[position][keyHead*headDim+dim];scores.push(score/Math.sqrt(headDim));}
    const weights=softmax(scores);attentionByHead.push(weights);for(let position=0;position<tokenIds.length;position++)for(let dim=0;dim<headDim;dim++)context[head*headDim+dim]+=weights[position]*values[position][keyHead*headDim+dim];
  }
  const attention=linear(tensors['model.layers.0.self_attn.o_proj.weight'],null,context),residual=add(input,attention),postNorm=rmsNorm(residual,tensors['model.layers.0.post_attention_layernorm.weight'],config.rms_norm_eps);
  const routerLogits=linear(tensors['model.layers.0.mlp.gate.weight'],null,postNorm),routerProbabilities=softmax([...routerLogits]),selectedExperts=topTwo(routerProbabilities),routed=new Float32Array(hidden),expertContributions=Array(shape.experts).fill(null);
  for(const expert of selectedExperts){const output=expertOutput(tensors,`model.layers.0.mlp.experts.${expert}`,postNorm),weight=routerProbabilities[expert],contribution=new Float32Array(hidden);for(let i=0;i<hidden;i++){contribution[i]=weight*output[i];routed[i]+=contribution[i];}expertContributions[expert]=contribution;}
  const shared=expertOutput(tensors,'model.layers.0.mlp.shared_expert',postNorm),sharedGate=linear(tensors['model.layers.0.mlp.shared_expert_gate.weight'],null,postNorm);for(let i=0;i<hidden;i++)shared[i]*=1/(1+Math.exp(-sharedGate[0]));
  const finalHidden=add(residual,routed,shared),finalNorm=rmsNorm(finalHidden,tensors['model.norm.weight'],config.rms_norm_eps),logits=linear(tensors['lm_head.weight'],null,finalNorm);
  return {input,normalized,q,keys,values,attentionByHead,context,attention,residual,postNorm,routerLogits,routerProbabilities,selectedExperts,expertContributions,routed,shared,finalHidden,finalNorm,logits};
}

export function namedMoveLogits(logits){return [...'abcdefghi!'].map(token=>{const id=token==='!'?4:68+token.charCodeAt(0)-97;return {token,id,logit:logits[id]};}).sort((a,b)=>b.logit-a.logit);}
