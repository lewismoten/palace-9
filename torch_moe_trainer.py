#!/usr/bin/env python3
import argparse,json,random,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

SQUARES='abcdefghi'
SENTINEL='!'
USE_DEFAULT_QUANTIZATION=object()
# Each permutation maps a source square index to its transformed destination.
D4_PERMUTATIONS=((0,1,2,3,4,5,6,7,8),(2,1,0,5,4,3,8,7,6),(8,7,6,5,4,3,2,1,0),(6,7,8,3,4,5,0,1,2),(0,3,6,1,4,7,2,5,8),(6,3,0,7,4,1,8,5,2),(8,5,2,7,4,1,6,3,0),(2,5,8,1,4,7,0,3,6))
LINES=((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,4,6),(0,4,8),(2,4,6))

def transform_example(example,permutation):
 def transform(values,width):
  output=[0]*len(values)
  for source,destination in enumerate(permutation):output[destination*width:destination*width+width]=values[source*width:source*width+width]
  return output
 return {**example,'features':transform(example['features'][:27],3)+example['features'][27:],'target':transform(example['target'][:9],1)+example['target'][9:]}

def inverse_permute_logits(values,permutation):
 output=list(values)
 for source,destination in enumerate(permutation):output[source]=values[destination]
 return output

def inverse_permute_logits_tensor(logits,permutations):
 restored=logits.clone()
 restored[:,:9]=logits[:,:9].gather(1,permutations)
 return restored

def canonicalize_example(example):
 candidates=[(tuple(transform_example(example,permutation)['features']),index,permutation) for index,permutation in enumerate(D4_PERMUTATIONS)]
 _,index,permutation=min(candidates)
 canonical=transform_example(example,permutation)
 return {**canonical,'transformIndex':index,'permutation':permutation}

def canonicalize_unique(examples):
 unique={}
 for example in examples:
  canonical=canonicalize_example(example);key=tuple(canonical['features'])
  if key in unique and unique[key]['target']!=canonical['target']:raise ValueError('canonical board state has conflicting policy target')
  unique[key]=canonical
 return list(unique.values())

def winner_history(history):
 board=[None]*9
 for index,token in enumerate(history):board[SQUARES.index(token)]='X' if index%2==0 else 'O'
 return any(board[a] and board[a]==board[b]==board[c] for a,b,c in LINES)

def invalid_history_examples(rows):
 examples=[]
 for row in rows:
  history=row['history'];features=row['features']
  if history:examples.append({'history':history+history[0],'features':features+[1,1],'target':[0]*9+[1],'category':'repeated-square'})
  examples.append({'history':history+'!','features':features+[1,0],'target':[0]*9+[1],'category':'non-square-token'})
  for square in SQUARES:
   if square in history:continue
   terminal=history+square
   if winner_history(terminal):
    continuation=next((candidate for candidate in SQUARES if candidate not in terminal),None)
    if continuation:examples.append({'history':terminal+continuation,'features':features+[1,0],'target':[0]*9+[1],'category':'post-terminal'})
    break
 return examples

def occupied_margin_loss(logits,legal,margin):
 if margin<=0:return logits.new_zeros(())
 best_legal=logits.masked_fill(~legal,float('-inf')).max(1).values
 worst_occupied=logits.masked_fill(legal,float('-inf')).max(1).values
 has_occupied=(~legal).any(1)
 return (torch.relu(margin+worst_occupied-best_legal)*has_occupied).sum()/has_occupied.sum().clamp_min(1)

def unsupported_policy_margin_loss(logits,probabilities,legal,margin):
 if margin<=0:return logits.new_zeros(())
 supported=(probabilities>0)&legal;unsupported=(probabilities==0)&legal
 best_supported=logits.masked_fill(~supported,float('-inf')).max(1).values
 worst_unsupported=logits.masked_fill(~unsupported,float('-inf')).max(1).values
 has_unsupported=unsupported.any(1);has_supported=supported.any(1)
 active=has_unsupported&has_supported
 return (torch.relu(margin+worst_unsupported-best_supported)*active).sum()/active.sum().clamp_min(1)

def fake_quantize_ste(weight,format,granularity='tensor'):
 if not format:return weight
 if format=='float16':
  quantized=weight.to(torch.float16).to(torch.float32)
  return weight+(quantized-weight).detach()
 bits=int(format[3:]) if isinstance(format,str) and format.startswith('int') else int(format);maximum=weight.detach().abs().amax(dim=1,keepdim=True) if granularity=='row' else weight.detach().abs().max()
 levels=(1<<(bits-1))-1;scale=maximum/levels
 scale=torch.where(scale==0,torch.ones_like(scale),scale)
 quantized=(weight/scale).round().clamp(-levels,levels)*scale
 return weight+(quantized-weight).detach()

def fake_quant_bits(format):
 return {'float16':16,'int8':8,'int4':4,'int2':2}.get(format,0)

def fake_quant_method(format,granularity):
 return 'IEEE-754 binary16 STE round/dequantize' if format=='float16' else f'symmetric per-{granularity} STE round/dequantize'

def synchronized_precision_formats():
 return (None,'float16','int8','int4')

def precision_granularity(format,default='tensor'):
 return 'row' if format=='int4' else default

class MoE(nn.Module):
 def __init__(self,hidden=16,expert_hidden=16,experts=8,inputs=29,outputs=9,fake_quant=0,fake_quant_granularity='tensor'):
  super().__init__();self.fake_quant=fake_quant;self.fake_quant_granularity=fake_quant_granularity;self.encoder=nn.Linear(inputs,9,bias=False);self.trunk=nn.ModuleList([nn.Linear(9,hidden,bias=False),nn.Linear(hidden,hidden,bias=False)]);self.router=nn.Linear(hidden,experts,bias=False);self.experts=nn.ModuleList([nn.Sequential(nn.Linear(hidden,expert_hidden,bias=False),nn.Tanh(),nn.Linear(expert_hidden,outputs,bias=False)) for _ in range(experts)])
 def linear(self,layer,x,format=USE_DEFAULT_QUANTIZATION):
  selected=self.fake_quant if format is USE_DEFAULT_QUANTIZATION else format
  return F.linear(x,fake_quantize_ste(layer.weight,selected,precision_granularity(selected,self.fake_quant_granularity)))
 def forward(self,x,format=USE_DEFAULT_QUANTIZATION):
  x=torch.tanh(self.linear(self.encoder,x,format))
  for layer in self.trunk:x=torch.tanh(self.linear(layer,x,format))
  gates=F.softmax(self.linear(self.router,x,format),dim=-1);logits=torch.stack([self.linear(expert[2],torch.tanh(self.linear(expert[0],x,format)),format) for expert in self.experts],dim=1)
  return (gates.unsqueeze(-1)*logits).sum(1),gates

def browser_model(model,steps):
 def matrix(layer):return layer.weight.detach().cpu().float().tolist()
 outputs=model.experts[0][2].out_features;inputs=model.encoder.in_features
 data={'format':f'palace-9-moe/v{3 if inputs==31 else 2 if outputs==10 else 1}','modelName':'palace-9','expansion':'Predictive Adversarial Learning and Contingency Evaluator','squares':9,'architecture':'board-state-moe','encoder':{'widths':[inputs,9],'weights':matrix(model.encoder)},'trunk':{'widths':[9,model.trunk[0].out_features,model.trunk[1].out_features],'weights':[matrix(x) for x in model.trunk]},'router':{'weights':matrix(model.router)},'experts':[{'widths':[model.trunk[-1].out_features,e[0].out_features,outputs],'weights':[matrix(e[0]),matrix(e[2])]} for e in model.experts],'steps':steps}
 if outputs==10:data['outputTokens']=list(SQUARES+SENTINEL);data['invalidToken']=SENTINEL
 return data

def metrics(logits,examples,has_sentinel=False):
 choice=logits.argmax(1).cpu().tolist();t={'draw':0,'win':0,'loss':0,'invalid':0}
 for i,row in enumerate(examples):
  p=row['policy'];pick=choice[i]
  if has_sentinel and pick==9:t['invalid']+=1
  elif p['probabilities'][pick]>0:t['win' if p['best']==10 else 'draw']+=1
  elif p['scores'][pick]==-10:t['loss']+=1
  else:t['invalid']+=1
 return t

def invalid_metrics(logits):
 choices=logits.argmax(1);correct=int((choices==9).sum().item());total=len(choices)
 return {'correct':correct,'incorrect':total-correct,'total':total}

def copy_overlap(layer,weights):
 tensor=torch.tensor(weights,dtype=layer.weight.dtype,device=layer.weight.device)
 rows=min(tensor.shape[0],layer.weight.shape[0]);cols=min(tensor.shape[1],layer.weight.shape[1])
 with torch.no_grad():layer.weight.zero_();layer.weight[:rows,:cols].copy_(tensor[:rows,:cols])
 return tuple(tensor.shape)!=tuple(layer.weight.shape)

def load_browser_model(model,path):
 data=json.loads(Path(path).read_text())
 if data.get('format') not in ('palace-9-moe/v1','palace-9-moe/v2','palace-9-moe/v3') or data.get('architecture')!='board-state-moe':raise ValueError('resume checkpoint must be a palace-9 board-state MoE')
 migrated=copy_overlap(model.encoder,data['encoder']['weights'])
 for layer,weights in zip(model.trunk,data['trunk']['weights']):migrated=copy_overlap(layer,weights) or migrated
 migrated=copy_overlap(model.router,data['router']['weights']) or migrated
 if len(data['experts'])!=len(model.experts):raise ValueError('resume expert count mismatch')
 for expert,stored in zip(model.experts,data['experts']):
  migrated=copy_overlap(expert[0],stored['weights'][0]) or migrated;migrated=copy_overlap(expert[2],stored['weights'][1]) or migrated
 annotations=dict(data.get('annotations',{}));annotations['modelMigrationApplied']=migrated
 return annotations

def optimizer_sidecar_path(snapshot_path):return Path(snapshot_path).with_suffix('.optimizer.pt')
def save_optimizer_state(optimizer,snapshot_path):
 target=optimizer_sidecar_path(snapshot_path);temporary=target.with_suffix('.optimizer.pt.tmp');torch.save(optimizer.state_dict(),temporary);temporary.replace(target)
def load_optimizer_state(optimizer,snapshot_path,device):
 sidecar=optimizer_sidecar_path(snapshot_path)
 if not sidecar.exists():return False
 try:optimizer.load_state_dict(torch.load(sidecar,map_location=device,weights_only=True));return True
 except (RuntimeError,ValueError):return False

def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=1000,help='0 runs until the exact frozen stop condition');p.add_argument('--batch-size',type=int,default=256);p.add_argument('--hidden-nodes',type=int,default=16);p.add_argument('--expert-nodes',type=int,default=16);p.add_argument('--experts',type=int,default=8);p.add_argument('--rate',type=float,default=.01);p.add_argument('--balance',type=float,default=.01);p.add_argument('--occupied-margin',type=float,default=0,help='margin by which the best legal logit must exceed occupied logits');p.add_argument('--unsupported-margin',type=float,default=0,help='margin by which supported legal moves must exceed legal zero-probability moves');p.add_argument('--symmetry',action='store_true',help='augment valid examples with all D4 board symmetries');p.add_argument('--canonicalize',action='store_true',help='canonicalize D4 board orientation and inverse-map output logits');p.add_argument('--output',default='torch-snapshots');p.add_argument('--seed',type=int,default=1983);p.add_argument('--snapshot-debounce-seconds',type=float,default=5.0);p.add_argument('--resume');p.add_argument('--fresh-optimizer',action='store_true',help='resume weights but intentionally reset optimizer state');p.add_argument('--qat-format',choices=('float16','int8','int4','int2'),help='fake-quantize every learned tensor with STE during training');p.add_argument('--qat-granularity',choices=('tensor','row'),default='tensor',help='fake-quantization scale per tensor or per linear output row');p.add_argument('--synchronize-precisions',action='store_true',help='jointly optimize FP32, FP16, INT8, and row-INT4 paths');p.add_argument('--sentinel',action='store_true',help='train ! as the invalid-history output class');a=p.parse_args()
 torch.manual_seed(a.seed);random.seed(a.seed);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');assert device.type=='cuda','CUDA GPU 0 is required'
 if a.canonicalize and a.symmetry:raise ValueError('--canonicalize already folds D4-equivalent states; do not also use --symmetry')
 rows=json.loads(Path('data/reachable-policy.json').read_text())['examples'];has_sentinel=a.sentinel;inputs=31 if has_sentinel else 29;outputs=10 if has_sentinel else 9
 valid_examples=[{'history':r['history'],'features':r['features']+([0,0] if has_sentinel else []),'target':r['policy']['probabilities']+([0] if has_sentinel else [])} for r in rows]
 evaluation_examples=[canonicalize_example(example) for example in valid_examples] if a.canonicalize else valid_examples
 training_examples=canonicalize_unique(valid_examples) if a.canonicalize else [transform_example(example,permutation) for example in valid_examples for permutation in D4_PERMUTATIONS] if a.symmetry else valid_examples
 invalid_examples=invalid_history_examples(rows) if has_sentinel else []
 if a.canonicalize:invalid_examples=canonicalize_unique(invalid_examples)
 x=torch.tensor([example['features'] for example in evaluation_examples],dtype=torch.float32,device=device);permutations=torch.tensor([example.get('permutation',D4_PERMUTATIONS[0]) for example in evaluation_examples],dtype=torch.long,device=device)
 tx=torch.tensor([example['features'] for example in training_examples],dtype=torch.float32,device=device);ty=torch.tensor([example['target'] for example in training_examples],dtype=torch.float32,device=device)
 legal=tx[:,2:27:3].bool()
 ix=torch.tensor([example['features'] for example in invalid_examples],dtype=torch.float32,device=device) if has_sentinel else None;iy=torch.tensor([example['target'] for example in invalid_examples],dtype=torch.float32,device=device) if has_sentinel else None
 model=MoE(a.hidden_nodes,a.expert_nodes,a.experts,inputs,outputs,a.qat_format,a.qat_granularity).to(device);resume_annotations=load_browser_model(model,a.resume) if a.resume else {};starting_round=int(resume_annotations.get('generatedRound',resume_annotations.get('rounds',0)));opt=torch.optim.AdamW(model.parameters(),lr=a.rate,weight_decay=1e-4);optimizer_state_restored=load_optimizer_state(opt,a.resume,device) if a.resume and not a.fresh_optimizer and not resume_annotations.get('modelMigrationApplied') else False;out=Path(a.output);out.mkdir(parents=True,exist_ok=True);saved=[];started=time.time();first_snapshot_at=None;first_snapshot_round=None;last_snapshot_at=None;suppressed_improvements=0;best=None;epoch=0;formats=synchronized_precision_formats() if a.synchronize_precisions else (USE_DEFAULT_QUANTIZATION,)
 while a.epochs==0 or epoch<a.epochs:
  epoch+=1;round_num=starting_round+epoch;order=torch.randperm(len(training_examples),device=device);model.train()
  for start in range(0,len(training_examples),a.batch_size):
   ids=order[start:start+a.batch_size];losses=[]
   for format in formats:
    logits,gates=model(tx[ids],format);term=-(ty[ids]*F.log_softmax(logits,dim=1)).sum(1).mean()+occupied_margin_loss(logits[:,:9],legal[ids],a.occupied_margin)+unsupported_policy_margin_loss(logits[:,:9],ty[ids,:9],legal[ids],a.unsupported_margin)
    if has_sentinel:
     sampled=torch.randint(len(ix),(len(ids),),device=device);invalid_logits,_=model(ix[sampled],format);term=term+F.cross_entropy(invalid_logits,iy[sampled].argmax(1))
    losses.append(term)
   loss=sum(losses)/len(losses);balance=((gates.mean(0)-1/a.experts)**2).mean();opt.zero_grad();(loss+a.balance*balance).backward();opt.step()
  model.eval()
  with torch.no_grad():
   logits,gates=model(x);logits=inverse_permute_logits_tensor(logits,permutations) if a.canonicalize else logits;t=metrics(logits,rows,has_sentinel);it=invalid_metrics(model(ix)[0]) if has_sentinel else {'correct':0,'incorrect':0,'total':0}
  score=t['loss']+t['invalid']+it['incorrect'];candidate=(score,t['loss'],t['invalid'],it['incorrect']);improved=best is None or candidate<best
  if improved:
   best=candidate;now=time.time();terminal=not t['loss'] and not t['invalid'] and not it['incorrect'];eligible=last_snapshot_at is None or now-last_snapshot_at>=a.snapshot_debounce_seconds or terminal
   if eligible:
    if first_snapshot_at is None:first_snapshot_at=now;first_snapshot_round=round_num
    elapsed=now-first_snapshot_at;rounds_since_first=round_num-first_snapshot_round;average=elapsed/rounds_since_first if rounds_since_first else 0.0;snapshot=browser_model(model,round_num*len(rows));snapshot['canonicalization']=({'group':'D4','selection':'lexicographically smallest transformed 27-feature board; lowest D4 index breaks ties','inputCoordinates':'canonical','outputCoordinates':'inverse-permuted to original'} if a.canonicalize else None);snapshot['annotations']={'rounds':round_num,'generatedRound':round_num,'losses':t['loss'],'invalid':t['invalid'],'wins':t['win'],'draws':t['draw'],'invalidHistoryCorrect':it['correct'],'invalidHistoryIncorrect':it['incorrect'],'invalidHistoryTotal':it['total'],'frozenMetrics':{k:100*v/len(rows) for k,v in t.items()},'gpu':'GPU 0','trainer':'torch_moe_trainer.py','resumedFrom':a.resume,'startingRound':starting_round,'modelMigrationApplied':bool(resume_annotations.get('modelMigrationApplied')),'optimizerStateRestored':optimizer_state_restored,'hiddenNodes':a.hidden_nodes,'expertNodes':a.expert_nodes,'experts':a.experts,'fakeQuant':({'format':a.qat_format,'method':fake_quant_method(a.qat_format,a.qat_granularity),'granularity':a.qat_granularity,'storageBits':fake_quant_bits(a.qat_format)} if a.qat_format else None),'symmetryAugmentation':a.symmetry,'trainingPositions':len(training_examples),'occupiedMargin':a.occupied_margin,'unsupportedPolicyMargin':a.unsupported_margin,'invalidHistoryCategories':sorted({example['category'] for example in invalid_examples}),'balanceLoss':float(((gates.mean(0)-1/a.experts)**2).mean().cpu()),'snapshotDebounceSeconds':a.snapshot_debounce_seconds,'elapsedSecondsSinceFirstSnapshot':elapsed,'averageSecondsPerRoundSinceFirstSnapshot':average};file=out/f'palace-9-round-{round_num:06d}.json';file.write_text(json.dumps(snapshot));save_optimizer_state(opt,file);saved.append(str(file));last_snapshot_at=now
   else:suppressed_improvements+=1
  if not t['loss'] and not t['invalid'] and not it['incorrect']:break
 torch.cuda.synchronize();print(json.dumps({'device':str(device),'resumedFrom':a.resume,'optimizerStateRestored':optimizer_state_restored,'startingRound':starting_round,'epochsCompleted':epoch,'finalRound':round_num,'seconds':round(time.time()-started,3),'best':best,'invalidHistoryTotal':len(invalid_examples),'symmetryAugmentation':a.symmetry,'trainingPositions':len(training_examples),'occupiedMargin':a.occupied_margin,'unsupportedPolicyMargin':a.unsupported_margin,'snapshots':saved,'suppressedImprovements':suppressed_improvements}))
if __name__=='__main__':main()
