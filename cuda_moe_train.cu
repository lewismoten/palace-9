#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstring>

constexpr int MAGIC=0x504D4F45, MAX_LAYERS=8, MAX_WIDTH=64, MAX_EXPERTS=8, MAX_HIDDEN=64;
struct Header { int magic,n,layers,experts,hidden,params,widths[MAX_LAYERS+1]; };
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(2);} } while(0)
__device__ float squash(float x){return tanhf(x);}
__device__ void softmax(float* a,int n){float m=a[0];for(int i=1;i<n;i++)m=fmaxf(m,a[i]);float s=0;for(int i=0;i<n;i++){a[i]=expf(a[i]-m);s+=a[i];}for(int i=0;i<n;i++)a[i]/=s;}
__device__ int offsets(const Header& h,int& trunk,int& router,int& expert){trunk=9*29;int p=trunk;for(int l=0;l<h.layers;l++)p+=h.widths[l+1]*h.widths[l];router=p;p+=h.experts*h.widths[h.layers];expert=p;return p;}
__global__ void gradients(const float* p,float* g,const float* x,const float* target,int start,int count,Header h){
 int sample=blockIdx.x*blockDim.x+threadIdx.x;if(sample>=count)return; int t0,ro,eo; offsets(h,t0,ro,eo); const float* in=x+(start+sample)*29; const float* y=target+(start+sample)*9;
 float enc[9],acts[MAX_LAYERS+1][MAX_WIDTH],router[MAX_EXPERTS],gates[MAX_EXPERTS],eh[MAX_EXPERTS][MAX_HIDDEN],elog[MAX_EXPERTS][9],logits[9],delta[9],trunkGrad[MAX_WIDTH];
 for(int r=0;r<9;r++){float v=0;for(int c=0;c<29;c++)v+=p[r*29+c]*in[c];enc[r]=squash(v);acts[0][r]=enc[r];}
 int off=t0; for(int l=0;l<h.layers;l++){for(int r=0;r<h.widths[l+1];r++){float v=0;for(int c=0;c<h.widths[l];c++)v+=p[off+r*h.widths[l]+c]*acts[l][c];acts[l+1][r]=squash(v);}off+=h.widths[l+1]*h.widths[l];}
 int last=h.widths[h.layers]; for(int e=0;e<h.experts;e++){float v=0;for(int c=0;c<last;c++)v+=p[ro+e*last+c]*acts[h.layers][c];router[e]=v;gates[e]=v;} softmax(gates,h.experts);
 for(int q=0;q<9;q++)logits[q]=0; int span=h.hidden*last+9*h.hidden;
 for(int e=0;e<h.experts;e++){int base=eo+e*span;for(int j=0;j<h.hidden;j++){float v=0;for(int c=0;c<last;c++)v+=p[base+j*last+c]*acts[h.layers][c];eh[e][j]=squash(v);}for(int q=0;q<9;q++){float v=0;for(int j=0;j<h.hidden;j++)v+=p[base+h.hidden*last+q*h.hidden+j]*eh[e][j];elog[e][q]=v;logits[q]+=gates[e]*v;}}
 softmax(logits,9);for(int q=0;q<9;q++)delta[q]=logits[q]-y[q];for(int c=0;c<last;c++)trunkGrad[c]=0;
 float gd[MAX_EXPERTS],mean=0;for(int e=0;e<h.experts;e++){gd[e]=0;for(int q=0;q<9;q++)gd[e]+=delta[q]*elog[e][q];mean+=gates[e]*gd[e];}
 for(int e=0;e<h.experts;e++){float rd=gates[e]*(gd[e]-mean);for(int c=0;c<last;c++){atomicAdd(g+ro+e*last+c,rd*acts[h.layers][c]);trunkGrad[c]+=p[ro+e*last+c]*rd;}}
 for(int e=0;e<h.experts;e++){int base=eo+e*span;float hd[MAX_HIDDEN];for(int j=0;j<h.hidden;j++)hd[j]=0;for(int q=0;q<9;q++){float od=delta[q]*gates[e];for(int j=0;j<h.hidden;j++){atomicAdd(g+base+h.hidden*last+q*h.hidden+j,od*eh[e][j]);hd[j]+=p[base+h.hidden*last+q*h.hidden+j]*od;}}for(int j=0;j<h.hidden;j++){hd[j]*=(1-eh[e][j]*eh[e][j]);for(int c=0;c<last;c++){atomicAdd(g+base+j*last+c,hd[j]*acts[h.layers][c]);trunkGrad[c]+=p[base+j*last+c]*hd[j];}}}
 off=ro; for(int l=h.layers-1;l>=0;l--){off-=h.widths[l+1]*h.widths[l];float prev[MAX_WIDTH];for(int c=0;c<h.widths[l];c++)prev[c]=0;for(int r=0;r<h.widths[l+1];r++){float d=trunkGrad[r]*(1-acts[l+1][r]*acts[l+1][r]);for(int c=0;c<h.widths[l];c++){atomicAdd(g+off+r*h.widths[l]+c,d*acts[l][c]);prev[c]+=p[off+r*h.widths[l]+c]*d;}}for(int c=0;c<h.widths[l];c++)trunkGrad[c]=prev[c];}
 for(int r=0;r<9;r++){float d=trunkGrad[r]*(1-enc[r]*enc[r]);for(int c=0;c<29;c++)atomicAdd(g+r*29+c,d*in[c]);}
}
__global__ void update(float* p,const float* g,int n,float lr){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]-=lr*g[i];}
int main(int argc,char** argv){if(argc!=6){fprintf(stderr,"usage: cuda_moe_train input output rounds batch rate\n");return 2;}std::ifstream f(argv[1],std::ios::binary);Header h;f.read((char*)&h,sizeof h);if(!f||h.magic!=MAGIC||h.layers<1||h.layers>MAX_LAYERS||h.experts<1||h.experts>MAX_EXPERTS||h.hidden<1||h.hidden>MAX_HIDDEN||h.widths[h.layers]>MAX_WIDTH){fprintf(stderr,"invalid training payload\n");return 2;}std::vector<float> params(h.params),inputs(h.n*29),targets(h.n*9);f.read((char*)params.data(),params.size()*4);f.read((char*)inputs.data(),inputs.size()*4);f.read((char*)targets.data(),targets.size()*4);if(!f){fprintf(stderr,"truncated training payload\n");return 2;}CHECK(cudaSetDevice(0));int device=-1;cudaDeviceProp prop;CHECK(cudaGetDevice(&device));CHECK(cudaGetDeviceProperties(&prop,device));printf("GPU %d: %s\n",device,prop.name);float *dp,*dg,*dx,*dy;CHECK(cudaMalloc(&dp,h.params*4));CHECK(cudaMalloc(&dg,h.params*4));CHECK(cudaMalloc(&dx,inputs.size()*4));CHECK(cudaMalloc(&dy,targets.size()*4));CHECK(cudaMemcpy(dp,params.data(),h.params*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dx,inputs.data(),inputs.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dy,targets.data(),targets.size()*4,cudaMemcpyHostToDevice));int rounds=atoi(argv[3]),batch=atoi(argv[4]);float rate=atof(argv[5]);for(int r=0;r<rounds;r++)for(int start=0;start<h.n;start+=batch){int count=(start+batch<=h.n?batch:h.n-start);CHECK(cudaMemset(dg,0,h.params*4));gradients<<<(count+127)/128,128>>>(dp,dg,dx,dy,start,count,h);CHECK(cudaGetLastError());update<<<(h.params+255)/256,256>>>(dp,dg,h.params,rate/count);CHECK(cudaGetLastError());}CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(params.data(),dp,h.params*4,cudaMemcpyDeviceToHost));CHECK(cudaFree(dp));CHECK(cudaFree(dg));CHECK(cudaFree(dx));CHECK(cudaFree(dy));std::ofstream o(argv[2],std::ios::binary);o.write((char*)&h,sizeof h);o.write((char*)params.data(),params.size()*4);return o?0:2;}
