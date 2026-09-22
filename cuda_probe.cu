#include <cstdio>
#include <cuda_runtime.h>
__global__ void square(const float* input,float* output){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<1024)output[i]=input[i]*input[i];}
int main(){cudaSetDevice(0);float *input,*output;cudaMallocManaged(&input,1024*sizeof(float));cudaMallocManaged(&output,1024*sizeof(float));for(int i=0;i<1024;i++)input[i]=float(i);square<<<4,256>>>(input,output);auto error=cudaDeviceSynchronize();if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));return 1;}std::printf("gpu=0 result=%.0f\n",output[31]);cudaFree(input);cudaFree(output);}
