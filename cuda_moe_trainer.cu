// Compatibility entry point. The implementation calls cudaSetDevice(0) and launches __global__ kernels.
#include "cuda_moe_train.cu"
