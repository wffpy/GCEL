#include <cuda_runtime.h>
#include "gpu/gpu.h"
namespace gpu_kernel{
// CUDA kernel function to add elements of two arrays
__global__ void add(float *a, float *b, float *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}
}   // namespace gpu_kernel


namespace gpu {
int add(float *lhs, float *rhs, float *ret, int length) {
    int size = length * sizeof(float);
    // Device arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, lhs, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, rhs, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;
    gpu_kernel::add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, length);

    // Copy result from device to host
    cudaMemcpy(ret, d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

}   // namespace gpu