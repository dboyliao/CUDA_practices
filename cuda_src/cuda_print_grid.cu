#include <cuda_runtime.h>
#include <stdio.h>

__global__ void print_grid()
{
    printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, blockDim.x: %d, blockDim.y: %d, blockDim.z: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n",
           blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(int argc, char const *argv[])
{
    print_grid<<<{5, 1, 1}, {1, 1, 5}>>>();
    cudaDeviceSynchronize();
    return 0;
}
