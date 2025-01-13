#include <cuda_runtime.h>
#include <stdio.h>

__global__ void print_grid()
{
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        // gridDim
        printf("gridDim.x: %d, gridDim.y: %d, gridDim.z: %d\n", gridDim.x, gridDim.y, gridDim.z);
        // blockDim
        printf("blockDim.x: %d, blockDim.y: %d, blockDim.z: %d\n", blockDim.x, blockDim.y, blockDim.z);
        // blockIdx
        printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
        // threadIdx
        printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    }
}

int main(int argc, char const *argv[])
{
    // the first list initialization is for gridDim
    // the second list initialization is for blockDim
    print_grid<<<{2, 3, 1}, {1, 1, 5}>>>();
    cudaDeviceSynchronize();
    return 0;
}
