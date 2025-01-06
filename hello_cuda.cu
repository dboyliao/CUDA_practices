#include "cuda_runtime.h"

#include <stdio.h>

__global__ void cuda_hello()
{
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        printf("Hello World from GPU!\n");
    }
}

int main()
{
    cuda_hello<<<1, 1>>>();
    ::cudaDeviceSynchronize(); // Wait for the GPU launched work to complete
    return 0;
}