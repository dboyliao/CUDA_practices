#include <cuda_runtime_api.h>
#include <stdio.h>

int main(int argc, char const *argv[])
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Device count: %d\n", deviceCount);
    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("Device %d: %s\n", i, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
        printf("  Memory bus width: %d bits\n", deviceProp.memoryBusWidth);
        printf("  Peak memory bandwidth: %f GB/s\n", 2.0 * deviceProp.memoryClockRate * deviceProp.memoryBusWidth / 8 / 1e6);
        printf("  Total global memory: %lu bytes\n", deviceProp.totalGlobalMem);
        printf("  Shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Warp size: %d\n", deviceProp.warpSize);
        // max active blocks per SM, which are the limit of blocks that can be executed simultaneously on an SM
        printf("  Max blocks per multiprocessor: %d\n", deviceProp.maxBlocksPerMultiProcessor);
        printf("  Max warps per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize);
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Max registers per block: %d\n", deviceProp.regsPerBlock);
        printf("  Max threads dimensions: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max grid size: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Clock rate: %d kHz\n", deviceProp.clockRate);
        printf("  Total constant memory: %lu bytes\n", deviceProp.totalConstMem);
        // max block per SM
    }
    return 0;
}
