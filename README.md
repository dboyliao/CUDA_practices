# Build
I use CMake to build the project. Please make sure you have installed CMake and CUDA Toolkit.
I also use `uv` to manage the Python dependencies and virtual environment. Please make sure you have installed it as well.

The CMake file is configured to build against the Python virtual environment that is created by `uv`. You have to setup the virtual environment before building the project.

If you don't want to use `uv`, you have to modify the CMake file to compile against the proper Python runtime.

## Python Dependencies
```bash
$ uv venv
$ uv sync --dev
```

## CMake Build
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```

# CUDA C++ Programming
- `<<<...>>>`: a kernel launch configuration
   - `<<< num_blocks, num_threads_per_block, shared_memory_size >>>`
   - `shared_memory_size` is optional
- Dimension of the grid and block:
    - `gridDim`: the number of blocks in the grid
        - use `blockIdx` to access the block index in the grid
    - `blockDim`: the number of threads per block
        - use `threadIdx` to access the thread index in the block
- a warp is a group of 32 threads that are executed in lockstep
- `__global__`: a function that runs on the device and can be called from the host
- `__device__`: a function that runs on the device and can be called from the device
- Application devided into blocks, blocks devided into wraps. Wraps will be mapped to SMs.
    - wraps are controled by the wrap scheduler
    - because each SM has fixed number of paritions. If the number of wraps is greater than the number of partitions, the wraps will be queued and scheduled by the wrap scheduler.
- Share memory is faster than global memory
    - `__syncthreads()`: a barrier that synchronizes threads in the same block
    - `__shared__`: a variable that is shared among threads in the same block
    - https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
- Error Handling
    - `cudaGetLastError()`: returns the last error that occured
        - normally used to check if there is an error after a kernel launch
    - `cudaGetErrorString(cudaError_t err)`: returns the error string
    - `cudaError_t`: an error code

# CUDA Computing Model
- Host: CPU
- Device: GPU
- Software
    - Grid: a group of blocks
    - Block: a group of threads that can cooperate with each other
    - Thread: a single execution unit
- Hardware
    - Grid is executed on the device
    - Block is executed on the Streaming Multiprocessor (SM)
        - Each SM has a fixed number of partitions
        - Threads in a block are grouped into warps and each partition can execute a warp
        - Warp scheduler schedules the warps to the partitions
    - Thread is executed on the CUDA core
        - the threads are executed in a SIMD fashion
- Occupancy: the ratio of active warps to the maximum number of warps that can be executed
    - The occupancy is a key factor that affects the performance of the GPU
    - The theoraetical occupancy can be calculated as follows:
        - `occupancy = active_warps / max_warps_per_SM`
            - `max_warps_per_SM = max_threads_per_SM / warp_size`
            - the `max_threads_per_SM` is the maximum number of threads that can be executed on the SM. It is determined by the GPU specifications.
        - `active_warps = num_threads_per_block / warp_size`
        - `num_threads_per_block = blockDim.x * blockDim.y * blockDim.z`
        - `warp_size = 32`
    - However, the achieved occupancy may be lower than the theoretical occupancy. This may be due to multiple factors such as the number of registers used by the kernel, the amount of shared memory used by the kernel, and the number of threads per block. Also, the cache and memory access patterns may affect the occupancy.
    - In order to maximize the occupancy, ideally the number of threads per block should be a multiple of the warp size, and the number of blocks should be a multiple of the number of SMs.
    - See `cuda_src/cuda_runtime_api.cu` for an example how to query the device properties at runtime.
- Allocated Active Blocks: the number of blocks that are allocated to the SM
    - Max Thread Blocks per SM: the maximum number of blocks that can be allocated to the SM
    - Max Warps per SM: the maximum number of warps that can be executed
        - this also defines the maximum number of threads that can be executed on the SM
        - As a result, the warps limit controls the number of blocks that can be allocated to the SM
            - the compiler may adjust the number of blocks allocated to the SM to meet the warps limit
            - See the "Block Limit Wraps" in Occupancy section of the `ncu` profiler output
        - It's crutial to know that all the blocks are distributed accross the SMs
    - Max registers per SM: the maximum number of registers that can be used by the threads in the SM
        - if the total number of registers used by the threads in a block exceeds this limit, the cuda program will use the local memory instead of the registers. This is called "register spilling" and will affect the performance.
    - With `ncu` profiler, you can find the limit of the number of blocks that can be allocated to the SM
        - under the `Occupancy` section, you can find
            - `Block Limit SM`: the limit of blocks according to the SMs limit
            - `Block Limit Warps`: the limit of blocks according to the warps limit
            - `Block Limit Registers`: the limit of blocks according to the registers limit
            - `Block Limit Shared Memory`: the limit of blocks according to the shared memory limit


# Tools
- CUDA-gdb: a debugger for CUDA programs
- Nsight: a profiler for CUDA programs
- CUDA-Memcheck: a memory checker for CUDA programs

# Key Paremeters of GPU
- memory bandwidth: the rate at which data can be read from or stored into the device memory. ex: 100GB/s
- Throughput: the number of operations that can be performed in a unit of time. ex: 5 TFLOPS
    - number of cores: the number of cores in the GPU. ex: 2560 cores
    - clock speed: the speed at which the cores operate. ex: 1.5 GHz

# How to Read NVIDIA GPU Whitepaper
The whitepaper is normally organized as follows:
- Introduction (New Features)
- Architecture (SM, CUDA Cores, Memory)
    - ex: "GV100 GPU Hardware Architecture In-Depth"
- Performance Benchmark (FLOPS, Memory Bandwidth)
- Technical Specifications (Clock Speed, Memory Size)


# Learning Materials
- https://www.youtube.com/watch?v=86FAWCzIe_4
- https://www.udemy.com/course/cuda-parallel-programming-on-nvidia-gpus-hw-and-sw
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- https://developer.nvidia.com/blog/cooperative-groups/
    - https://www.youtube.com/watch?v=k7K-h7P1Bdk
    - https://leimao.github.io/blog/CUDA-Cooperative-Groups/
- https://stackoverflow.com/questions/22278631/what-does-pragma-unroll-do-exactly-does-it-affect-the-number-of-threads
- Debugging in VSCode: https://www.youtube.com/watch?v=gN3XeFwZ4ng
- Optimal Blocksize: https://forums.developer.nvidia.com/t/how-to-choose-how-many-threads-blocks-to-have/55529
- [1D, 2D and 3D thread allocation in CUDA](https://erangad.medium.com/1d-2d-and-3d-thread-allocation-for-loops-in-cuda-e0f908537a52)
- https://siboehm.com/articles/22/CUDA-MMM
- https://medium.com/@limyoonaxi/introduction-to-cuda-optimization-with-practical-examples-707e5b06bef8
