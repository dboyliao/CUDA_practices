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
- `__global__`: a function that runs on the device and can be called from the host
- `__device__`: a function that runs on the device and can be called from the device
- Application devided into blocks, blocks devided into wraps. Wraps will be mapped to SMs.
    - wraps are controled by the wrap scheduler
    - because each SM has fixed number of paritions. If the number of wraps is greater than the number of partitions, the wraps will be queued and scheduled by the wrap scheduler.
- Share memory is faster than global memory
    - `__syncthreads()`: a barrier that synchronizes threads in the same block
    - `__shared__`: a variable that is shared among threads in the same block
    - https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/

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
