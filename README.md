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