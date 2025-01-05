cmake_minimum_required(VERSION 3.10)
project(CUDA_Practices LANGUAGES CXX CUDA)
cmake_policy(SET CMP0104 OLD)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)

add_executable(hello_cuda hello_cuda.cu)