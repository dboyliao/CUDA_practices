cmake_minimum_required(VERSION 3.10)
project(CUDA_Practices LANGUAGES CXX CUDA)
cmake_policy(SET CMP0104 OLD)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)

execute_process(
    COMMAND uv run python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(PYBIND11_INCLUDES)
message(STATUS "PYBIND11_INCLUDES: ${PYBIND11_INCLUDES}")

add_executable(hello_cuda hello_cuda.cu)