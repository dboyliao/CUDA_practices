cmake_minimum_required(VERSION 3.10)
project(CUDA_Practices LANGUAGES CXX CUDA)
cmake_policy(SET CMP0104 OLD)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)

# setup hints for pybind11
set(Python_ROOT_DIR ${CMAKE_SOURCE_DIR}/.venv/)
execute_process(
    COMMAND uv run python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found")
endif()

message(STATUS "python interpretor: ${Python_EXECUTABLE}")
message(STATUS "python includes: ${Python_INCLUDE_DIRS}")
message(STATUS "python libs: ${Python_LIBRARIES}")

add_executable(hello_cuda cuda_src/hello_cuda.cu)
add_executable(cuda_print_grid cuda_src/cuda_print_grid.cu)
add_executable(
    section3_001
    ${CMAKE_SOURCE_DIR}/submodules/CUDA_Course/section3_001.cu
)
pybind11_add_module(
    py_vec_add
    cuda_src/py_vec_add.cu
)
