#include "pybind11/numpy.h"
#include <cuda_runtime.h>
#include <stdio.h>

namespace py = pybind11;

__global__ void kernel_vec_add(int *a, int *b, int *c, int n)
{
    int i = threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

py::array_t<int> vec_add_naive_int(py::array_t<int> arr_a, py::array_t<int> arr_b)
{
    py::buffer_info buf_a = arr_a.request();
    py::buffer_info buf_b = arr_b.request();

    if (buf_a.ndim != 1 || buf_b.ndim != 1)
    {
        throw std::runtime_error("Number of dimensions must be one");
    }

    if (buf_a.size != buf_b.size)
    {
        throw std::runtime_error("Input shapes must be compatible.");
    }

    auto result = py::array_t<int>(buf_a.size);
    py::buffer_info buf_result = result.request();

    int *ptr_a = (int *)buf_a.ptr;
    int *ptr_b = (int *)buf_b.ptr;
    int *ptr_result = (int *)buf_result.ptr;

    int *d_a, *d_b, *d_result;
    cudaMalloc((void **)&d_a, buf_a.size * sizeof(int));
    cudaMalloc((void **)&d_b, buf_b.size * sizeof(int));
    cudaMalloc((void **)&d_result, buf_result.size * sizeof(int));

    cudaMemcpy(d_a, ptr_a, buf_a.size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, ptr_b, buf_b.size * sizeof(int), cudaMemcpyHostToDevice);

    int chunk_size = buf_a.size > 256 ? 256 : buf_a.size;
    int remain_cnt = buf_a.size;
    int idx = 0;
    while (remain_cnt > 0)
    {
        int add_cnt = remain_cnt > chunk_size ? chunk_size : remain_cnt;
        int offset = idx * chunk_size;
        kernel_vec_add<<<1, chunk_size>>>(d_a + offset, d_b + offset, d_result + offset, add_cnt);
        cudaDeviceSynchronize();
        remain_cnt -= chunk_size;
        idx++;
    }

    cudaMemcpy(ptr_result, d_result, buf_result.size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return result;
}

PYBIND11_MODULE(py_vec_add, m)
{
    m.def("vec_add_naive_int", &vec_add_naive_int, "Add two vectors of integers (Naive version)", py::arg("arr_a"), py::arg("arr_b"));
}