#pragma once
#include <stdlib.h>
#include <iostream>
#include <functional>
#include "cuda_runtime.h"

inline void check_status(cudaError_t cudaStatus, const char* message)
{
    if (cudaStatus != cudaSuccess) {
        std::cerr << message << std::endl;
        exit(cudaStatus);
    }
}
template <typename T>
using cu_unique_ptr = std::unique_ptr <T, std::function<void(T*)> >;
template <typename T>
cu_unique_ptr<T> cu_make_unique(int size)
{
    void* ptr;
    if (cudaSuccess == cudaMalloc((void**)&ptr, size * sizeof(T))) {
        return cu_unique_ptr<T>(static_cast<T*>(ptr), [](T* ptr) {cudaFree(ptr); });
    }
    throw "cudaMalloc failed";
}

template <typename T>
cu_unique_ptr<T> cu_make_unique_memcpy(T* h_ptr, int size)
{
    auto d_ptr = cu_make_unique<T>(size);
    const auto status = cudaMemcpy(d_ptr.get(), h_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        throw "cudaMemcpy failed!";
    }
    return d_ptr;
}
__device__ inline unsigned int getLaneId()
{
    unsigned int laneid;
    //This command gets the lane ID within the current warp
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}
