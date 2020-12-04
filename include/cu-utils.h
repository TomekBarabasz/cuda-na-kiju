#pragma once
#include <stdlib.h>
#include <iostream>
#include <functional>
#include <string>
#include "cuda_runtime.h"

using std::string;

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
    const auto status = cudaMalloc((void**)&ptr, size * sizeof(T));
    if (cudaSuccess == status) {
        return cu_unique_ptr<T>(static_cast<T*>(ptr), [](T* ptr) {cudaFree(ptr); });
    }
    throw string("cudaMalloc failed") + cudaGetErrorString(status);
}

template <typename T>
cu_unique_ptr<T> cu_make_unique_memcpy(T* h_ptr, int size)
{
    auto d_ptr = cu_make_unique<T>(size);
    const auto status = cudaMemcpy(d_ptr.get(), h_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        throw string("cudaMemcpy HostToDevice failed : ") + cudaGetErrorString(status);
    }
    return d_ptr;
}
template <typename T>
cu_unique_ptr<T> cu_make_unique_memcpy(std::unique_ptr<T[]>& h_ptr, int size){
    return cu_make_unique_memcpy(h_ptr.get(), size);
}
template <typename T>
cu_unique_ptr<T> cu_make_unique_memcpy(std::unique_ptr<T>& h_ptr, int size) {
    return cu_make_unique_memcpy(h_ptr.get(), size);
}
template <typename T>
void cu_copy_to_host(T* d_ptr, T* h_ptr, int size)
{
    const auto status = cudaMemcpy(h_ptr, d_ptr, size*sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw string("cudaMemcpy DeviceToHost failed : ") + cudaGetErrorString(status);
    }
}
template <typename T>
void cu_copy_to_host(cu_unique_ptr<T>& d_ptr, std::unique_ptr<T[]>& h_ptr, int size) {
    cu_copy_to_host(d_ptr.get(), h_ptr.get(), size);
}
template <typename T>
void cu_copy_to_host(cu_unique_ptr<T>& d_ptr, std::unique_ptr<T>& h_ptr, int size) {
    cu_copy_to_host(d_ptr.get(), h_ptr.get(), size);
}

template <typename T>
void cu_copy_to_device(T* h_ptr, T* d_ptr, int size)
{
    const auto status = cudaMemcpy(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        throw string("cudaMemcpy DeviceToHost failed : ") + cudaGetErrorString(status);
    }
}
template <typename T>
void cu_copy_to_device(std::unique_ptr<T[]>& h_ptr, cu_unique_ptr<T>& d_ptr, int size) {
    cu_copy_to_device(h_ptr.get(), d_ptr.get(), size);
}
template <typename T>
void cu_copy_to_device(std::unique_ptr<T>& h_ptr, cu_unique_ptr<T>& d_ptr, int size) {
    cu_copy_to_device(h_ptr.get(), d_ptr.get(), size);
}

inline void cu_device_synchronize() 
{
    const auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        throw string("cudaDeviceSynchronize failed : ") + cudaGetErrorString(status);
    }
}

__device__ inline unsigned int getLaneId()
{
    unsigned int laneid;
    //This command gets the lane ID within the current warp
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}

