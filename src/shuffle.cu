#include <cuda.h>
#include "cu-utils.h"
#include "perf-measure.h"
#include <chrono>
#include <thread>

__global__ void shuffle_kernel(int* in, int* out, int nElem)
{
    auto laneId = getLaneId();
    int value = in[threadIdx.x];
    int vs = __shfl_up_sync(0xffffffff, value, 1, 8);
    printf("thread %d 1st shuffle value %d\n", threadIdx.x, vs);

    if (0 == laneId) printf("\n");

    vs = __shfl_up_sync(0xffffffff, value, 2, 8);
    printf("thread %d 2nd shuffle value %d\n", threadIdx.x, vs);

    if (0 == laneId) printf("\n");

    vs = __shfl_up_sync(0xffffffff, value, 4, 8);
    printf("thread %d 3rd shuffle value %d\n", threadIdx.x, vs);

    out[threadIdx.x] = value;
}

int shuffle(int argc, char**argv)
{
    constexpr int nElem = 32;
    int a_in[nElem];
    int a_out[nElem];

    for (int i = 0; i < nElem; ++i) {
        a_in[i] = i;
    }

    // Choose which GPU to run on, change this on a multi-GPU system.
    auto cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        throw "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
    }

    // device pointers    
    //auto d_in  = cuda_make_unique<int>(nElem);
    auto d_in = cu_make_unique_memcpy<int>(a_in, nElem);
    auto d_out = cu_make_unique<int>(nElem);

    auto& del1 = d_in.get_deleter();
    auto& del2 = d_out.get_deleter();

    Measurements mm;
    mm.start();
    std::this_thread::sleep_for(std::chrono::microseconds(10'000));
    const auto tm = mm.elapsed();
    std::cout << "sleep 100 uses time : " << tm << std::endl;
    //std::cout << "sleep 100 uses time : " << mm.elapsed() << std::endl;

    mm.start();
    shuffle_kernel << <1, 32 >> > (d_in.get(), d_out.get(), nElem);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::string("addKernel launch failed: ") + cudaGetErrorString(cudaStatus);
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        throw std::string("cudaDeviceSynchronize returned error code: ") + cudaGetErrorString(cudaStatus);
    }
    std::cout << "kernel exec time : " << mm.elapsed() << std::endl;

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(a_out, d_out.get(), nElem * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        throw "cudaMemcpy failed!";
    }

    std::cout << "output:";
    for (int i = 0; i < nElem; ++i) {
        std::cout << a_out[i] << ", ";
    }
    std::cout << std::endl;
    return 0;
}