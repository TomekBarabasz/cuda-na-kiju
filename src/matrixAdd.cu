#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <random>
#include <map>
#include <functional>
#include "cu-utils.h"
#include "perf-measure.h"

template <typename T>
__global__ void addKernel(T* mA, T* mB, T* mC, int Nx, int Ny)
{
	const unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (ix < Nx && iy < Ny) {
		const unsigned idx = iy * Nx + ix;
		mC[idx] = mA[idx] + mB[idx];
	}
}

struct IRngGen
{
	virtual bool is_device_gen() = 0;
	virtual void generate_int	(int*,	 int size) = 0;
	virtual void generate_float	(float*, int size) = 0;
	virtual void generate_double(double*,int size) = 0;
	virtual ~IRngGen() {};
};

struct CudaRndGen : public IRngGen
{
	curandGenerator_t m_gen;
	static curandRngType_t genName2Type(const char* name)
	{
		std::map<string, curandRngType_t> types = {
			{"DEFAULT", CURAND_RNG_PSEUDO_DEFAULT},
			{"XORWOW", CURAND_RNG_PSEUDO_XORWOW},
			{"MRG32K3A", CURAND_RNG_PSEUDO_MRG32K3A},
			{"MTGP32",CURAND_RNG_PSEUDO_MTGP32},
			{"MT19937",CURAND_RNG_PSEUDO_MT19937},
			{"PHILOX4", CURAND_RNG_PSEUDO_PHILOX4_32_10},
			{"QUASI_DEFAULT", CURAND_RNG_QUASI_DEFAULT},
			{"QUASI_SOBOL32", CURAND_RNG_QUASI_SOBOL32},
			{"SCRAMBLED_SOBOL32", CURAND_RNG_QUASI_SCRAMBLED_SOBOL32} };
		auto it = types.find(name);
		if (it != types.end()) return it->second;
		else return CURAND_RNG_PSEUDO_DEFAULT;
	}
	CudaRndGen(const char* type)
	{
		const auto status = curandCreateGenerator(&m_gen, genName2Type(type));
		if (status != CURAND_STATUS_SUCCESS) {
			throw "curandCreateGenerator failed";
		}
	}
	~CudaRndGen()
	{
		curandDestroyGenerator(m_gen);
	}
	bool is_device_gen() override { return true; }
	void generate_int(int* ptr, int size) override
	{
		const auto status = curandGenerate(m_gen, (unsigned int*)ptr, size);
		if (status != CURAND_STATUS_SUCCESS) {
			throw "curandGenerate failed";
		}
	}
	void generate_float(float* ptr, int size) override
	{
		const auto status = curandGenerateUniform(m_gen, ptr, size);
		if (status != CURAND_STATUS_SUCCESS) {
			throw "curandGenerateUniform failed";
		}
	}
	void generate_double(double* ptr, int size) override
	{
		const auto status = curandGenerateUniformDouble(m_gen, ptr, size);
		if (status != CURAND_STATUS_SUCCESS) {
			throw "curandGenerateUniformDouble failed";
		}
	}
};

struct CpuRndGen : public IRngGen
{
	std::random_device dev;
	std::mt19937 rng;
	CpuRndGen() : rng(dev()) {}

	bool is_device_gen() override { return false; }
	void generate_int(int* ptr, int size) override
	{
		std::uniform_int_distribution<int> dist(-10, 10);
		while (size-- > 0) { *ptr++ = dist(rng); }
	}
	void generate_float(float* ptr, int size) override
	{
		std::uniform_real_distribution<float> dist(-1, 1);
		while (size-- > 0) { *ptr++ = dist(rng); }
	}
	void generate_double(double* ptr, int size) override
	{
		std::uniform_real_distribution<float> dist(-1, 1);
		while (size-- > 0) { *ptr++ = dist(rng); }
	}
};
template <typename T>
__host__ void doMatrixAdd(int one_dim_size, int blockx, int blocky, int dimx, int dimy, std::function<void(T*,int)> rng, bool rng_on_device)
{
	const auto tot_size = one_dim_size * one_dim_size;
	auto host_a = std::make_unique<T[]>(tot_size);
	auto host_b = std::make_unique<T[]>(tot_size);
	auto host_c = std::make_unique<T[]>(tot_size);

	auto dev_a = cu_make_unique<T>(tot_size);
	auto dev_b = cu_make_unique<T>(tot_size);
	auto dev_c = cu_make_unique<T>(tot_size);

	Measurements mm;
	if (rng_on_device) {
		mm.start();
		std::cout << "start generating random numbers ...";
		rng(dev_a.get(), tot_size);
		rng(dev_b.get(), tot_size);
		cu_device_synchronize();
		std::cout << " done " << mm.elapsed() << std::endl;
	}
	else {
		mm.start();
		std::cout << "start generating random numbers ...";
		rng(host_a.get(), tot_size);
		rng(host_b.get(), tot_size);
		std::cout << " done " << mm.elapsed() << std::endl;

		mm.start();
		std::cout << "start copying data to device ...";
		cu_copy_to_device(host_b, dev_b, tot_size);
		cu_copy_to_device(host_c, dev_c, tot_size);
		cu_device_synchronize();
		std::cout << " done " << mm.elapsed() << std::endl;
	}
	
	mm.start();
	addKernel<T> << <dim3(blockx, blocky), dim3(dimx, dimy) >> > (dev_a.get(), dev_b.get(), dev_c.get(), one_dim_size, one_dim_size);
	if (auto status = cudaGetLastError(); status != cudaSuccess) {
		std::cout << "kernel launch error " << cudaGetErrorString(status) << std::endl;
	}
	cu_device_synchronize();
	std::cout << "kernel exec time : " << mm.elapsed() << std::endl;

	cu_copy_to_host(dev_c, host_c, tot_size);
}
IRngGen* makeRndGen(const char* _gen_type)
{
	string gen_type(_gen_type);
	std::transform(gen_type.begin(), gen_type.end(), gen_type.begin(), std::toupper);
	if (gen_type == "CPU") {
		return new CpuRndGen();
	}
	else {
		return new CudaRndGen(gen_type.c_str());
	}
}
int matrixAdd(int argc, char** argv)
{
	const int oneDimMatrixSize = argc > 1 ? 1 << std::atoi(argv[0]) : 1 << 14;
	const int dimx = argc > 2 ? std::atoi(argv[1]) : 32;
	const int dimy = argc > 3 ? std::atoi(argv[2]) : 32;
	const char* type = argc > 4 ? argv[3] : "float";
	const char* RNG_type = argc > 5 ? argv[4] : "CPU";
	const int blockx = (oneDimMatrixSize + dimx - 1) / dimx;
	const int blocky = (oneDimMatrixSize + dimy - 1) / dimy;
	auto rng = makeRndGen(RNG_type);

	printf("matrix size (%d,%d) grid (%d,%d) block (%d,%d)\n", oneDimMatrixSize, oneDimMatrixSize, blockx, blocky, dimx, dimy);
	if (0 == strcmp(type, "float")) {
		auto gen = [&](float* ptr, int size) {
			rng->generate_float(ptr, size);
		};
		doMatrixAdd<float>(oneDimMatrixSize,blockx,blocky,dimx,dimy, gen, rng->is_device_gen());
	}
	else if (0 == strcmp(type, "double")) {
		auto gen = [&](double* ptr, int size) {
			rng->generate_double(ptr, size);
		};
		doMatrixAdd<double>(oneDimMatrixSize, blockx, blocky, dimx, dimy, gen, rng->is_device_gen());
	}
	else if (0 == strcmp(type, "int")) {
		auto gen = [&](int* ptr, int size) {
			rng->generate_int(ptr, size);
		};
		doMatrixAdd<int>(oneDimMatrixSize, blockx, blocky, dimx, dimy, gen, rng->is_device_gen());
	}
	delete rng;
	return 0;
}