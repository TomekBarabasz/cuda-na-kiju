cmake_minimum_required(VERSION 3.10)

project(cuda-test LANGUAGES CXX CUDA)

# specify the C++ standard
#set(CMAKE_CUDA_STANDARD 17) -- this it not supported yet, highest version is 14

add_executable(cuda-test 
	../src/cuda-test.cpp
	../src/add.cu
	../src/sorting.cu
	../src/shuffle.cu
	../src/misc.cu
	../src/matrixAdd.cu
	../src/matrixMult.cu
)

target_include_directories( cuda-test PUBLIC
	../include
	../src
)

target_link_directories(cuda-test PUBLIC
	/usr/local/cuda/lib64
)

target_link_libraries( cuda-test
	curand
)

#this shit does not work!		     
set_target_properties(
        cuda-test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
	std c++17
	arch sm_35)
