cmake_minimum_required(VERSION 3.16.3)  # 需要CMake 3.20+支持现代CUDA特性
project(cudaLearn LANGUAGES)  # 关键：必须声明CUDA语言支持
enable_language(CUDA)
enable_language(CXX)
# 设置CUDA相关配置
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80")  # 根据显卡计算能力设置：70(Volta),75(Turing),80(Ampere)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda-11.8/include")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_EXE_LINKER_FLAGS}")
endif()
# 可执行文件配置
add_executable(C_project
                cuda_learn/src/add/add.cpp
                cuda_learn/src/spmm/spmm_CPU.cpp
                src/Struct/struct.h
                src/readMtx/mmio.h
                src/readMtx/utils.h
                src/include/Liner_Library.h
                src/include/compare.h
                lcuda-project/SpLibrary/common/sp_handle.cpp
                lcuda-project/SpLibrary/common/sp_utils.cpp
                lcuda-project/SpLibrary/src/backends/mlu_backend.cpp
                lcuda-project/SpLibrary/src/backends/nvidia_backend.cpp
                lcuda-project/SpLibrary/src/interface/AXPYB.cpp
                lcuda-project/SpLibrary/src/interface/spgemm.cpp
                lcuda-project/SpLibrary/src/interface/spmv.cpp
                lcuda-project/SpLibrary/src/interface/SpVV.cpp
                lcuda-project/SpLibrary/src/kernels/cpu/AXPYB_cpu.cpp
                lcuda-project/SpLibrary/src/kernels/cpu/spmv_cpu.cpp
                lcuda-project/SpLibrary/src/kernels/cpu/SpVV_cpu.cpp
                lcuda-project/SpLibrary/src/utils/mmio.cpp
                lcuda-project/SpLibrary/src/utils/readmtx.cpp
                lcuda-project/SpLibrary/tests/test_AXPYB.cpp
                lcuda-project/SpLibrary/tests/test_spmv.cpp
                lcuda-project/SpLibrary/tests/test_SpVV.cpp)

add_executable(cudaTest cuda_learn/cuda_example/add.cu
                        cuda_learn/cuda_example/profile.cu
                        cuda_learn/cuda_example/test_mma.cu
                        cuda_learn/cuda_example/test.cu
                        cuda_learn/cuda_example/test_PTX.cu
                        cuda_learn/cuda_example/test_PTX_trans.cu)

add_executable(cudaLibrary src/main.cu
                            src/include/compare_cu.cuh
                            src/include/Liner_Library.cuh)

add_executable(lcudaproject lcuda-project/SpLibrary/src/kernels/cuda/AXPYB_nvidia.cu
                            lcuda-project/SpLibrary/src/kernels/cuda/spmv_nvidia.cu
                            lcuda-project/SpLibrary/src/kernels/cuda/SpVV_nvidia.cu)

add_executable(cuda_spmm src/spmm/spmm.cu
                        src/spmm/kernel/spmm_kernel.cu
                        src/spmm/utils/utils.cu
                        src/spmm/kernel/spmm_kernel_shared.cu
                        src/spmm/kernel/spmm_kernel_V3_128x4.cu
                        src/spmm/kernel/spmm_kernel_m8n8k4.cu
                        src/spmm/kernel/spmm_kernel_m16n8k8_shared.cu
                        src/spmm/kernel/spmm_kernel_m16n8k8_cuda_shared.cu
                        src/spmm/kernel/spmm_cuda.cu
                        src/spmm/kernel/translations_HTC.cu
                        src/spmm/kernel/spmm_kernel_async.cu
                        src/spmm/kernel/spmm_memory.cu)

add_executable(RoDe cuda_learn/RoDe/utils/matrix_utils.cu
                    cuda_learn/RoDe/eval/eval_spmm_f32_n128.cu
                    cuda_learn/RoDe/utils/basic_utils.h
                    cuda_learn/RoDe/utils/common_utils.h
                    cuda_learn/RoDe/RoDe_SpMM/RoDeSpmm.cu
                    cuda_learn/RoDe/utils/matrix_utils.h
                    cuda_learn/RoDe/RoDe_SpMM/RoDeSpmm.h)


