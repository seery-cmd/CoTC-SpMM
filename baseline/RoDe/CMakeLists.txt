cmake_minimum_required(VERSION 3.13)
project(SPC LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

add_definitions(-w)

SET(CMAKE_CUDA_ARCHITECTURES 80 90)  # 仅支持 sm_80 和 sm_90

find_package(CUDAToolkit 11.8 REQUIRED)
message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit Path: ${CUDAToolkit_ROOT}")
message(STATUS "NVCC Path: ${CUDAToolkit_NVCC_EXECUTABLE}")

list(APPEND SPC_LIBS 
  CUDA::cudart_static        # CUDA 运行时库（静态）
  CUDA::cublas_static        # cuBLAS 库（静态）
  CUDA::cusparse_static      # cuSPARSE 库（静态）
)
add_subdirectory(third_party/abseil-cpp)
list(APPEND SPC_LIBS "absl::random_random")

set(UTILS_LIBS ${PROJECT_SOURCE_DIR}/utils)
set(Sputnik_SPMM_LIBS ${PROJECT_SOURCE_DIR}/spmm/Sputnik)
set(CUSPARSE_SPMM_LIBS ${PROJECT_SOURCE_DIR}/spmm/cuSparse)
set(RoDe_SPMM_LIBS ${PROJECT_SOURCE_DIR}/spmm/RoDe)


# 添加到全局头文件搜索路径中，确保所有子目录的源文件能
# 直接通过相对路径引用根目录下的头文件
include_directories(${PROJECT_SOURCE_DIR})

MESSAGE( STATUS "this var key = ${CMAKE_CUDA_FLAGS}.")

# 进入各个子目录，执行相应的 CMakeLists.txt 文件
add_subdirectory(utils)
add_subdirectory(spmm)
add_subdirectory(eval)