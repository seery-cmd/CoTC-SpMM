//
// Created by lenovo on 2025/4/17.
//

#include <cooperative_groups/memcpy_async.h>
#include "../Struct/struct.h"
#include "cuda.h"
#include "cuda_pipeline.h"


#ifndef CUDALEARN_LINER_LIBRARY_H
#define CUDALEARN_LINER_LIBRARY_H
/** 基本稀疏线性代数库
 * AXPYB: y = αx + βy
 * SpVV:稀疏向量 - 稀疏向量乘 result = op(x) * y OP表示是否需要转置, 转置结果为scalar
 * SpMM:稀疏矩阵 - 稠密矩阵乘法
 * __global__函数不能放.h文件里面,放到.cu文件里面给nvcc编译,.h文件运行回满
 * **/
 /* add */
__global__ void add(int n,Data_type *x,Data_type *y,Data_type *rec);
void add_cpu(int n,const Data_type *x,const Data_type *y,Data_type *rec);
/* AXPYB */
__constant__ Data_type alpha_GPU[1];// GPU常量内存alpha
__constant__ Data_type beta_GPU[1];// GPU的常量内存beta
__global__ void L_cuda_AXPYB(const int * __restrict__ Sp_vec_GPU_index,const Data_type *__restrict__ Sp_vec_GPU_val,const Data_type *__restrict__ Dense_Vector_GPU,Data_type *Sp_rec_vector_GPU,int length);
void L_cuda_AXPYB_cpu(Sp_vector *Sp_Vec,Data_type *Dense_Vector,Data_type *Sp_rec_vector,Data_type alpha,Data_type beta);
/* SpVV */
constexpr int shared_size = 1024; //编译计算共享内存大小,事先定义
__global__ void L_cuda_SpVV_T(const int * __restrict__ Sp_vec_GPU_index,const Data_type *__restrict__ Sp_vec_GPU_val,const Data_type *__restrict__ Sparse_Vector_GPU,Data_type *__restrict__ rec_scalar_GPU,int length);
void L_cuda_SpVV_cpu_T(Sp_vector *Sp_Vec,Data_type *Sparse_Vector,Data_type *rec_scalar);

/** 工具类
 * AXPYB,SPVV创建向量元素
 * 从矩阵里面创建稀疏向量
 **/
void creat_SpVector_CPU(Sp_vector *Sp_vector,Matrix_data_CSR *MtxCSR);

/** SPMM格式转换
 * 转换为SPMM存储格式,CPU和GPU两个版本
 * **/
void trans_SPMM_format();

__global__
void add(int n,Data_type *x,Data_type *y,Data_type *rec)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;//gridDim.x=param1,blockDim.x=param2
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i+=stride)
        rec[i] = x[i] + y[i];
}
void add_cpu(int n,const Data_type *x,const Data_type *y,Data_type *rec)
{
    for(int i=0;i<n;i++)
    {
        rec[i] = x[i] + y[i];
    }
}
void creat_SpVector_CPU(Sp_vector *Sp_vector,Matrix_data_CSR *MtxCSR)
{
    Sp_vector->length = MtxCSR->nnz;
    Sp_vector->index = (int *)malloc(sizeof(int)*Sp_vector->length);
    Sp_vector->val = (Data_type *) malloc(sizeof(Data_type) * Sp_vector->length);
    int start,end;
    for(int i=0;i<MtxCSR->row;i++)
    {
        start = MtxCSR->row_ptr[i],end = MtxCSR->row_ptr[i+1];
        for(int j = start;j < end;j++)
        {
            Sp_vector->index[j] = MtxCSR->row * i + MtxCSR->col_index[j];
            Sp_vector->val[j] = MtxCSR->val[j];
        }
    }
    printf("creat sp_vec success!\n");
}
void L_cuda_AXPYB_cpu(Sp_vector *Sp_Vec,Data_type *Dense_Vector,Data_type *Sp_rec_vector,Data_type alpha,Data_type beta)
{
    for(int i = 0 ;i<Sp_Vec->length;i++)
    {
        Sp_rec_vector[i] = alpha *Sp_Vec->val[i] + beta * Dense_Vector[Sp_Vec->index[i]];
    }
}
__global__
void L_cuda_AXPYB(const int * __restrict__ Sp_vec_GPU_index,const Data_type *__restrict__ Sp_vec_GPU_val,const Data_type *__restrict__ Dense_Vector_GPU,Data_type *Sp_rec_vector_GPU,int length)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;//gridDim.x=param1,blockDim.x=param2
    int stride = blockDim.x * gridDim.x;
    for(int i=index;i<length;i+=stride)
    {
        Sp_rec_vector_GPU[i] = alpha_GPU[0] * Sp_vec_GPU_val[i] + beta_GPU[0] * Dense_Vector_GPU[Sp_vec_GPU_index[i]];
    }
}
void L_cuda_SpVV_cpu_T(Sp_vector *Sp_Vec,Data_type *Sparse_Vector,Data_type *rec_scalar)
{
    for(int i=0;i<Sp_Vec->length;i++)
    {
        *rec_scalar += Sp_Vec->val[i] * Sparse_Vector[Sp_Vec->index[i]];
    }
}

__global__
void L_cuda_SpVV_T(const int * __restrict__ Sp_vec_GPU_index,const Data_type *__restrict__ Sp_vec_GPU_val,const Data_type *__restrict__ Sparse_Vector_GPU,Data_type *__restrict__ rec_scalar_GPU,int length)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;//全局线程编号
    int stride = blockDim.x * gridDim.x;//线程组规模
    int local_index = threadIdx.x;  //本地线程编号
    int local_block = blockIdx.x;
//    auto grid = cooperative_groups::this_grid();
//    auto block = cooperative_groups::this_thread_block();
//    constexpr int length_cache = 6 * 1024;
//    size_t size = length_cache * block.group_index().x;
//    cooperative_groups::memcpy_async(block, cache, Sp_vec_GPU_index + size,sizeof(int) * length_cache); //加载1024个到共享内存中
//    cooperative_groups::wait(block);   非阻塞的global memory - > share memory
    __shared__ Data_type cache[shared_size];
    Data_type temp = 0.0;
    for(int i = index;i<length ;i+=stride)
    {
        temp += Sp_vec_GPU_val[i] * Sparse_Vector_GPU[Sp_vec_GPU_index[i]];
    }
    /** compute **/
    cache[local_index] = temp;
    __syncthreads();
    int abs =  shared_size / 2;
    while(abs != 0)
    {
        if(local_index < abs)
        {
            cache[local_index] += cache[local_index + abs];
        }
        abs /= 2;
        __syncthreads();
    }
    if(local_index == 0) {
        rec_scalar_GPU[local_block] = cache[0];
    }
}
#endif //CUDALEARN_LINER_LIBRARY_H
