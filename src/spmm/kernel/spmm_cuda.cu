#include "../../Struct/struct.h"
#include <mma.h>
#include <cuda.h>
#include <fstream>
#ifndef CHECK_H
#define CHECK_H
inline void check(cudaError_t call, const char* file,  int line);
#define CHECK(call) (check(call, __FILE__, __LINE__))
inline void check(cudaError_t call, const char* file,  int line)
{
    if (call != cudaSuccess)
    {
        std::cout << "cuda error: " << cudaGetErrorName(call) << std::endl;
        std::cout << "at file: " << file << ", line: " << line << std::endl;
        std::cout << cudaGetErrorString(call) << std::endl;
    }
}
#endif //CHECK_H
__global__ void SpMM_GPU_kernel_cudaCore(int bN,int row,const int * __restrict__ dev_row_ptr,const int * __restrict__ dev_col_idx,Data_type *dev_val,const Data_type *dev_dense_val,Data_type *dev_rec_val)
{
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num = blockDim.x * gridDim.x;
    unsigned int i,j;
    if(thread_id > row * bN)
    {
        return;
    }
    Data_type temp;
    unsigned int row_I;
    unsigned int col_I;
    for(i = thread_id;i < row * bN ;i += num)
    {
        temp = 0.0;
        row_I = i/bN;
        col_I = i%bN;
        for(j = dev_row_ptr[row_I]; j < dev_row_ptr[row_I+1] ; j++)
        {
            temp += dev_val[j] * dev_dense_val[dev_col_idx[j]*bN + col_I];
        }
        dev_rec_val[row_I*bN + col_I] = temp;
    }
}
void SpMM_cuda(const Matrix_data_CSR *Sparse_mtx,const Matrix_data_CSR *Dense_mtx_CSR,Matrix_data_CSR *rec_mtx)
{
    /** 事件计时 **/
    cudaEvent_t start,end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    /* GPU内存开辟 */
    //nscore格式分配内存
    int *dev_col_idx,*dev_row_ptr;
    Data_type *dev_val;
    CHECK(cudaMalloc(&dev_col_idx,sizeof(int) * Sparse_mtx->nnz));
    CHECK(cudaMalloc(&dev_row_ptr,sizeof(int) * (Sparse_mtx->row + 1)));
    CHECK(cudaMalloc(&dev_val,sizeof(Data_type) * Sparse_mtx->nnz));
    //dense matrix分配内存
    Data_type *dev_dense_val;
    CHECK(cudaMalloc(&dev_dense_val,Dense_mtx_CSR->nnz * sizeof(Data_type)));
    //rec matrix分配内存
    Data_type *dev_rec_val;
    CHECK(cudaMalloc(&dev_rec_val,rec_mtx->nnz * sizeof(Data_type)));

    /* 数据传输GPU */
    CHECK(cudaMemcpy(dev_col_idx,Sparse_mtx->col_index,sizeof(int) * Sparse_mtx->nnz,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_row_ptr,Sparse_mtx->row_ptr,sizeof(int) * (1 + Sparse_mtx->row),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_val,Sparse_mtx->val,sizeof(Data_type) * Sparse_mtx->nnz,cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(dev_dense_val,Dense_mtx_CSR->val,sizeof(Data_type) * Dense_mtx_CSR->nnz,cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(dev_rec_val,rec_mtx->val,sizeof(Data_type) * rec_mtx->nnz,cudaMemcpyHostToDevice));

    /* v1 kernel */
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    CHECK(cudaEventRecord(start,0));

    SpMM_GPU_kernel_cudaCore<<<1024, 1024>>>(BN, Sparse_mtx->row, dev_row_ptr, dev_col_idx, dev_val,
                                                          dev_dense_val, dev_rec_val);

    CHECK(cudaEventRecord(end,0));
    CHECK(cudaEventSynchronize(end));
    float timecost_GPU;
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("timecost_cuda:%fms\tcuda core验证程序 Gflops:%lf\n",timecost_GPU,Sparse_mtx->nnz / 1e6 / timecost_GPU * BN * 2.0);

//    /** write **/
//    fstream f;
//    f.open("/data/seery/src/log/reslut/3_251113.txt",ios::out|ios::app);
//    f<<timecost_GPU<<" "<<Sparse_mtx->nnz / 1e6 / timecost_GPU * BN * 2.0<<endl;
//    f.close();

    CHECK(cudaMemcpy(rec_mtx->val,dev_rec_val,sizeof(Data_type) * rec_mtx->nnz,cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    CHECK(cudaFree(dev_col_idx));
    CHECK(cudaFree(dev_row_ptr));
    CHECK(cudaFree(dev_val));
    CHECK(cudaFree(dev_dense_val));
    CHECK(cudaFree(dev_rec_val));
}