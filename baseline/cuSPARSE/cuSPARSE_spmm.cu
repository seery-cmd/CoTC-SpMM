#include <cuda_runtime.h>
#include "cusparse.h"
#include "stdio.h"
#include <torch/extension.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t e_status = (func);                                               \
    if (e_status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(e_status), e_status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t e_status = (func);                                          \
    if (e_status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(e_status), e_status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

class cuSparse_SPMM {
    public:
    void Preprocess(int m,int k,int nonzeros,
        int *row_offsets,int* column_indices,at::Half* values);
        
        void Process(int n,at::Half * B,float * C); 
        
        private:
        cusparseStatus_t status;
        cusparseHandle_t handle=0;   
        cusparseSpMatDescr_t matA;  
        cusparseDnMatDescr_t matB, matC; 
        float alpha = 1.0f,beta = 1.0f;    
        
        int m_,k_;
};


void cuSparse_SPMM::Preprocess(int m,int k,int nonzeros,
                    int *row_offsets,int* column_indices,at::Half* values) {
        
    m_ = m; k_ = k;
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return ;
    }
    cusparseCreateCsr(&matA, m, k, nonzeros,
                    row_offsets, column_indices, values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    
}

void cuSparse_SPMM::Process(int n,at::Half * B,float * C) {
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    // Create dense matrix B
    int ldb = n;
    int ldc = n;

    cusparseCreateDnMat(&matB, k_, n, ldb, B,
                                       CUDA_R_16F, CUSPARSE_ORDER_ROW);
    // Create dense matrix C
    cusparseCreateDnMat(&matC, m_, n, ldc, C,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW);
    // allocate an external buffer if needed
    cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);


    cusparseSpMM(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
                                
    // CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    // CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    // CHECK_CUDA( cudaFree(dBuffer) )
}

float cuSPARSE_spmm_csr(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM,
    const long dimK,
    const long dimN,
    const long nnz,
    int epoches,
    int warmup){

    cuSparse_SPMM cu_sp;

    cu_sp.Preprocess(dimM,dimK,nnz,
                    row_offsets.data<int>(),col_indices.data<int>(),values.data<at::Half>());

    cudaDeviceSynchronize();
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    
    auto output_matrix = torch::zeros({dimM,dimN}, torch::kCUDA);
    for(int i=0; i < epoches; ++i)
        cu_sp.Process(dimN,rhs_matrix.data<at::Half>(),output_matrix.data<float>());

	cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    spmm_ms_avg = spmm_ms/(float)epoches;

    // printf(", %f, %f",spmm_ms_avg, gflops);


    // for(int iter=0; iter<warmup; ++iter){
    //     cuSPARSE_spmm_csr_kernel(
    //         row_offsets.data<int>(),
    //         col_indices.data<int>(),
    //         values.data<at::Half>(),
    //         rhs_matrix.data<at::Half>(),
    //         output_matrix.data<float>(),
    //         dimM,
    //         dimN,
    //         nnz
    //     );
    // }

    // cudaDeviceSynchronize();
    // float spmm_ms_avg = 0.0f;
    // float spmm_ms = 0.0f;
    // cudaEvent_t spmm_start;
    // cudaEvent_t spmm_end;
    // cudaEventCreate(&spmm_start);
    // cudaEventCreate(&spmm_end);
    // cudaEventRecord(spmm_start);
    // for(int iter=0; iter<epoches; ++iter){
    //     cuSPARSE_spmm_csr_kernel(
    //         row_offsets.data<int>(),
    //         col_indices.data<int>(),
    //         values.data<at::Half>(),
    //         rhs_matrix.data<at::Half>(),
    //         output_matrix.data<float>(),
    //         dimM,
    //         dimN,
    //         nnz
    //     );
    // }
	// cudaEventRecord(spmm_end);
    // cudaEventSynchronize(spmm_end);
    // cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    // cudaEventDestroy(spmm_start);
    // cudaEventDestroy(spmm_end);
    
    // //计算时间 ms
    // spmm_ms_avg = spmm_ms/(float)epoches;

    return spmm_ms_avg;
}