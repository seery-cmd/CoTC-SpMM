#include "../../Struct/struct.h"
#include "../utils/utils.cu"
#include <cooperative_groups/memcpy_async.h>
#include "../../readMtx/utils.h"
#include <mma.h>
#include <cuda.h>
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
#define WARP_SIZE 32

#define mma_m8n8k4(C0,C1,C2,C3,A0,A1,B0,B1) asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 {%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};" : "+r"(C0), "+r"(C1), "+r"(C2), "+r"(C3): "r"(A0), "r"(A1),"r"(B0),"r"(B1))

__global__ void SpMM_GPU_kernel_v4(int bN,int row,const int * __restrict__ dev_rowT_ptr,const int * __restrict__ dev_colT_idx,Data_type *dev_valT,Data_type *dev_dense_val,Data_type *dev_rec_val) {
    unsigned int warp_id = threadIdx.x / 32; //0,1,2,3....
    unsigned int A_addr = threadIdx.x % 32;//sparse -- global->寄存器
    unsigned int i = blockIdx.x;
    if(i > row/8)
    {
        return;
    }
    unsigned int j;
    unsigned int bn = bN;
    unsigned int line_b = threadIdx.x % 4;
    unsigned int laneID_A = A_addr / 16 * 16 + A_addr % 4 * 4;
    unsigned int laneId_B = A_addr / 4 % 4 * 8 + A_addr / 16 * 4 + warp_id * 32; //离散
    unsigned int landId_C = i * bn * 8 + A_addr / 4 % 4 * 8 + A_addr % 4 * bn + A_addr / 16 * 1024 + warp_id * 32;

    uint32_t mtx_A[2];//存储4个元素,一个warp存储8 * 16个元素,需要32位寄存器2个
    uint32_t mtx_B[2];//存储4个元素,一个warp存储8 * 16个元素,需要32位寄存器2个
    uint32_t mtx_c[4];//存储4个元素,一个warp存储16 * 16个元素,需要32位寄存器4个
    //赋值C
    mtx_c[0] = 0, mtx_c[1] = 0, mtx_c[2] = 0, mtx_c[3]=0;
    //offset
    unsigned int col_idx_0 ;
    /** 这里执行for循环,把一行的Tcore元素算完,定义的mtx_c位于寄存器 **/
    const unsigned int TcoreNum = dev_rowT_ptr[i + 1] - dev_rowT_ptr[i]; //一个block负责一行
    const unsigned int tcore_start = dev_rowT_ptr[i];
    const unsigned int tcore_end = dev_rowT_ptr[i + 1];
    if (TcoreNum == 0) {
        return;
    }
    else {
        for(j=tcore_start;j < tcore_end;j++)
        {
            col_idx_0 = dev_colT_idx[j * 4 + line_b] * bn;
            //加载A,__ldg函数,或者ld的PTX语句
            mtx_A[0] = *reinterpret_cast<uint32_t *>(&dev_valT[j * 32 + laneID_A]);
            mtx_A[1] = *reinterpret_cast<uint32_t *>(&dev_valT[j * 32 + laneID_A + 2]);
            //mma加载b
            mtx_B[0] = *reinterpret_cast<uint32_t *>(&dev_dense_val[col_idx_0 + laneId_B]);
            mtx_B[1] = *reinterpret_cast<uint32_t *>(&dev_dense_val[col_idx_0 + laneId_B + 2]);

            //执行mma
            clock_t clock_start=clock64();
            mma_m8n8k4(mtx_c[0], mtx_c[1], mtx_c[2], mtx_c[3], mtx_A[0], mtx_A[1], mtx_B[0], mtx_B[1]);
            clock_t clock_end=clock64();
            if(j == tcore_start)
            {
                printf("%ld\n",clock_end - clock_start);
            }
        }
    }
    /** 直接强制转换,与移位效果相同,写会global C**/
    *(uint32_t*)&dev_rec_val[landId_C] = mtx_c[0];
    *(uint32_t*)&dev_rec_val[landId_C + 2] = mtx_c[1];
    *(uint32_t*)&dev_rec_val[landId_C + 4] = mtx_c[2];
    *(uint32_t*)&dev_rec_val[landId_C + 6] = mtx_c[3];
}
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
__global__ void add(int n, float * __restrict__ x,float  *rec)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i+=stride)
        rec[i] = x[i] + 1.0f;
}
void SpMM_GPU(const Matrix_data_CSR *Sparse_mtx,const Matrix_data_CSR *Dense_mtx_CSR,Matrix_data_CSR *rec_mtx)
{
    /** 格式转换 **/
    Matrix_data_NScore *mtx_nscore = (Matrix_data_NScore *)malloc(sizeof(Matrix_data_NScore));
    trans_format_NScoreV2(Sparse_mtx,mtx_nscore);

//    Printf_matrix_nscore(mtx_nscore);

    /** 事件计时 **/
    cudaEvent_t start,end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    /* GPU内存开辟 */
    //nscore格式分配内存
    int *dev_colT_idx,*dev_rowT_ptr;
    Data_type *dev_valT;
    CHECK(cudaMalloc(&dev_colT_idx,mtx_nscore->nnz_tcore * ShapeX_col * sizeof(int)));
    CHECK(cudaMalloc(&dev_rowT_ptr,(1 + mtx_nscore->row/ShapeX_row + (mtx_nscore->row%ShapeX_row > 0 ? 1 : 0))*sizeof(int)));
    CHECK(cudaMalloc(&dev_valT,mtx_nscore->nnz_tcore * (ShapeX_row * ShapeX_col) * sizeof(Data_type)));
    //dense matrix分配内存
    Data_type *dev_dense_val;
    CHECK(cudaMalloc(&dev_dense_val,Dense_mtx_CSR->nnz * sizeof(Data_type)));
    //rec matrix分配内存
    Data_type *dev_rec_val;
    CHECK(cudaMalloc(&dev_rec_val,rec_mtx->nnz * sizeof(Data_type)));

    /* 数据传输GPU */
    CHECK(cudaMemcpy(dev_colT_idx,mtx_nscore->colT_ide,sizeof(int) * mtx_nscore->nnz_tcore * ShapeX_col,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_rowT_ptr,mtx_nscore->rowT_ptr,sizeof(int) * (1 + mtx_nscore->row/ShapeX_row + (mtx_nscore->row%ShapeX_row > 0 ? 1 : 0)),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_valT,mtx_nscore->valT,sizeof(Data_type) * mtx_nscore->nnz_tcore * (ShapeX_row * ShapeX_col),cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(dev_dense_val,Dense_mtx_CSR->val,sizeof(Data_type) * Dense_mtx_CSR->nnz,cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(dev_rec_val,rec_mtx->val,sizeof(Data_type) * rec_mtx->nnz,cudaMemcpyHostToDevice));


//    cudaStream_t stream1,stream2;
//    cudaStreamCreate(&stream1);
//    cudaStreamCreate(&stream2);
    float timecost_GPU;
    /* add */
    int length = 1024 * 512;
    float *host_x = (float *)malloc(sizeof(float)*length);
    float *x,*rec;
    cudaMalloc(&x,length * sizeof(float));
    cudaMalloc(&rec,length * sizeof(float));
    cudaMemcpy(x,host_x,sizeof(float) * length,cudaMemcpyHostToDevice);
    CHECK(cudaEventRecord(start,0));
    add<<<1024,1024>>>(length,x,rec);
    CHECK(cudaEventRecord(end,0));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("timecost_add:%fms\n",timecost_GPU);
    /* kernel */
    dim3 grid_Dim(1,1,1);
    dim3 block_Dim(256,1,1);
    CHECK(cudaEventRecord(start,0));

    SpMM_GPU_kernel_v4<<<grid_Dim, block_Dim>>>(BN, mtx_nscore->row, dev_rowT_ptr, dev_colT_idx, dev_valT,
                                                dev_dense_val, dev_rec_val);
    CHECK(cudaEventRecord(end,0));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("timecost_Nscore:%fms\tNscore m8n8k4 Gflops:%lf\n",timecost_GPU,Sparse_mtx->nnz * BN * 2.0 / timecost_GPU / 1e6);


    CHECK(cudaMemcpy(rec_mtx->val,dev_rec_val,sizeof(Data_type) * rec_mtx->nnz,cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    free(mtx_nscore);
    CHECK(cudaFree(dev_colT_idx));
    CHECK(cudaFree(dev_rowT_ptr));
    CHECK(cudaFree(dev_valT));
    CHECK(cudaFree(dev_dense_val));
    CHECK(cudaFree(dev_rec_val));
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

    SpMM_GPU_kernel_cudaCore<<<1024, 1024,0,stream1>>>(BN, Sparse_mtx->row, dev_row_ptr, dev_col_idx, dev_val,
                                                       dev_dense_val, dev_rec_val);

    CHECK(cudaEventRecord(end,0));
    CHECK(cudaEventSynchronize(end));
    float timecost_GPU;
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("timecost_cuda:%fms\tcuda core验证程序 Gflops:%lf\n",timecost_GPU,Sparse_mtx->nnz * BN * 2.0 / timecost_GPU / 1e6);

    CHECK(cudaMemcpy(rec_mtx->val,dev_rec_val,sizeof(Data_type) * rec_mtx->nnz,cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    CHECK(cudaFree(dev_col_idx));
    CHECK(cudaFree(dev_row_ptr));
    CHECK(cudaFree(dev_val));
    CHECK(cudaFree(dev_dense_val));
    CHECK(cudaFree(dev_rec_val));
}