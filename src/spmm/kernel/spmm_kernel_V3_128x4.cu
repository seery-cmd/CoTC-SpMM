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

#define mma_m16n8k8(C0,C1,A0,A1,B0) asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};" : "+r"(C0), "+r"(C1) : "r"(A0), "r"(A1),"r"(B0))

__global__ void SpMM_GPU_kernel_v3(int bN,int row,const int * __restrict__ dev_rowT_ptr,const int * __restrict__ dev_colT_idx,Data_type *dev_valT,const Data_type *dev_dense_val,Data_type *dev_rec_val) {
    unsigned int warp_id = threadIdx.x / 32; //0,1,2,3....
    unsigned int A_addr = threadIdx.x % 32;//sparse -- global->寄存器
    unsigned int i = blockIdx.x;
    if(i > row/8)
    {
        return;
    }
    unsigned int j;
    unsigned int bn = bN;
    unsigned int line_b = threadIdx.x % 4 * 2;
    unsigned int laneId_B = threadIdx.x / 4 + warp_id * 8;
    unsigned int landId_C = i * bn * 8 + threadIdx.x % 4 * bn * 2 + laneId_B;

    uint32_t mtx_A[1];//A矩阵是1个8 * 8,每个线程存储2个元素,需要32位寄存器1个
    uint32_t mtx_B[4][2];//B矩阵是2个8 * 16,每个线程存储4个元素,需要32位寄存器2个
    uint32_t mtx_c[4][2];//C矩阵是2个8 * 16,每个线程存储4个元素,需要32位寄存器2个
    //赋值C
    mtx_c[0][0] = 0, mtx_c[0][1] = 0;
    mtx_c[1][0] = 0, mtx_c[1][1] = 0;
    mtx_c[2][0] = 0, mtx_c[2][1] = 0;
    mtx_c[3][0] = 0, mtx_c[3][1] = 0;
    //half2
    half2 temp_b[2][4];
    //offset
    const unsigned int b_offset1 = laneId_B,b_offset2 = b_offset1 + 8;
    const unsigned int b_offset3 = laneId_B + bn/4,b_offset4 = b_offset3 + 8;
    const unsigned int b_offset5 = laneId_B + bn/2,b_offset6 = b_offset5 + 8;
    const unsigned int b_offset7 = laneId_B + bn/4 * 3,b_offset8 = b_offset7 + 8;
    unsigned int col_idx_0 ,col_idx_1;
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
            col_idx_0 = dev_colT_idx[j * 8 + line_b] * bn,col_idx_1 = dev_colT_idx[j * 8 + line_b + 1] * bn;
            //加载A,__ldg函数,或者ld的PTX语句
            mtx_A[0] = *reinterpret_cast<uint32_t *>(&dev_valT[j * 64 + A_addr * 2]);
            //mma加载b
            temp_b[0][0].x=dev_dense_val[col_idx_0 + b_offset1];
            temp_b[0][0].y=dev_dense_val[col_idx_1 + b_offset1];
            temp_b[0][1].x=dev_dense_val[col_idx_0 + b_offset2];
            temp_b[0][1].y=dev_dense_val[col_idx_1 + b_offset2];
            temp_b[0][2].x=dev_dense_val[col_idx_0 + b_offset3];
            temp_b[0][2].y=dev_dense_val[col_idx_1 + b_offset3];
            temp_b[0][3].x=dev_dense_val[col_idx_0 + b_offset4];
            temp_b[0][3].y=dev_dense_val[col_idx_1 + b_offset4];

            temp_b[1][0].x=dev_dense_val[col_idx_0 + b_offset5];
            temp_b[1][0].y=dev_dense_val[col_idx_1 + b_offset5];
            temp_b[1][1].x=dev_dense_val[col_idx_0 + b_offset6];
            temp_b[1][1].y=dev_dense_val[col_idx_1 + b_offset6];
            temp_b[1][2].x=dev_dense_val[col_idx_0 + b_offset7];
            temp_b[1][2].y=dev_dense_val[col_idx_1 + b_offset7];
            temp_b[1][3].x=dev_dense_val[col_idx_0 + b_offset8];
            temp_b[1][3].y=dev_dense_val[col_idx_1 + b_offset8];

            mtx_B[0][0] = *reinterpret_cast<uint32_t *>(&temp_b[0][0]);
            mtx_B[0][1] = *reinterpret_cast<uint32_t *>(&temp_b[0][1]);
            mtx_B[1][0] = *reinterpret_cast<uint32_t *>(&temp_b[0][2]);
            mtx_B[1][1] = *reinterpret_cast<uint32_t *>(&temp_b[0][3]);

            mtx_B[2][0] = *reinterpret_cast<uint32_t *>(&temp_b[1][0]);
            mtx_B[2][1] = *reinterpret_cast<uint32_t *>(&temp_b[1][1]);
            mtx_B[3][0] = *reinterpret_cast<uint32_t *>(&temp_b[1][2]);
            mtx_B[3][1] = *reinterpret_cast<uint32_t *>(&temp_b[1][3]);

            mma_m16n8k8(mtx_c[0][0], mtx_c[0][1], mtx_B[0][0], mtx_B[0][1], mtx_A[0]);
            mma_m16n8k8(mtx_c[1][0], mtx_c[1][1], mtx_B[1][0], mtx_B[1][1], mtx_A[0]);

            mma_m16n8k8(mtx_c[2][0], mtx_c[2][1], mtx_B[2][0], mtx_B[2][1], mtx_A[0]);
            mma_m16n8k8(mtx_c[3][0], mtx_c[3][1], mtx_B[3][0], mtx_B[3][1], mtx_A[0]);
        }
    }
    /** 直接强制转换,与移位效果相同,写会global C**/
    const half2 *bits_1 = reinterpret_cast<half2 *>(&mtx_c[0][0]);
    const half2 *bits_2 = reinterpret_cast<half2 *>(&mtx_c[0][1]);
    const half2 *bits_3 = reinterpret_cast<half2 *>(&mtx_c[1][0]);
    const half2 *bits_4 = reinterpret_cast<half2 *>(&mtx_c[1][1]);

    const half2 *bits_5 = reinterpret_cast<half2 *>(&mtx_c[2][0]);
    const half2 *bits_6 = reinterpret_cast<half2 *>(&mtx_c[2][1]);
    const half2 *bits_7 = reinterpret_cast<half2 *>(&mtx_c[3][0]);
    const half2 *bits_8 = reinterpret_cast<half2 *>(&mtx_c[3][1]);
    dev_rec_val[landId_C] = bits_1->x;
    dev_rec_val[landId_C + bn] = bits_1->y;
    dev_rec_val[landId_C + 8] = bits_2->x;
    dev_rec_val[landId_C + bn + 8] = bits_2->y;
    dev_rec_val[landId_C + bn/4] = bits_3->x;
    dev_rec_val[landId_C + bn/4 + bn] = bits_3->y;
    dev_rec_val[landId_C + bn/4 + 8] = bits_4->x;
    dev_rec_val[landId_C + bn/4 + bn + 8] = bits_4->y;

    dev_rec_val[landId_C + bn/2] = bits_5->x;
    dev_rec_val[landId_C + bn + bn/2] = bits_5->y;
    dev_rec_val[landId_C + 8 + bn/2] = bits_6->x;
    dev_rec_val[landId_C + bn + 8 + bn/2] = bits_6->y;
    dev_rec_val[landId_C + bn/4 * 3] = bits_7->x;
    dev_rec_val[landId_C + bn/4 * 3 + bn] = bits_7->y;
    dev_rec_val[landId_C + bn/4 * 3 + 8] = bits_8->x;
    dev_rec_val[landId_C + bn/4 * 3 + bn + 8] = bits_8->y;
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
    printf("timecost_Nscore:%fms\n",timecost_GPU);
    /* v1 kernel */
    dim3 grid_Dim(Sparse_mtx->row/8,1,1);
    dim3 block_Dim(128,1,1);
    CHECK(cudaEventRecord(start,0));

    SpMM_GPU_kernel_v3<<<grid_Dim, block_Dim>>>(BN, mtx_nscore->row, dev_rowT_ptr, dev_colT_idx, dev_valT,
                                                dev_dense_val, dev_rec_val);
    CHECK(cudaEventRecord(end,0));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("timecost_Nscore:%fms\tNscore v3 Gflops:%lf\n",timecost_GPU,Sparse_mtx->nnz * BN * 2.0 / timecost_GPU / 1e6);


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