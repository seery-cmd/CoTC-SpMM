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

#define LD_mtx_Btrans_m16n8k8(R0,R1,addr)  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "l"(addr))// "=" 表示写入 "+" 表示可读可写

#define CP_ASYNC_CG_16B(dst, src) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;" ::"r"(dst), "l"(src))//有cg和ca的差别,cg只在L2设立cache,ca在l2,l1都设立cache

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;" ::"n"(N))

__global__ void SpMM_GPU_kernel_v2(int bN,int row,const int * __restrict__ dev_rowT_ptr,const int * __restrict__ dev_colT_idx,Data_type *dev_valT,const Data_type *dev_dense_val,Data_type *dev_rec_val) {
    unsigned int warp_id = threadIdx.x / 32; //0,1,2,3....
    unsigned int dense_col_ptr = threadIdx.x / 16;//denseB -- global->shared
    unsigned int dense_row_ptr = threadIdx.x % 16;//denseB -- global->shared
    unsigned int A_addr = threadIdx.x % 32;//sparseA -- global->寄存器
    unsigned int thread_id = threadIdx.x;//线程号
    unsigned int i = blockIdx.x;//排除无用block
    if(i > row/8)
    {
        return;
    }
    unsigned int b;
    const size_t bn = bN;//256,block 128执行双缓冲
//mma中数据量,寄存器
    uint32_t mtx_A[1];
    uint32_t mtx_B[2][4];
    uint32_t mtx_c[2][4];
    mtx_c[0][0]=0.0,mtx_c[0][1]=0.0,mtx_c[0][2]=0.0,mtx_c[0][3]=0.0;
    mtx_c[1][0]=0.0,mtx_c[1][1]=0.0,mtx_c[1][2]=0.0,mtx_c[1][3]=0.0;
//shared空间存储B
    const size_t shared_lane_B = dense_row_ptr % 8 * 128 + dense_row_ptr / 8 * 8 + warp_id * 16;//shared空间B首地址
    __shared__ half shared_B[2][1024];
    uint32_t b_col_addr_0 = __cvta_generic_to_shared(&shared_B[0][thread_id * 8]);
    uint32_t b_col_addr_1 = __cvta_generic_to_shared(&shared_B[1][thread_id * 8]);
//C地址
    unsigned int landId_C1 = i * bn * 8 + thread_id % 4 * bn * 2 + thread_id / 4 + warp_id * 8;//global空间C首地址
    /** 这里执行for循环,把一行的Tcore元素算完,定义的mtx_c位于寄存器 **/
    const unsigned int TcoreNum = dev_rowT_ptr[i + 1] - dev_rowT_ptr[i]; //一个block负责一行
    const unsigned int tcore_start = dev_rowT_ptr[i];
    const unsigned int tcore_end = dev_rowT_ptr[i + 1];
    if (TcoreNum == 0) {
        return;
    }
    else
    {
        /** 流水线start **/
        //前8 * 128
        CP_ASYNC_CG_16B(b_col_addr_0, &dev_dense_val[dev_colT_idx[tcore_start * 8 + dense_col_ptr] * bN +
                                                     dense_row_ptr * 8]);//传输0
        CP_ASYNC_COMMIT_GROUP();//提交异步
        //加载A,__ldg函数,或者ld的PTX语句
        mtx_A[0] = *reinterpret_cast<uint32_t *>(&dev_valT[tcore_start * 64 + A_addr * 2]);
        //后8 *128
        CP_ASYNC_CG_16B(b_col_addr_1, &dev_dense_val[dev_colT_idx[tcore_start * 8 + dense_col_ptr] * bN +
                                                     dense_row_ptr * 8 + bn/2]);//传输1
        CP_ASYNC_COMMIT_GROUP();//提交异步
        //mma,计算shared(0)
        CP_ASYNC_WAIT_GROUP(1);//等待0
        __syncthreads();
        LD_mtx_Btrans_m16n8k8(mtx_B[0][0], mtx_B[0][1], __cvta_generic_to_shared(&shared_B[0][shared_lane_B]));
        LD_mtx_Btrans_m16n8k8(mtx_B[0][2], mtx_B[0][3], __cvta_generic_to_shared(&shared_B[0][shared_lane_B + 64]));
        mma_m16n8k8(mtx_c[0][0], mtx_c[0][1], mtx_B[0][0], mtx_B[0][1], mtx_A[0]);
        mma_m16n8k8(mtx_c[0][2], mtx_c[0][3], mtx_B[0][2], mtx_B[0][3], mtx_A[0]);
        for(b = tcore_start + 1;b<tcore_end;b++)
        {
            //前8 * 128
            CP_ASYNC_CG_16B(b_col_addr_0, &dev_dense_val[dev_colT_idx[b * 8 + dense_col_ptr] * bN +
                                                         dense_row_ptr * 8]);//传输0
            CP_ASYNC_COMMIT_GROUP();//提交异步
            //mma,计算shared(1)
            CP_ASYNC_WAIT_GROUP(1);//等待0
            __syncthreads();
            LD_mtx_Btrans_m16n8k8(mtx_B[1][0], mtx_B[1][1], __cvta_generic_to_shared(&shared_B[1][shared_lane_B]));
            LD_mtx_Btrans_m16n8k8(mtx_B[1][2], mtx_B[1][3], __cvta_generic_to_shared(&shared_B[1][shared_lane_B + 64]));
            mma_m16n8k8(mtx_c[1][0], mtx_c[1][1], mtx_B[1][0], mtx_B[1][1], mtx_A[0]);
            mma_m16n8k8(mtx_c[1][2], mtx_c[1][3], mtx_B[1][2], mtx_B[1][3], mtx_A[0]);

            //后8 * 128
            CP_ASYNC_CG_16B(b_col_addr_1, &dev_dense_val[dev_colT_idx[b * 8 + dense_col_ptr] * bN +
                                                         dense_row_ptr * 8 + bn/2]);//传输0
            CP_ASYNC_COMMIT_GROUP();//提交异步

            //加载A,__ldg函数,或者ld的PTX语句
            mtx_A[0] = *reinterpret_cast<uint32_t *>(&dev_valT[b * 64 + A_addr * 2]);

            //mma,计算shared(0)
            CP_ASYNC_WAIT_GROUP(1);//等待0
            __syncthreads();
            LD_mtx_Btrans_m16n8k8(mtx_B[0][0], mtx_B[0][1], __cvta_generic_to_shared(&shared_B[0][shared_lane_B]));
            LD_mtx_Btrans_m16n8k8(mtx_B[0][2], mtx_B[0][3], __cvta_generic_to_shared(&shared_B[0][shared_lane_B + 64]));
            mma_m16n8k8(mtx_c[0][0], mtx_c[0][1], mtx_B[0][0], mtx_B[0][1], mtx_A[0]);
            mma_m16n8k8(mtx_c[0][2], mtx_c[0][3], mtx_B[0][2], mtx_B[0][3], mtx_A[0]);
        }
        //mma,计算shared(1)
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
        LD_mtx_Btrans_m16n8k8(mtx_B[1][0], mtx_B[1][1], __cvta_generic_to_shared(&shared_B[1][shared_lane_B]));
        LD_mtx_Btrans_m16n8k8(mtx_B[1][2], mtx_B[1][3], __cvta_generic_to_shared(&shared_B[1][shared_lane_B + 64]));
        mma_m16n8k8(mtx_c[1][0], mtx_c[1][1], mtx_B[1][0], mtx_B[1][1], mtx_A[0]);
        mma_m16n8k8(mtx_c[1][2], mtx_c[1][3], mtx_B[1][2], mtx_B[1][3], mtx_A[0]);
        //C 寄存器 -> global
        const half2 *bits_1_0 = reinterpret_cast<half2 *>(&mtx_c[0][0]);
        const half2 *bits_2_0 = reinterpret_cast<half2 *>(&mtx_c[0][1]);
        const half2 *bits_3_0 = reinterpret_cast<half2 *>(&mtx_c[0][2]);
        const half2 *bits_4_0 = reinterpret_cast<half2 *>(&mtx_c[0][3]);

        const half2 *bits_1_1 = reinterpret_cast<half2 *>(&mtx_c[1][0]);
        const half2 *bits_2_1 = reinterpret_cast<half2 *>(&mtx_c[1][1]);
        const half2 *bits_3_1 = reinterpret_cast<half2 *>(&mtx_c[1][2]);
        const half2 *bits_4_1 = reinterpret_cast<half2 *>(&mtx_c[1][3]);
        dev_rec_val[landId_C1] = bits_1_0->x;
        dev_rec_val[landId_C1 + bn] = bits_1_0->y;
        dev_rec_val[landId_C1 + 8] = bits_2_0->x;
        dev_rec_val[landId_C1 + bn + 8] = bits_2_0->y;
        dev_rec_val[landId_C1 + 64] = bits_3_0->x;
        dev_rec_val[landId_C1 + bn + 64] = bits_3_0->y;
        dev_rec_val[landId_C1 + 72] = bits_4_0->x;
        dev_rec_val[landId_C1 + bn + 72] = bits_4_0->y;

        dev_rec_val[landId_C1 + bn/2] = bits_1_1->x;
        dev_rec_val[landId_C1 + bn + bn/2] = bits_1_1->y;
        dev_rec_val[landId_C1 + 8 + bn/2] = bits_2_1->x;
        dev_rec_val[landId_C1 + bn + 8 + bn/2] = bits_2_1->y;
        dev_rec_val[landId_C1 + 64 + bn/2] = bits_3_1->x;
        dev_rec_val[landId_C1 + bn + 64 + bn/2] = bits_3_1->y;
        dev_rec_val[landId_C1 + 72 + bn/2] = bits_4_1->x;
        dev_rec_val[landId_C1 + bn + 72 + bn/2] = bits_4_1->y;
    }
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
    CHECK(cudaMalloc(&dev_colT_idx,mtx_nscore->nnz_tcore * 8 * sizeof(int)));
    CHECK(cudaMalloc(&dev_rowT_ptr,(1 + mtx_nscore->row/ShapeX_row + (mtx_nscore->row%ShapeX_row > 0 ? 1 : 0))*sizeof(int)));
    CHECK(cudaMalloc(&dev_valT,mtx_nscore->nnz_tcore * 64 * sizeof(Data_type)));
    //dense matrix分配内存
    Data_type *dev_dense_val;
    CHECK(cudaMalloc(&dev_dense_val,Dense_mtx_CSR->nnz * sizeof(Data_type)));
    //rec matrix分配内存
    Data_type *dev_rec_val;
    CHECK(cudaMalloc(&dev_rec_val,rec_mtx->nnz * sizeof(Data_type)));

    /* 数据传输GPU */
    CHECK(cudaMemcpy(dev_colT_idx,mtx_nscore->colT_ide,sizeof(int) * mtx_nscore->nnz_tcore * 8,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_rowT_ptr,mtx_nscore->rowT_ptr,sizeof(int) * (1 + mtx_nscore->row/ShapeX_row + (mtx_nscore->row%ShapeX_row > 0 ? 1 : 0)),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_valT,mtx_nscore->valT,sizeof(Data_type) * mtx_nscore->nnz_tcore * 64,cudaMemcpyHostToDevice));

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

    SpMM_GPU_kernel_v2<<<grid_Dim, block_Dim>>>(BN, mtx_nscore->row, dev_rowT_ptr, dev_colT_idx, dev_valT,
                                                dev_dense_val, dev_rec_val);
    CHECK(cudaEventRecord(end,0));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("timecost_Nscore:%fms\tNscore v2 Gflops:%lf\n",timecost_GPU,Sparse_mtx->nnz / 1e6 * BN * 2.0 / timecost_GPU);

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
/** kernel内核 **/
//        for(z=0;z<2;z++) {
//            mtx_c[0] = 0,mtx_c[1] = 0,mtx_c[2] = 0,mtx_c[3] = 0;
//            for (j = tcore_start; j < tcore_end; j++) {
//                col_idx_0 = dev_colT_idx[j * 8 + line_b] * bn,col_idx_1 = dev_colT_idx[j * 8 + line_b + 1] * bn;
//                //加载A,__ldg函数,或者ld的PTX语句
//                mtx_A[0] = *reinterpret_cast<uint32_t *>(&dev_valT[j * 64 + A_addr * 2]);
//                //mma加载b0
//                temp_b[0].x = dev_dense_val[col_idx_0 + b_offset1];
//                temp_b[0].y = dev_dense_val[col_idx_1 + b_offset1];
//                temp_b[1].x = dev_dense_val[col_idx_0 + b_offset1 + 8];
//                temp_b[1].y = dev_dense_val[col_idx_1 + b_offset1 + 8];
//                temp_b[2].x = dev_dense_val[col_idx_0 + b_offset1 + 64];
//                temp_b[2].y = dev_dense_val[col_idx_1 + b_offset1 + 64];
//                temp_b[3].x = dev_dense_val[col_idx_0 + b_offset1 + 72];
//                temp_b[3].y = dev_dense_val[col_idx_1 + b_offset1 + 72];
//                mtx_B[0] = *reinterpret_cast<uint32_t *>(&temp_b[0]);
//                mtx_B[1] = *reinterpret_cast<uint32_t *>(&temp_b[1]);
//                mtx_B[2] = *reinterpret_cast<uint32_t *>(&temp_b[2]);
//                mtx_B[3] = *reinterpret_cast<uint32_t *>(&temp_b[3]);
//                mma_m16n8k8(mtx_c[0], mtx_c[1], mtx_B[0], mtx_B[1], mtx_A[0]);
//                mma_m16n8k8(mtx_c[2], mtx_c[3], mtx_B[2], mtx_B[3], mtx_A[0]);
//            }
//            const half2 *bits_1 = reinterpret_cast<half2 *>(&mtx_c[0]);
//            const half2 *bits_2 = reinterpret_cast<half2 *>(&mtx_c[1]);
//            const half2 *bits_3 = reinterpret_cast<half2 *>(&mtx_c[2]);
//            const half2 *bits_4 = reinterpret_cast<half2 *>(&mtx_c[3]);
//            dev_rec_val[landId_C1] = bits_1->x;
//            dev_rec_val[landId_C1 + bn] = bits_1->y;
//            dev_rec_val[landId_C1 + 8] = bits_2->x;
//            dev_rec_val[landId_C1 + bn + 8] = bits_2->y;
//            dev_rec_val[landId_C1 + 64] = bits_3->x;
//            dev_rec_val[landId_C1 + bn + 64] = bits_3->y;
//            dev_rec_val[landId_C1 + 72] = bits_4->x;
//            dev_rec_val[landId_C1 + bn + 72] = bits_4->y;
//            b_offset1 += bn / 2;
//            landId_C1 += bn / 2;
//        }