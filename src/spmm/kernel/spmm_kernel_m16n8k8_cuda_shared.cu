#include "translations_HTC.cu"
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
#define WARP_SIZE 32

#define mma_m16n8k8(C0,C1,A0,A1,B0) asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};" : "+r"(C0), "+r"(C1) : "r"(A0), "r"(A1),"r"(B0))

#define LD_mtx_Btrans_m16n8k8(R0,R1,addr)  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "l"(addr))// "=" 表示写入 "+" 表示可读可写

#define CP_ASYNC_CG_16B(dst, src) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" ::"r"(dst), "l"(src))//有cg和ca的差别,cg只在L2设立cache,ca在l2,l1都设立cache

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;" ::"n"(N))

__global__ void SpMM_GPU_kernel_v6(int bN,int row,const int * __restrict__ dev_rowT_ptr,const int * __restrict__ dev_colT_idx,Data_type *dev_valT,const Data_type *dev_dense_val,Data_type *dev_rec_val) {
    unsigned int i = blockIdx.x;
    if(i > row/8)
    {
        return;
    }
    unsigned int warp_id = threadIdx.x / 32; //0,1,2,3....
    unsigned int thread_32 = threadIdx.x % 32;//sparse -- global->寄存器
    uint32_t mtx_A;//A矩阵是1个8 * 8,每个线程存储2个元素,需要32位寄存器1个
    uint32_t mtx_B[4];//B矩阵是2个8 * 16,每个线程存储4个元素,需要32位寄存器2个
    uint32_t mtx_c[4];//C矩阵是2个8 * 16,每个线程存储4个元素,需要32位寄存器2个
    //赋值C
    mtx_c[0] = 0, mtx_c[1] = 0;
    mtx_c[2] = 0, mtx_c[3] = 0;
    size_t landId_B = (threadIdx.x % 4 * bN << 1) + (threadIdx.x >> 2) + (warp_id << 3);
    size_t landId_C = (i * bN << 3) + landId_B;
    /** shared **/
    __shared__ Data_type shared_mtxB[2][2048];
    size_t shared_IdB = (thread_32 % 8 << 8) + (thread_32 >> 3 << 3) + (warp_id << 4);
    uint32_t b_shared_addr; //控制shared位置 0 & 1
    /** core kernel **/
    int j = dev_rowT_ptr[i], tcore_end = dev_rowT_ptr[i + 1];
    int switch_id = 0;// 0 & 1
    if (dev_rowT_ptr[i + 1] == dev_rowT_ptr[i]) {
        return;
    }
    else {
        //global -> shared
        b_shared_addr = __cvta_generic_to_shared(&shared_mtxB[switch_id][threadIdx.x << 3]);
        CP_ASYNC_CG_16B(b_shared_addr,
                        &dev_dense_val[dev_colT_idx[(j << 3) + warp_id] * bN + (thread_32 << 3)]);
        CP_ASYNC_COMMIT_GROUP();
        //A
        mtx_A = *reinterpret_cast<uint32_t *>(&dev_valT[(j << 6) + (thread_32 << 1)]);
        while(j < tcore_end - 1){
            j++;
            //global -> shared
            switch_id ^= 1;
            b_shared_addr = __cvta_generic_to_shared(&shared_mtxB[switch_id][threadIdx.x << 3]);
            CP_ASYNC_CG_16B(b_shared_addr,
                            &dev_dense_val[dev_colT_idx[(j << 3) + warp_id] * bN + (thread_32 << 3)]);
            CP_ASYNC_COMMIT_GROUP();
            //shared -> 寄存器 加载B
            CP_ASYNC_WAIT_GROUP(1);//等待0
            __syncthreads();
            LD_mtx_Btrans_m16n8k8(mtx_B[0], mtx_B[1], __cvta_generic_to_shared(&shared_mtxB[switch_id ^ 1][shared_IdB]));
            LD_mtx_Btrans_m16n8k8(mtx_B[2], mtx_B[3], __cvta_generic_to_shared(&shared_mtxB[switch_id ^ 1][shared_IdB + bN/2]));
            //mma
            mma_m16n8k8(mtx_c[0], mtx_c[1], mtx_B[0], mtx_B[1], mtx_A);
            mma_m16n8k8(mtx_c[2], mtx_c[3], mtx_B[2], mtx_B[3], mtx_A);
            mtx_A = *reinterpret_cast<uint32_t *>(&dev_valT[(j << 6) + (thread_32 << 1)]);
        }
        CP_ASYNC_WAIT_GROUP(0);//等待0
        __syncthreads();
        LD_mtx_Btrans_m16n8k8(mtx_B[0], mtx_B[1], __cvta_generic_to_shared(&shared_mtxB[switch_id][shared_IdB]));
        LD_mtx_Btrans_m16n8k8(mtx_B[2], mtx_B[3], __cvta_generic_to_shared(&shared_mtxB[switch_id][shared_IdB + bN/2]));
        //mma
        mma_m16n8k8(mtx_c[0], mtx_c[1], mtx_B[0], mtx_B[1], mtx_A);
        mma_m16n8k8(mtx_c[2], mtx_c[3], mtx_B[2], mtx_B[3], mtx_A);
    }
    /** 直接强制转换,与移位效果相同,写会global C**/
    const half2 *bits_1 = reinterpret_cast<half2 *>(&mtx_c[0]);
    const half2 *bits_2 = reinterpret_cast<half2 *>(&mtx_c[1]);
    const half2 *bits_3 = reinterpret_cast<half2 *>(&mtx_c[2]);
    const half2 *bits_4 = reinterpret_cast<half2 *>(&mtx_c[3]);
    dev_rec_val[landId_C] = bits_1->x;
    dev_rec_val[landId_C + bN] = bits_1->y;
    dev_rec_val[landId_C + 8] = bits_2->x;
    dev_rec_val[landId_C + bN + 8] = bits_2->y;
    dev_rec_val[landId_C + (bN >> 1)] = bits_3->x;
    dev_rec_val[landId_C + (bN >> 1) + bN] = bits_3->y;
    dev_rec_val[landId_C + (bN >> 1) + 8] = bits_4->x;
    dev_rec_val[landId_C + (bN >> 1) + bN + 8] = bits_4->y;
}
__global__ void SpMM_GPU_kernel_CUDA(int bN,const int * __restrict__ dev_row_merge,const int * __restrict__ dev_row_ptr,const int * __restrict__ dev_col_idx,Data_type *dev_val,const Data_type *dev_dense_val,Data_type *dev_rec_val)
{
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;
    Data_type temp = 0.0;
    for(int i = dev_row_ptr[row];i<dev_row_ptr[row + 1];i++)
    {
        temp += dev_val[i] * dev_dense_val[dev_col_idx[i]*bN + col];
    }
    dev_rec_val[dev_row_merge[blockIdx.x]*bN + col] = temp;
}
void SpMM_GPU(const Matrix_data_CSR *Sparse_mtx,const Matrix_data_CSR *Dense_mtx_CSR,Matrix_data_CSR *rec_mtx)
{
    /** 格式转换 **/
    Matrix_data_NScore *mtx_nscore = (Matrix_data_NScore *)malloc(sizeof(Matrix_data_NScore));
    trans_format_NScoreV3(Sparse_mtx,mtx_nscore);
//    Printf_matrix_nscore(mtx_nscore);
    printf("cuda core=%d,TCU=%d\n",mtx_nscore->nnz_core,Sparse_mtx->nnz - mtx_nscore->nnz_core);
    /** 事件计时 **/
    cudaEvent_t start,end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    /* GPU内存开辟 */
    //nscore TCU格式分配内存
    int *dev_col_idx_TCU,*dev_row_ptr_TCU;
    Data_type *dev_val_TCU;
    CHECK(cudaMalloc(&dev_col_idx_TCU,mtx_nscore->nnz_tcore * 8 * sizeof(int)));
    CHECK(cudaMalloc(&dev_row_ptr_TCU,(1 + mtx_nscore->row/ShapeX_row + (mtx_nscore->row%ShapeX_row > 0 ? 1 : 0))*sizeof(int)));
    CHECK(cudaMalloc(&dev_val_TCU,mtx_nscore->nnz_tcore * 64 * sizeof(Data_type)));

    //nscore cuda core 格式分配内存
    int *dev_col_idx_CUDA,*dev_row_ptr_CUDA_merge,*dev_row_merge;
    Data_type *dev_val_CUDA;
    CHECK(cudaMalloc(&dev_col_idx_CUDA,mtx_nscore->nnz_core * sizeof(int)));
    CHECK(cudaMalloc(&dev_row_ptr_CUDA_merge,(1 + mtx_nscore->merge_row_count)*sizeof(int)));
    CHECK(cudaMalloc(&dev_row_merge,mtx_nscore->merge_row_count*sizeof(int)));
    CHECK(cudaMalloc(&dev_val_CUDA,mtx_nscore->nnz_core * sizeof(Data_type)));

    //dense matrix分配内存
    Data_type *dev_dense_val_TCU,*dev_dense_val_CUDA;
    CHECK(cudaMalloc(&dev_dense_val_TCU,Dense_mtx_CSR->nnz * sizeof(Data_type)));
    CHECK(cudaMalloc(&dev_dense_val_CUDA,Dense_mtx_CSR->nnz * sizeof(Data_type)));
    //rec matrix分配内存
    Data_type *dev_rec_val_TCU,*dev_rec_val_CUDA;
    CHECK(cudaMalloc(&dev_rec_val_TCU,rec_mtx->nnz * sizeof(Data_type)));
    CHECK(cudaMalloc(&dev_rec_val_CUDA,rec_mtx->nnz * sizeof(Data_type)));
    /** 数据传输GPU **/
    //TCU
    CHECK(cudaMemcpy(dev_col_idx_TCU,mtx_nscore->colT_ide,sizeof(int) * mtx_nscore->nnz_tcore * 8,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_row_ptr_TCU,mtx_nscore->rowT_ptr,sizeof(int) * (1 + mtx_nscore->row/ShapeX_row + (mtx_nscore->row%ShapeX_row > 0 ? 1 : 0)),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_val_TCU,mtx_nscore->valT,sizeof(Data_type) * mtx_nscore->nnz_tcore * 64,cudaMemcpyHostToDevice));
    //CUDA
    CHECK(cudaMemcpy(dev_col_idx_CUDA,mtx_nscore->col_index,sizeof(int) * mtx_nscore->nnz_core,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_row_ptr_CUDA_merge,mtx_nscore->merge_row_ptr,sizeof(int) * (1 + mtx_nscore->merge_row_count),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_val_CUDA,mtx_nscore->val,sizeof(Data_type) * mtx_nscore->nnz_core,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_row_merge,mtx_nscore->merge_row,sizeof(int) * mtx_nscore->merge_row_count,cudaMemcpyHostToDevice));
    //dense B
    CHECK(cudaMemcpy(dev_dense_val_TCU,Dense_mtx_CSR->val,sizeof(Data_type) * Dense_mtx_CSR->nnz,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_dense_val_CUDA,Dense_mtx_CSR->val,sizeof(Data_type) * Dense_mtx_CSR->nnz,cudaMemcpyHostToDevice));
    //rec C
    CHECK(cudaMemcpy(dev_rec_val_TCU,rec_mtx->val,sizeof(Data_type) * rec_mtx->nnz,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_rec_val_CUDA,rec_mtx->val,sizeof(Data_type) * rec_mtx->nnz,cudaMemcpyHostToDevice));

    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    float timecost_GPU;
    /* v6 kernel */
    dim3 grid_Dim_TCU(Sparse_mtx->row / 8,1,1);
    dim3 block_Dim_TCU(BN,1,1);

    dim3 grid_Dim_CUDA(mtx_nscore->merge_row_count,1,1);
    dim3 block_Dim_CUDA(BN,1,1);
    SpMM_GPU_kernel_v6<<<grid_Dim_TCU, block_Dim_TCU>>>(BN, mtx_nscore->row, dev_row_ptr_TCU, dev_col_idx_TCU, dev_val_TCU,dev_dense_val_TCU, dev_rec_val_TCU);
    cudaDeviceSynchronize();
    //TCU
    CHECK(cudaEventRecord(start));
    SpMM_GPU_kernel_v6<<<grid_Dim_TCU, block_Dim_TCU>>>(BN, mtx_nscore->row, dev_row_ptr_TCU, dev_col_idx_TCU, dev_val_TCU,dev_dense_val_TCU, dev_rec_val_TCU);
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("warm up time TCU:%fms\n",timecost_GPU);

//    /** write **/
//    fstream f;
//    f.open("/data/seery/src/log/reslut/3_251113.txt",ios::out|ios::app);
//    f<<timecost_GPU<<" ";
//    f.close();

    //CUDA
    CHECK(cudaEventRecord(start));
    SpMM_GPU_kernel_CUDA<<<grid_Dim_CUDA, block_Dim_CUDA>>>(BN, dev_row_merge,dev_row_ptr_CUDA_merge, dev_col_idx_CUDA,dev_val_CUDA,dev_dense_val_CUDA, dev_rec_val_CUDA);
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("warm up time CUDA:%fms\n",timecost_GPU);

//    /** write **/
//    f.open("/data/seery/src/log/reslut/3_251113.txt",ios::out|ios::app);
//    f<<timecost_GPU<<" ";
//    f.close();

    //TCU + CUDA
    CHECK(cudaEventRecord(start));
    SpMM_GPU_kernel_v6<<<grid_Dim_TCU, block_Dim_TCU,0,stream1>>>(BN, mtx_nscore->row, dev_row_ptr_TCU, dev_col_idx_TCU, dev_val_TCU,dev_dense_val_TCU, dev_rec_val_TCU);

    SpMM_GPU_kernel_CUDA<<<grid_Dim_CUDA, block_Dim_CUDA,0,stream2>>>(BN, dev_row_merge,dev_row_ptr_CUDA_merge, dev_col_idx_CUDA,dev_val_CUDA,dev_dense_val_CUDA, dev_rec_val_CUDA);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
//    cudaDeviceSynchronize();
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("timecost_Nscore:%fms\tNscore v6 Gflops:%lf\n",timecost_GPU,Sparse_mtx->nnz / 1e6 / timecost_GPU * BN * 2.0);

//    f.open("/data/seery/src/log/reslut/3_251113.txt",ios::out|ios::app);
//    f<<timecost_GPU<<" "<<Sparse_mtx->nnz / 1e6 / timecost_GPU * BN * 2.0<<" ";
//    f.close();

    Data_type *rec_TCU,*rec_CUDA;
    rec_TCU = (Data_type *)malloc(sizeof (Data_type) * rec_mtx->nnz);
    rec_CUDA = (Data_type *)malloc(sizeof (Data_type) * rec_mtx->nnz);
    CHECK(cudaMemcpy(rec_TCU,dev_rec_val_TCU,sizeof(Data_type) * rec_mtx->nnz,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(rec_CUDA,dev_rec_val_CUDA,sizeof(Data_type) * rec_mtx->nnz,cudaMemcpyDeviceToHost));

    for(int i = 0 ;i < rec_mtx->nnz;i++)
    {
        rec_mtx->val[i] =__float2half(__half2float(rec_TCU[i]) + __half2float(rec_CUDA[i]));
    }
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    free(mtx_nscore);
    free(rec_TCU);
    free(rec_CUDA);
    CHECK(cudaFree(dev_col_idx_TCU));
    CHECK(cudaFree(dev_row_ptr_TCU));
    CHECK(cudaFree(dev_val_TCU));

    CHECK(cudaFree(dev_col_idx_CUDA));
    CHECK(cudaFree(dev_row_ptr_CUDA_merge));
    CHECK(cudaFree(dev_val_CUDA));

    CHECK(cudaFree(dev_dense_val_TCU));
    CHECK(cudaFree(dev_dense_val_CUDA));

    CHECK(cudaFree(dev_rec_val_TCU));
    CHECK(cudaFree(dev_rec_val_CUDA));
}