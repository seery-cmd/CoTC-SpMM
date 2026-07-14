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

__global__ void SpMM_GPU_kernel_v1(int bN,int row,const int * __restrict__ dev_rowT_ptr,const int * __restrict__ dev_colT_idx,Data_type *dev_valT,const Data_type *dev_dense_val,Data_type *dev_rec_val) {
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
    uint32_t mtx_B[2][2];//B矩阵是2个8 * 16,每个线程存储4个元素,需要32位寄存器2个
    uint32_t mtx_c[2][2];//C矩阵是2个8 * 16,每个线程存储4个元素,需要32位寄存器2个
    //赋值C
    mtx_c[0][0] = 0, mtx_c[0][1] = 0;
    mtx_c[1][0] = 0, mtx_c[1][1] = 0;
    //half2
    half2 temp_b[4];
    //offset
    const unsigned int b_offset1 = laneId_B,b_offset3 = b_offset1 + 8;
    const unsigned int b_offset2 = laneId_B + bn/2,b_offset4 = b_offset2 + 8;
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
            col_idx_0 = dev_colT_idx[(j << 3) + line_b] * bn,col_idx_1 = dev_colT_idx[(j << 3) + line_b + 1] * bn;
            //加载A,__ldg函数,或者ld的PTX语句
            mtx_A[0] = *reinterpret_cast<uint32_t *>(&dev_valT[(j << 6) + (A_addr << 1)]);
            //mma加载b
            temp_b[0].x=dev_dense_val[col_idx_0 + b_offset1];
            temp_b[0].y=dev_dense_val[col_idx_1 + b_offset1];
            temp_b[1].x=dev_dense_val[col_idx_0 + b_offset3];
            temp_b[1].y=dev_dense_val[col_idx_1 + b_offset3];
            temp_b[2].x=dev_dense_val[col_idx_0 + b_offset2];
            temp_b[2].y=dev_dense_val[col_idx_1 + b_offset2];
            temp_b[3].x=dev_dense_val[col_idx_0 + b_offset4];
            temp_b[3].y=dev_dense_val[col_idx_1 + b_offset4];
            mtx_B[0][0] = *reinterpret_cast<uint32_t *>(&temp_b[0]);
            mtx_B[0][1] = *reinterpret_cast<uint32_t *>(&temp_b[1]);
            mtx_B[1][0] = *reinterpret_cast<uint32_t *>(&temp_b[2]);
            mtx_B[1][1] = *reinterpret_cast<uint32_t *>(&temp_b[3]);
            //mma
            mma_m16n8k8(mtx_c[0][0], mtx_c[0][1], mtx_B[0][0], mtx_B[0][1], mtx_A[0]);
            mma_m16n8k8(mtx_c[1][0], mtx_c[1][1], mtx_B[1][0], mtx_B[1][1], mtx_A[0]);
        }
    }
    /** 直接强制转换,与移位效果相同,写会global C**/
    const half2 *bits_1 = reinterpret_cast<half2 *>(&mtx_c[0][0]);
    const half2 *bits_2 = reinterpret_cast<half2 *>(&mtx_c[0][1]);
    const half2 *bits_3 = reinterpret_cast<half2 *>(&mtx_c[1][0]);
    const half2 *bits_4 = reinterpret_cast<half2 *>(&mtx_c[1][1]);
    dev_rec_val[landId_C] = bits_1->x;
    dev_rec_val[landId_C + bn] = bits_1->y;
    dev_rec_val[landId_C + 8] = bits_2->x;
    dev_rec_val[landId_C + bn + 8] = bits_2->y;
    dev_rec_val[landId_C + bn/2] = bits_3->x;
    dev_rec_val[landId_C + bn/2 + bn] = bits_3->y;
    dev_rec_val[landId_C + bn/2 + 8] = bits_4->x;
    dev_rec_val[landId_C + bn/2 + bn + 8] = bits_4->y;
}
void SpMM_GPU(const Matrix_data_CSR *Sparse_mtx,const Matrix_data_CSR *Dense_mtx_CSR,Matrix_data_CSR *rec_mtx)
{
    int epoch = 1;
    /** 格式转换 **/
    Matrix_data_NScore *mtx_nscore = (Matrix_data_NScore *)malloc(sizeof(Matrix_data_NScore));
    trans_format_NScoreV2(Sparse_mtx,mtx_nscore);

    //Printf_matrix_nscore(mtx_nscore);
    //GPU
    //trans_format_NScore_GPU(Sparse_mtx);
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

    dim3 grid_Dim(Sparse_mtx->row/8,1,1);
    dim3 block_Dim(BN,1,1);
    float timecost_GPU;
    /* warm up */
    for(int i=0;i<epoch;i++) {
        SpMM_GPU_kernel_v1<<<grid_Dim, block_Dim>>>(BN, mtx_nscore->row, dev_rowT_ptr, dev_colT_idx, dev_valT,
                                                    dev_dense_val, dev_rec_val);
    }
    cudaDeviceSynchronize();
    /* v1 kernel */
    CHECK(cudaEventRecord(start));

    SpMM_GPU_kernel_v1<<<grid_Dim, block_Dim>>>(BN, mtx_nscore->row, dev_rowT_ptr, dev_colT_idx, dev_valT,
                                                    dev_dense_val, dev_rec_val);

    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));

    printf("timecost_Nscore:%fms\tNscore v1 Gflops:%lf\n",timecost_GPU, Sparse_mtx->nnz / 1e6 / timecost_GPU * BN * 2.0);

//    /** write **/
//    fstream f;
//    f.open("/data/seery/src/log/reslut/suitesparse.txt",ios::out|ios::app);
//    f<<timecost_GPU<<" "<<Sparse_mtx->nnz / 1e6 / timecost_GPU * BN * 2.0<<" ";
//    f.close();

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