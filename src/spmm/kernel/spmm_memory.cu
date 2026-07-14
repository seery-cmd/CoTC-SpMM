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
    /** 格式转换 **/
//    Matrix_data_NScore *mtx_nscore_DTC = (Matrix_data_NScore *)malloc(sizeof(Matrix_data_NScore));
//
//    trans_format_NScoreVDTC(Sparse_mtx,mtx_nscore_DTC);
//    printf("DTC=CSR NNZ:%d,TC memory:%d\n",Sparse_mtx->nnz,mtx_nscore_DTC->rowT_ptr[Sparse_mtx->row/16]*16*ShapeX_col);
//
//    Matrix_data_NScore *mtx_nscore_flash = (Matrix_data_NScore *)malloc(sizeof(Matrix_data_NScore));
//    trans_format_NScoreV2(Sparse_mtx,mtx_nscore_flash);
//    printf("flashsparse=CSR NNZ:%d,TC memory:%d\n",Sparse_mtx->nnz,mtx_nscore_flash->rowT_ptr[Sparse_mtx->row/ShapeX_row]*ShapeX_row*ShapeX_col);
//
//    Matrix_data_NScore *mtx_nscore_COTC = (Matrix_data_NScore *)malloc(sizeof(Matrix_data_NScore));
//    trans_format_NScoreV3(Sparse_mtx,mtx_nscore_COTC);
//    printf("COTC=CSR NNZ:%d,TC memory:%d,CUDA memory:%d\n",Sparse_mtx->nnz,
//           mtx_nscore_COTC->rowT_ptr[Sparse_mtx->row/ShapeX_row]*ShapeX_row*ShapeX_col,mtx_nscore_COTC->merge_row_ptr[mtx_nscore_COTC->merge_row_count]);
    trans_format_NScore_GPU(Sparse_mtx);
//    /** write **/
//    fstream f;
//    f.open("/data/seery/src/log/reslut/memory/memory.txt",ios::out|ios::app);
//    f<<mtx_nscore_DTC->rowT_ptr[Sparse_mtx->row/16]*16*ShapeX_col<<" "<<mtx_nscore_flash->rowT_ptr[Sparse_mtx->row/ShapeX_row]*ShapeX_row*ShapeX_col<<" "<<mtx_nscore_COTC->rowT_ptr[Sparse_mtx->row/ShapeX_row]*ShapeX_row*ShapeX_col<<" "<<mtx_nscore_COTC->merge_row_ptr[mtx_nscore_COTC->merge_row_count]<<"\n";
//    f.close();


//    free(mtx_nscore_flash);
//    free(mtx_nscore_COTC);
}