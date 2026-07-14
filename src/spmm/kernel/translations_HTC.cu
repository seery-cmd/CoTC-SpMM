#include "../../Struct/struct.h"
#include "../utils/utils.cu"
#include "../../readMtx/utils.h"
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
__device__ int sort(int* arr, int left, int right) {
    int i = left;
    int j = right;
    int temp = arr[left];
    while (i != j) {
        while (temp <= arr[j] && i < j)
            j--;
        while (temp >= arr[i] && i < j)
            i++;
        if (i < j) {
            int t = arr[i];
            arr[i] = arr[j];
            arr[j] = t;
        }
    }
    arr[left] = arr[i];
    arr[i] = temp;
    return i;
}
__device__ void Quick_sort(int *arr, int *stack,unsigned int left,unsigned int right)
{
    int start = left;
    int count = 0;
    if (left < right) {
        stack[start + count] = left;
        stack[start + count + 1] = right;
        count += 2;
    }
    while (count) {
        int r_pop = stack[start + count - 1];
        int l_pop = stack[start + count - 2];
        count -= 2;
        int i = sort(arr, l_pop, r_pop);
        if (l_pop < i - 1) {
            stack[start + count] = l_pop;
            stack[start + count + 1] = i - 1;
            count += 2;
        }
        if (r_pop > i + 1) {
            stack[start + count] = i + 1;
            stack[start + count + 1] = r_pop;
            count += 2;
        }
    }
}
/**
 *
 * @param num : 需要处理的行,按照8行计算,未处理corner
 * @param recode_col : 返回每个8行带有非零元的列值(可复用)
 * @param recode_num : 返回具有非零元的列,其非零元的个数(可复用)
 * @param Mtx_ptr : 偏移
 * @param sort_col : 排序(可复用)
 * @param stack_col : 栈空间,辅助排序
 * @param rowT_ptr : TC block的ptr
 * @param nnz_core : 返回每个block的cuda core非零元
 * @param nnz_tcore : 返回每个block的TC core非零元
 */
__global__ void trans_format_GPU_step1(int num,int *recode_col,int *recode_num,const int *Mtx_ptr,int *sort_col,int *stack_col,
                                 int *rowT_ptr,
                                 int *nnz_tcore,int *nnz_core) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num) {
        return;
    }
    __shared__ int TC_block_num[1];
    __shared__ int CUDA_block_num[1];
    TC_block_num[0] = 0;
    CUDA_block_num[0] = 0;
    __syncthreads();
    const unsigned int start_row = i * ShapeX_row, end_row = (i + 1) * ShapeX_row;//起始行 结束行
    const unsigned int start = Mtx_ptr[start_row], end = Mtx_ptr[end_row];//起始偏移 结束偏移
    unsigned int j;
    int temp = -1;
    int count = -1;
    //init
    for (j = start; j < end; j++) {
        recode_col[j] = 0;
        recode_num[j] = 0;
    }
    /** 排序 **/
    Quick_sort(sort_col, stack_col, start, end - 1);
    /** 记录列向量 **/
    for (j = start; j < end; j++) {
        if(sort_col[j] != temp)
        {
            temp = sort_col[j];
            count++;
            recode_col[start + count] = sort_col[j];
        }
        recode_num[start + count]++;
    }
    /** 记录TC block **/
    int temp_sum = 0;//临时变量
    int get_ColNumber = 0;
    int all_sum = 0;
    for(j=0;j<=count;j++)
    {
        if(recode_num[start + j] >= Col_size)
        {
            temp_sum += recode_num[start + j];
            get_ColNumber++;
            if(get_ColNumber % ShapeX_col == 0)
            {
                all_sum += temp_sum;//组装成功
                temp_sum = 0;
            }
        }
    }
    rowT_ptr[i+1] = get_ColNumber/ShapeX_col;//CPU端还需要进行累加操作
    atomicAdd(&TC_block_num[0], all_sum);
    atomicAdd(&CUDA_block_num[0], end - start - all_sum);
    __syncthreads();

    if (threadIdx.x == 0) {
        nnz_tcore[blockIdx.x] = TC_block_num[0];
        nnz_core[blockIdx.x] = CUDA_block_num[0];
    }
}
__global__ void trans_format_GPU_step2(int num,int *recode_col,int *recode_num,int *tensorCol_record,int *cudaCol_record,Data_type *recode_val,
                                       const int *Mtx_ptr,const int *sort_col,
                                       const int *static_mtx_col,const Data_type *static_mtx_val,
                                       int *CUDA_rowPtr,int *CUDA_colIdx,Data_type *CUDA_Val,
                                       const int *TC_ptr,int *TC_colIdx,Data_type *TC_Val)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num) {
        return;
    }
    //init
    const unsigned int start_row = i * ShapeX_row, end_row = (i + 1) * ShapeX_row;//起始行 结束行
    const unsigned int start = Mtx_ptr[start_row], end = Mtx_ptr[end_row];//起始偏移 结束偏移
    unsigned int j;
    for (j = start; j < end; j++) {
        recode_col[j] = 0;
        recode_num[j] = 0;
    }
    int temp = -1;
    int count = -1;
    /** 记录列向量 (可以优化的地方)**/
    for (j = start; j < end; j++) {
        if(sort_col[j] != temp)
        {
            temp = sort_col[j];
            count++;
            recode_col[start + count] = sort_col[j];
        }
        recode_num[start + count]++;
    }
    int offset_val = (ShapeX_row * ShapeX_col) * TC_ptr[i];
    int offset_col = ShapeX_col * TC_ptr[i];
    unsigned int offset_row;
    int num_Tcore = TC_ptr[i+1] - TC_ptr[i];
    int count_TC = 0;
    int count_CUDA = 0;
    int get_ColNumber = 0;
    int count_CUDA_VAL;
    /** 记录稠密列 and 稀疏列 **/
    for(j=0;j<=count;j++)
    {
        if(recode_num[start + j] > 0 && recode_num[start + j] < Col_size)
        {
            cudaCol_record[start + count_CUDA] = recode_col[start + j];
            count_CUDA++;
        }
        else if(recode_num[start + j] >= Col_size)
        {
            if(get_ColNumber < num_Tcore * ShapeX_col) {
                tensorCol_record[start + count_TC] = recode_col[start + j]; //记录列元素
                count_TC++;
                TC_colIdx[offset_col] = recode_col[start + j]; //先来先选入
                offset_col++;
            }
            // cuda core nnz剩余情况
            else
            {
                cudaCol_record[start + count_CUDA] = recode_col[start + j];
                count_CUDA++;
            }
            get_ColNumber++;
        }
    }
    /** 写入val **/
    for(j=start_row;j<end_row;j++) {
        count_CUDA_VAL = 0;
        //重置
        for(int ii=0;ii<num_Tcore * ShapeX_col;ii++)
        {
            recode_val[start + ii] = 0.0;//重置
        }
        //begin
        for(int k=Mtx_ptr[j];k<Mtx_ptr[j+1];k++)
        {
            //TC
            for(int z = 0;z < count_TC;z++)
            {
                if(static_mtx_col[k] == tensorCol_record[start + z])
                {
                    recode_val[start + z] = static_mtx_val[k];//记录元素按行存储位置
                    break;
                }
            }
            //CUDA
            for(int z = 0;z<count_CUDA;z++)
            {
                if(static_mtx_col[k] == cudaCol_record[start + z])
                {
                    CUDA_Val[Mtx_ptr[j] + count_CUDA_VAL] = static_mtx_val[k];
                    CUDA_colIdx[Mtx_ptr[j] + count_CUDA_VAL] = static_mtx_col[k];
                    count_CUDA_VAL++;
                    break;
                }
            }
        }
        CUDA_rowPtr[j + 1] = count_CUDA_VAL;
        offset_row = (j-start_row) * ShapeX_col;
        //转换
        for(int k=0;k<num_Tcore;k++)
        {
            for(int kk=0;kk<ShapeX_col;kk++)
            {
                TC_Val[offset_val + k*(ShapeX_row * ShapeX_col) + offset_row + kk] = recode_val[start + k*ShapeX_col + kk];
            }
        }
    }
}
/**
 * out put Tensor core TC_rowPtr,TC_colIdx,TC_Val
 * out put CUDA core merge_row,merge_rowPtr,merge_col,merge_val
 */
void trans_format_NScore_GPU(const Matrix_data_CSR *Sparse_mtx)
{
    cudaEvent_t start,end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    //Out Put
    int *TC_rowPtr,*TC_colIdx;
    Data_type *TC_Val;
    int *CUDA_rowPtr,*CUDA_colIdx;
    Data_type *CUDA_Val;
    int *merge_row,*merge_rowPtr,*merge_col;
    Data_type *merge_val;
    int *CUDA_rowPtr_host,*CUDA_colIdx_host;
    Data_type *CUDA_Val_host;
    int merge_count=0;
    int TC_num=0,CUDA_num=0;
    int *TC_num_host = (int *)malloc(sizeof(int)*(Sparse_mtx->row/256/ShapeX_row + 1));
    int *CUDA_num_host = (int *)malloc(sizeof(int)*(Sparse_mtx->row/256/ShapeX_row + 1));
    int *TC_rowPtr_host = (int *)malloc(sizeof(int)*(1 + Sparse_mtx->row/ShapeX_row + (Sparse_mtx->row%ShapeX_row > 0 ? 1 : 0)));
    int *TC_col_host;
    Data_type *TC_val_host;
    //Init
    CHECK(cudaMalloc(&TC_rowPtr,(1 + Sparse_mtx->row/ShapeX_row + (Sparse_mtx->row%ShapeX_row > 0 ? 1 : 0))*sizeof(int)));
    int *TC_num_block;//初始 cuda core 数量
    int *CUDA_num_block;//初始 tc block 数量
    CHECK(cudaMalloc(&TC_num_block,(Sparse_mtx->row/256/ShapeX_row + 1)*sizeof(int)));
    CHECK(cudaMalloc(&CUDA_num_block,(Sparse_mtx->row/256/ShapeX_row + 1)*sizeof(int)));
    //Immediate
    int *recode_col,*recode_num;//记录
    int *Mtx_ptr,*sort_col,*stack_col,*static_Mtx_col;//const Ptr and sort Col
    Data_type *static_mtx_val;

    CHECK(cudaMalloc(&recode_col,Sparse_mtx->nnz*sizeof(int)));
    CHECK(cudaMalloc(&recode_num,Sparse_mtx->nnz*sizeof(int)));
    CHECK(cudaMalloc(&Mtx_ptr,(Sparse_mtx->row+1)*sizeof(int)));
    CHECK(cudaMalloc(&sort_col,Sparse_mtx->nnz*sizeof(int)));
    CHECK(cudaMalloc(&stack_col,Sparse_mtx->nnz*sizeof(int)));
    CHECK(cudaMalloc(&static_Mtx_col,Sparse_mtx->nnz*sizeof(int)));
    CHECK(cudaMalloc(&static_mtx_val,Sparse_mtx->nnz*sizeof(Data_type)));
    CHECK(cudaMemcpy(Mtx_ptr,Sparse_mtx->row_ptr,sizeof(int) * (Sparse_mtx->row+1),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sort_col,Sparse_mtx->col_index,sizeof(int) * Sparse_mtx->nnz,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(stack_col,Sparse_mtx->col_index,sizeof(int) * Sparse_mtx->nnz,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(static_Mtx_col,Sparse_mtx->col_index,sizeof(int) * Sparse_mtx->nnz,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(static_mtx_val,Sparse_mtx->val,sizeof(Data_type) * Sparse_mtx->nnz,cudaMemcpyHostToDevice));

    dim3 girdDimId(Sparse_mtx->row/256/ShapeX_row + 1,1,1);
    dim3 blockDimId(256,1,1);

    float timecost_GPU;
    //返回
    CHECK(cudaEventRecord(start));
    trans_format_GPU_step1<<<girdDimId,blockDimId>>>(Sparse_mtx->row/ShapeX_row,recode_col,recode_num,Mtx_ptr,sort_col,stack_col,
                                               TC_rowPtr,
                                               TC_num_block,CUDA_num_block);
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("step1:%fms\n",timecost_GPU);
    /** 主机端累加num **/
    CHECK(cudaMemcpy(TC_num_host,TC_num_block,sizeof(int) * (Sparse_mtx->row/256/ShapeX_row + 1),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(CUDA_num_host,CUDA_num_block,sizeof(int) * (Sparse_mtx->row/256/ShapeX_row + 1),cudaMemcpyDeviceToHost));
    for(int i = 0; i < (Sparse_mtx->row/256/ShapeX_row + 1);i++)
    {
        TC_num += TC_num_host[i];
        CUDA_num += CUDA_num_host[i];
    }
    /** 主机端返回ptr,完成累加再返回设备端**/
    CHECK(cudaMemcpy(TC_rowPtr_host,TC_rowPtr,sizeof(int) * (1 + Sparse_mtx->row/ShapeX_row + (Sparse_mtx->row%ShapeX_row > 0 ? 1 : 0)),cudaMemcpyDeviceToHost));
    TC_rowPtr_host[0] = 0;
    for(int i=1;i<=(Sparse_mtx->row/ShapeX_row);i++)
    {
        TC_rowPtr_host[i] += TC_rowPtr_host[i-1];
    }
    CHECK(cudaMemcpy(TC_rowPtr,TC_rowPtr_host,sizeof(int) * (1 + Sparse_mtx->row/ShapeX_row + (Sparse_mtx->row%ShapeX_row > 0 ? 1 : 0)),cudaMemcpyHostToDevice));
    /****************** step2 ********************/
    //CUDA core malloc
    CHECK(cudaMalloc(&CUDA_colIdx,Sparse_mtx->nnz*sizeof(int)));
    CHECK(cudaMalloc(&CUDA_Val,Sparse_mtx->nnz*sizeof(Data_type)));
    CHECK(cudaMalloc(&CUDA_rowPtr,(Sparse_mtx->row+1)*sizeof(int)));
    //TC block malloc
    CHECK(cudaMalloc(&TC_colIdx,TC_rowPtr_host[Sparse_mtx->row/ShapeX_row]*ShapeX_col*sizeof(int)));
    CHECK(cudaMalloc(&TC_Val,TC_rowPtr_host[Sparse_mtx->row/ShapeX_row]*ShapeX_col*ShapeX_row*sizeof(Data_type)));
    //record_TC_block_val
    Data_type *record_TC_block_val;
    int *tensorCol_record,*cudaCol_record;
    CHECK(cudaMalloc(&tensorCol_record,Sparse_mtx->nnz*sizeof(int)));
    CHECK(cudaMalloc(&cudaCol_record,Sparse_mtx->nnz*sizeof(int)));
    CHECK(cudaMalloc(&record_TC_block_val,Sparse_mtx->nnz*sizeof(Data_type)));


    CHECK(cudaEventRecord(start));
    trans_format_GPU_step2<<<girdDimId,blockDimId>>>(Sparse_mtx->row/ShapeX_row,recode_col,recode_num,tensorCol_record,cudaCol_record,record_TC_block_val,
                                                     Mtx_ptr,sort_col,
                                                     static_Mtx_col,static_mtx_val,
                                                     CUDA_rowPtr,CUDA_colIdx,CUDA_Val,
                                                     TC_rowPtr,TC_colIdx,TC_Val);

    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("step2:%fms\n",timecost_GPU);

    /** write **/
    fstream f;
    f.open("/data/seery/src/log/reslut/preprogress/CoTC_pre.txt",ios::out|ios::app);
    f<<timecost_GPU<<"\n";
    f.close();

    /** merge **/
    CUDA_rowPtr_host = (int *)malloc(sizeof(int)*(Sparse_mtx->row+1));
    CUDA_colIdx_host = (int *)malloc(sizeof(int)*Sparse_mtx->nnz);
    CUDA_Val_host = (Data_type *)malloc(sizeof(Data_type)*Sparse_mtx->nnz);
    CHECK(cudaMemcpy(CUDA_rowPtr_host,CUDA_rowPtr,sizeof(int) * (Sparse_mtx->row+1),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(CUDA_colIdx_host,CUDA_colIdx,sizeof(int) * Sparse_mtx->nnz,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(CUDA_Val_host,CUDA_Val,sizeof(Data_type) * Sparse_mtx->nnz,cudaMemcpyDeviceToHost));
    int *merge_row_host,*merge_rowPtr_host,*merge_col_host;
    Data_type *merge_val_host;
    merge_row_host = (int *)malloc(sizeof(int)*(Sparse_mtx->row));
    merge_rowPtr_host = (int *)malloc(sizeof(int)*(Sparse_mtx->row + 1));
    merge_col_host = (int *)malloc(sizeof(int)*Sparse_mtx->nnz);
    merge_val_host = (Data_type *)malloc(sizeof(Data_type)*Sparse_mtx->nnz);
    int Ptr_temp=0,all_count_val=0;
    CUDA_rowPtr_host[0]=0;
    merge_rowPtr_host[0]=0;
    for(int i=1;i<=(Sparse_mtx->row / 8 * 8);i++)
    {
        if(CUDA_rowPtr_host[i]!=0)
        {
            Ptr_temp += CUDA_rowPtr_host[i];
            merge_row_host[merge_count] = i - 1;
            merge_rowPtr_host[merge_count+1] = Ptr_temp;
            merge_count++;
            for(int j=0;j<CUDA_rowPtr_host[i];j++)
            {
                merge_val_host[all_count_val]=CUDA_Val_host[Sparse_mtx->row_ptr[i-1]+j];
                merge_col_host[all_count_val]=CUDA_colIdx_host[Sparse_mtx->row_ptr[i-1]+j];
                all_count_val++;
            }
        }
    }


    /** printf **/

    TC_col_host = (int *)malloc(sizeof(int)*TC_rowPtr_host[Sparse_mtx->row/ShapeX_row]*ShapeX_col);
    TC_val_host = (Data_type *)malloc(sizeof(Data_type)*TC_rowPtr_host[Sparse_mtx->row/ShapeX_row]*ShapeX_col*ShapeX_row);

//    CHECK(cudaMemcpy(TC_rowPtr_host,TC_rowPtr,sizeof(int) * (1 + Sparse_mtx->row/ShapeX_row + (Sparse_mtx->row%ShapeX_row > 0 ? 1 : 0)),cudaMemcpyDeviceToHost));
//    CHECK(cudaMemcpy(TC_col_host,TC_colIdx,sizeof(int) * (TC_rowPtr_host[Sparse_mtx->row/ShapeX_row]*ShapeX_col),cudaMemcpyDeviceToHost));
//    CHECK(cudaMemcpy(TC_val_host,TC_Val,sizeof(Data_type) * (TC_rowPtr_host[Sparse_mtx->row/ShapeX_row]*ShapeX_col*ShapeX_row),cudaMemcpyDeviceToHost));


//    printf("****************TCU********************\n");
//    printf("----------------colT_idx-------------------\n");
//    for(int i=0;i<Sparse_mtx->row/ShapeX_row;i++)
//    {
//        printf("第%d行\n",i);
//        for(int j=TC_rowPtr_host[i];j<TC_rowPtr_host[i+1];j++)
//        {
//            for(int k=0;k<ShapeX_col;k++)
//            {
//                printf("%d ",TC_col_host[j*ShapeX_col + k]);
//            }
//            printf("\n");
//        }
//    }
//    printf("----------------rowT_ptr-------------------\n");
//    for(int i = 0;i<=Sparse_mtx->row/ShapeX_row;i++)
//    {
//        printf("%d\n",TC_rowPtr_host[i]);
//    }
//    printf("----------------valT-------------------\n");
//    for(int i=0;i<Sparse_mtx->row/ShapeX_row;i++)
//    {
//        printf("第%d行tensorcore块\n", i);
//        for(int jj=TC_rowPtr_host[i];jj<TC_rowPtr_host[i+1];jj++)
//        {
//            for (int j = 0; j < ShapeX_row; j++) {
//                for (int k = 0; k < ShapeX_col; k++) {
//                    printf("%f ", __half2float(TC_val_host[jj * (ShapeX_row * ShapeX_col) + j * ShapeX_col + k]));
//                }
//                printf("\n");
//            }
//            printf("\n\n");
//        }
//    }
//    printf("****************CUDA CORE********************\n");
//    printf("----------------merge_ptr-------------------\n");
//    for(int i=0;i<=merge_count;i++)
//    {
//        printf("%d=%d\n",i,merge_rowPtr_host[i]);
//    }
//    printf("----------------merge_row-------------------\n");
//    for(int i=0;i<merge_count;i++)
//    {
//        printf("%d=%d\n",i,merge_row_host[i]);
//    }
//    printf("----------------merge_col-------------------\n");
//    for(int i =0;i<merge_count;i++)
//    {
//        for(int j = merge_rowPtr_host[i];j<merge_rowPtr_host[i+1];j++)
//        {
//            printf("<%d %f> ",merge_col_host[j],__half2float(merge_val_host[j]));
//        }
//        printf("\n");
//    }
    free(TC_num_host);
    free(CUDA_num_host);
    free(TC_rowPtr_host);

    free(CUDA_rowPtr_host);
    free(CUDA_colIdx_host);
    free(CUDA_Val_host);

    free(merge_row_host);
    free(merge_rowPtr_host);
    free(merge_col_host);
    free(merge_val_host);

    free(TC_col_host);
    free(TC_val_host);

    cudaFree(CUDA_Val);
    cudaFree(CUDA_rowPtr);
    cudaFree(CUDA_colIdx);

    cudaFree(TC_num_block);
    cudaFree(CUDA_num_block);
    cudaFree(recode_col);
    cudaFree(recode_num);
    cudaFree(Mtx_ptr);
    cudaFree(sort_col);
    cudaFree(stack_col);
    cudaFree(static_Mtx_col);
    cudaFree(static_mtx_val);

    cudaFree(record_TC_block_val);
    cudaFree(tensorCol_record);
    cudaFree(cudaCol_record);
}