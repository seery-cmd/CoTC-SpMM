#include "../../Struct/struct.h"

/**
 * SpMM_CPU 计算方式: row * col
 */
void SpMM_cpu(const Matrix_data_CSR *Sparse_mtx,const Matrix_data_CSR *Dense_mtx_csr,Matrix_data_CSR *rec_mtx)
{
    int start_sp,end_sp;
    float temp;
    for(int i=0;i<Sparse_mtx->row;i++)
    {
        start_sp = Sparse_mtx->row_ptr[i],end_sp = Sparse_mtx->row_ptr[i+1];
        for(int k = 0;k<Dense_mtx_csr->col;k++) {
            temp = (float) 0.0;
            for (int j = start_sp; j < end_sp; j++) {
                temp += __half2float(Sparse_mtx->val[j]) * __half2float(Dense_mtx_csr->val[k + Sparse_mtx->col_index[j] * BN]);
            }
            rec_mtx->val[i * BN + k] = __float2half(temp);
        }
    }
}
/**
 * 转换矩阵的格式 csr to csc
 * **/
void trans_format_CSRtoCSC(Matrix_data_CSR *Dense_mtx_csr,Matrix_data_CSC *Dense_mtx_csc)
{
    Dense_mtx_csc->row = Dense_mtx_csr->row;
    Dense_mtx_csc->col = Dense_mtx_csr->col;
    Dense_mtx_csc->nnz = Dense_mtx_csr->nnz;
    Dense_mtx_csc->col_ptr = (int *)malloc(sizeof(int)*(Dense_mtx_csc->col + 1));
    Dense_mtx_csc->row_index = (int *)malloc(sizeof(int)*Dense_mtx_csc->nnz);
    Dense_mtx_csc->val = (Data_type *)malloc(sizeof(Data_type)*Dense_mtx_csc->nnz);
    Dense_mtx_csc->col_ptr[0] = 0;
    int startRow,endRow;
    int count = 0;
    for(int i =0;i<Dense_mtx_csc->col;i++)
    {
        for(int j=0;j<Dense_mtx_csr->row;j++)
        {
            startRow = Dense_mtx_csr->row_ptr[j],endRow = Dense_mtx_csr->row_ptr[j+1];
            for(int z=startRow;z<endRow;z++)
            {
                if(Dense_mtx_csr->col_index[z] < i)
                {
                    continue;
                }
                else if(Dense_mtx_csr->col_index[z] == i)
                {
                    Dense_mtx_csc->val[count] = Dense_mtx_csr->val[z];
                    Dense_mtx_csc->row_index[count] = j;
                    count++;
                }
                else
                {
                    break;
                }
            }
        }
        Dense_mtx_csc->col_ptr[i + 1] = count;
    }
}
/**
 * 格式转换GPU专用格式
 */
void trans_format_NScoreV2(const Matrix_data_CSR *Sparse_mtx,Matrix_data_NScore *mtx_nscore)
{
    mtx_nscore->row = Sparse_mtx->row;
    mtx_nscore->col = Sparse_mtx->col;
    mtx_nscore->nnz_all = Sparse_mtx->nnz;
    mtx_nscore->row_ptr = (int *)malloc(sizeof(int) * (1 + mtx_nscore->row));
    mtx_nscore->rowT_ptr = (int *)malloc(sizeof (int)* (1 + mtx_nscore->row/ShapeX_row + (mtx_nscore->row%ShapeX_row > 0 ? 1 : 0)));
    mtx_nscore->row_ptr[0] = 0;
    mtx_nscore->rowT_ptr[0] = 0;
    mtx_nscore->nnz_core = Sparse_mtx->nnz;//初始
    mtx_nscore->nnz_tcore = 0;//初始

    int *colNumber_record = (int *)malloc(sizeof(int) * mtx_nscore->col);//记录列有多少元素

    int start_row,end_row;
    int get_ColNumber;
    int b = mtx_nscore->row/ShapeX_row + (mtx_nscore->row%ShapeX_row > 0 ? 1 : 0);//按照8-4-8行的方式进行筛选
    start_row = 0,end_row = ShapeX_row;
    //第一遍统计
    for(int z=0;z<b;z++)
    {
        //重置
        for(int i=0;i<Sparse_mtx->col;i++)
        {
            colNumber_record[i] = 0;
        }
        get_ColNumber = 0;
        //记录元素
        for(int i=start_row;i<end_row;i++)
        {
            for(int j=Sparse_mtx->row_ptr[i];j<Sparse_mtx->row_ptr[i+1];j++)
            {
                colNumber_record[Sparse_mtx->col_index[j]]++;
            }
        }
        //统计列量
        for(int i=0;i<Sparse_mtx->col;i++)
        {
            if(colNumber_record[i] >= 1)
            {
                mtx_nscore->nnz_core -= colNumber_record[i];
                get_ColNumber++;
            }
        }
        mtx_nscore->nnz_tcore += (get_ColNumber/ShapeX_col + (get_ColNumber % ShapeX_col> 0 ? 1 : 0));//有剩余情况,剩余情况提交给TCU
        mtx_nscore->rowT_ptr[z+1] = mtx_nscore->nnz_tcore;
        start_row += ShapeX_row;
        end_row += ShapeX_row;
        if(z == b - 2)//行尾巴不存了
        {
            end_row = Sparse_mtx->row;
        }
    }
    //为cuda core 分配内存


    /** 为tensor core 分配内存 **/
    mtx_nscore->valT = (Data_type *)malloc(sizeof(Data_type) * mtx_nscore->nnz_tcore * (ShapeX_row * ShapeX_col));
    //多余部分也要设置为0
    mtx_nscore->colT_ide = (int *)malloc(sizeof(int) * mtx_nscore->nnz_tcore * ShapeX_col);//一个TC block对应shape_col个列值于A,shape_col个行值于B
    int count;
    int count_all_col = 0;
    int k_x;
    //第二遍统计,再次遍历得到 元素值 和 列值
    Data_type *tensorVal_record = (Data_type *)malloc(sizeof(Data_type) * (mtx_nscore->col + ShapeX_col));//一行的
    int *tensorCol_record = (int *)malloc(sizeof(int) * (mtx_nscore->col + ShapeX_col));
    start_row = 0,end_row = ShapeX_row;
    for(int z=0;z<b;z++)
    {
        //重置
        for(int i=0;i<Sparse_mtx->col;i++)
        {
            colNumber_record[i] = 0;
        }
        //记录元素
        for(int i=start_row;i<end_row;i++)
        {
            for(int j=Sparse_mtx->row_ptr[i];j<Sparse_mtx->row_ptr[i+1];j++)
            {
                colNumber_record[Sparse_mtx->col_index[j]]++;
            }
        }
        //统计列量
        int offset = (ShapeX_row * ShapeX_col) * mtx_nscore->rowT_ptr[z];//偏移
        int offset_row;
        int num_Tcore = mtx_nscore->rowT_ptr[z + 1] - mtx_nscore->rowT_ptr[z];//个数
        count = 0;//记录列个数
        //写入列
        for(int i=0;i<Sparse_mtx->col;i++)
        {
            if(colNumber_record[i] >= 1)
            {
                tensorCol_record[count] = i; //记录列元素
                count++;
                mtx_nscore->colT_ide[count_all_col] = i; //先来先选入
                count_all_col++;
            }
        }
        k_x=0;
        for(int i=count;i<num_Tcore * ShapeX_col;i++)
        {
            tensorCol_record[count] = -1;
            count++;
            mtx_nscore->colT_ide[count_all_col] = Sparse_mtx->col + k_x++;
            count_all_col++;
        }
        //写入val
        for(int i=start_row;i<end_row;i++)
        {
            for(int ii=0;ii<num_Tcore * ShapeX_col;ii++)
            {
                tensorVal_record[ii] = __float2half(0.0f);//重置
            }
            for(int j=Sparse_mtx->row_ptr[i];j<Sparse_mtx->row_ptr[i+1];j++)
            {
                for(int k = 0;k<num_Tcore * ShapeX_col;k++)
                {
                    if(Sparse_mtx->col_index[j] == tensorCol_record[k])
                    {
                        tensorVal_record[k] = Sparse_mtx->val[j];//记录元素按行存储位置
                        break;
                    }
                }
            }
            //转换 (ShapeX_row * ShapeX_col)
            offset_row = (i-start_row) * ShapeX_col;
            for(int j=0;j<num_Tcore;j++)
            {
                for(int k=0;k<ShapeX_col;k++)
                {
                    mtx_nscore->valT[offset + j*(ShapeX_row * ShapeX_col) + offset_row + k] = tensorVal_record[j*ShapeX_col + k];
                }
            }
        }
        start_row += ShapeX_row;
        end_row += ShapeX_row;
        if(z == b - 2)
        {
            end_row = Sparse_mtx->row;
        }
    }

    free(colNumber_record);
    free(tensorVal_record);
    free(tensorCol_record);
}
void trans_format_NScoreVDTC(const Matrix_data_CSR *Sparse_mtx,Matrix_data_NScore *mtx_nscore)
{
    mtx_nscore->row = Sparse_mtx->row;
    mtx_nscore->col = Sparse_mtx->col;
    mtx_nscore->nnz_all = Sparse_mtx->nnz;
    mtx_nscore->row_ptr = (int *)malloc(sizeof(int) * (1 + mtx_nscore->row));
    mtx_nscore->rowT_ptr = (int *)malloc(sizeof (int)* (1 + mtx_nscore->row/16 + (mtx_nscore->row%16 > 0 ? 1 : 0)));
    mtx_nscore->row_ptr[0] = 0;
    mtx_nscore->rowT_ptr[0] = 0;
    mtx_nscore->nnz_core = Sparse_mtx->nnz;//初始
    mtx_nscore->nnz_tcore = 0;//初始

    int *colNumber_record = (int *)malloc(sizeof(int) * mtx_nscore->col);//记录列有多少元素

    int start_row,end_row;
    int get_ColNumber;
    int b = mtx_nscore->row/16 + (mtx_nscore->row%16 > 0 ? 1 : 0);//按照8-4-8行的方式进行筛选
    start_row = 0,end_row = 16;
    //第一遍统计
    for(int z=0;z<b;z++)
    {
        //重置
        for(int i=0;i<Sparse_mtx->col;i++)
        {
            colNumber_record[i] = 0;
        }
        get_ColNumber = 0;
        //记录元素
        for(int i=start_row;i<end_row;i++)
        {
            for(int j=Sparse_mtx->row_ptr[i];j<Sparse_mtx->row_ptr[i+1];j++)
            {
                colNumber_record[Sparse_mtx->col_index[j]]++;
            }
        }
        //统计列量
        for(int i=0;i<Sparse_mtx->col;i++)
        {
            if(colNumber_record[i] >= 1)
            {
                mtx_nscore->nnz_core -= colNumber_record[i];
                get_ColNumber++;
            }
        }
        mtx_nscore->nnz_tcore += (get_ColNumber/ShapeX_col + (get_ColNumber % ShapeX_col> 0 ? 1 : 0));//有剩余情况,剩余情况提交给TCU
        mtx_nscore->rowT_ptr[z+1] = mtx_nscore->nnz_tcore;
        start_row += 16;
        end_row += 16;
        if(z == b - 2)//行尾巴不存了
        {
            end_row = Sparse_mtx->row;
        }
    }
    //为cuda core 分配内存


    /** 为tensor core 分配内存 **/
    mtx_nscore->valT = (Data_type *)malloc(sizeof(Data_type) * mtx_nscore->nnz_tcore * (16 * ShapeX_col));
    //多余部分也要设置为0
    mtx_nscore->colT_ide = (int *)malloc(sizeof(int) * mtx_nscore->nnz_tcore * ShapeX_col);//一个TC block对应shape_col个列值于A,shape_col个行值于B
    int count;
    int count_all_col = 0;
    int k_x;
    //第二遍统计,再次遍历得到 元素值 和 列值
    Data_type *tensorVal_record = (Data_type *)malloc(sizeof(Data_type) * (mtx_nscore->col + ShapeX_col));//一行的
    int *tensorCol_record = (int *)malloc(sizeof(int) * (mtx_nscore->col + ShapeX_col));
    start_row = 0,end_row = 16;
    for(int z=0;z<b;z++)
    {
        //重置
        for(int i=0;i<Sparse_mtx->col;i++)
        {
            colNumber_record[i] = 0;
        }
        //记录元素
        for(int i=start_row;i<end_row;i++)
        {
            for(int j=Sparse_mtx->row_ptr[i];j<Sparse_mtx->row_ptr[i+1];j++)
            {
                colNumber_record[Sparse_mtx->col_index[j]]++;
            }
        }
        //统计列量
        int offset = (16 * ShapeX_col) * mtx_nscore->rowT_ptr[z];//偏移
        int offset_row;
        int num_Tcore = mtx_nscore->rowT_ptr[z + 1] - mtx_nscore->rowT_ptr[z];//个数
        count = 0;//记录列个数
        //写入列
        for(int i=0;i<Sparse_mtx->col;i++)
        {
            if(colNumber_record[i] >= 1)
            {
                tensorCol_record[count] = i; //记录列元素
                count++;
                mtx_nscore->colT_ide[count_all_col] = i; //先来先选入
                count_all_col++;
            }
        }
        k_x=0;
        for(int i=count;i<num_Tcore * ShapeX_col;i++)
        {
            tensorCol_record[count] = -1;
            count++;
            mtx_nscore->colT_ide[count_all_col] = Sparse_mtx->col + k_x++;
            count_all_col++;
        }
        //写入val
        for(int i=start_row;i<end_row;i++)
        {
            for(int ii=0;ii<num_Tcore * ShapeX_col;ii++)
            {
                tensorVal_record[ii] = __float2half(0.0f);//重置
            }
            for(int j=Sparse_mtx->row_ptr[i];j<Sparse_mtx->row_ptr[i+1];j++)
            {
                for(int k = 0;k<num_Tcore * ShapeX_col;k++)
                {
                    if(Sparse_mtx->col_index[j] == tensorCol_record[k])
                    {
                        tensorVal_record[k] = Sparse_mtx->val[j];//记录元素按行存储位置
                        break;
                    }
                }
            }
            //转换 (ShapeX_row * ShapeX_col)
            offset_row = (i-start_row) * ShapeX_col;
            for(int j=0;j<num_Tcore;j++)
            {
                for(int k=0;k<ShapeX_col;k++)
                {
                    mtx_nscore->valT[offset + j*(16 * ShapeX_col) + offset_row + k] = tensorVal_record[j*ShapeX_col + k];
                }
            }
        }
        start_row += 16;
        end_row += 16;
        if(z == b - 2)
        {
            end_row = Sparse_mtx->row;
        }
    }
    free(colNumber_record);
    free(tensorVal_record);
    free(tensorCol_record);
}
/**
 * 格式转换GPU专用格式
 */
void trans_format_NScoreV3(const Matrix_data_CSR *Sparse_mtx,Matrix_data_NScore *mtx_nscore)
{
    mtx_nscore->row = Sparse_mtx->row;
    mtx_nscore->col = Sparse_mtx->col;
    mtx_nscore->nnz_all = Sparse_mtx->nnz;
    mtx_nscore->row_ptr = (int *)malloc(sizeof(int) * (1 + mtx_nscore->row));
    mtx_nscore->rowT_ptr = (int *)malloc(sizeof (int)* (1 + mtx_nscore->row/ShapeX_row + (mtx_nscore->row%ShapeX_row > 0 ? 1 : 0)));
    mtx_nscore->row_ptr[0] = 0;
    mtx_nscore->rowT_ptr[0] = 0;
    mtx_nscore->nnz_core = Sparse_mtx->nnz;//初始 cuda core 数量
    mtx_nscore->nnz_tcore = 0;//初始 tc block 数量

    int *colNumber_record = (int *)malloc(sizeof(int) * mtx_nscore->col);//记录列有多少元素

    int start_row,end_row;
    int get_ColNumber;
    int b = mtx_nscore->row/ShapeX_row + (mtx_nscore->row%ShapeX_row > 0 ? 1 : 0);//按照8-4-8行的方式进行筛选
    start_row = 0,end_row = ShapeX_row;
    //第一遍统计
    for(int z=0;z<b;z++)
    {
        //重置
        for(int i=0;i<Sparse_mtx->col;i++)
        {
            colNumber_record[i] = 0;
        }
        get_ColNumber = 0;
        //记录元素
        for(int i=start_row;i<end_row;i++)
        {
            for(int j=Sparse_mtx->row_ptr[i];j<Sparse_mtx->row_ptr[i+1];j++)
            {
                colNumber_record[Sparse_mtx->col_index[j]]++;
            }
        }
        //统计列量
        int temp_sum = 0;//临时变量
        for(int i=0;i<Sparse_mtx->col;i++)
        {
            if(colNumber_record[i] >= Col_size)
            {
                temp_sum += colNumber_record[i];
                get_ColNumber++;
                if(get_ColNumber % ShapeX_col == 0)
                {
                    mtx_nscore->nnz_core -= temp_sum;//记录稀疏列nnz个数
                    temp_sum = 0;
                }
            }
        }
        mtx_nscore->nnz_tcore += get_ColNumber/ShapeX_col;//有剩余情况,剩余情况提交给cuda core
        mtx_nscore->rowT_ptr[z+1] = mtx_nscore->nnz_tcore;
        start_row += ShapeX_row;
        end_row += ShapeX_row;
        if(z == b - 2)//行尾巴不存了
        {
            end_row = Sparse_mtx->row;
        }
    }
    /** 为cuda core 分配内存 **/
    mtx_nscore->val = (Data_type *)malloc(sizeof(Data_type) * mtx_nscore->nnz_core);
    mtx_nscore->col_index = (int *)malloc(sizeof(int)  * mtx_nscore->nnz_core);
    int cuda_count_all = 0;
    /** 为tensor core 分配内存 **/
    mtx_nscore->valT = (Data_type *)malloc(sizeof(Data_type) * mtx_nscore->nnz_tcore * (ShapeX_row * ShapeX_col));
    mtx_nscore->colT_ide = (int *)malloc(sizeof(int) * mtx_nscore->nnz_tcore * ShapeX_col);//一个TC block对应shape_col个列值于A,shape_col个行值于B
    int count;//tcu
    int count_all_col = 0;//tcu
    int count_cuda;//cuda
    //第二遍统计,再次遍历得到 元素值 和 列值
    Data_type *tensorVal_record = (Data_type *)malloc(sizeof(Data_type) * (mtx_nscore->col));//一行的
    int *tensorCol_record = (int *)malloc(sizeof(int) * (mtx_nscore->col));//稠密列
    int *cudaCol_record = (int *)malloc(sizeof(int) * (mtx_nscore->col));//稀疏列
    start_row = 0,end_row = ShapeX_row;
    for(int z=0;z<b;z++)
    {
        //重置
        for(int i=0;i<Sparse_mtx->col;i++)
        {
            colNumber_record[i] = 0;
        }
        //记录元素
        for(int i=start_row;i<end_row;i++)
        {
            for(int j=Sparse_mtx->row_ptr[i];j<Sparse_mtx->row_ptr[i+1];j++)
            {
                colNumber_record[Sparse_mtx->col_index[j]]++;
            }
        }
        //统计列量
        int offset = (ShapeX_row * ShapeX_col) * mtx_nscore->rowT_ptr[z];//偏移
        int offset_row;
        int num_Tcore = mtx_nscore->rowT_ptr[z + 1] - mtx_nscore->rowT_ptr[z];//个数
        count = 0;//记录列个数
        count_cuda = 0;
        //写入列
        get_ColNumber = 0;
        for(int i=0;i<Sparse_mtx->col;i++)
        {
            //稀疏列
            if(colNumber_record[i] > 0 && colNumber_record[i] < Col_size)
            {
                cudaCol_record[count_cuda] = i; //记录列元素
                count_cuda++;
            }
            //稠密列
            else if(colNumber_record[i] >= Col_size)
            {
                if(get_ColNumber < num_Tcore * ShapeX_col) {
                    tensorCol_record[count] = i; //记录列元素
                    count++;
                    mtx_nscore->colT_ide[count_all_col] = i; //先来先选入
                    count_all_col++;
                }
                // cuda core nnz剩余情况
                else
                {
                    cudaCol_record[count_cuda] = i; //记录列元素
                    count_cuda++;
                }
                get_ColNumber++;
            }
        }
        //写入val
        for(int i=start_row;i<end_row;i++)
        {
            for(int ii=0;ii<num_Tcore * ShapeX_col;ii++)
            {
                tensorVal_record[ii] = __float2half(0.0f);//重置
            }
            for(int j=Sparse_mtx->row_ptr[i];j<Sparse_mtx->row_ptr[i+1];j++)
            {
                for(int k = 0;k < count;k++)
                {
                    if(Sparse_mtx->col_index[j] == tensorCol_record[k])
                    {
                        tensorVal_record[k] = Sparse_mtx->val[j];//记录元素按行存储位置
                        break;
                    }
                }
                for(int k = 0;k<count_cuda;k++)
                {
                    if(Sparse_mtx->col_index[j] == cudaCol_record[k])
                    {
                        //记录cuda core元素
                        mtx_nscore->val[cuda_count_all] = Sparse_mtx->val[j];
                        mtx_nscore->col_index[cuda_count_all] = Sparse_mtx->col_index[j];
                        cuda_count_all++;
                        break;
                    }
                }
            }
            mtx_nscore->row_ptr[i + 1] = cuda_count_all;
            //转换 (ShapeX_row * ShapeX_col)
            offset_row = (i-start_row) * ShapeX_col;
            for(int j=0;j<num_Tcore;j++)
            {
                for(int k=0;k<ShapeX_col;k++)
                {
                    mtx_nscore->valT[offset + j*(ShapeX_row * ShapeX_col) + offset_row + k] = tensorVal_record[j*ShapeX_col + k];
                }
            }
        }
        start_row += ShapeX_row;
        end_row += ShapeX_row;
        if(z == b - 2)
        {
            end_row = Sparse_mtx->row;
        }
    }

    //merge csr
    int merge_count = 0;
    for(int i=0;i<mtx_nscore->row;i++)
    {
        if((mtx_nscore->row_ptr[i+1] - mtx_nscore->row_ptr[i])!=0)
        {
            merge_count++;
        }
    }
    mtx_nscore->merge_row_count = merge_count;
    mtx_nscore->merge_row = (int *)malloc(sizeof(int)*merge_count);
    mtx_nscore->merge_row_ptr = (int *)malloc(sizeof(int)*(merge_count + 1));
    mtx_nscore->merge_row_ptr[0] = 0;
    merge_count = 0;
    for(int i=0;i<mtx_nscore->row;i++)
    {
        if((mtx_nscore->row_ptr[i+1] - mtx_nscore->row_ptr[i])!=0)
        {
            mtx_nscore->merge_row[merge_count] = i;
            mtx_nscore->merge_row_ptr[merge_count + 1] = mtx_nscore->row_ptr[i + 1];
            merge_count++;
        }
    }
    free(colNumber_record);
    free(tensorVal_record);
    free(tensorCol_record);
    free(cudaCol_record);
}
/**
 * 创造稠密矩阵
 */
void creat_dense_matrix_cpu(Matrix_data_CSR *Dense_mtx,const int row,const int col)
{
    Dense_mtx->row = row;
    Dense_mtx->col = col;
    Dense_mtx->nnz = row * col;
    Dense_mtx->val = (Data_type *)malloc(sizeof(Data_type) * Dense_mtx->nnz);
    Dense_mtx->col_index = (int *) malloc(sizeof(int) * Dense_mtx->nnz);
    Dense_mtx->row_ptr = (int *) malloc(sizeof(int) * (row + 1));
    Dense_mtx->row_ptr[0] = 0;
    for(int i=0;i<row - 8;i++)
    {
        for(int j=0;j<col;j++)
        {
#if Length==16
            Dense_mtx->val[i * col + j] = __float2half((float)(i%10) * 0.1f); //一行的值不变,值随行递增而递增
#else
            Dense_mtx->val[i * col + j] = (Data_type)((i%10) * 1.1f);
#endif
            Dense_mtx->col_index[i * col + j] = j;
        }
        Dense_mtx->row_ptr[i+1] = col * (i + 1);
    }
    for(int i=row - 8;i<row;i++)
    {
        for(int j=0;j<col;j++)
        {
#if Length==16
            Dense_mtx->val[i * col + j] = __float2half(0.0); //一行的值不变,值随行递增而递增
#else
            Dense_mtx->val[i * col + j] = (Data_type)((i%10) * 1.1f);
#endif
            Dense_mtx->col_index[i * col + j] = j;
        }
        Dense_mtx->row_ptr[i+1] = col * (i + 1);
    }
}
/**
 * 创造结果稠密矩阵
 */
void creat_rec_dense_matrix(Matrix_data_CSR *rec_Dense_mtx,const int row,const int col)
{
    rec_Dense_mtx->row = row;
    rec_Dense_mtx->col = col;
    rec_Dense_mtx->nnz = row * col;
    rec_Dense_mtx->val = (Data_type *)malloc(sizeof(Data_type) * rec_Dense_mtx->nnz);
    rec_Dense_mtx->col_index = (int *) malloc(sizeof(int) * rec_Dense_mtx->nnz);
    rec_Dense_mtx->row_ptr = (int *) malloc(sizeof(int) * (row + 1));
    rec_Dense_mtx->row_ptr[0] = 0;
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<col;j++)
        {
#if Length==16
            rec_Dense_mtx->val[i * col + j] = __float2half(0.0f);
#else
            rec_Dense_mtx->val[i * col + j] = (Data_type)0.0;
#endif
            rec_Dense_mtx->col_index[i * col + j] = j;
        }
        rec_Dense_mtx->row_ptr[i+1] = col * (i + 1);
    }
}
/**
 * 销毁cpuCSR
 */
void Destroy_matrixCSR_cpu(Matrix_data_CSR *mtx)
{
    free(mtx->row_ptr);
    free(mtx->col_index);
    free(mtx->val);
}
/**
 * 销毁cpuCSC
 */
void Destroy_matrixCSC_cpu(Matrix_data_CSC *mtx)
{
    free(mtx->col_ptr);
    free(mtx->row_index);
    free(mtx->val);
}

