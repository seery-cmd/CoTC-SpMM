//
// Created by lenovo on 2025/4/16.
//
#include <cassert>
#include "../Struct/struct.h"
#include "mmio.h"


#ifndef CUDALEARN_UTILS_H
#define CUDALEARN_UTILS_H
static int diag_num=0;
void read_COO_format(Matrix_data_COO *matrixDataCoo,char *filename);
void Format_trans_COO_to_CSR(Matrix_data_COO *matrixDataCoo,Matrix_data_CSR *matrixDataCsr);
void Read_sparse_matrix(Matrix_data_CSR *matrixDataCsr,char *filename);
void Printf_matrix_CSR(Matrix_data_CSR *matrixDataCsr);
void printSparseMatrix(Matrix_data_CSR * mtx);
void Printf_matrix_CSC(Matrix_data_CSC * mtx);
void Printf_matrix_nscore(Matrix_data_NScore *mtx_nscore);
void read_COO_format(Matrix_data_COO *matrixDataCoo, char *mm_filename) {
    int ret_code;
    MM_typecode matcode;
    FILE *fid;
    int M, N, nz;
    int  *I, *J;
    Data_type *val;

    fid = fopen(mm_filename, "r");

    if(fid == NULL){
        std::cout << "Unable to open file: "<< mm_filename << std::endl;
        exit(1);
    }

    if (mm_read_banner(fid, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if(!mm_is_valid(matcode)){
        std::cout << "Invalid Matrix" << std::endl;
        exit(1);
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) && mm_is_sparse(matcode) ) ){
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        printf("Only sparse real-valued or pattern coordinate matrices are supported\n");
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(fid, &M, &N, &nz)) != 0)
    {
        std::cout << "The line of rows, cols, nnzs is in wrong format" << std::endl;
        exit(1);
    }

    matrixDataCoo->row = (Index_type) M;
    matrixDataCoo->col = (Index_type) N;
    matrixDataCoo->nnz = (Index_type) nz;

    /* reseve memory for matrices */

    I = (Index_type *)malloc(nz * sizeof(Index_type));
    J = (Index_type *)malloc(nz * sizeof(Index_type));
    val = (Data_type *)malloc(nz * sizeof(Data_type));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    std::cout << "- Reading sparse matrix from file: "<< mm_filename << std::endl;
    fflush(stdout);

// #if Length == 64
//     for (i = 0; i < nz; i++) {
//         fscanf(f, "%d %d %lf\n", &I[i], &J[i], &val[i]);
//         I[i]--; /* adjust from 1-based to 0-based */
//         J[i]--;
//     }
// #elif Length == 32
//     for (i = 0; i < nz; i++) {
//         fscanf(f, "%d %d %f\n", &I[i], &J[i], &val[i]);
//         I[i]--; /* adjust from 1-based to 0-based */
//         J[i]--;
//     }
// #endif

    if(mm_is_pattern(matcode)){
        for (Index_type i = 0; i < matrixDataCoo->nnz; i++)
        {
#if IndexLength == 64
            assert(fscanf(fid,"%lld %lld\n", &(I[i]), &(J[i])) == 2);
#elif IndexLength == 32
            assert(fscanf(fid,"%d %d\n", &(I[i]), &(J[i])) == 2);
#endif
            // adjust from 1-based to 0-based indexing
            --(I[i]);
            --(J[i]);
            val[i] = 1.0;
        }
    }else if (mm_is_real(matcode) || mm_is_integer(matcode)){
        for( Index_type i = 0; i < matrixDataCoo->nnz; i++ ){
            Index_type row_id, col_id;
            double V; // read in double and convert to ValueType

#if IndexLength == 64
            assert(fscanf(fid, "%lld %lld %lf\n", &row_id, &col_id, &V) == 3);
#elif IndexLength == 32
            assert(fscanf(fid, "%d %d %lf\n", &row_id, &col_id, &V) == 3);
#endif

            I[i]      = (Index_type) row_id - 1;
            J[i]      = (Index_type) col_id - 1;
            #if Length==16
                val[i] = __float2half((float)(V - floor(V) + (int)V%100));
            #else
                val[i] = (Data_type) V;
            #endif
        }
    }else{
        std::cout << "Unsupported data type" << std::endl;
        exit(1);
    }

    fclose(fid);
    std::cout << "- Finish Reading data from " << mm_filename << std::endl;

    matrixDataCoo->row_index = I;
    matrixDataCoo->col_index = J;
    matrixDataCoo->val       = val;

    // 处理对称情况 duplicate off diagonal entries
    if( mm_is_symmetric(matcode) ){
        Index_type off_diagonals = 0;
        for (Index_type i = 0; i < matrixDataCoo->nnz; i++){
            if(I[i] != J[i])
                off_diagonals++;
        }
        // realNNZ = 2*off_diagonals + (coo.num_nonzeros - off_diagonals)
        Index_type true_nnz = off_diagonals + matrixDataCoo->nnz;

        Index_type* new_rowindex = (Index_type *)malloc(true_nnz * sizeof(Index_type));
        Index_type* new_colindex = (Index_type *)malloc(true_nnz * sizeof(Index_type));
        Data_type* new_V = (Data_type *)malloc(true_nnz * sizeof(Data_type));

        Index_type ptr = 0;
        for (Index_type i = 0; i < matrixDataCoo->nnz; i++)
        {
            if(I[i] != J[i]){
                new_rowindex[ptr] = I[i];
                new_colindex[ptr] = J[i];
                new_V[ptr]        = val[i];
                ptr++;
                new_colindex[ptr] = I[i];
                new_rowindex[ptr] = J[i];
                new_V[ptr]        = val[i];
                ptr++;
            }else {
                new_rowindex[ptr] = I[i];
                new_colindex[ptr] = J[i];
                new_V[ptr]        = val[i];
                ptr++;
            }
        }
        // delete_array(coo.row_index);
        // delete_array(coo.col_index);
        // delete_array(coo.values);
        free(I); free(J); free(val);
        matrixDataCoo->row_index = new_rowindex;
        matrixDataCoo->col_index = new_colindex;
        matrixDataCoo->val       = new_V;
        matrixDataCoo->nnz       = true_nnz;
    }

    // matrixDataCoo->row_index = (int *)malloc(sizeof(int) * nz);
    // matrixDataCoo->col_index = (int *)malloc(sizeof(int) * nz);
    // matrixDataCoo->val = (Data_type *)malloc(sizeof(Data_type) * nz);

    // matrixDataCoo->row = M;
    // matrixDataCoo->col = N;
    // matrixDataCoo->nnz = nz;

    // memcpy(matrixDataCoo->row_index, I, nz * sizeof(int));
    // memcpy(matrixDataCoo->col_index, J, nz * sizeof(int));
    // memcpy(matrixDataCoo->val, val, nz * sizeof(Data_type));

    // free(I);
    // free(J);
    // free(val);
}

void Format_trans_COO_to_CSR(Matrix_data_COO *matrixDataCoo, Matrix_data_CSR *matrixDataCsr) {
    /** 改写CSR格式 **/
    matrixDataCsr->row = matrixDataCoo->row;
    matrixDataCsr->col = matrixDataCoo->col;
    matrixDataCsr->nnz = matrixDataCoo->nnz;

    // matrixDataCsr->nnz =
    //     issy == 1 ? 2 * matrixDataCoo->nnz - matrixDataCoo->row : matrixDataCoo->nnz;


    // matrixDataCsr->col_index = (int *)malloc(sizeof(int) * matrixDataCsr->nnz);
    // matrixDataCsr->row_ptr = (int *)malloc(sizeof(int) * (matrixDataCsr->row + 1));
    // matrixDataCsr->row_ptr[0] = 0;
    // matrixDataCsr->val = (Data_type *)malloc(sizeof(Data_type) * matrixDataCsr->nnz);
    matrixDataCsr->row_ptr = (Index_type *)malloc((matrixDataCsr->row +1)* sizeof(Index_type));
    matrixDataCsr->col_index = (Index_type *)malloc(matrixDataCsr->nnz * sizeof(Index_type));
    matrixDataCsr->val = (Data_type *)malloc(matrixDataCsr->nnz * sizeof(Data_type));

//    int rowPtr = 1;
//    int count_all = 0;

    //========== Rowoffset calculation ==========
    for (Index_type i = 0; i < matrixDataCsr->row; i++){
        matrixDataCsr->row_ptr[i] = 0;
    }

    // Get each row's nnzs
    for (Index_type i = 0; i < matrixDataCsr->nnz; i++){
        matrixDataCsr->row_ptr[ matrixDataCoo->row_index[i] ]++;
    }

    //  sum to get row_offset
    for(Index_type i = 0, cumsum = 0; i < matrixDataCsr->row; i++){
        Index_type temp = matrixDataCsr->row_ptr[i];
        matrixDataCsr->row_ptr[i] = cumsum;
        cumsum += temp;
    }
    matrixDataCsr->row_ptr[matrixDataCsr->row] = matrixDataCsr->nnz;

    // ========== write col_index and values ==========
    for (Index_type i = 0; i < matrixDataCsr->nnz; i++){
        Index_type rowIndex  = matrixDataCoo->row_index[i];
        Index_type destIndex = matrixDataCsr->row_ptr[rowIndex];

        matrixDataCsr->col_index[destIndex] = matrixDataCoo->col_index[i];
        matrixDataCsr->val[destIndex]       = matrixDataCoo->val[i];

        matrixDataCsr->row_ptr[rowIndex]++;  // row_offset move behind
    }

    // Restore the row_offset
    for(Index_type i = 0, last = 0; i <= matrixDataCsr->row; i++){
        Index_type temp = matrixDataCsr->row_ptr[i];
        matrixDataCsr->row_ptr[i] = last;
        last = temp;
    }
}

void Read_sparse_matrix(Matrix_data_CSR *matrixDataCsr, char *filename) {
    Matrix_data_COO *matrixCOO;
    matrixCOO = (Matrix_data_COO *)alloca(sizeof(Matrix_data_COO));
    read_COO_format(matrixCOO, filename);
    Format_trans_COO_to_CSR(matrixCOO, matrixDataCsr);
}
void Printf_matrix_CSR(Matrix_data_CSR *matrixDataCsr)
{
    int start,end;
    for(int i=0;i<matrixDataCsr->row;i++)
    {
        start=matrixDataCsr->row_ptr[i],end=matrixDataCsr->row_ptr[i+1];
        //printf("%d,%d\n",start,end);
        printf("row=%d:",i);
        for(int j=start;j<end;j++)
        {
            printf("<%d-%f>  ",matrixDataCsr->col_index[j],__half2float(matrixDataCsr->val[j]));
        }
        printf("\n");
    }
}
void printSparseMatrix(Matrix_data_CSR * mtx)
{
    int rowIndex = 0;
    int nzIndex = 0;

    for (rowIndex = 0; rowIndex < mtx->row; rowIndex++)
    {
        for (int colIndex = 0; colIndex < mtx->col; colIndex++)
        {
            if (colIndex == mtx->col_index[nzIndex] && nzIndex < mtx->row_ptr[rowIndex + 1])
            {
                printf("%f ", __half2float(mtx->val[nzIndex]));
                nzIndex++;
            }
            else
            {
                printf("* ");
            }
        }
        printf("\n");
    }
}
void Printf_matrix_CSC(Matrix_data_CSC * mtx)
{
    int start,end;
    for(int i=0;i<mtx->col;i++)
    {
        start=mtx->col_ptr[i],end=mtx->col_ptr[i+1];
        //printf("%d,%d\n",start,end);
        printf("col=%d:",i);
        for(int j=start;j<end;j++)
        {
            printf("<%d-%f>  ",mtx->row_index[j],__half2float(mtx->val[j]));
        }
        printf("\n");
    }
}
void Printf_matrix_nscore(Matrix_data_NScore *mtx_nscore)
{
    printf("****************TCU********************\n");
    printf("----------------colT_idx-------------------\n");
    for(int i=0;i<mtx_nscore->row/ShapeX_row;i++)
    {
        printf("第%d行\n",i);
        for(int j=mtx_nscore->rowT_ptr[i];j<mtx_nscore->rowT_ptr[i+1];j++)
        {
            for(int k=0;k<ShapeX_col;k++)
            {
                printf("%d ",mtx_nscore->colT_ide[j*ShapeX_col + k]);
            }
            printf("\n");
        }
    }
    printf("----------------rowT_ptr-------------------\n");
    for(int i = 0;i<=mtx_nscore->row/ShapeX_row;i++)
    {
        printf("%d\n",mtx_nscore->rowT_ptr[i]);
    }
    printf("----------------valT-------------------\n");
    for(int i=0;i<mtx_nscore->row/ShapeX_row;i++)
    {
        printf("第%d行tensorcore块\n", i);
        for(int jj=mtx_nscore->rowT_ptr[i];jj<mtx_nscore->rowT_ptr[i+1];jj++)
        {
            for (int j = 0; j < ShapeX_row; j++) {
                for (int k = 0; k < ShapeX_col; k++) {
                    printf("%f ", __half2float(mtx_nscore->valT[jj * (ShapeX_row * ShapeX_col) + j * ShapeX_col + k]));
                }
                printf("\n");
            }
            printf("\n\n");
        }
    }
    printf("****************CUDA CORE********************\n");
    printf("----------------row_ptr-------------------\n");
    for(int i = 0;i<=mtx_nscore->row;i++)
    {
        printf("%d\n",mtx_nscore->row_ptr[i]);
    }
    printf("----------------merge_row-------------------\n");
    for(int i = 0;i<mtx_nscore->merge_row_count;i++)
    {
        printf("%d,%d\n",mtx_nscore->merge_row[i],mtx_nscore->merge_row_ptr[i+1]);
    }
    printf("----------------col_idx-------------------\n");
    for(int i=0;i<mtx_nscore->row;i++)
    {
        printf("第%d行: ",i);
        for(int j=mtx_nscore->row_ptr[i];j<mtx_nscore->row_ptr[i+1];j++)
        {
            printf("<%d-%f>  ",mtx_nscore->col_index[j],__half2float(mtx_nscore->val[j]));
        }
        printf("\n");
    }
}
#endif //CUDALEARN_UTILS_H