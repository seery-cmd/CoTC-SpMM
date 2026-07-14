//
// Created by lenovo on 2025/4/16.
//

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <cuda_fp16.h>


#ifndef CUDALEARN_STRUCT_H
#define CUDALEARN_STRUCT_H

using namespace std;
#define BN 256
#define Col_size 2//如果列的nnz个数大于Col_size则归于TC block
#define ShapeX_row 8 //TC block 的行大小
#define ShapeX_col 8 //TC block 的列大小

#define Length 16 //64--double 32--float 16--halt 8--int
#if Length==64
#define Data_type double
#elif Length==32
#define Data_type float
#elif Length==16
#define Data_type half
#elif Length==8
    #define Data_type int
#endif

#define IndexLength 32
#if IndexLength == 64
#define Index_type long long
#elif IndexLength == 32
#define Index_type int
#endif

typedef struct Matrix_data_CSR
{
    int nnz;
    int row;
    int col;
    int *row_ptr;
    int *col_index;
    Data_type *val;
}Matrix_data_CSR;
typedef struct Matrix_data_COO
{
    int nnz;
    int row;
    int col;
    int *row_index;
    int *col_index;
    Data_type *val;
}Matrix_data_COO;
typedef struct Sp_vector
{
    int length;
    int *index;
    Data_type *val;
}Sp_vector;
typedef struct Matrix_data_CSC
{
    int nnz;
    int row;
    int col;
    int *col_ptr;
    int *row_index;
    Data_type *val;
}Matrix_data_CSC;
typedef struct Matrix_data_NScore
{
    //classics data
    int row;
    int col;
    int nnz_all; //总共元素
    //leave and cuda nnz
    int nnz_core;
    int *row_ptr;
    int *merge_row;//row_ptr消除0元素
    int *merge_row_ptr;
    int merge_row_count;
    int *col_index;
    Data_type *val;
    //t_core
    int nnz_tcore;
    int *colT_ide;
    int *rowT_ptr;//一行
    Data_type *valT;
}Matrix_data_NScore;
#endif //CUDALEARN_STRUCT_H
