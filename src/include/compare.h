//
// Created by lenovo on 2025/4/21.
//

#ifndef CUDALEARN_COMPARE_H
#define CUDALEARN_COMPARE_H
#include "Liner_Library.h"

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

void main_add_compare(int length);
const Data_type alpha[1] = {3.61};
const Data_type beta[1] = {4.95};
void main_AXPYB_compare(char *mtx_path);
void main_SpVV_T_compare(char *mtx_path);


void main_add_compare(int length)
{
    int N = length;
    Data_type *x, *y,*rec;
    Data_type *dev_x,*dev_y,*dev_rec;
    Data_type *final_rec_cpu;
    timeval t1,t2;
    cudaEvent_t start,end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));

    /**内存分配，在GPU或者CPU上统一分配内存**/
    // cudaMallocManaged(&x, N*sizeof(float));
    // cudaMallocManaged(&y, N*sizeof(float));
    /** CPU malloc memory **/
    x=(Data_type *)malloc(sizeof(Data_type)*N);
    y=(Data_type *)malloc(sizeof(Data_type)*N);
    rec=(Data_type *)malloc(sizeof(Data_type)*N);
    final_rec_cpu=(Data_type *)malloc(sizeof(Data_type)*N);
    /** GPU malloc memory **/
    CHECK(cudaMalloc(&dev_x,N*sizeof(Data_type)));
    CHECK(cudaMalloc(&dev_y,N*sizeof(Data_type)));
    CHECK(cudaMalloc(&dev_rec,N*sizeof(Data_type)));
    /* 初始化 */
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    /* 数据传输GPU */
    CHECK(cudaMemcpy(dev_x,x,sizeof(Data_type)*N,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_y,y,sizeof(Data_type)*N,cudaMemcpyHostToDevice));

    CHECK(cudaEventRecord(start,0));
    /* blockNum,blockSize */
    add<<<1024, 1024>>>(N, dev_x, dev_y,dev_rec);
    CHECK(cudaEventRecord(end,0));
    CHECK(cudaEventSynchronize(end));
    float timecost_GPU;
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("timecost_GPU:%fms\n",timecost_GPU);


    /* CPU计算 */
    gettimeofday(&t1,NULL);
    add_cpu(N,x,y,rec);
    gettimeofday(&t2,NULL);
    double timecost_cpu = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
    std::cout << "CPU time: " << timecost_cpu << "ms" << std::endl;


    CHECK(cudaMemcpy(final_rec_cpu,dev_rec,N*sizeof(Data_type),cudaMemcpyDeviceToHost));

    for(int i=0;i<5;i++)
    {
        printf("%lf,%lf\n",final_rec_cpu[i],rec[i]);
    }

    /*释放内存*/
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    free(x);
    free(y);
    free(rec);
    free(final_rec_cpu);

    CHECK(cudaFree(dev_x));
    CHECK(cudaFree(dev_y));
    CHECK(cudaFree(dev_rec));
}
void main_AXPYB_compare(char *mtx_path)
{
    struct timeval t1,t2;
    Matrix_data_CSR *matrixDataCsr = (Matrix_data_CSR *)malloc(sizeof (Matrix_data_CSR));
    Read_sparse_matrix(matrixDataCsr,mtx_path);//读取矩阵
    Sp_vector *Sp_Vec = (Sp_vector *)malloc (sizeof(Sp_vector));
    creat_SpVector_CPU(Sp_Vec,matrixDataCsr);// 通过稀疏矩阵构造的稀疏向量
    /** 构造稠密向量 **/
    int length = (matrixDataCsr->row * matrixDataCsr->col);
    Data_type *Dense_Vector = (Data_type *)malloc(sizeof(Data_type) * length);//CPU稠密向量
    for(int i=0;i<length;i++)
    {
        Dense_Vector[i] = 1.5 * (i%100) - i;
    }
    /** 构造结果向量 **/
    Data_type *Sp_rec_vector = (Data_type *)malloc(sizeof(Data_type) * Sp_Vec->length);//结果向量
    for(int i=0;i<Sp_Vec->length;i++)
    {
        Sp_rec_vector[i] = 0;
    }
    /** CPU 计算 **/
    gettimeofday(&t1,NULL);
    L_cuda_AXPYB_cpu(Sp_Vec,Dense_Vector,Sp_rec_vector,alpha[0],beta[0]);//计算
    gettimeofday(&t2,NULL);
    double timecost_cpu = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
    std::cout << "CPU time: " << timecost_cpu << "ms" << std::endl;
    /** GPU 计算 **/
    cudaEvent_t start,end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    int *Sp_vec_GPU_index;
    Data_type *Sp_vec_GPU_val;
    Data_type *Dense_Vector_GPU;
    Data_type *Sp_rec_vector_GPU;
    Data_type *Sp_rec_vector_GPU_to_CPU = (Data_type *)malloc(sizeof (Data_type) * Sp_Vec->length);

    CHECK(cudaMalloc(&Sp_vec_GPU_index,Sp_Vec->length*sizeof(int)));
    CHECK(cudaMalloc(&Sp_vec_GPU_val,Sp_Vec->length*sizeof(Data_type)));
    CHECK(cudaMalloc(&Dense_Vector_GPU,length*sizeof(Data_type)));
    CHECK(cudaMalloc(&Sp_rec_vector_GPU,Sp_Vec->length*sizeof(Data_type)));


    /* 数据传输GPU */
    CHECK(cudaMemcpy(Sp_vec_GPU_index,Sp_Vec->index,sizeof(int)*Sp_Vec->length,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Sp_vec_GPU_val,Sp_Vec->val,sizeof(Data_type)*Sp_Vec->length,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Dense_Vector_GPU,Dense_Vector,sizeof(Data_type)*length,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(alpha_GPU,alpha,sizeof(Data_type) * 1));
    CHECK(cudaMemcpyToSymbol(beta_GPU,beta,sizeof(Data_type) * 1));



    CHECK(cudaEventRecord(start,0));

    L_cuda_AXPYB<<<1024, 1024>>>(Sp_vec_GPU_index,Sp_vec_GPU_val,Dense_Vector_GPU,Sp_rec_vector_GPU,Sp_Vec->length);

    CHECK(cudaEventRecord(end,0));
    CHECK(cudaEventSynchronize(end));
    float timecost_GPU;
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("timecost_GPU:%fms\n",timecost_GPU);

    CHECK(cudaMemcpy(Sp_rec_vector_GPU_to_CPU,Sp_rec_vector_GPU,sizeof (Data_type)*Sp_Vec->length,cudaMemcpyDeviceToHost));


    /** compare **/
    for(int i=0;i<Sp_Vec->length;i+=100)
    {
        if(fabs(Sp_rec_vector[i]-Sp_rec_vector_GPU_to_CPU[i]) > 1e-5) {
            printf("error!\n");
            printf("error index=%d: %lf,%lf\n",i,Sp_rec_vector[i],Sp_rec_vector_GPU_to_CPU[i]);
            exit(1);
        }
    }
    printf("compare success!\n");
    /** 释放内存 **/
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    free(matrixDataCsr);
    free(Sp_Vec);
    free(Dense_Vector);
    free(Sp_rec_vector);

    free(Sp_rec_vector_GPU_to_CPU);
    CHECK(cudaFree(Sp_rec_vector_GPU));
    CHECK(cudaFree(Sp_vec_GPU_index));
    CHECK(cudaFree(Sp_vec_GPU_val));
    CHECK(cudaFree(Dense_Vector_GPU));
}
void main_SpVV_T_compare(char *mtx_path)
{
    struct timeval t1,t2;
    Matrix_data_CSR *matrixDataCsr = (Matrix_data_CSR *)malloc(sizeof (Matrix_data_CSR));
    Read_sparse_matrix(matrixDataCsr,mtx_path);//读取矩阵
    Sp_vector *Sp_Vec = (Sp_vector *)malloc (sizeof(Sp_vector));
    creat_SpVector_CPU(Sp_Vec,matrixDataCsr);// 通过稀疏矩阵构造的稀疏向量
    /** 构造稀疏(稠密,非零元)向量 **/
    int length = (matrixDataCsr->row * matrixDataCsr->col);
    Data_type *Sparse_Vector = (Data_type *)malloc(sizeof(Data_type) * length);//稀疏向量,按照稠密向量存
    for(int i=0;i<length;i++)
    {
        if(i % 4 == 0) {
            Sparse_Vector[i] = 1.5 * (i % 16) - (i % 4);
        }
        else
        {
            Sparse_Vector[i] = 0.0;
        }
    }
    /** 构造结果标量 **/
    Data_type rec_scalar=0.0;//结果向量
    /** CPU 计算 **/
    gettimeofday(&t1,NULL);
    L_cuda_SpVV_cpu_T(Sp_Vec,Sparse_Vector,&rec_scalar);//计算
    gettimeofday(&t2,NULL);
    double timecost_cpu = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
    std::cout << "CPU time: " << timecost_cpu << "ms" << std::endl;

    /** GPU 计算 **/
    cudaEvent_t start,end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    int *Sp_vec_GPU_index;
    Data_type *Sp_vec_GPU_val;
    Data_type *Sparse_Vector_GPU;
    Data_type *rec_scalar_GPU;
    int thread_block_num = 1024;
    Data_type *rec_scalar_GPU_to_CPU =(Data_type *)malloc(sizeof (Data_type)* thread_block_num);
    Data_type rec_scalar_GPU_final = 0.0;

    CHECK(cudaMalloc(&Sp_vec_GPU_index,Sp_Vec->length*sizeof(int)));
    CHECK(cudaMalloc(&Sp_vec_GPU_val,Sp_Vec->length*sizeof(Data_type)));
    CHECK(cudaMalloc(&Sparse_Vector_GPU,length*sizeof(Data_type)));
    CHECK(cudaMalloc(&rec_scalar_GPU,sizeof(Data_type) * thread_block_num));


    /* 数据传输GPU */
    CHECK(cudaMemcpy(Sp_vec_GPU_index,Sp_Vec->index,sizeof(int)*Sp_Vec->length,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Sp_vec_GPU_val,Sp_Vec->val,sizeof(Data_type)*Sp_Vec->length,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Sparse_Vector_GPU,Sparse_Vector,sizeof(Data_type)*length,cudaMemcpyHostToDevice));



    CHECK(cudaEventRecord(start,0));

    L_cuda_SpVV_T<<<thread_block_num, 1024>>>(Sp_vec_GPU_index,Sp_vec_GPU_val,Sparse_Vector_GPU,rec_scalar_GPU,Sp_Vec->length);

    CHECK(cudaEventRecord(end,0));
    CHECK(cudaEventSynchronize(end));
    float timecost_GPU;
    CHECK(cudaEventElapsedTime(&timecost_GPU,start,end));
    printf("timecost_GPU:%fms\n",timecost_GPU);

    CHECK(cudaMemcpy(rec_scalar_GPU_to_CPU,rec_scalar_GPU,sizeof (Data_type) * thread_block_num,cudaMemcpyDeviceToHost));
    /** 规约 **/
    for(int i=0;i<thread_block_num;i++)
    {
        rec_scalar_GPU_final+=rec_scalar_GPU_to_CPU[i];
    }

    /** compare **/
    if(fabs(rec_scalar_GPU_final - rec_scalar) > 1e-5)
    {
        printf("compute error!\n");
        printf("recVal: %lf,%lf\n",rec_scalar_GPU_final,rec_scalar);
    }
    else {
        printf("compare success!\n");
    }
    /** 释放内存 **/
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    free(matrixDataCsr);
    free(Sp_Vec);
    free(Sparse_Vector);

    free(rec_scalar_GPU_to_CPU);
    CHECK(cudaFree(Sp_vec_GPU_index));
    CHECK(cudaFree(Sp_vec_GPU_val));
    CHECK(cudaFree(Sparse_Vector_GPU));
    CHECK(cudaFree(rec_scalar_GPU));
}

#endif //CUDALEARN_COMPARE_H
