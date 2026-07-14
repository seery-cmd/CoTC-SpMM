#include "kernel/spmm_memory.cu"
#include "kernel/spmm_cuda.cu"
int main(int argc,char *argv[])
{
    printf("test matrix = %s\n",argv[2]);
    Matrix_data_CSR *sparse_matrix = (Matrix_data_CSR *)malloc(sizeof (Matrix_data_CSR));
    Read_sparse_matrix(sparse_matrix,argv[2]);//读取矩阵
    printf("%d-%d-%d\n",sparse_matrix->row,sparse_matrix->col,sparse_matrix->nnz);
    /** write **/
    fstream f;
    f.open("/data/seery/src/log/reslut/preprogress/CoTC_pre.txt",ios::out|ios::app);
    f<<argv[2]<<" "<<sparse_matrix->row<<" "<<sparse_matrix->col<<" "<<sparse_matrix->nnz<<" ";
    f.close();
//    for(int i =0;i<sparse_matrix->nnz;i++)
//    {
//        sparse_matrix->val[i] = (Data_type) 1.0;
//    }
//    Printf_matrix_CSR(sparse_matrix);
    /** 构造稠密矩阵 size: sparse_mtx * BN **/
    Matrix_data_CSR *Dense_mtx_CSR = (Matrix_data_CSR *)malloc(sizeof (Matrix_data_CSR));
    creat_dense_matrix_cpu(Dense_mtx_CSR,sparse_matrix->col + 8,BN);
    cout<<"creat dense matrix success!\n\n"<<endl;
//    printSparseMatrix(Dense_mtx_CSR);
    /** 构造结果(rec)矩阵 cpu和gpu**/
    //Matrix_data_CSR *rec_Dense_mtx_cpu = (Matrix_data_CSR *)malloc(sizeof (Matrix_data_CSR));//CPU
    Matrix_data_CSR *rec_Dense_mtx_GPU = (Matrix_data_CSR *)malloc(sizeof (Matrix_data_CSR));//nscore
    Matrix_data_CSR *rec_Dense_mtx_cuda = (Matrix_data_CSR *)malloc(sizeof (Matrix_data_CSR));//cuda core
    //creat_rec_dense_matrix(rec_Dense_mtx_cpu,sparse_matrix->row,BN);
    creat_rec_dense_matrix(rec_Dense_mtx_GPU,sparse_matrix->row,BN);
    creat_rec_dense_matrix(rec_Dense_mtx_cuda,sparse_matrix->row,BN);
    cout<<"creat rec dense matrix CPU and GPU success!\n\n"<<endl;
    /** CPU 计算 **/
//    struct timeval t1,t2;
//    gettimeofday(&t1,NULL);
//    SpMM_cpu(sparse_matrix,Dense_mtx_CSR,rec_Dense_mtx_cpu);//cpu计算Spmm
//    gettimeofday(&t2,NULL);
//    double timecost_cpu = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
//    std::cout << "CPU time: " << timecost_cpu << "ms" << std::endl;
    /** GPU nscore计算 **/
    SpMM_GPU(sparse_matrix,Dense_mtx_CSR,rec_Dense_mtx_GPU);
    /** GPU cuda core验证计算 **/
    SpMM_cuda(sparse_matrix,Dense_mtx_CSR,rec_Dense_mtx_cuda);
    /** compare cpu and gpu **/
    //输出前100个
    for(int i=0;i<BN * 100;i+=BN)
    {
        if(i > BN * sparse_matrix->row)
        {
            break;
        }
        printf("tid=%d:%f,%f\n", i, __half2float(rec_Dense_mtx_GPU->val[i]),__half2float(rec_Dense_mtx_cuda->val[i]));
    }
    /** 释放内存 **/
    Destroy_matrixCSR_cpu(sparse_matrix);
    Destroy_matrixCSR_cpu(Dense_mtx_CSR);
//    Destroy_matrixCSR_cpu(rec_Dense_mtx_cpu);
    Destroy_matrixCSR_cpu(rec_Dense_mtx_GPU);
    Destroy_matrixCSR_cpu(rec_Dense_mtx_cuda);
}