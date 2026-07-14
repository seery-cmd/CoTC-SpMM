#include "../spmm_utils/dense_tile.h"
#include "../spmm_utils/compute_utils.h"
#include "../spmm_utils/output_tile.h"
#include <stdio.h>
#include <mma.h>
#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>


/*
TF16-8x1-balance
*/
template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_fp16_balance(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const int* t_window_row,
    const int* t_atomic,
    const double* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int parts_t,
    long nOri,
    int mOri,
    int splitk)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=parts_t)
    return;

    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    // if((dimN_index+((lane_id/32+1)*16))>dimN)
    // return;
    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_fp16_map dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    // mmaDenseTile_fp16_test dense_tile_loader(row_offset_vec, values, col_indices,
    //     nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    // );
    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    half * output_fragment_half = reinterpret_cast<half *>(output_fragment);
    mmaComputeUtils_fp16_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            computer.TileMAC();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    int cur_t_atomic = __ldg(t_atomic + m_index_vec);
    int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
    int col=dimN_index + warp_id*16 + + (warpin_id/4)*2;

    if(row<mOri)
    {
        float * output_matrix_ = output_matrix +(row*nOri)+col;
        if(cur_t_atomic==0)
        {
            if(col<nOri)
            *(output_matrix_ ) = __half2float(output_fragment_half[0]);
            if((col+1)<nOri)
            *(output_matrix_+1) =  __half2float(output_fragment_half[2]);
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                if(col<nOri)
                *(output_matrix_) = __half2float(output_fragment_half[1]);
                if((col+1)<nOri)
                *(output_matrix_+1) = __half2float( output_fragment_half[3]);
            }
        }else{
            if(col<nOri)
            atomicAdd(output_matrix_ ,__half2float(output_fragment_half[0]));
            if((col+1)<nOri)
            atomicAdd(output_matrix_+1, __half2float(output_fragment_half[2]));
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                if(col<nOri)
                atomicAdd(output_matrix_ , __half2float(output_fragment_half[1]));
                if((col+1)<nOri)
                atomicAdd(output_matrix_+1 , __half2float(output_fragment_half[3]));
            }
        }
    }
//    mmaOutputTile_fp16 output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
//     output_tile_storer.Store(cur_m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri,cur_t_atomic);
}

float spmm_forward_cuda_fp16_balance(
    int * row_offsets,
    int * col_indices, 
    double * values,
    int* t_window_row,
    int * t_atomic,
    int parts_t, 
    double * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    int n1=dimN;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    //预热
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim(grid_x, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim(128, 1, 1);
    for(int iter=0; iter<1; ++iter){
        spmm_forward_cuda_kernel_fp16_balance<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            t_window_row,
            t_atomic,
            rhs_matrix, 
            output_matrix,
            n1, parts_t, dimN, mOri,splitk_t);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_cuda_kernel_fp16_balance<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            t_window_row,
            t_atomic,
            rhs_matrix, 
            output_matrix,
            n1, parts_t, dimN, mOri,splitk_t);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    
    return spmm_ms_avg;
}


/*
TF16-16x1
*/
template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_fp16_16(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const double* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri)
{
    //每个block5个warp，最后一个warp用于计算
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*8))>dimN)
    return;
    if((blockIdx.z*100+blockIdx.y)>=dimM)
    return;

    // if(blockIdx.x==0 & (blockIdx.z*8+blockIdx.y)==0 & threadIdx.x==0)
    // {
    //     const half * p = reinterpret_cast<const half*>(rhs_matrix);
    //     for(int i=0;i<7;i++){
    //     printf("%f ", __half2float(*p));
    //     p+=1;}
    //     printf("\n");
    // }

    int m_index_vec = (blockIdx.z*100)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[2] = {0.0, 0.0};
    float dense_fragment[1] = {0.0};
    mmaDenseTile_fp16_16 dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index/4, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );

    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    mmaComputeUtils_fp16_16 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            // sparse_tile_loader.Load();
            // __syncwarp();
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            // __syncthreads();
            computer.TileMAC();
            // __syncwarp();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        
        __syncwarp();
        computer.TileMACResidue();
    }  
   mmaOutputTile_fp16_16 output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
    // if(blockIdx.x==0 & (blockIdx.z*8+blockIdx.y)==0 & threadIdx.x==0)
    // {
    //     const half * p = reinterpret_cast<const half*>(output_fragment);
    //     for(int i=0;i<4;i++){
    //     printf("%f ", __half2float(*p));
    //     p+=1;}
    //     printf("\n");
    // }
}

float spmm_forward_cuda_fp16_16(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    double * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    // int splitk = 0;
    // if(dimM<500000) splitk=8;
    // else splitk=((dimM/1250000)+1)*20;

    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN%8)!=0) n1=((dimN>>3)+1)<<3;

    int grid_x = (n1>>5)+1;
    if(n1%32==0) grid_x-=1;
    dim3 grid_dim(grid_x, 100 ,((dimM/100)+1));
    dim3 block_dim(128, 1, 1);

    for(int iter=0; iter<1; ++iter){

        spmm_forward_cuda_kernel_fp16_16<32><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_cuda_kernel_fp16_16<32><<<grid_dim, block_dim>>>(
        row_offsets, 
        col_indices, 
        values, 
        rhs_matrix, 
        output_matrix,
        n1, dimM, dimN, mOri);
    
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    
    return spmm_ms_avg;
}


//map
__global__ void spmm_forward_cuda_kernel_fp16_map(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const double* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri,
    int Tile_N)
{
    //每个block5个warp，最后一个warp用于计算
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*16))>dimN)
    return;
    if((blockIdx.z*200+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*200)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_fp16_map dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );

    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    mmaComputeUtils_fp16_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            computer.TileMAC();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
   mmaOutputTile_fp16_map output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
}

float spmm_forward_cuda_fp16_map(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    double * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches,
    int warps)
{
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    int Tile_N = warps*16;
    int grid_x = (n1/Tile_N)+1;
    if(n1%Tile_N==0) grid_x-=1;
    dim3 grid_dim(grid_x, 200 ,((dimM/200)+1));
    dim3 block_dim(warps*32, 1, 1);
    for(int iter=0; iter<1; ++iter){
        spmm_forward_cuda_kernel_fp16_map<<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri,Tile_N);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_cuda_kernel_fp16_map<<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri,Tile_N);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    
    return spmm_ms_avg;
}


__global__ void spmm_forward_cuda_kernel_fp16_test(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const float2* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri,
    int Tile_N)
{
    //每个block5个warp，最后一个warp用于计算
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*16))>dimN)
    return;
    if((blockIdx.z*200+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*200)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[260];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};

    mmaDenseTile_fp16_test dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );

    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    mmaComputeUtils_fp16_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            computer.TileMAC();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
//    mmaOutputTile_fp16 output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
//     output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
    mmaOutputTile_fp16_test output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, reinterpret_cast< float2 *>(output_matrix) ,mOri,nOri);
}

float spmm_forward_cuda_fp16_test(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    float2 * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches,
    int warps)
{
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;

    int Tile_N = warps*16;
    int grid_x = (n1/Tile_N)+1;
    if(n1%Tile_N==0) grid_x-=1;
    dim3 grid_dim(grid_x, 200 ,((dimM/200)+1));
    dim3 block_dim(warps*32, 1, 1);
    for(int iter=0; iter<1; ++iter){
        spmm_forward_cuda_kernel_fp16_test<<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri,Tile_N);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_cuda_kernel_fp16_test<<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri,Tile_N);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    
    return spmm_ms_avg;
}

/*
TF32-8x1
*/
template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_tf32(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri)
{
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*16))>dimN)
    return;
    if((blockIdx.z*200+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*200)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_tf32_v2 dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    //output_fragment必须为float
    float output_fragment[4] = {0.0,0.0,0.0,0.0};
    mmaComputeUtils_tf32_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    int steps = nonzeros>>2;
    int residue = nonzeros%4;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            // sparse_tile_loader.Load();
            // __syncwarp();
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            // if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     for(int i=0; i<16;i++)
            //     for(int j=0; j<4;j++){
            //     printf("%f ", dense_tile[64+i*4+j]);
            //     if(j==3)
            //     printf("\n");}
            // }
            // __syncthreads();
            computer.TileMAC();
            // __syncwarp();
        }
    }
//     if(threadIdx.x==32 & blockIdx.y==0)
// {
//     printf("%f\n", output_fragment[0]);
//     printf("%f\n", output_fragment[1]);
//     printf("%f\n", output_fragment[2]);
//     printf("%f\n", output_fragment[3]);
// }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri, dimN_index,residue);
        __syncwarp();
        // if(threadIdx.x==32 & blockIdx.y==0)
        //     {
        //         printf("%f\n", dense_tile[16]);
        //         printf("%f\n", dense_tile[17]);
        //     }
        //    if(threadIdx.x==5 & blockIdx.y==0)
        //     {
        //         printf("%f\n", sparse_fragment[0]);
        //         printf("%f\n", dense_tile[0]);
        //         printf("%f\n", dense_tile[1]);
        //     }
        computer.TileMACResidue();
    }  
   mmaOutputTile_tf32 output_tile_storer(lane_id,output_fragment);
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
}

float spmm_forward_cuda_tf32(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    // if(dimM<500000) splitk=8;
    // else splitk=((dimM/1250000)+1)*20;
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    dim3 grid_dim(grid_x, 200 ,((dimM/200)+1));
    dim3 block_dim(128, 1, 1);

    for(int iter=0; iter<1; ++iter){
        
        spmm_forward_cuda_kernel_tf32<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
            spmm_forward_cuda_kernel_tf32<64><<<grid_dim, block_dim>>>(
        row_offsets, 
        col_indices, 
        values, 
        rhs_matrix, 
        output_matrix,
        n1, dimM, dimN, mOri);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_tf32_map(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri)
{
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*16))>dimN)
    return;
    if((blockIdx.z*200+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*200)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_tf32_v2_map dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    //output_fragment必须为float
    float output_fragment[4] = {0.0,0.0,0.0,0.0};
    mmaComputeUtils_tf32_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    int steps = nonzeros>>2;
    int residue = nonzeros%4;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            // sparse_tile_loader.Load();
            // __syncwarp();
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            // if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     for(int i=0; i<16;i++)
            //     for(int j=0; j<4;j++){
            //     printf("%f ", dense_tile[64+i*4+j]);
            //     if(j==3)
            //     printf("\n");}
            // }
            // __syncthreads();
            computer.TileMAC();
            // __syncwarp();
        }
    }
//     if(threadIdx.x==32 & blockIdx.y==0)
// {
//     printf("%f\n", output_fragment[0]);
//     printf("%f\n", output_fragment[1]);
//     printf("%f\n", output_fragment[2]);
//     printf("%f\n", output_fragment[3]);
// }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri, dimN_index,residue);
        __syncwarp();
        // if(threadIdx.x==32 & blockIdx.y==0)
        //     {
        //         printf("%f\n", dense_tile[16]);
        //         printf("%f\n", dense_tile[17]);
        //     }
        //    if(threadIdx.x==5 & blockIdx.y==0)
        //     {
        //         printf("%f\n", sparse_fragment[0]);
        //         printf("%f\n", dense_tile[0]);
        //         printf("%f\n", dense_tile[1]);
        //     }
        computer.TileMACResidue();
    }  
   mmaOutputTile_tf32_map output_tile_storer(lane_id,output_fragment);
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
}

float spmm_forward_cuda_tf32_map(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    // if(dimM<500000) splitk=8;
    // else splitk=((dimM/1250000)+1)*20;
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    dim3 grid_dim(grid_x, 200 ,((dimM/200)+1));
    dim3 block_dim(128, 1, 1);

    for(int iter=0; iter<1; ++iter){
        
        spmm_forward_cuda_kernel_tf32_map<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
            spmm_forward_cuda_kernel_tf32_map<64><<<grid_dim, block_dim>>>(
        row_offsets, 
        col_indices, 
        values, 
        rhs_matrix, 
        output_matrix,
        n1, dimM, dimN, mOri);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


/*
TF32-8x1 balance
*/
template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_tf32_balance(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const int* t_window_row,
    const int* t_atomic,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int parts_t,
    long nOri,
    int mOri,
    int splitk)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=parts_t)
    return;

    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;

    // //排除部分warp
    // if((blockIdx.z*200+blockIdx.y)>=dimM)
    // return;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_tf32_v2_map dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    //output_fragment必须为float
    float output_fragment[4] = {0.0,0.0,0.0,0.0};
    mmaComputeUtils_tf32_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    int steps = nonzeros>>2;
    int residue = nonzeros%4;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            // sparse_tile_loader.Load();
            // __syncwarp();
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            // if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     for(int i=0; i<16;i++)
            //     for(int j=0; j<4;j++){
            //     printf("%f ", dense_tile[64+i*4+j]);
            //     if(j==3)
            //     printf("\n");}
            // }
            // __syncthreads();
            computer.TileMAC();
            // __syncwarp();
        }
    }


    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri, dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
        //原子写入gloabl
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + (warpin_id/4)*2;

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = output_fragment[0];
                if((col+1)<nOri)
                *(output_matrix_+1) =  output_fragment[2];
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = output_fragment[1];
                    if((col+1)<nOri)
                    *(output_matrix_+1) =  output_fragment[3];
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ , output_fragment[0]);
                if((col+1)<nOri)
                atomicAdd(output_matrix_+1, output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , output_fragment[1]);
                    if((col+1)<nOri)
                    atomicAdd(output_matrix_+1 , output_fragment[3]);
                }
            }
        }
//    mmaOutputTile_tf32 output_tile_storer(lane_id,output_fragment);
//     output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
}

float spmm_forward_cuda_tf32_balance(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    int* t_window_row,
    int * t_atomic,
    int parts_t,
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim(grid_x, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim(128, 1, 1);

    for(int iter=0; iter<1; ++iter){
        
        spmm_forward_cuda_kernel_tf32_balance<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            t_window_row,
            t_atomic,
            rhs_matrix, 
            output_matrix,
            n1, parts_t, dimN, mOri,splitk_t);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
            spmm_forward_cuda_kernel_tf32_balance<64><<<grid_dim, block_dim>>>(
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        t_atomic,
        rhs_matrix, 
        output_matrix,
        n1, parts_t, dimN, mOri,splitk_t);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

/*
TF32-16x1
*/
template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_tf32_16(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri)
{
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*8))>dimN)
    return;
    if((blockIdx.z*100+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*100)+blockIdx.y;
    // if(blockIdx.x==0 and blockIdx.y==1 and threadIdx.x==0)
    // printf("%d\n", m_index_vec);
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
  
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[2] = {0.0, 0.0};
    float dense_fragment[1] = {0.0};
    mmaDenseTile_tf32_16 dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    //output_fragment必须为float
    float output_fragment[4] = {0.0,0.0,0.0,0.0};
    mmaComputeUtils_tf32_16 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    int steps = nonzeros>>2;
    int residue = nonzeros%4;
    if(steps > 0){
        // #pragma unroll
        for(int i = 0; i < steps; i++){
            // sparse_tile_loader.Load();
            // __syncwarp();
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            
            // if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     for(int i=0; i<16;i++)
            //     for(int j=0; j<4;j++){
            //     printf("%f ", dense_tile[64+i*4+j]);
            //     if(j==3)
            //     printf("\n");}
            // }
            // __syncthreads();
            computer.TileMAC();
            // __syncwarp();
        }
    }
//     if(threadIdx.x==32 & blockIdx.y==0)
// {
//     printf("%f\n", output_fragment[0]);
//     printf("%f\n", output_fragment[1]);
//     printf("%f\n", output_fragment[2]);
//     printf("%f\n", output_fragment[3]);
// }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri, dimN_index,residue);
        __syncwarp();
        // if(threadIdx.x==32 & blockIdx.y==0)
        //     {
        //         printf("%f\n", dense_tile[16]);
        //         printf("%f\n", dense_tile[17]);
        //     }
        //    if(threadIdx.x==5 & blockIdx.y==0)
        //     {
        //         printf("%f\n", sparse_fragment[0]);
        //         printf("%f\n", dense_tile[0]);
        //         printf("%f\n", dense_tile[1]);
        //     }
        computer.TileMACResidue();
    }  
   mmaOutputTile_tf32_16 output_tile_storer(lane_id,output_fragment);
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
    // if(blockIdx.x==0 and blockIdx.y==1 and threadIdx.x==0)
    // printf("%f\n", output_fragment[0]);
}

float spmm_forward_cuda_tf32_16(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    // int splitk = 200;
    // if(dimM<500000) splitk=8;
    // else splitk=((dimM/1250000)+1)*20;
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN%8)!=0) n1=((dimN>>3)+1)<<3;
    int grid_x = (n1>>5)+1;
    if(n1%32==0) grid_x-=1;
    dim3 grid_dim(grid_x, 100 ,((dimM/100)+1));
    dim3 block_dim(128, 1, 1);

    for(int iter=0; iter<1; ++iter){
        
        spmm_forward_cuda_kernel_tf32_16<32><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
            spmm_forward_cuda_kernel_tf32_16<32><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    
    return spmm_ms_avg;
}


