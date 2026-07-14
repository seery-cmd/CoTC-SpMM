import os
import sys
from fs_fp16.mdataset2 import *
import FS_SpMM

# 8x1
def fs_fp16_8_1(data, epoches, dimN, data_path,  window, wide):

    inputInfo = dataSet_fp16(dimN, data_path, window, wide)
    
    X_prime, spmm_ms_avg  = FS_SpMM.forward_fp16_test(   
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        inputInfo.degrees, 
        inputInfo.x, 
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, epoches, 4)
    
    spmm_ms_avg = round((spmm_ms_avg.item()),5)
    print(str(dimN) + '-' + data + 'tcu_8_1' + '-' +str(spmm_ms_avg))
    return [spmm_ms_avg, inputInfo.exe]

def fs_fp16_8_1_map(data, epoches, dimN, data_path,  window, wide):

    inputInfo = dataSet_fp16(dimN, data_path, window, wide)
    
    X_prime, spmm_ms_avg  = FS_SpMM.forward_fp16_map(   
        inputInfo.row_pointers, 
        inputInfo.column_index, 
        inputInfo.degrees, 
        inputInfo.x, 
        inputInfo.num_nodes, 
        inputInfo.x.size(1), 
        inputInfo.num_nodes_ori, epoches, 4)
    
    spmm_ms_avg = round((spmm_ms_avg.item()),5)
    print(str(dimN) + '-' + data + 'tcu_8_1_map' + '-' +str(spmm_ms_avg))
    return [spmm_ms_avg, inputInfo.exe]