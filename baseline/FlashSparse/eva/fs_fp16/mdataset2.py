#!/usr/bin/env python3
import torch
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import *
import scipy.io as sio
import FS_Block_gpu

# fp16
class dataSet_fp16(torch.nn.Module):

    def __init__(self, dimN, mm_path, window, wide):
        super(dataSet_fp16, self).__init__()

        self.graph = sio.mmread(mm_path)
        self.num_features = dimN #256 & 128

        self.init_edges(window, wide) # 32,8,8
        self.init_embedding()
         
        
    def init_edges(self, window, wide):
        self.num_nodes_ori =  self.graph.shape[0]
        self.num_nodes_dst =  self.graph.shape[1]
        
        self.num_nodes = self.num_nodes_ori
        if self.num_nodes_ori%16 !=0 :
            self.num_nodes = (self.num_nodes_ori + 16 - 1) // 16 * 16 # 上取整
        
        self.num_edges = self.graph.nnz

        src_li = self.graph.row
        dst_li = self.graph.col

        self.edge_index = np.stack([src_li, dst_li])

        self.avg_degree = self.num_edges / self.num_nodes
        val = [1] * self.num_edges #val = row + 1

        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr() 
        
        self.column_index = torch.IntTensor(adj.indices) #col_idx
        self.row_pointers = torch.IntTensor(adj.indptr) #row_ptr
        self.degrees = torch.randn(self.num_edges).half() #nnz长度
        
        self.row_pointers, \
        self.column_index, \
        self.degrees, self.exe = FS_Block_gpu.preprocess_gpu_fs(self.row_pointers, self.column_index, self.num_nodes, self.num_edges, window, wide)

    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)
        self.x = self.x.half()
       