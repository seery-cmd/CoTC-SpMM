### We reuse the code from TC-GNN ()
#!/usr/bin/env python3
import torch
import numpy as np
import time
import scipy.io as sio
from scipy.sparse import *

torch.manual_seed(0)
class DTC_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, path, verbose=False):
        super(DTC_dataset, self).__init__()
        self.nodes = set()
        self.num_nodes = 0
        self.edge_index = None
        self.verbose_flag = verbose
        self.init_sparse(path)
    
    def init_sparse(self, path):
        #start = time.perf_counter()
        self.graph = sio.mmread(path)

        src_li = self.graph.row
        dst_li = self.graph.col

        self.num_nodes_ori =  self.graph.shape[0]
        self.num_nodes_dst =  self.graph.shape[1]

        self.num_nodes = (self.num_nodes_ori + 15) // 16 * 16
        self.num_edges = self.graph.nnz

        self.avg_degree = self.num_edges / self.num_nodes
        val = [1] * self.num_edges
        
        scipy_coo = coo_matrix((val, (src_li, dst_li)), shape=(self.num_nodes, self.num_nodes_dst))
        adj = scipy_coo.tocsr()

        #build_csr = time.perf_counter() - start
        #if self.verbose_flag:
        #    print("# Build CSR (s): {:.3f}".format(build_csr))

        self.column_index = torch.IntTensor(adj.indices)
        self.row_pointers = torch.IntTensor(adj.indptr)
        