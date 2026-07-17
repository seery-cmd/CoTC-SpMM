import os.path as osp
import argparse
import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import *
import torch.cuda as cuda
from scipy.io import *
import HCSPMM
from dataset import *
from GNN_model import *
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='DD_A_our_3', help="dataset")
parser.add_argument("--dim", type=int, default=128, help="input embedding dimension")
parser.add_argument("--num_layers", type=int, default=6, help="num layers")
parser.add_argument("--hidden", type=int, default=32, help="hidden dimension")
parser.add_argument("--classes", type=int, default=22, help="number of output classes")
parser.add_argument("--epochs", type=int, default=200, help="number of epoches")
parser.add_argument("--model", type=str, default='gcn', help='GNN model', choices=['gcn', 'gin'])
parser.add_argument("--single_kernel", action='store_true', help="whether to profile a single SAG kernel")
args = parser.parse_args()
print(args)

dataset = args.dataset
path = osp.join("./Dataset/", dataset + ".txt")
dataset = HCSPMM_dataset(path, args.dim, args.classes, load_from_txt=True)
num_nodes = dataset.num_nodes
num_edges = dataset.num_edges
column_index =  dataset.column_index
row_pointers = dataset.row_pointers
print(num_nodes,num_edges)
num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
output = torch.zeros(num_nodes * args.hidden, dtype=torch.float).reshape(num_nodes, args.hidden)

column_index = column_index.cuda()
row_pointers = row_pointers.cuda()
output = output.cuda()

start = time.perf_counter()
blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = HCSPMM.preprocess(column_index, row_pointers, num_nodes, num_edges, num_row_windows)
build_neighbor_parts = time.perf_counter() - start
print("Prep. (ms):\t{:.3f}".format(build_neighbor_parts*1e3))

SAG_obj = SAG(row_pointers, column_index,blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
X = torch.randn(num_nodes, args.dim).cuda()
time_cost = SAG_obj.profile(X)
print("Gflops={:.3f}".format(num_edges/1e6/time_cost*2.0*args.dim))
#貌似只会执行一个行窗口前面3个TC block,同时如果dim=96的话dense B也只会执行48行,因为只有3个warp在工作.详细在spmm_forward_cuda_kernel_arbi_warps_hybrid_32文件中
#在应用到suitesparse Matrix collection上面会出现报错
#重新编写了single kernel 内核
#区分cuda还是tensor执行的方式只是一个简单的判别式

exit(0)