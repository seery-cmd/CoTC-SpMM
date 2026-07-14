import os
import sys
import csv
import time
import torch
import DTCSpMM
from dataset import *

BLK_H = 16
BLK_W = 8

dimN = int(sys.argv[1])
print('dimN: ' + str(dimN))

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(current_dir))
dataset_dir = os.path.join(project_dir, 'dataset')

# reference csv file
input_file = os.path.join(dataset_dir, 'dataset.csv')
# output csv file
output_file = os.path.join(project_dir, 'result', 'baseline', f'DTC_{str(dimN)}.csv')

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', newline='') as outfile, open(input_file, 'r') as infile:
    csv_reader = csv.DictReader(infile)
    fieldnames = csv_reader.fieldnames[1:] + ['DTC_pre(ms)', 'DTC(ms)', 'DTC(GFLOPS)']

    csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    csv_writer.writeheader()

    for row in csv_reader:
        try:
            new_row = {key: row[key] for key in fieldnames if key in row.keys()}

            mm_name = row['name']
            mm_path = os.path.join(dataset_dir, row['path'])

            matrix = DTC_dataset(mm_path)
            num_rows = matrix.num_nodes
            num_nnz = matrix.num_edges

            column_index =  matrix.column_index 
            row_pointers =  matrix.row_pointers 

            # Process data.
            num_row_windows = (num_rows + BLK_H - 1) // BLK_H
            edgeToColumn = torch.zeros(num_nnz, dtype=torch.int)
            edgeToRow = torch.zeros(num_nnz, dtype=torch.int)
            blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
            column_index_ori  = column_index.cuda()
            row_pointers_ori = row_pointers.cuda()

            blockPartition_cuda  = blockPartition.cuda()
            edgeToColumn_cuda = edgeToColumn.cuda()
            edgeToRow_cuda  = edgeToRow.cuda()

            # Optimize GPU.
            RowWindowOffset, TCblockRowid,\
                TCblocktileId, TCblockoffset, SparseAToXindex,\
                    block_count, pre_ms = DTCSpMM.preprocess_gpu(column_index_ori, row_pointers_ori, num_rows, BLK_H, BLK_W, blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)

            new_row['DTC_pre(ms)'] = f'{pre_ms:.5f}'

            X = torch.ones((num_rows, dimN)).cuda()
            # Run test.

            balance_choice = True
            exeplan = 'float4' + '_' + 'split'
            if balance_choice == False:
                _, dtc_spmm = DTCSpMM.run_DTCSpMM(X, RowWindowOffset, TCblocktileId, TCblockoffset, SparseAToXindex, num_rows, num_nnz, exeplan)
            else:
                _, dtc_spmm = DTCSpMM.run_DTCSpMM_balance(X, TCblockRowid, TCblocktileId, TCblockoffset, SparseAToXindex, num_rows, exeplan)

            spmm_ms = dtc_spmm.item()
            gflops = 2 * num_nnz * dimN / spmm_ms / 1e6

            new_row['DTC(ms)'] = f'{spmm_ms:.5f}'
            new_row['DTC(GFLOPS)'] = f'{gflops:.2f}'

            csv_writer.writerow(new_row)

            print(f'{mm_name} is success')
        except Exception as e:
            print(f'Error: {str(e)}')