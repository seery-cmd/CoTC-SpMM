import os
import sys
import csv
import torch
import scipy.io as sio
from scipy.sparse import coo_matrix
import cusparse_spmm_csr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dimN = int(sys.argv[1])
print('dimN: ' + str(dimN))

epoches = 10

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(current_dir))
dataset_dir = os.path.join(project_dir, 'dataset')

# reference csv file
input_file = os.path.join(dataset_dir, 'dataset.csv')
# output csv file
output_file = os.path.join(project_dir, 'result', 'baseline', f'cuSPARSE_{str(dimN)}.csv')

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', newline='') as outfile, open(input_file, 'r') as infile:
    csv_reader = csv.DictReader(infile)
    fieldnames = csv_reader.fieldnames[1:] + ['cuSPARSE(ms)', 'cuSPARSE(GFLOPS)']
    csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    csv_writer.writeheader()

    for row in csv_reader:
        try:
            new_row = {key: row[key] for key in fieldnames if key in row.keys()}

            mm_name = row['name']
            mm_path = os.path.join(dataset_dir, row['path'])
            mm = sio.mmread(mm_path)
            val = [1] * mm.nnz

            scipy_coo = coo_matrix((val, (mm.row, mm.col)), shape=mm.shape)
            adj = scipy_coo.tocsr()

            row_ptr = torch.IntTensor(adj.indptr)
            col_inx = torch.IntTensor(adj.indices)
            a_value = torch.HalfTensor(adj.data)
            b_value = torch.randn(mm.shape[1], dimN).half()

            row_ptr = row_ptr.to(device)
            col_inx = col_inx.to(device)
            a_value = a_value.to(device)
            b_value = b_value.to(device)

            ms_avg = cusparse_spmm_csr.cuSPARSE_SPMM_CSR(row_ptr, col_inx, a_value, b_value, mm.shape[0], mm.shape[1], dimN, mm.nnz, epoches, 1)

            gflops = mm.nnz * 2 * dimN / ms_avg / 1e6

            new_row['cuSPARSE(ms)'] = f'{ms_avg:.5f}'
            new_row['cuSPARSE(GFLOPS)'] = f'{gflops:.2f}'

            csv_writer.writerow(new_row)
            
            print(f'{mm_name} is success: {ms_avg:.5f}(ms), {gflops:.2f}(GFLOPS)')
        except Exception as e:
            print(f'error: {str(e)}')

    print('All is success')