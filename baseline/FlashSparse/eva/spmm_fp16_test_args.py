import os
import sys
import csv
from fs_fp16 import test_fs

def fs_fp16_8_1(data, dimN, epoches, data_path, window, wide) :     
    spmm = test_fs.fs_fp16_8_1(data, epoches, dimN, data_path, window, wide)
    return spmm

def fs_fp16_8_1_map(data, dimN, epoches, data_path, window, wide) :     
    spmm = test_fs.fs_fp16_8_1_map(data, epoches, dimN, data_path, window, wide)
    return spmm
             
if __name__ == "__main__":
    dimN = int(sys.argv[1]) #256 & 128
    print('dimN: ' + str(dimN))
 
    epoches = 1
    partsize_t = 32

    current_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    dataset_dir = os.path.join(project_dir, 'dataset')

    # reference csv file
    input_file = os.path.join(dataset_dir, 'dataset.csv')
    # output csv file
    output_file = os.path.join(project_dir, 'result', 'baseline', f'FlashSparse_{str(dimN)}.csv')

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='') as outfile, open(input_file, 'r') as infile:
        csv_reader = csv.DictReader(infile)

        fieldnames = csv_reader.fieldnames[1:] + ['FlashSparse_pre(ms)', 'FlashSparse(ms)', 'FlashSparse(GFLOPS)', 
                                            'FlashSparse_map_pre(ms)', 'FlashSparse_map(ms)', 'FlashSparse_map(GFLOPS)']
        csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        csv_writer.writeheader()

        for row in csv_reader:
            try:
                if row['rows'] != row['cols']:
                    print(f"Skipping {row['name']} as it is not square.")
                    continue
                new_row = {key: row[key] for key in fieldnames if key in row.keys()}

                n_nnzs = int(row['nnzs'])
                mm_name = row['name']
                mm_path = os.path.join(dataset_dir, row['path'])

                # 8x1
                ms_8_1, pre_8_1 = fs_fp16_8_1(mm_name, dimN, epoches, mm_path, 8, 8)
                new_row['FlashSparse_pre(ms)'] = f'{pre_8_1:.5f}'
                new_row['FlashSparse(ms)'] = f'{ms_8_1:.5f}'
                gflops_8_1 = 2 * n_nnzs * dimN / ms_8_1 / 1e6
                new_row['FlashSparse(GFLOPS)'] = f'{gflops_8_1:.2f}'

                # test-map
                ms_8_1_map, pre_8_1_map = fs_fp16_8_1_map(mm_name, dimN, epoches, mm_path, 8, 8)
                new_row['FlashSparse_map_pre(ms)'] = f'{pre_8_1_map:.5f}'
                new_row['FlashSparse_map(ms)'] = f'{ms_8_1_map:.5f}'
                gflops_8_1_map = 2 * n_nnzs * dimN / ms_8_1_map / 1e6
                new_row['FlashSparse_map(GFLOPS)'] = f'{gflops_8_1_map:.2f}'

                csv_writer.writerow(new_row)
                print(mm_name + ' is success')
            except Exception as e:
                print(f"Error processing {row['name']}: {str(e)}")
    print('All is success')