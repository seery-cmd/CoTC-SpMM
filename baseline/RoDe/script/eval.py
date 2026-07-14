import os
import csv
import sys
import subprocess

DimN = int(sys.argv[1])  # 256 & 128
print('DimN: ' + str(DimN))

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
dataset_dir = os.path.join(project_dir, 'dataset')

# reference csv file
input_file = os.path.join(dataset_dir, 'dataset.csv')
# output csv file
output_file = os.path.join(project_dir, 'result', 'baseline', f'RoDe_{str(DimN)}.csv')

os.makedirs(os.path.dirname(output_file), exist_ok=True)

fieldnames = ['name','rows','cols','nnzs','Sputnik(ms)','Sputnik(GFLOPS)','cuSPARSE(ms)','cuSPARSE(GFLOPS)','RoDe(ms)','RoDe(GFLOPS)']

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    csv_reader = csv.DictReader(infile)
    csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    csv_writer.writeheader()

    for row in csv_reader:
        new_row = {key: row[key] for key in fieldnames if key in row.keys()}

        file_path = os.path.join(dataset_dir, row['path'])
        shell_command = f"{project_dir}/baseline/RoDe/build/eval/eval_spmm_f32_n{DimN} {file_path}"
        
        result = subprocess.run(shell_command, capture_output=True, text=True, shell=True)
        output_values = result.stdout.strip().split()
        new_row['Sputnik(ms)'] = output_values[0]
        new_row['Sputnik(GFLOPS)'] = output_values[1]
        new_row['cuSPARSE(ms)'] = output_values[2]
        new_row['cuSPARSE(GFLOPS)'] = output_values[3]
        new_row['RoDe(ms)'] = output_values[4]
        new_row['RoDe(GFLOPS)'] = output_values[5]
        csv_writer.writerow(new_row)
            
        print(f'{row["name"]} processed.')

print(f'DimN({DimN})Evaluation completed.')
