#!/bin/bash
while IFS= read -r line; do
  first_part="${line%% *}"
  make run input_matrix="none" test_matrix="/data/suitesparse_collection/${first_part}.mtx" log_path="/data/seery/src/log/reslut/cuda.log";
  #echo ${first_part}
done < "/data/seery/src/log/reslut/suitesparse.txt"