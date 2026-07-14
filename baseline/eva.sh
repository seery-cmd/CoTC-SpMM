#!/bin/bash

# Evaluate FlashSparse
cd FlashSparse/eva
python spmm_fp16_test_args.py 256
cd ../..

# Evaluate DTC-SpMM
cd DTC-SpMM
python run_DTC_SpMM.py 256
cd ..

# Evaluate cuSPARSE
cd cuSPARSE
python eva.py 256
cd ..

# Evaluate RoDe Sputnik cuSPARSE
cd RoDe/script
python eval.py 256
cd ../..