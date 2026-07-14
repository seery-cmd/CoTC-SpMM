#!/bin/bash

# Build RoDe Sputnik cuSPARSE
cd RoDe
rm -rf build
mkdir build
cd build
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc ..
make

# Install FlashSparse, DTC-SpMM, cuSPARSE
cd ../../FlashSparse
rm -rf build
python setup.py install

# Back to root directory
cd ../..