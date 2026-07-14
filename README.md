# CoTC-SpMM
CoTC-SpMM: A Cooperative Tensor–CUDA Cores Scheme for Efficient Sparse Matrix Multiplication

The baseline folder contains the work of ```RoDe(CUDA core), TCGNN(TC core), DTC-SpMM(TC core), HC-SpMM(CUDA and TC core), Flashsparse(TC core), and cuSPARSE(CUDA core)```

CoTC-SpMM is a hybrid kernel work, and its overviews are as follows:

![overview](overview.pdf)

For the MMA matrix multiplication instruction used in sparse matrices, improving memory access for sparse data is crucial. 

![MMA in sparse matrix](mma.pdf)

Therefore, we propose the HTC format.

![An example of HTC](HTC.pdf)

At the same time, we designed two PTX kernels.

CUDA kernel:

![CUDA kernel](CUDA kernel.pdf)

And TC kernel:

![TC kernel](TC kernel.pdf)

Experimental results on NVIDIA A100 and H800 GPUs demonstrate that CoTC-SpMM achieves substantial performance speedups over state-of-the-art implementations.

```cd src/spmm``` and using ```make``` to run the CoTC-SpMM. Note that, you need to change the ```makefile``` *test_Matrix* variable to correct it to your matrix data path
