from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='MBaseline_kernel',
    ext_modules=[
       CUDAExtension(
            name='cusparse_spmm_csr', 
            sources=[
            'cuSPARSE/cuSPARSE_spmm.cu',
            'cuSPARSE/spmm_csr.cpp'
            ]
        ),
        CUDAExtension(
            name='DTCSpMM',
            sources=[
            'DTC-SpMM/DTCSpMM.cpp',
            'DTC-SpMM/DTCSpMM_kernel.cu',
            ]
        ),
        CUDAExtension(
            name='FS_SpMM', 
            sources=[
            'FlashSparse/src/SpMM/src/benchmark.cpp',
            'FlashSparse/src/SpMM/src/spmmKernel.cu',
            ]
        ),
        CUDAExtension(
            name='FS_Block_gpu', 
            sources=[
            'FlashSparse/src/Block_gpu/block.cpp',
            'FlashSparse/src/Block_gpu/block_kernel.cu',
            ]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


