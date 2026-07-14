from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name='FlashSparse_kernel',
    ext_modules=[
       CUDAExtension(
            name='FS_SpMM', 
            sources=[
            './src/SpMM/src/benchmark.cpp',
            './src/SpMM/src/spmmKernel.cu',
            ]
         ),
       CUDAExtension(
            name='FS_Block_gpu', 
            sources=[
            './src/Block_gpu/block.cpp',
            './src/Block_gpu/block_kernel.cu',
            ]
         ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


