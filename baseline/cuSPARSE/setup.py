from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cusparse_spmm_csr',
    ext_modules=[
       CUDAExtension(
            name='cusparse_spmm_csr', 
            sources=[
            'cuSPARSE_spmm.cu',
            'spmm_csr.cpp'
            ]
            # extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
            # libraries=["numa", "tcmalloc_minimal"]
         ) 
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


