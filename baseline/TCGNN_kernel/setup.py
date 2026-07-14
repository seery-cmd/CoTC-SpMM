from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='TCGNN_kernel',
    ext_modules=[
        CUDAExtension('TCGNN_kernel', [
            'TCGNN.cpp',
            'TCGNN_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
