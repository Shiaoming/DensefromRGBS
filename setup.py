from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        Extension(
            "nnfc",
            ["utils/nnfc/nnfc.pyx"],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
        )),
    include_dirs=[numpy.get_include()]
)