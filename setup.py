from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["utils/nnfc/nnfc.pyx"],
                          annotate=True),
)
