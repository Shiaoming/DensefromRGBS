from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# setup(
#     ext_modules=cythonize(
#         Extension(
#             "nnfc",
#             ["utils/nnfc/nnfc.pyx"],
#              # "nyuv2/nyuv2_utility.py"]
#         )),
#     include_dirs=[numpy.get_include()]
# )

setup(
    ext_modules=cythonize(["utils/nnfc/nnfc.pyx"],
                          annotate=True),
    include_dirs=[numpy.get_include()]
)
