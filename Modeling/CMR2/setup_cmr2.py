from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    name='CMR2',
    ext_modules = cythonize("CMR2_pack_cyth.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)
