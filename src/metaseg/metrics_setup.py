"""
Compile metrics.pyx via:
python3 metrics_setup.py build_ext --inplace
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

ext_core = Extension(
    "metrics",
    sources=["metrics.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3"])

setup(ext_modules=cythonize([ext_core]))
