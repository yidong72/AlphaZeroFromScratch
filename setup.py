# setup.py

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Othello Cython',
    ext_modules=cythonize("othello_cython.pyx"),
    include_dirs=[numpy.get_include()],  # add this line
    zip_safe=False,
)