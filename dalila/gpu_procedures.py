from __future__ import division

import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda import driver
import pycuda.autoinit
import skcuda.cublas as cublas
import skcuda.misc as misc
import skcuda.linalg as linalg
linalg.init()


def _non_negative_projection_GPU(x, nn, matrix_type=None):
    if nn == "both" or nn==matrix_type:
        zeros = gpuarray.zeros_like(x)
        return  gpuarray.maximum(zeros, x)
    return x