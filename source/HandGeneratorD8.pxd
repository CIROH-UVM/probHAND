import numpy as np
# Import compile-time info about Numpy; okay to name the same as numpy import
# https://cython.readthedocs.io/en/stable/src/tutorial/numpy.html
cimport numpy as np
np.import_array()

cimport cqueue
from cqueue cimport Queue


cdef class HandGeneratorD8:
    cdef readonly Queue Q
    cdef readonly int nx, ny
    cdef double[:, ::1] z
    cdef unsigned char[:, ::1] mask
    cdef readonly double nodataval
    cpdef ravel_index(self, int row, int col)
    cpdef unravel_index(self, int idx)
    cpdef check_indices(self, int row, int col)