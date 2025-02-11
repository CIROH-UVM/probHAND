import numpy as np
# Import compile-time info about Numpy; okay to name the same as numpy import
# https://cython.readthedocs.io/en/stable/src/tutorial/numpy.html
cimport numpy as np
np.import_array()

cimport cqueue
from cqueue cimport Queue

# Compile-time type definition
ctypedef np.float64_t DTYPEF_t
ctypedef np.int64_t DTYPEI_t
ctypedef np.uint8_t DTYPEU_t

cdef class HandGeneratorD8:
    cdef readonly Queue Q
    cdef readonly int nx, ny
    cdef np.ndarray z
    cdef np.ndarray mask
    cdef readonly DTYPEF_t nodataval
    cpdef ravel_index(self, int row, int col)
    cpdef unravel_index(self, int idx)
    cpdef check_indices(self, int row, int col)