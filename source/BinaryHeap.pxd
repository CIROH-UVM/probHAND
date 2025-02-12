cdef class BinaryHeap:
    cdef int ny, nx, qmaxlength, length
    cdef int[::1] pqueue
    cdef double[::1] pval
    cdef double [::1] z
    cdef bubble_up(self, int idx)
    cdef bubble_emptyroot(self)
    cpdef pq_push(self, int newval)
    cpdef int pq_pop(self)
    cpdef print_queue(self)
    cpdef int queue_length(self)

