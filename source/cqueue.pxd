cimport cqueue_ as cqueue

cdef class Queue:
    cdef cqueue.Queue* _c_queue
    cpdef append(self, int value)
    cpdef extend(self, values)
    cdef extend_ints(self, int* values, size_t count)
    cpdef int peek(self) except? -1
    cpdef int pop(self) except? -1
