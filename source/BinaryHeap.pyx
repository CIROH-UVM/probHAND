import numpy as np

cimport numpy as np
np.import_array()

from libc.math cimport pow
from libc.stdio cimport printf
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)  # use C-style integer division (rounds towards 0)

cdef class BinaryHeap:
    """A binary heap on array indices with priority determined by a secondary array.
    """    

    def __init__(self, double[:, ::1] topo):
        self.nx = topo.shape[1]
        self.ny = topo.shape[0]
        self.qmaxlength = self.nx * self.ny  # max possible queue length; if this is exceeded, something bad happened
        self.length = 0  # the length of the current queue state
        self.z = np.asarray(topo).ravel()  # memoryview of flattened topographic elevation array (ASSUMED TO BE HYDROCORRECTED ALREADY)

        # initialize queue vectors
        self.pval = np.full([self.qmaxlength,], -1.0, dtype=np.float64)
        self.pqueue = np.full([self.qmaxlength,], -1, dtype=np.int32)

    cdef bubble_up(self, int idx):
        # Bubble the queue entry at index idx up the binary heap until order is valid
        cdef int itemp, parent, i
        cdef double ztemp

        i = idx
        while (i > 0):
            parent = (i - 1) / 2  # C-style floor division
            if (self.pval[i] < self.pval[parent]):  # heap order is broken
                # swap their places
                itemp = self.pqueue[i]
                self.pqueue[i] = self.pqueue[parent]
                self.pqueue[parent] = itemp
                ztemp = self.pval[i]
                self.pval[i] = self.pval[parent]
                self.pval[parent] = ztemp

                i = parent
            else:  # heap-order is valid
                break

    cdef bubble_emptyroot(self):
        # Bubbles empty root down to a leaf node, swaps with last queue entry, bubbles up swapped entry to restore heap order        
        cdef int lchild, rchild, smallest, i
        i = 0
        while (i < self.length / 2):
            lchild = 2 * i + 1
            rchild = 2 * i + 2
            if ((rchild < self.length) and (self.pval[rchild] <= self.pval[lchild])):
                smallest = rchild
            else:
                smallest = lchild
            self.pqueue[i] = self.pqueue[smallest]
            self.pval[i] = self.pval[smallest]
            i = smallest
            # upon exit, i is index of a leaf node
        if (i < self.length):
            # fix a gap in the final level of the heap
            self.pqueue[i] = self.pqueue[self.length - 1]
            self.pval[i] = self.pval[self.length - 1]
            self.length -= 1
            self.bubble_up(i)  # make sure heap order is valid
        else:
            # empty root landed at the final entry of the queue; nothing to do but decrement queue size
            self.length -= 1
        
    cpdef pq_push(self, int newval):
        # Add an element to the end of the priority queue and restore heap-order property
        self.length += 1
        if self.length >= self.qmaxlength:
            raise ValueError("The priority queue is full, something went wrong")
        self.pqueue[self.length - 1] = newval
        self.pval[self.length - 1] = self.z[newval]
        # Bubble the newly-added final element of the priority queue up the binary heap
        self.bubble_up(self.length - 1)

    cpdef int pq_pop(self):
        # Pops off the root (guaranteed to be minimum topography in the binary heap) and restores the heap-order property
        cdef int rootval
        if (self.length <=0):
            raise IndexError("Attempted to pop an item off an empty queue!")
        rootval = self.pqueue[0]
        self.bubble_emptyroot()  # Bubble empty root down to restore heap-order; slightly more efficient than generic bubble-down alg (~log2+k vs ~2*log2)
        return rootval

    cpdef print_queue(self):
        cdef int i
        for i in range(self.length):
            printf("%d %lf\n", self.pqueue[i], self.pval[i])
    
    cpdef int queue_length(self):
        return self.length
