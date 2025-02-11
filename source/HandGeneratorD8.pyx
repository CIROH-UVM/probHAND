import numpy as np

cimport cqueue
from cqueue cimport Queue

# Import compile-time info about Numpy; okay to name the same as numpy import
# https://cython.readthedocs.io/en/stable/src/tutorial/numpy.html
cimport numpy as np
np.import_array()

from libc.math cimport pow
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)  # use C-style integer division (rounds towards 0)

cdef class HandGeneratorD8:

    def __init__(self, double[:, ::1] topo, unsigned char[:, ::1] channel_mask, double ndval):

        # Check that topo and channel_mask have same shape
        if (topo.shape[0] != channel_mask.shape[0]) or (topo.shape[1] != channel_mask.shape[1]):
            raise ValueError(
                f'HandGeneratorD8 init error: input topo and channel_mask arrays must be same shape; got {(topo.shape[0],topo.shape[1])} and {(channel_mask.shape[0],channel_mask.shape[1])} resp.'
            )

        self.Q = cqueue.Queue()  # FIFO queue from C source code wrapped in Cython
        self.nx = topo.shape[1]  # Cython doesn't like topo.shape, leads to compile error "Cannot convert 'npy_intp *' to Python object"
        self.ny = topo.shape[0]
        self.z = topo  # memoryview of topographic elevation array (ASSUMED TO BE HYDROCORRECTED ALREADY)
        self.mask = channel_mask  # memoryview of mask array of 1s (indicating channel) and 0s
        self.nodataval = ndval  # no-data value of topo


    def compute_hand(self):
        ''' Computes the height above nearest drainage in every valid grid cell that drains to a channel.

            Pseudocode of the process
            The queue in this case (self.Q) will hold the flattened 1D indices of the 2D arrays

            1. Initialize queue with channel nodes (i.e. channel_mask == 1)
                i) Sort these nodes by elevation (increasing order)
                ii) Set hand of these nodes to 0.0
                iii) Push them onto queue with self.Q.extend(values)
            
            2. While queue is not empty:
                i) Pop() off first element, node i
                ii) For each neighbor j of i:
                    a. check that the node is valid (i.e. indices in-bounds, not equal to nodata)
                    b. check that j is upslope from i
                    c. check that either the hand hasn't been set yet OR that the current distance to stream is less than the existing distance
                    d. assuming a-c, set the new hand and dist using values relative to node i values
                    e. push j onto queue
            
            Note that as with TauDEM, this methodology will return NoData for any cell that doesn't have a downslope path to a channel.
            e.g., cells that drain to the DEM edge or internal non-channel depressions

        '''
        cdef double[:, ::1] hand = np.full([self.ny, self.nx], self.nodataval, dtype=np.float64)
        cdef double[:, ::1] dist = np.full([self.ny, self.nx], self.nodataval, dtype=np.float64)
        cdef double dd, dz
        cdef int x, y, t, xx, yy, num
        cdef int neighbor_ys[8]
        cdef int neighbor_xs[8]

        # Set up neighbor node arrays
        neighbor_ys[0] = 0
        neighbor_ys[1] = 0
        neighbor_ys[2] = -1
        neighbor_ys[3] = 1
        neighbor_ys[4] = -1
        neighbor_ys[5] = 1
        neighbor_ys[6] = -1
        neighbor_ys[7] = 1
        neighbor_xs[0] = -1
        neighbor_xs[1] = 1
        neighbor_xs[2] = 0
        neighbor_xs[3] = 0
        neighbor_xs[4] = -1
        neighbor_xs[5] = -1
        neighbor_xs[6] = 1
        neighbor_xs[7] = 1

        print("Initializing queue")
        for t in range(self.ny * self.nx):
            y, x = self.unravel_index(t)
            if self.z[y, x] == self.nodataval:
                continue
            if self.mask[y, x] == 1:
                hand[y, x] = 0.0
                dist[y, x] = 0.0
                self.Q.append(t)
        print("Done initializing! Starting to pop")
        num = 0
        while (self.Q.__bool__()):  # while the queue is not empty
            t = self.Q.pop()  # popped elements will be valid nodes with hand, dist already set
            y, x = self.unravel_index(t)

            # num += 1
            # if (num % 1000000 == 0):
            #     print(num)
            # print(y, x)

            for t in range(8):
                yy = y + neighbor_ys[t]
                xx = x + neighbor_xs[t]
                if self.check_indices(yy, xx):
                    if (self.z[yy, xx] != self.nodataval) and (self.z[yy, xx] >= self.z[y, x]):
                        dd = dist[y, x] + pow(pow(yy - y, 2.) + pow(xx - x, 2.), 0.5)
                        dz = self.z[yy, xx] - self.z[y, x] + hand[y, x]
                        if (hand[yy, xx] == self.nodataval) or ((dd < dist[yy, xx]) and (dz <= hand[yy, xx])):
                            hand[yy, xx] = dz
                            dist[yy, xx] = dd
                            self.Q.append(self.ravel_index(yy, xx))

        return hand

    cpdef ravel_index(self, int row, int col):
        """ Convert 2D array indices to corresponding 1D raveled array index.
        
        Given indices (row,col) that correspond to the (y,x) coordinates, respectively, of a 2D array
        (i.e., accessed in Python with array[col, row]), return the corresponding index in a flattened 1D array.

        Defined with cpdef for speed-up of C-level function (like cdef) and for accessibility on Python side (for e.g. testing)

        Parameters
        ----------
        row : int
            Row index of the 2D array
        col : int
            Column index of the 2D array

        Returns
        -------
        int
            _description_
        """
        cdef int idx
        if not self.check_indices(row, col):
            raise ValueError('input row and col out of bounds!')
        idx = row * self.nx + col
        return idx

    cpdef unravel_index(self, int idx):
        '''
            Given an index in the flattened 1D array, return corresponding 2D indices.
            Defined with cpdef for speed-up of C-level function (like cdef) and for accessibility on Python side (for e.g. testing)
        '''
        cdef int row, col
        row = idx / self.nx  # integer division rounds down with cdivision turned on (see cython compiler directives near top)
        col = idx - row * self.nx
        if not self.check_indices(row, col):
            raise ValueError('input raveled index out of bounds!')
        return row, col

    cpdef check_indices(self, int row, int col):
        if (row < 0) or (col < 0) or (row >= self.ny) or (col >= self.nx):
            return False
        return True
