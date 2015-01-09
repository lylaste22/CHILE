# Test OpenMP python feature
#
# Juan Carlos Maureira
from cython.parallel cimport prange
import numpy as np

def run():
    cdef int i, j, n

    print "testing omp..."

    x = np.zeros((200, 2000), float)

    n = x.shape[0]
    for i in prange(n, nogil=True):
        with gil:
            for j in range(5000):
                x[i,:] = np.cos(x[i,:])

    print "done"

    return x

