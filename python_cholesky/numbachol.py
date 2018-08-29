from math import sqrt
import numpy as np
import numba
import llvmlite.binding as llvm

#llvm.set_option('','--debug-only=loop-vectorize')

@numba.jit('(float64[:, :], float64[:, :])',nopython=True, nogil=True, parallel=True)
def cholesky_numba(M, cholesky):
    """Lower triangular version of a Cholesky decomposition of a posiive-definite M = L L^T
    M must be symmetric, positive definite. 
    M is a numpy array"""
    n = M.shape[0]

    # Perform the Cholesky decomposition
    for i in range(n):
        for j in range(i+1):
            sum = 0
            for m in range(j):
                sum += (cholesky[i, m] * cholesky[j, m])

            val = M[i, j] - sum
            
            if (i == j): # Calculate diagonal elements
                cholesky[i, j] = sqrt(val)
            else:        # Calculate lower-diagonal elements 
                cholesky[i, j] = val / cholesky[j, j]
