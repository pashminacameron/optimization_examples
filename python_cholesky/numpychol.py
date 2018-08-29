from math import sqrt
import numpy as np

#@profile
def cholesky_numpy(M, cholesky):
    """Lower triangular version of a Cholesky decomposition of a posiive-definite M = L L^T
    M must be symmetric, positive definite. 
    M is a numpy array"""
    n = M.shape[0]

    # Perform the Cholesky decomposition
    for i in range(n):
        for j in range(i+1):
            val = M[i, j] - np.dot(cholesky[i, :j], cholesky[j, :j] )

            if (i == j): # Calculate diagonal elements
                cholesky[i, j] = sqrt(val)
            else:        # Calculate below-diagonal elements
                cholesky[i, j] = (val / cholesky[j, j])
    return
