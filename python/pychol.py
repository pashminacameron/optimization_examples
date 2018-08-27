from math import sqrt

#@profile
def cholesky_py(M, cholesky):
    """Lower triangular version of a Cholesky decomposition of a posiive-definite A
    A must be symmetric, positive definite. """
    n = len(M)

    # Perform the Cholesky decomposition
    for i in range(n):
        for j in range(i+1):
            val = M[i][j] - sum(cholesky[i][m] * cholesky[j][m] for m in range(j))
            
            if (i == j): # Calculate diagonal elements
                cholesky[i][j] = sqrt(val)
            else:        # Calculate below-diagonal elements
                cholesky[i][j] = (val / cholesky[j][j])
    return

# An enum based implementation is slightly better than Python loop implementation 
# but also far less readable
