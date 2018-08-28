from pychol import cholesky_py
from numpychol import cholesky_numpy
from numbachol import cholesky_numba
import timeit
import time
import sys
from pprint import pprint
import numpy as np
import scipy as sp
from sklearn.datasets import make_spd_matrix as make_spd_matrix

from scipy import random, linalg

# To profile, use: kernprof -l -v test.py, with @profile decorators around functions to profile

def genRandPosDefMatrix(n):    
    return make_spd_matrix(n)   

def accuracyCheck():
    # Accuracy test on a small matrix
    M = np.matrix('4 12 -16; 12 37 -43; -16 -43 98')
    E = np.matrix('2 0 0; 6 1 0; -8 5 3')
    L1 = [[0.0] * 3 for i in range(3)]
    cholesky_py(M.tolist(), L1)
    if not np.allclose(E.tolist(), L1):
        print("Accuracy test failed")
        sys.exit()
    return

def timeCholesky(matSize, resultsFile, fastOnly):

    with open(resultsFile,'a') as fh:
        accuracyCheck()

        numRuns = 5
        # Generate a test matrix 
        M = genRandPosDefMatrix(matSize)

        # Convert to list of lists for pure Python implementation
        A = M.tolist()
        # Initialize L to zero matrix

        fh.write(str(matSize)+",")
        if not fastOnly:
            times = []
            # Warm up runs
            L1 = [[0.0] * matSize for i in range(matSize)]
            cholesky_py(A, L1)
            cholesky_py(A, L1)
            # Timed runs
            for _ in range(numRuns):                
                start1 = time.clock()
                L1 = [[0.0] * matSize for i in range(matSize)]                
                cholesky_py(A, L1)
                times.append(time.clock() - start1)
            print("Size = %d, Python Cholesky time      = \t %10.10f ms" % (matSize, min(times)*1000))
            fh.write("{:10.10f}".format(min(times)*1000)+",")

            times = []
            An = np.asmatrix(M, dtype=float)
            # Warm up runs
            L2 = np.zeros((matSize, matSize),dtype=float)
            cholesky_numpy(An, L2)
            cholesky_numpy(An, L2)
            # Timed runs
            for _ in range(numRuns):                
                start2 = time.clock()
                L2 = np.zeros((matSize, matSize),dtype=float)                
                cholesky_numpy(An, L2)
                times.append(time.clock() - start2)
            print("Size = %d, Numpy (hand) Cholesky time =\t %10.10f ms" % (matSize, min(times)*1000))
            fh.write("{:10.10f}".format(min(times)*1000)+",")

        else:
            fh.write(" , ,")        
        

        An = np.asmatrix(M, dtype=float)
        times = []
        # Warm up runs
        L3 = np.zeros((matSize, matSize), dtype=float)
        cholesky_numba(An, L3)
        cholesky_numba(An, L3)
        # Timed runs
        for _ in range(numRuns):
            start3 = time.clock()
            L3 = np.zeros((matSize, matSize), dtype=float)            
            # Numba is happier with all variables already allocated
            cholesky_numba(An, L3)
            times.append(time.clock() - start3)
        print("Size = %d, Numba Cholesky time      =\t %10.10f ms" % (matSize, min(times)*1000))
        fh.write("{:10.10f}".format(min(times)*1000)+",")

        # numpy library function
        times = []
        # Warm up runs
        L4 = np.linalg.cholesky(An)
        L4 = np.linalg.cholesky(An)
        # Timed runs
        for _ in range(numRuns):
            start4 = time.clock()
            L4 = np.linalg.cholesky(An)
            times.append(time.clock() - start4)
        print("Size = %d, numpy.linalg.cholesky time =\t %10.10f ms" % (matSize, min(times)*1000))
        fh.write("{:10.10f}".format(min(times)*1000)+",")

        # numpy library function
        times = []
        # Warm up runs
        L5 = sp.linalg.cholesky(M, True)
        L5 = sp.linalg.cholesky(M, True)
        # Timed runs
        for _ in range(numRuns):
            start5 = time.clock()
            L5 = sp.linalg.cholesky(M, True)
            times.append(time.clock() - start5)
        print("Size = %d, scipy.linalg.cholesky time =\t %10.10f ms" % (matSize, min(times)*1000))
        fh.write("{:10.10f}".format(min(times)*1000)+",")

        times=[]
        # Warm up runs
        (L6 ,pd) = sp.linalg.lapack.spotrf(M, True)
        (L6 ,pd) = sp.linalg.lapack.spotrf(M, True)
        # Timed runs
        for _ in range(numRuns):
            start6 = time.clock()
            (L6 ,pd) = sp.linalg.lapack.spotrf(M, True)
            times.append(time.clock() - start6)
        print("Size = %d, scipy.linalg.lapack.cholesky time =\t %10.10f ms" % (matSize, min(times)*1000))
        fh.write("{:10.10f}".format(min(times)*1000)+",")

        fh.write("\n")

        #print(np.allclose(L1, L2))
        #print(np.allclose(L1, L3))
        #print(np.allclose(L1, L4))
        #print(np.allclose(L1, L5))
        #print(np.allclose(L1, L6))

        fh.close()
    return

# Use smallSizes over the whole range (2,13) when doing a benchmark but this takes a very long time
smallSizes = [pow(2,i) for i in range(2,11)] # 4 to 1024
largerSizes = [pow(2,i) for i in range(11,13)] # 2048, 4096


resultsFile = "timings2.csv"
# Clear results file
open(resultsFile, 'w').close()
# Write headers to results file
with open(resultsFile,'a') as fh:
    fh.write("Size,Python,NumPy,Numba,np.linalg,sp.linalg,sp.linalg.lapack\n")
fh.close()

# Time all algorithms for matrix sizes 4-256
for n in smallSizes:
    timeCholesky(n, resultsFile, False)
# Time only the efficient, C-based algorithms for matrix sizes 512-4096
for n in largerSizes:
    timeCholesky(n, resultsFile, True)
