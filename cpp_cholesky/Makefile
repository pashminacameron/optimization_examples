# Run as follows
# make clean && make && LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64_lin/ ./testCholesky
# See MKL Ubuntu installation instructions in installMKL.sh 
# Make sure you export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64_lin/ on Linux or have it in PATH on Windows 
# so that mkl_rt can load it's dependencies
# or you make have to run as follows:
# make clean && make
# ./testCholesky # to run without MKL
# LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64_lin/ ./testCholeskyMKL # to run with MKL
# -ffast-math turns on -funsafe-math-optimizations which is the real problem/speedup depending on what you care about: accuracy and compliance or speed
# not having -ffast-math is equivalent to having -fno-unsafe-math-optimizations, rest of the flags have very little effect on this example

NAME = testCholesky
CC := /usr/bin/g++
LD := /usr/bin/ld
CCFLAGS := -m64 --std=c++11 -O3 #-ffast-math -v -pg -fprofile-use
MKL_CCFLAGS := -DHAVE_MKL
LDFLAGS := # -pg -fprofile-use

AVX_CCFLAGS = -mavx 

INCLUDES += -I.. 
MKL_INCLUDES += -I/opt/intel/mkl/include/

LIBS += -lmkl_rt 
LIBS_PATH += -L/opt/intel/mkl/lib/intel64_lin/

all: testCholesky testCholeskyMKL

cholesky_avx.o: cholesky_avx.cpp
	$(CC) $(INCLUDES) $(CCFLAGS) $(AVX_CCFLAGS) -c $< -o $@

testCholesky.o: testCholesky.cpp
	$(CC) $(INCLUDES) $(CCFLAGS) -c $< -o $@

testCholesky: testCholesky.o cholesky_avx.o
	$(CC) $(INCLUDES) $(LDFLAGS)  $+ -o $@ 

cholesky_avxMKL.o: cholesky_avx.cpp
	$(CC) $(INCLUDES) $(MKL_INCLUDES) $(CCFLAGS) $(MKL_CCFLAGS) $(AVX_CCFLAGS) -c $< -o $@

testCholeskyMKL.o: testCholesky.cpp
	$(CC) $(INCLUDES) $(MKL_INCLUDES) $(CCFLAGS) $(MKL_CCFLAGS) -c $< -o $@

testCholeskyMKL: testCholeskyMKL.o cholesky_avxMKL.o
	$(CC) $(INCLUDES) $(MKL_INCLUDES) $(MKL_CCFLAGS) $(LDFLAGS)  $+ -o $@ $(LIBS_PATH) $(LIBS)

clean:
	rm -f testCholesky testCholesky.o cholesky_avx.o
	rm -f testCholeskyMKL testCholeskyMKL.o cholesky_avxMKL.o

