# Cholesky example for optimization benchmarking

## Setup

The repo contains code for Python as well as C++. See below for language-specific setup instructions. 

## Python setup

Install Anaconda Python 3.6 using pre-built installers at 
- Linux: https://www.anaconda.com/download/#linux 
- Windows: https://www.anaconda.com/download/ 

Install numba using 
```
conda install numba
```

## C++ setup

Install gcc and optionally Intel MKL. This code uses C++11. Use appropriate sections of Makefile that build the MKL/non-MKL versions. This is controlled by the macro `HAVE_MKL`. GCC versions 4.8, 5.5, 6.4, 7.3 and 8.1 have been tested with Ubuntu 18.04 and GCC versions 4.9, 5.5 and 6.4 have been tested with Ubuntu 16.04. MKL version used was 2018/update 3. 

## Benchmarking

Benchmarks were taken on an Intel Xeon E5 processor (Windows 10). This processor has SSE/AVX instructions but not AVX2/AVX-512. 

## Turbo Boost
`Turbo Boost` was disabled by following instructions [here](https://www.tautvidas.com/blog/2011/04/disabling-intel-turbo-boost/). With `Turbo Boost` enabled, which is the default, timings are not consistent because the CPU clock frequency changes (as per temperature of the machine). 
Anaconda uses Intel MKL and I have not tested this on an ARM processor. All timings were taken in Ubuntu 18.04 (Windows Subsystem for Linux). 
To verify that Turbo Boost is off, you may find it useful to download Intel Extreme tuning utility and check that the `Max Core Frequency` stays roughly constant when running benchmarks. 