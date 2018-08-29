#ifndef _LINALG_MATRIX_HPP_
#define _LINALG_MATRIX_HPP_

#include <cstdlib> //for aligned_alloc
#include <new>
#include <cassert> //for assert
#include <cstring> //for memcpy
#include <iomanip>
#include <iostream>

#define MEM_ALIGNMENT 128 //upto AVX-512 code-friendly, valid values are 8,16,32,64,128

#ifdef HAVE_MKL
#include <mkl.h>
#endif

// Selective aligned allocation is preferred over -fpack-struct[=n] because -fpack-struct option generates suboptimal code
// Eigen uses something similar, see https://stackoverflow.com/questions/16376942/best-cross-platform-method-to-get-aligned-memory 


namespace linalg{
namespace util {

	static unsigned int roundTo(unsigned int value, unsigned int roundTo)
	{
		return (value + (roundTo - 1)) & ~(roundTo - 1);
	}

	template<typename T> 
	static inline T * alignedCalloc(unsigned int n, size_t size, size_t alignment)
	{
		void * mem;
		mem = aligned_alloc(alignment, roundTo(n * size, alignment));
		if (!mem)
			throw std::bad_alloc();
		return static_cast<T*>(mem);
	}

	static int getColStride(int cols)
	{
		// No padding case
		// return cols;
		// Pad to nearest multiple of 4 
		return (cols + 3) & ~3;
	}
}

// Data is stored row-wise, each row is padded to nearest multiple of 4
// If a matrix has 3 cols, the fourth entry in first row is assumed to be padding
// fifth entry is the first column of the second row
template<typename T>
class Matrix
{
public:
	Matrix(int r, int c) : rows(r), cols(c), stride(util::getColStride(cols))
	{ 
		data = linalg::util::alignedCalloc<T>(rows * stride, sizeof(T), MEM_ALIGNMENT);
	}

	Matrix(const Matrix & mat) : rows(mat.rows), cols(mat.cols), stride(mat.stride)
	{
		data = linalg::util::alignedCalloc<T>(mat.rows * mat.stride, sizeof(T), MEM_ALIGNMENT);
		memcpy(data, mat.data, rows * stride * sizeof(T));
	}

	~Matrix() 
	{
		free(data);
	}

	Matrix & operator=(const Matrix & mat)
	{
		assert(this->rows == mat.rows);
		assert(this->cols == mat.cols);
		memcpy(data, mat.data, rows * stride * sizeof(T));
        return *this;
	}

	//data can only be owned by one copy
	Matrix & operator=(Matrix && mat)
	{
        assert(rows == mat.rows);
        assert(cols == mat.cols);
        free(data);
        data = mat.data;
        mat.data = 0;
        return *this;
	}

	T & operator()(int r, int c) { return data[r * stride + c]; }
	const T & operator()(int r, int c) const { return data[r * stride + c]; }
	
	int rows;
	int cols;
    int stride; // same as leading dimension (lda) in MKL/LAPACK terms

	T * data;
};

// initialise Matrix from an array, but arr must include same padding as eventual matrix
template<typename T>
void setMatrix(Matrix<T> & m, const T * arr)
{
	memcpy(m.data, arr, m.rows * m.stride * sizeof(T));
}

// Calculates out of place transpose, using BLAS, if available
template<typename T>
inline Matrix<T> transpose(const Matrix<T> & M)
{
	Matrix<T> MT(M.cols, M.rows);
	for (int i = 0; i < M.rows; i++)
		for (int j = 0; j < M.cols; j++)
			MT(j, i) = M(i, j);
	return MT;
}

#ifdef HAVE_MKL
template<>
inline Matrix<float> transpose(const Matrix<float> & M)
{
	Matrix<float> MT(M.cols, M.rows);
	mkl_somatcopy('R', 'T', M.rows, M.cols, 1.0, M.data, M.stride, MT.data, MT.stride);
	return MT;
}

template<>
inline Matrix<double> transpose(const Matrix<double> & M)
{
	Matrix<double> MT(M.cols, M.rows);
	mkl_domatcopy('R', 'T', M.rows, M.cols, 1.0, M.data, M.stride, MT.data, MT.stride);
	return MT;
}
#endif

//Calculates C = A*B, using BLAS, if available
template<typename T>
inline void product(const Matrix<T> & A, const Matrix<T> &B, Matrix<T> & C)
{
	for (int i = 0; i < A.rows; i++)
	{
		for (int j = 0; j < B.cols; j++)
		{
			T sum = 0;
			for (int k = 0; k < A.cols; k++)
				sum += A(i, k) * B(k, j);
			C(i, j) = sum;
		}
	}
}

#ifdef HAVE_MKL
template<>
inline void product(const Matrix<float> & A, const Matrix<float> &B, Matrix<float> & C)
{
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows, B.rows, C.rows, 1.0,
		A.data, A.stride, B.data, B.stride, 0, C.data, C.stride);
}

template<>
inline void product(const Matrix<double> & A, const Matrix<double> &B, Matrix<double> & C)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows, B.rows, C.rows, 1.0,
		A.data, A.stride, B.data, B.stride, 0, C.data, C.stride);
}
#endif 

template<class T>
inline void print(std::string name, const Matrix<T> & m)
{
	std::cout << name << " = [" << std::endl;
	for (int r = 0; r < m.rows; r++)
	{
		for (int c = 0; c < m.cols; c++)
		{
			std::cout << m(r, c) << ",\t";
		}
		std::cout << "\n";
	}
	std::cout << "]" << std::endl;
}

} //namespace linalg
	
#endif 
