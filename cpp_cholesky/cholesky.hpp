#ifndef _LINALG_CHOLESKY_HPP_
#define _LINALG_CHOLESKY_HPP_

#include "matrix.hpp"
#include <iostream>
#include <functional>
#include <cmath>
#include <vector>

namespace linalg{

#ifdef HAVE_MKL
	enum class CholeskyImpl { CPP, AVX, BLAS };
#else
	enum class CholeskyImpl { CPP, AVX };
#endif

	//Define a function pointer to choose between two implementations
	typedef std::function<float(const float * u, const float * v, const float * d, int size)> func_type_LDLt;
	typedef std::function<float(const float * u, const float * v, int size)> func_type_LLt;
 
	float sum3VecProductAVX(const float * u, const float * v, const float * d, int size);
	float sum2VecProductAVX(const float * u, const float * v, int size);

    static float sum3VecProduct(const float * u, const float * v, const float * d, int size)
    {
        float dp = 0;
		for (int i = 0; i < size; i++)        
			dp += u[i] * v[i] * d[i];
		return dp;
    }

	static float sum2VecProduct(const float * u, const float * v, int size)
	{
		float dp = 0;
		for (int i = 0; i < size; i++)
			dp += u[i] * v[i];
		return dp;
	}

#ifdef HAVE_MKL
	static float sum2VecProductBLAS(const float * u, const float * v, int size)
	{
		return cblas_sdot(size, u, 1, v, 1);
	}
#endif

    static float sum3VecProductWrapper(const float * row1, const float * row2, const float * diag, int size, func_type_LDLt computeFunc)
    {              
        return computeFunc(row1, row2, diag, size); //either sumPairwiseProduct or sumPairwiseProductSSE
    }

	static float sum2VecProductWrapper(const float * row1, const float * row2, int size, func_type_LLt computeFunc)
	{
		return computeFunc(row1, row2, size); //either sumPairwiseProduct or sumPairwiseProductSSE
	}
    
/// Decomposes a symmetric, positive semi-definite matrix A as L D L^T, 
/// uses tricks from Numerical recipes in C, section 2.9 Cholesky Decomposition
/// D is diagonal of m_chol (squared entries)
/// L is lower diagonal m_chol with each (i,j)th element multiplied by sqrt(m_chol(j,j))
/// and the diagonal consists of sqrt(m_chol(j,j))
/// This implementation delays the computation of sqrt(D) until after the main loop

class Cholesky
{
public:	
	explicit Cholesky(int size, CholeskyImpl impl) : m_chol(size, size)
	{
		diag.resize(size);

		//Define function pointers pointing to SSE optimized and naive CPP implementation
		//and decide which one to use based on argument in constructor
		switch (impl)
		{
#ifdef HAVE_MKL
		case CholeskyImpl::BLAS:
			m_LDLt_Impl = &sum3VecProduct; //BLAS does not have 3 vector product, fallback to CPP
			m_LLt_Impl = &sum2VecProductBLAS;
			break;
#endif
		case CholeskyImpl::AVX:
			m_LDLt_Impl = &sum3VecProductAVX;
			m_LLt_Impl = &sum2VecProductAVX;
			break;
		case CholeskyImpl::CPP:
		default:
			m_LDLt_Impl = &sum3VecProduct;
			m_LLt_Impl = &sum2VecProduct;
			break;
		}
	}
	
    /// Compute the LDL^T decomposition of mat, given mat 
	void calculateCholeskyLDLt(const Matrix<float>& M)
	{
		//Setup
		m_chol = M;

        int stride = m_chol.stride;
		
		for (int j = 0; j < m_chol.cols; j++)
		{
			float sum = 0;
			//for (int k = 0; k < j; k++)
				//sum += m_chol(j, k) * m_chol(j, k) * diag[k];
			//m_chol(j, j) = m_chol(j, j) - sum;
			m_chol(j, j) = m_chol(j, j) - sum3VecProductWrapper(&m_chol.data[j * stride], &m_chol.data[j * stride], &diag[0], j, m_LDLt_Impl);
			diag[j] = m_chol(j, j);

			float invDiag = 1 / m_chol(j, j);
			for (int i = j + 1; i < m_chol.rows; i++)
			{	// i > j, i.e. lower diagonal
				//float sum = 0;
				//for (int k = 0; k < j; k++)
				//	sum += m_chol(i, k) * m_chol(j, k) * diag[k];
				m_chol(i, j) = invDiag * (m_chol(i, j) - sum3VecProductWrapper(&m_chol.data[i * stride], &m_chol.data[j * stride], &diag[0], j, m_LDLt_Impl));
			}
		}
    }

	/// Compute the LL^T decomposition of mat, given mat 
	void calculateCholeskyLLt(const Matrix<float>& M)
	{
		//Setup
		m_chol = M;

		int stride = m_chol.stride;

		for (int j = 0; j < m_chol.cols; j++)
		{
			//float sum = 0;
			//for (int k = 0; k < j; k++)
			//	sum += m_chol(j, k) * m_chol(j, k);
			
			m_chol(j, j) = std::sqrt(m_chol(j, j) - sum2VecProductWrapper(&m_chol.data[j*stride], &m_chol.data[j*stride], j, m_LLt_Impl));

			float invDiag = 1 / m_chol(j, j);
			for (int i = j + 1; i < m_chol.rows; i++)
			{	// i > j
				//float sum = 0;
				//for (int k = 0; k < j; k++)
				//	sum += m_chol(i, k) * m_chol(j, k);
				//m_chol(i, j) = invDiag * (m_chol(i, j) - sum);

				m_chol(i, j) = invDiag * (m_chol(i, j) - sum2VecProductWrapper(&m_chol.data[i*stride], &m_chol.data[j*stride], j, m_LLt_Impl));
			}
		}
	}

	Matrix<float> getCholeskyMatrix()
	{
		//This populates the upper-triangular L^T part of the LDL^T matrix
		Matrix<float> chol(m_chol);
		for (int i = 0; i < m_chol.rows; i++)
			for (int j = i + 1; j < m_chol.cols; j++)
				chol(i, j) = m_chol(j, i);

		return chol;
	}

private:
	// We store Cholesky in-place
	Matrix<float> m_chol;
	std::vector<float> diag;
	// Function pointer that chooses the implementation dynamically
	func_type_LDLt m_LDLt_Impl = NULL;
	func_type_LLt m_LLt_Impl = NULL;

};
	
}
#endif

