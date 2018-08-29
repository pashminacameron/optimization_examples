// needs c++11
#include <chrono>
#include <iostream>
#include <cstdlib>
#include "cholesky.hpp"

using namespace linalg;
namespace cn = std::chrono;

cn::time_point<cn::high_resolution_clock> startTimer()
{
	return cn::high_resolution_clock::now();
}

double endTimer(cn::time_point<cn::high_resolution_clock> t1)
{
	auto t2 = cn::high_resolution_clock::now();
	cn::duration<double> diff = t2 - t1;
	return diff.count() * 1000;
}

Matrix<float> genTestMatrix()
{
	Matrix<float> M(3, 3);
	float data[12] = { 4, 12, -16, 0,
					  12, 37, -43, 0,
					 -16, -43, 98, 0 };
	setMatrix<float>(M, &data[0]);
	return M;
}

Matrix<float> genExpectedLLt()
{
	Matrix<float> E(3, 3);
	//LLt
	float data[12] = { 2,  6, -8,  0,
					   6,  1,  5,  0,
					  -8,  5,  3,  0 };
	setMatrix<float>(E, &data[0]);
	return E;
}

Matrix<float> genExpectedLDLt()
{
	Matrix<float> E(3, 3);
	//LDLt
	float data[12] = { 4,  3, -4, 0, 
					   3,  1,  5, 0,
					  -4,  5,  9, 0 }; 
	setMatrix<float>(E, &data[0]);
	return E;
}

Matrix<float> genRandomPosDefMatrix(int size)
{
	srand(111970);
	// Generate random matrix
	Matrix<float> M(size, size);
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			M(i, j) = (float)rand() / RAND_MAX;
	// prod = M^T M to guarantee positive-definite
	Matrix<float> MT = transpose(M);
	Matrix<float> prod(size, size);
	product<float>(MT, M, prod);
	return prod;
}

bool accuracyCheck()
{
	Matrix<float> M = genTestMatrix();
	Matrix<float> E_LLt = genExpectedLLt();

	Cholesky chol = Cholesky(M.rows, CholeskyImpl::AVX);
	chol.calculateCholeskyLLt(M);
	Matrix<float> LLt = chol.getCholeskyMatrix();
	bool correct = true;
	for (int i = 0; i < M.rows; i++)
		for (int j = 0; j < M.cols; j++)
			if (LLt(i, j) != E_LLt(i, j))
				correct = false;

	if (!correct)
	{
		print("Matrix", M);
		print("Expected LLt", E_LLt);
		print("Actual LLt", LLt);
	}

	return correct;
}

#ifdef HAVE_MKL
void callMKLBlockCholesky(Matrix<float> M)
{
	// Note: This returns L for M = L L^T in M
	// Hence provide a calling wrapper that makes a copy of the matrix by value
	int info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', M.rows, M.data, M.stride);
	// M may be printed here, for timings we can discard it
}
#endif

int main(int argc, const char * argv[])
{
	// Timings are stable across 5/10/20 runs, so use 5 runs
	// Timings are not very reliable for small matrices

	if (!accuracyCheck())
	{
		throw std::runtime_error("Accuracy check failed, exiting");
		return 1;
	}
	else
	{
		std::cout << "accuracy check passed\n";
	}
	int numRuns = 2; 
	int startMSize = 4;
	int endMSize = 4096;

	char sep = ',';

	std::cout << "Size" << sep << "CPP-LLt" << sep << "AVX-LLt" << sep << "CPP-LDLt" << sep << "AVX-LDLt";
#ifdef HAVE_MKL
	std::cout << sep << "BLAS-LLt" << sep << "LAPACK";
#endif
	std::cout << std::endl;

	for (int mSize = startMSize; mSize <= endMSize; mSize *= 2)
	{
		Matrix<float> M = genRandomPosDefMatrix(mSize);
	
		//Warmup run
		Cholesky(mSize, CholeskyImpl::CPP).calculateCholeskyLLt(M);
		auto t1 = startTimer();
		for (int i = 0; i < numRuns; i++)
			Cholesky(mSize, CholeskyImpl::CPP).calculateCholeskyLLt(M);
		double time1 = endTimer(t1) / numRuns;
		
		//Warmup run
		Cholesky(mSize, CholeskyImpl::AVX).calculateCholeskyLLt(M);
		auto t2 = startTimer();
		for (int i = 0; i < numRuns; i++)
			Cholesky(mSize, CholeskyImpl::AVX).calculateCholeskyLLt(M);
		double time2 = endTimer(t2) / numRuns;

		//Warmup run
		Cholesky(mSize, CholeskyImpl::CPP).calculateCholeskyLDLt(M);
		auto t3 = startTimer();
		for (int i = 0; i < numRuns; i++)
			Cholesky(mSize, CholeskyImpl::CPP).calculateCholeskyLDLt(M);
		double time3 = endTimer(t3) / numRuns;

		//Warmup run
		Cholesky(mSize, CholeskyImpl::AVX).calculateCholeskyLDLt(M);
		auto t4 = startTimer();
		for (int i = 0; i < numRuns; i++)
			Cholesky(mSize, CholeskyImpl::AVX).calculateCholeskyLDLt(M);
		double time4 = endTimer(t4) / numRuns;
		
#ifdef HAVE_MKL
		//Warmup run
		Cholesky(mSize, CholeskyImpl::BLAS).calculateCholeskyLLt(M);
		auto t5 = startTimer();
		for (int i = 0; i < numRuns; i++)
			Cholesky(mSize, CholeskyImpl::BLAS).calculateCholeskyLLt(M);
		double time5 = endTimer(t5) / numRuns;

		//Warmup run
		callMKLBlockCholesky(M);
		auto t6 = startTimer();
		for (int i = 0; i < numRuns; i++)
			callMKLBlockCholesky(M);
		double time6 = endTimer(t6) / numRuns;

#endif
		
		// times are in milliseconds
		std::cout << mSize << sep << time1 << sep << time2 << sep << time3 << sep << time4 << sep;
#ifdef HAVE_MKL
		std::cout << time5 << sep << time6;
		//std::cout << "\tLAPACK\t" << diff6 / (numRuns) << "\t";
#endif
		std::cout << std::endl;
	}
	
	return 0;
}