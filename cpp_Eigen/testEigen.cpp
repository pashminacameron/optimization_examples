#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <iostream>
#include <chrono>
#include <cstdlib>

/*
This code is entirely based on the examples from Eigen. 
This file simply creates dynamic sized symmetric positive definite matrices and calls Eigen/Cholesky and times the calls. 
Eigen can use MKL if installed, but for the purposes of this test, MKL was not used as MKL was tested separately already. 
If using Eigen in a real-life project, you may want to use Eigen with MKL. 
See instruction on Eigen's website: https://eigen.tuxfamily.org/dox/TopicUsingIntelMKL.html
*/


using namespace Eigen;
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

MatrixXf genRandomPosDefMatrix(int size)
{
	srand(111970);
	// Generate random matrix
	MatrixXf M(size, size);
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			M(i, j) = (float)rand() / RAND_MAX;
	// prod = M^T M to guarantee positive-definite
	MatrixXf MT = M.transpose();
	MatrixXf prod(size, size);
	prod = MT * M;
	return prod;
}

int main(int argc, const char * argv[])
{
	int numRuns = 10;
	for(int mSize = 4; mSize <= 4096; mSize *= 2)
	{
		MatrixXf A = genRandomPosDefMatrix(mSize);
		MatrixXf L(mSize,mSize);

		//Warm up run
		L = A.ldlt().matrixL(); 

		auto t1 = startTimer();	
		for(int i = 0; i < numRuns; i++)
			L = A.llt().matrixL(); 
		auto time1 = endTimer(t1)/numRuns;
		//std::cout << "The Cholesky factor L is" << std::endl << L << std::endl;
		std::cout << mSize << "\t\t" << time1 << " ms"<< std::endl;
	}
	return 0;
}

