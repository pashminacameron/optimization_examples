#include <immintrin.h> //AVX

#include "cholesky.hpp"

// Assumes the machine has AVX instructions
// If you are not sure of this (i.e. you run on very old machines), 
// you may wish to add runtime checks around this code

namespace linalg{

    float sum3VecProductAVX(const float * u, const float * v, const float * d, int size)
    {
		float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
		int groups_8 = size / 8;                  // groups of 8 elements
		int groups_1 = size % 8;	                // remaining groups of 1

		// Process 8 elements in one lane
		__m256 singleLane = _mm256_setzero_ps();
		_mm256_zeroall();
		for (int it = 0; it < groups_8; it++)
		{
			__m256 a1 = _mm256_load_ps(u + 8 * it);
			__m256 b1 = _mm256_load_ps(v +  8 * it);
			__m256 c1 = _mm256_load_ps(d +  8 * it);
			b1 = _mm256_mul_ps(a1, b1);
			c1 = _mm256_mul_ps(b1, c1);
			singleLane = _mm256_add_ps(singleLane, c1);
		}
		singleLane = _mm256_hadd_ps(singleLane, singleLane);
		singleLane = _mm256_hadd_ps(singleLane, singleLane);
		singleLane = _mm256_hadd_ps(singleLane, singleLane);
		_mm256_storeu_ps(&acc[0], singleLane); // we have the answer in acc[0] as we have already done the horizontal add

		// Add last few after multiples of 8
		if (groups_1)
			for (int i = groups_8 * 8; i < size; i++)
				acc[0] += u[i] * v[i] * d[i];

		return acc[0];
    }

	float sum2VecProductAVX(const float * u, const float * v, int size)
	{
		float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
		int groups_8 = size / 8;                  // groups of 8 elements
		int groups_1 = size % 8;	                // remaining groups of 1

													// Process 8 elements in one lane
		__m256 singleLane = _mm256_setzero_ps();
		_mm256_zeroall();
		for (int it = 0; it < groups_8; it++)
		{
			__m256 a1 = _mm256_load_ps(u + 8 * it);
			__m256 b1 = _mm256_load_ps(v + 8 * it);
			singleLane = _mm256_add_ps(singleLane, _mm256_mul_ps(a1, b1));
		}
		singleLane = _mm256_hadd_ps(singleLane, singleLane);
		singleLane = _mm256_hadd_ps(singleLane, singleLane);
		singleLane = _mm256_hadd_ps(singleLane, singleLane);
		_mm256_storeu_ps(&acc[0], singleLane); // we have the answer in acc[0] as we have already done the horizontal add

											   // Add last few after multiples of 8
		if (groups_1)
			for (int i = groups_8 * 8; i < size; i++)
				acc[0] += u[i] * v[i];

		return acc[0];
	}

}
      
