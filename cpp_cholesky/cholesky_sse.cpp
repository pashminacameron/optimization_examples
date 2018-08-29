#include <xmmintrin.h> //SSE
#include <pmmintrin.h> //SSE3 for hadd

#include "cholesky.hpp"

// Assumes the machine has SSE/SSE3 instructions
// If you are not sure of this (i.e. you run on very old machines), 
// you may wish to add runtime checks around this code

namespace linalg{

    float sumPairwiseProductSSE(const float * u, const float * v, const float * d, int size)
    {
		float acc = 0;
		int groups_16 = size / 16;                  // groups of 16 elements
		int groups_4 = (size - groups_16 * 16) / 4; // remaining groups of 4
		int groups_1 = size % 4;	                // remaining groups of 1

		//Process 16 elements across 4 lanes
		__m128 result = _mm_setzero_ps();
		for (int i = 0; i < groups_16; i++)
		{	
			__m128 lane[4];
			// computes u[i] * v[i] * d[i]
			lane[0] = _mm_mul_ps(_mm_mul_ps(_mm_load_ps(u + i * 16), _mm_load_ps(v + i * 16)), _mm_load_ps(d + i * 16));
			lane[1] = _mm_mul_ps(_mm_mul_ps(_mm_load_ps(u + i * 16 + 4), _mm_load_ps(v + i * 16 + 4)), _mm_load_ps(d + i * 16 + 4));
			lane[2] = _mm_mul_ps(_mm_mul_ps(_mm_load_ps(u + i * 16 + 8), _mm_load_ps(v + i * 16 + 8)), _mm_load_ps(d + i * 16 + 8));
			lane[3] = _mm_mul_ps(_mm_mul_ps(_mm_load_ps(u + i * 16 + 12), _mm_load_ps(v + i * 16 + 12)), _mm_load_ps(d + i * 16 + 12));

			lane[0] = _mm_add_ps(lane[0], lane[1]);
			lane[2] = _mm_add_ps(lane[2], lane[3]);
			lane[0] = _mm_add_ps(lane[0], lane[2]);
			result = _mm_add_ps(result, lane[0]); //add the result of all four lanes to the accumulator
		}

		// Process 4 elements in one lane
		__m128 singleLane = _mm_setzero_ps();
		for (int it = 0; it < groups_4; it++)
		{
			__m128 a1 = _mm_load_ps(u + 16 * groups_16 + 4 * it);
			__m128 b1 = _mm_load_ps(v + 16 * groups_16 + 4 * it);
			__m128 c1 = _mm_load_ps(d + 16 * groups_16 + 4 * it);
			singleLane = _mm_add_ps(singleLane, _mm_mul_ps(_mm_mul_ps(a1, b1),c1));
		}
		result = _mm_add_ps(result, singleLane);
		result = _mm_hadd_ps(result, result);
		result = _mm_hadd_ps(result, result);

		_mm_store_ss(&acc, result);

		// Add last few after multiples of 4
		if (groups_1)
			for (int i = groups_16 * 16 + groups_4 * 4; i < size; i++)
				acc += u[i] * v[i] * d[i];
			
		return acc;
    }

}
      
