#include "immintrin.h"

//This function is in a file called sample_sse41.cpp and is compiled with /arch:sse2

//length is guaranteed to be a multiple of 4
void add_sse41(const float * vec1, const float * vec2, const float * vec3, int length, float * result)
{
	const int PARALLEL = 4;
	for (int j = 0; j < length; j = j + PARALLEL)
	{
		__m128 a = _mm_loadu_ps(&vec1[j]);
		__m128 b = _mm_loadu_ps(&vec2[j]);
		__m128 c = _mm_add_ps(_mm_loadu_ps(&vec3[j]), _mm_add_ps(a,b));
		_mm_storeu_ps(&result[j], c);
	}
}

