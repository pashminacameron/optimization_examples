#include "immintrin.h"

//This function is in a file called sample_avx.cpp and is compiled with /arch:avx

//length is guaranteed to be a multiple of 8
void add_avx(const float * vec1, const float * vec2, const float * vec3, int length, float * result)
{
	const int PARALLEL = 8;
	_mm256_zeroall();
	for (int j = 0; j < length; j = j + PARALLEL)
	{
		__m256 a = _mm256_loadu_ps(&vec1[j]);
		__m256 b = _mm256_loadu_ps(&vec2[j]);
		__m256 c = _mm256_add_ps(_mm256_loadu_ps(&vec3[j]), _mm256_add_ps(a,b));

		_mm256_storeu_ps(&result[j], c);
//		_mm256_stream_ps(&result[j], c); 
	}
	_mm256_zeroall();
}
