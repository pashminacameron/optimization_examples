

#include <inttypes.h>
#include <stdio.h>
#include <memory.h>
#include <sys/time.h>
#include <x86intrin.h>

#define SIZE 128*8*100
#define NUM_RUNS 10000


//length is guaranteed to be a multiple of 8
void add_avx(const float * vec1, const float * vec2, const float * vec3, int length, float * result);

//length is guaranteed to be a multiple of 4
void add_sse41(const float * vec1, const float * vec2, const float * vec3, int length, float * result);


unsigned getTickCount()
{
        struct timeval tv;
        if(gettimeofday(&tv, NULL) != 0)
                return 0;

        return (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
}

__attribute__ ((aligned(128))) float vec1[SIZE];
__attribute__ ((aligned(128))) float vec2[SIZE];
__attribute__ ((aligned(128))) float vec3[SIZE];
__attribute__ ((aligned(128))) float result[SIZE];


int main()
{
	printf("SIZE = %d \n", SIZE);
	
	// It could be useful to measure actual cycles using __rdtsc(), but then mapping that to real time is non-trivial
	unsigned start1, start2, end1, end2;
	uint64_t startTick, stopTick;

	memset(vec1, 1, sizeof(vec1));
	memset(vec2, 1, sizeof(vec1));
	memset(vec3, 1, sizeof(vec1));

	start1 = getTickCount();
	for (int i=0; i<NUM_RUNS; i++)
	{
		add_avx(vec1, vec2, vec3, SIZE, result);
	}
	end1 = getTickCount();
	
	start2 = getTickCount();
	for (int i=0; i<NUM_RUNS; i++)
	{
		add_sse41(vec1, vec2, vec3, SIZE, result);
	}
	end2 = getTickCount();
	
	printf("add_avx took %lld ms\n", end1 - start1);
	printf("add_sse took %lld ms\n", end2 - start2);
}


