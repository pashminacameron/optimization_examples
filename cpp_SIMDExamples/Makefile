CC := /usr/bin/g++
CCFLAGS := -m64 --std=c++11 -O3 
LDFLAGS := 
INCLUDES+= -I .
LIBS += 

all: build
build: testSIMD

sample_avx.o: sample_avx.cpp
	$(CC) $(INCLUDES) $(CCFLAGS) -mavx -c $< -o $@

sample_sse41.o: sample_sse41.cpp
	$(CC) $(INCLUDES) $(CCFLAGS) -msse4.1 -c $< -o $@

testSIMD.o: testSIMD.cpp
	$(CC) $(INCLUDES) $(CCFLAGS) -c $< -o $@

testSIMD: sample_sse41.o sample_avx.o testSIMD.o
	$(CC) $(INCLUDES) $(LDFLAGS) -o $@ $+ $(LIBS)

clean:
	rm -f testSIMD sample_sse41.o sample_avx.o testSIMD.o




