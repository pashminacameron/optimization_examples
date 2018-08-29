# Run as follows
# make clean && make ./featureMatching

NAME = featureMatching
CC := /usr/bin/g++
LD := /usr/bin/ld
CCFLAGS := -m64 --std=c++11 -O3 -mavx

INCLUDES += -I..

all: featureMatching

featureMatching.o: featureMatching.cpp
	$(CC) $(INCLUDES) $(CCFLAGS) -c $< -o $@

featureMatching: featureMatching.o
	$(CC) $(INCLUDES) $(LDFLAGS)  $+ -o $@

clean:
	rm -f featureMatching featureMatching.o

