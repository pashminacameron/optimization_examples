# Run as follows
# make clean && make ./testEigen

NAME = testEigen
CC := /usr/bin/g++
LD := /usr/bin/ld
CCFLAGS := -m64 --std=c++11 -O3

INCLUDES += -I.. -I/mnt/c/dev/eigen

all: testEigen

testEigen.o: testEigen.cpp
	$(CC) $(INCLUDES) $(CCFLAGS) -c $< -o $@

testEigen: testEigen.o
	$(CC) $(INCLUDES) $(LDFLAGS)  $+ -o $@

clean:
	rm -f testEigen testEigen.o

