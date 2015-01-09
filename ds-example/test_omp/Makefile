CC=icc
CYTHON=cython
CFLAGS=-fPIC -fopenmp -xhost -O3
TC_HOME=/home/apps/astro
INCLUDES=-I$(TC_HOME)/include/python2.7

all:
	$(CYTHON) test_omp.pyx
	$(CC) $(CFLAGS) $(INCLUDES) -shared -o test_omp.so test_omp.c 
clean:
	rm -rf test_omp.c
	rm -rf test_omp.so
