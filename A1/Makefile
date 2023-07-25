CC=g++ ### make file variables
CFLAGS=-std=c++11 -O3 -fopenmp -Wall

# export LD_LIBRARY_PATH := ./

exec:main.o
	$(CC) $(CFLAGS)  main.o -L. -l:library.so -o exec

run: exec
	export LD_LIBRARY_PATH=./ ; ./exec input output

main.o: main.cpp library.hh
	$(CC) $(CFLAGS) -c main.cpp

clean:
	rm *.o exec