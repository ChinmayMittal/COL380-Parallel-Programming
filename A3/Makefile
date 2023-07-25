CC=g++ ### make file variables
CFLAGS=-std=c++11 -O2 -fopenmp
sources=main.cpp
objects=$(sources:.cpp=.o)


a3:$(objects)
	mpic++ $(CFLAGS) $^ -o $@

run: a3
	mpirun -np 8 ./a3 --taskid=2 --inputpath=/tmp/COL380/A3/test6/test-input-6.gra --headerpath=/tmp/COL380/A3/test6/test-header-6.dat --outputpath=./out.txt --verbose=1 --startk=4 --endk=4 --p=6

%.o: %.cpp
	mpic++ $(CFLAGS) -c $<

clean:
	rm *.o a3 


