CC=g++ ### make file variables
CFLAGS=-std=c++11 -Wall -O2
sources=main.cpp
objects=$(sources:.cpp=.o)

a2:$(objects)
	mpic++ $^ -o $@
#$(CC) $(CFLAGS) $^ -o $@

run: a2
	mpirun -np 4 ./a2 --taskid==1 --inputpath=../test3/test-input-3.gra --headerpath=../test3/test-header-3.dat --outputpath=./out.txt --verbose=1 --startk=1 --endk=5 --p=1

%.o: %.cpp
# $(CC) $(CFLAGS) -c $<
	mpic++ -c $<

clean:
	rm *.o a2