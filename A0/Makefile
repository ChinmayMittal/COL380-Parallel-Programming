#g++ -std=c++11 -O2 -mavx2 -c classify.cpp
## $@ => target
### $^ => all prerequisits
### $? => Outputs all prerequisites newer than the target
## $< the first prerequisite
CC=g++ ### make file variables
CFLAGS=-std=c++11 -O2 -fopenmp -g
NT=4
NITER=10
sources=main.cpp classify.cpp
objects=$(sources:.cpp=.o)

classify:$(objects)
	$(CC) $(CFLAGS) $^ -o $@

run: classify
	./classify rfile dfile 1009072 $(NT) $(NITER)

#classify.o: classify.h
#	touch classify.o
#	$(CC) $(CFLAGS) -c $<
#main.o: classify.h
#	$(CC) $(CFLAGS) -c $<

%.o: %.cpp classify.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm *.o classify
