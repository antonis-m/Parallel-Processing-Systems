CC=g++
CFLAGS=-O3 -fopenmp -Wall -g -std=c++0x
LIBS= -ltbb -lrt
EXECUTABLES=  recursive

all: $(EXECUTABLES)

recursive: recursive_tasks.o 
	$(CC) $^  -o recursive $(LIBS)

recursive_tasks.o: recursive_tasks.cpp
	$(CC) $(CFLAGS) -c $< 

clean:
	rm  *.o

distclean:
	rm *.o res* $(EXECUTABLES)
	
