#!/usr/bin/make

dds:	dds.h dds.c io_setup.o convergence.o
	gcc *.o dds.c -o dds -I/home/willmore/usr/include -L/home/willmore/lib -lftw -lm

covergence.o:	convergence.h convergence.c
	gcc -c convergence.c -L/home/willmore/lib -I/home/willmore/usr/include

io_setup.o:	io_setup.c io_setup.h
	gcc -Wall -O3 -c io_setup.c -I/home/willmore/usr/include

install:	dds
	cp dds /home/willmore/bin/

clean:
	rm -v *.o

cg:	crystal_generator.c
	gcc crystal_generator.c -o cg

