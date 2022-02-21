#!/usr/bin/make

ddsg:	ddsg.h ddsg.c io_setup.o convergence.o
	gcc *.o ddsg.c -o ddsg -I/home/willmore/usr/include -L/home/willmore/lib -lftw -lm

covergence.o:	convergence.h convergence.c
	gcc -L/home/willmore/lib -I/home/willmore/usr/include -c convergence.c

io_setup.o:	io_setup.c io_setup.h
	gcc -Wall -O3 -c io_setup.c -I/home/willmore/usr/include

install:	ddsg
	cp ddsg /home/willmore/bin/

clean:
	rm -v *.o
