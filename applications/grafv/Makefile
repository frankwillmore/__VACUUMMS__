#!/usr/bin/make

grafv:	grafv.c io_setup.o
	gcc *.o grafv.c -o grafv -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 

io_setup.o:	io_setup.c io_setup.h
	gcc -Wall -O3 -c io_setup.c -I/home/willmore/usr/include

install:	grafv clean
	cp grafv /home/willmore/bin/

clean:
	rm -v *.o


