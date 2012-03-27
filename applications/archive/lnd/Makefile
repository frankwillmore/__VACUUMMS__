#!/usr/bin/make

lnd:	lnd.h lnd.c io_setup.o
	gcc *.o lnd.c -o lnd -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 

sea:	lnd.h sea.c io_setup.o
	gcc *.o sea.c -o sea -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 
	mv sea $(HOME)/bin/sea

io_setup.o:	io_setup.c io_setup.h
	gcc -Wall -O3 -c io_setup.c -I/home/willmore/usr/include -lftw -lm

install:	lnd sea
	cp lnd /home/willmore/bin/
	cp sea /home/willmore/bin/

clean:
	rm -v *.o
