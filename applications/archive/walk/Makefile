#!/usr/bin/make

walk:	walk.c io_setup.o
	gcc *.o walk.c -o walk -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 

walk2:	walk2.c io_setup.o
	gcc *.o walk2.c -o walk2 -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 

io_setup.o:	io_setup.c io_setup.h
	gcc -Wall -O3 -c io_setup.c -I/home/willmore/usr/include

install:	walk walk2	
	cp walk /home/willmore/bin/
	cp walk2 /home/willmore/bin/

clean:
	rm -v *.o


