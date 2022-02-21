#!/usr/bin/make

mfp:	mfp.c io_setup.o
	gcc *.o mfp.c -o mfp -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 
	cp mfp ~/bin/mfp

io_setup.o:	io_setup.c io_setup.h
	gcc -Wall -O3 -c io_setup.c -I/home/willmore/usr/include

install:	mfp	
	cp mfp /home/willmore/bin/

clean:
	rm -v *.o


