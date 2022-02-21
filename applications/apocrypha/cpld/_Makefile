#!/usr/bin/make

INCLUDES=-I. -I/home/willmore/dev/general 

CFLAGS=-Wall -O3

CC=gcc

LIBS=-L/usr/X11R6/lib -lm -lX11

cpld:	cpld.h cpld.c io.o edges.o
	gcc $(LIBS) $(INCLUDES) cpld.c io.o edges.o -o cpld

io.o:	io.h io.c
	gcc $(LIBS) $(INCLUDES) io.c -c

edges.o:	edges.h edges.c
	gcc $(LIBS) $(INCLUDES) edges.c -c

install:	cpld
	cp cpld /home/willmore/bin/cpld

clean:
	rm -v *.o
