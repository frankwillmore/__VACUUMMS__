#!/usr/bin/make

INCLUDES=-I. 

CFLAGS=-c -O3 -Wall

CC=gcc

LIBS=-L/usr/X11R6/lib -lm -lX11
LDFLAGS=

circles:	circles.c
	$(CC) $(INCLUDES) -DBG_COLOR=65535 -DFG_COLOR=255 $(LIBS) $(INCLUDES) circles.c

chains:	chains.c
	$(CC) $(INCLUDES) -DBG_COLOR=65535 -DFG_COLOR=255 $(LIBS) $(INCLUDES) chains.c
clean:
	rm *.o

