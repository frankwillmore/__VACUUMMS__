#!/usr/bin/make

INCLUDES=-I. -I/home/willmore/dev/general 

CFLAGS=-Wall

LIBS= -lm 

CC=gcc $(CFLAGS) $(INCLUDES)

csd:	csd.h csd.c command_line_parser.o 
	$(CC) $(LIBS) /home/willmore/dev/general/ftw_rng.o command_line_parser.o csd.c -o csd

command_line_parser.o:	command_line_parser.c
	$(CC) -c command_line_parser.c

install:	csd
	cp csd /home/willmore/bin/

clean:
	rm -v *.o
