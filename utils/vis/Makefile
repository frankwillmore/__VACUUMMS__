#!/bin/make

HOME_DIR=/ranger/share/home/00479/willmore/

INCLUDES=-I. -I$(HOME_DIR)/usr/include
CFLAGS=-c -O2 -Wall
CC=gcc
LIBS=-L/usr/X11R6/lib -lm -lX11 

vis:	io_setup.o graphics.o command_line_parser.o vis.c vis.h
	gcc -O3 -Wall $(INCLUDES) $(LIBS) *.o vis.c -o vis

io_setup.o:	io_setup.c io_setup.h
	$(CC) $(CFLAGS) $(INCLUDES) io_setup.c

graphics.o:	graphics.c graphics.h
	$(CC) $(CFLAGS) $(INCLUDES) graphics.c

command_line_parser.o:	command_line_parser.c command_line_parser.h
	$(CC) $(CFLAGS) $(INCLUDES) command_line_parser.c

install: vis
	cp vis /home/willmore/bin/

clean:
	rm *.o

cav2vis:	cav2vis.c
	gcc cav2vis.c -o cav2vis
	cp cav2vis /home/willmore/bin/cav2vis
