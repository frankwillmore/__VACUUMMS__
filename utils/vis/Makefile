#!/bin/make

# 	By default, the install will be based from the users' home directory.  
# 	Change the value here to use any other directory. 

HOME_DIR:=$$HOME

INSTALL_DIR:=$(HOME_DIR)/usr/bin

INCLUDES:=-I. -I$(HOME_DIR)/usr/include

CFLAGS:=-O2 -Wall
CC:=gcc
LIBS:=-L/usr/X11R6/lib -lm -lX11 

vis:	io_setup.o graphics.o command_line_parser.o vis.c vis.h
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) *.o vis.c -o vis

io_setup.o:	io_setup.c io_setup.h
	$(CC) $(CFLAGS) $(INCLUDES) -c io_setup.c

graphics.o:	graphics.c graphics.h
	$(CC) $(CFLAGS) $(INCLUDES) -c graphics.c

command_line_parser.o:	command_line_parser.c command_line_parser.h
	$(CC) $(CFLAGS) $(INCLUDES) -c command_line_parser.c

install: vis
	mv vis $(INSTALL_DIR)
	make clean

clean:
	rm -f *.o
	rm -f vis

cav2vis:	cav2vis.c
	gcc cav2vis.c -o cav2vis
	cp cav2vis /home/willmore/bin/cav2vis

