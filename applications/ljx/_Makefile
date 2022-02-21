#!/usr/bin/make

HOME_DIR=/home/willmore

INCLUDES=-I$(HOME_DIR)/usr/include/
CFLAGS=-O3 -Wall
CC=gcc
LIBS=-lm -lX11 -lftw 
LIB_DIRS= -L$(HOME_DIR)/lib/ -L/usr/X11R6/lib/

ljx:	io_setup.o graphics.o command_line_parser.o energy.o ljx_main.c ljx_main.h
	gcc *.o ljx_main.c -o ljx -I$(HOME_DIR)/usr/include/ -L$(HOME_DIR)/lib/ -L/usr/X11R6/lib -lm -lX11 -lftw -O3 -Wall

io_setup.o:	io_setup.c
	gcc -c io_setup.c -I$(HOME_DIR)/usr/include -L$(HOME_DIR)/lib -lftw -O3 -Wall

graphics.o:	graphics.c
	gcc -c graphics.c -I$(HOME_DIR)/usr/include -L/usr/X11R6/lib -O3 -Wall

command_line_parser.o:	command_line_parser.c command_line_parser.h
	$(CC) $(CFLAGS) $(INCLUDES) $(LIB_DIRS) $(LIBS) -c command_line_parser.c

energy.o:	energy.c energy.h
	$(CC) $(CFLAGS) $(INCLUDES) $(LIB_DIRS) $(LIBS) -c energy.c

install: ljx
	cp ljx $(HOME_DIR)/bin
	make clean

clean:
	rm *.o
