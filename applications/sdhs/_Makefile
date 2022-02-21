#!/usr/bin/make

HOME_DIR=/home/willmore

INCLUDES=-I$(HOME_DIR)/usr/include/
CFLAGS=-O3 -Wall
CC=gcc
LIBS=-lm -lX11 -lftw 
LIB_DIRS= -L$(HOME_DIR)/lib/ -L/usr/X11R6/lib/

sdhs:	io_setup.o graphics.o sdhs_main.c sdhs_main.h
	gcc *.o sdhs_main.c -o sdhs -I$(HOME_DIR)/usr/include/ -L$(HOME_DIR)/lib/ -L/usr/X11R6/lib -lm -lX11 -lftw -O3 -Wall

io_setup.o:	io_setup.c
	gcc -c io_setup.c -I$(HOME_DIR)/usr/include -L$(HOME_DIR)/lib -lftw -O3 -Wall

graphics.o:	graphics.c
	gcc -c graphics.c -I$(HOME_DIR)/usr/include -L/usr/X11R6/lib -O3 -Wall

install: sdhs
	cp sdhs $(HOME_DIR)/bin
	make clean

clean:
	rm *.o
