#!/usr/bin/make

# Defualt install is to user home

HOME_DIR:=/home1/00479/willmore

INSTALL_DIR:=$(HOME_DIR)/usr/bin
LIBRARY_DIR:=$(HOME_DIR)/usr/lib
INCLUDE_DIR:=$(HOME_DIR)/usr/include

CC:=gcc
CFLAGS:=-Wall -O2 -lftw -lm

ddx:	ddx.h ddx.c io_setup.o 
	$(CC) $(CFLAGS) io_setup.o ddx.c -o ddx -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) 

# trace_ddx outlines the path followed from insertion to minimum
trace_ddx:	ddx.h trace_ddx.c io_setup.o 
	$(CC) $(CFLAGS) io_setup.o trace_ddx.c -O3 -o trace_ddx -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) 

io_setup.o:	io_setup.c io_setup.h
	$(CC) $(CCFLAGS) -c io_setup.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR)

install:	ddx trace_ddx
	mv ddx trace_ddx $(INSTALL_DIR) 

clean:
	rm -f *.o ddx dddx trace_ddx cull 
