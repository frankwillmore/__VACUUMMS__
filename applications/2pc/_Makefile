#!/usr/bin/make

HOME=/home1/00479/willmore
LIBRARY_DIR=$(HOME)/usr/lib
INCLUDE_DIR=$(HOME)/usr/include
INSTALL_DIR=$(HOME)/usr/bin

2pc:	2pc.c 2pc.h
	gcc -O3 -lm -Wall -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) 2pc.c -o 2pc 

install: 2pc 
	mv 2pc $(INSTALL_DIR)

clean:
	rm *.o
