#!/usr/bin/make

HOME=/home1/00479/willmore

INCLUDE_DIR=$(HOME)/usr/include
LIBRARY_DIR=$(HOME)/usr/lib
INSTALL_DIR=$(HOME)/usr/bin

fv2fv:	fv2fv.c 
	gcc fv2fv.c -O3 -o fv2fv -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm 

install: fv2fv
	mv fv2fv $(INSTALL_DIR)

