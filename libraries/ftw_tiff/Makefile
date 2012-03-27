#!/bin/make

HOME=/home1/00479/willmore

LIBRARY_DIR=$(HOME)/usr/lib
INCLUDE_DIR=$(HOME)/usr/include
INSTALL_DIR=$(HOME)/usr/bin

INCLUDES=-I$(INCLUDE_DIR) 
LIBS=-L$(LIBRARY_DIR) 

CC=gcc

ftw_makeTIFF.o:       ftw_makeTIFF.c ftw_makeTIFF.h
	$(CC) ftw_makeTIFF.h ftw_makeTIFF.c -c -I. -fPIC

libftw_tiff.so:	ftw_makeTIFF.o
	$(CC) -shared -o libftw_tiff.so ftw_makeTIFF.o -fPIC 

example:  example.c libftw_tiff.so
	$(CC) -L$(LIBRARY_DIR) -I$(INCLUDE_DIR) example.c -ltiff -lftw_tiff -o example

clean:
	rm -f ftw_makeTIFF.o libftw_tiff.so

install:	libftw_tiff.so ftw_makeTIFF.h
	cp libftw_tiff.so $(LIBRARY_DIR)
	cp ftw_makeTIFF.h $(INCLUDE_DIR)

