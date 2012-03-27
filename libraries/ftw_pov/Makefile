#!/bin/make

HOME=/home1/00479/willmore

LIBRARY_DIR=$(HOME)/usr/lib
INCLUDE_DIR=$(HOME)/usr/include
INSTALL_DIR=$(HOME)/usr/bin

INCLUDES=-I$(INCLUDE_DIR) 
LIBS=-L$(LIBRARY_DIR) 

CC=gcc

ftw_povheader.o:       ftw_povheader.c ftw_pov.h
	$(CC) ftw_povheader.c -c -fPIC -L$(LIBRARY_DIR) -I$(INCLUDE_DIR)

libftw_pov.so:	ftw_povheader.o
	$(CC) -shared -o libftw_pov.so ftw_povheader.o -fPIC 

#example:  example.c libftw_tiff.so
#	$(CC) -L$(LIBRARY_DIR) -I$(INCLUDE_DIR) example.c -ltiff -lftw_tiff -o example

clean:
	rm -f ftw_povheader.o libftw_pov.so

install-header:	ftw_pov.h
	cp ftw_pov.h $(INCLUDE_DIR)

install:	libftw_pov.so ftw_pov.h
	cp libftw_pov.so $(LIBRARY_DIR)
	cp ftw_pov.h $(INCLUDE_DIR)

