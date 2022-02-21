#!/bin/make

#  Install to user home by default
HOME_DIR:=$$HOME

LIBRARY_DIR:=$(HOME_DIR)/usr/lib
INCLUDE_DIR:=$(HOME_DIR)/usr/include
INSTALL_DIR:=$(HOME_DIR)/usr/bin

INCLUDES:=-I$(INCLUDE_DIR) 
LIBS:=-L$(LIBRARY_DIR) 

CC:=gcc
CFLAGS:=-O2

echo:	
	@echo "Please specify a target, e.g. 'make install' "
	@echo $(HOME_DIR)

ftw_povheader.o:       ftw_povheader.c ftw_pov.h
	$(CC) $(CFLAGS) ftw_povheader.c -c -fPIC -L$(LIBRARY_DIR) -I$(INCLUDE_DIR)

libftw_pov.so:	ftw_povheader.o
	$(CC) $(CFLAGS) -shared -o libftw_pov.so ftw_povheader.o -fPIC 

#example:  example.c libftw_tiff.so
#	$(CC) -L$(LIBRARY_DIR) -I$(INCLUDE_DIR) example.c -ltiff -lftw_tiff -o example

install:  install-header install-lib

install-header:	ftw_pov.h
	cp ftw_pov.h $(INCLUDE_DIR)

install-lib:	libftw_pov.so ftw_pov.h
	mv libftw_pov.so $(LIBRARY_DIR)

clean:
	rm -f ftw_povheader.o libftw_pov.so

.PHONY:  clean install install-header

