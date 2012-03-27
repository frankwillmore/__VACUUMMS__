#!/usr/bin/make

HOME=/home1/00479/willmore/

INCLUDE_DIR=$(HOME)/usr/include
INSTALL_DIR=$(HOME)/usr/bin
LIBRARY_DIR=$(HOME)/usr/lib

CC=gcc

cv:	cv.c
	$(CC) cv.c -Wall -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lm -lftw -o cv 
	mv cv $(INSTALL_DIR)

csa:	csa.c
	gcc csa.c -Wall -I/home/willmore/usr/include -L/home/willmore/lib -lm -lftw -o csa 
	mv csa $(HOME_DIR)/bin

spackle:	spackle.c
	gcc spackle.c -Wall -I/home/willmore/usr/include -L/home/willmore/lib -lm -lftw -o spackle
	mv spackle $(HOME_DIR)/bin

cell_volume:	cell_volume.c
	gcc cell_volume.c -Wall -I/home/willmore/usr/include -L/home/willmore/lib -lm -lftw -o cell_volume 
	mv cell_volume $(HOME_DIR)/bin

boxv:	boxv.c
	gcc boxv.c -Wall -I/home/willmore/usr/include -L/home/willmore/lib -lm -lftw -o boxv 
	mv boxv $(HOME_DIR)/bin

asph:	asph.c
	gcc asph.c -Wall -I/home/willmore/usr/include -L/home/willmore/lib -lm -lftw -o asph
	mv asph $(HOME_DIR)/bin

sample:	sample.c
	gcc sample.c -Wall -I/home/willmore/usr/include -L/home/willmore/lib -lm -lftw -o sample

install:	csa cv asph

