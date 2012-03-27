#!/usr/bin/make

HOME_DIR=/home/willmore

rog:	rog.c
	gcc rog.c -I/home/willmore/usr/include -L/home/willmore/lib -lm -lftw -o rog

install:	rog
	mv rog $(HOME_DIR)/bin
