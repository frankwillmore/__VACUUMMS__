#!/usr/bin/make

HOME_DIR=/home/willmore

size:	size.c
	gcc size.c -I/home/willmore/usr/include -L/home/willmore/lib -lm -lftw -o size

install:	size
	mv size $(HOME_DIR)/bin
