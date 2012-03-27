#!/usr/bin/make

HOME_DIR=/home/willmore

center:	center.c
	gcc center.c -I$(HOME_DIR)/usr/include -L$(HOME_DIR)/lib -lftw -o center

install:	center
	mv center $(HOME_DIR)/bin
