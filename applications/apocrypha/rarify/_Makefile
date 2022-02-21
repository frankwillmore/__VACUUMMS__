#!/usr/bin/make

HOME_DIR=/home/willmore

rarify:	rarify.c
	gcc rarify.c -I$(HOME_DIR)/usr/include -L$(HOME_DIR)/lib -lftw -o rarify

install:	rarify
	mv rarify $(HOME_DIR)/bin
