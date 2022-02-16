#!/usr/bin/make

HOME_DIR=/home/willmore

nanopore:	nanopore.c
	gcc nanopore.c -I$(HOME_DIR)/usr/include -L$(HOME_DIR)/lib -lftw -o nanopore

install:	nanopore
	mv nanopore $(HOME_DIR)/bin
