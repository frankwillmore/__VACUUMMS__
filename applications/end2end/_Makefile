#!/usr/bin/make

HOME=/home1/00479/willmore

INSTALL_DIR=$(HOME)/usr/bin
LIBRARY_DIR=$(HOME)/usr/lib
INCLUDE_DIR=$(HOME)/usr/include

CC=icc


end2end:	end2end.c
	$(CC) end2end.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lm -lftw -o end2end

install:	end2end
	mv end2end $(INSTALL_DIR)
