#!/usr/bin/make

HOME_DIR:=/home1/00479/willmore

INSTALL_DIR:=$(HOME_DIR)/usr/bin
LIBRARY_DIR:=$(HOME_DIR)/usr/lib
INCLUDE_DIR:=$(HOME_DIR)/usr/include

rattle:	rattle.c 
	gcc rattle.c -O2 -o rattle -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -Wall

rattlj:	rattlj.c 
	gcc rattlj.c -O2 -o rattlj -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -Wall

install:	rattle rattlj
	mv rattle rattlj $(INSTALL_DIR) 

clean:
	rm -f rattle rattlj
