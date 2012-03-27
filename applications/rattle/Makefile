#!/usr/bin/make

HOME=/home1/00479/willmore

INSTALL_DIR=$(HOME)/usr/bin
LIBRARY_DIR=$(HOME)/usr/lib
INCLUDE_DIR=$(HOME)/usr/include

rattle:	rattle.c 
	gcc rattle.c -O2 -o rattle -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -Wall

rattlj:	rattlj.c 
	gcc rattlj.c -O2 -o rattlj -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -Wall

install:	rattle rattlj
	mv rattle rattlj $(INSTALL_DIR) 

clean:
	rm -f *.o rattle 
