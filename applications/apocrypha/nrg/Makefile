#!/usr/bin/make

HOME_DIR=/home/willmore

nrg:	io_setup.o energy.o nrg.c nrg.h
	gcc -O3 -Wall -I$(HOME_DIR)/usr/include -L$(HOME_DIR)/lib io_setup.o energy.o nrg.c -lm -lftw -o nrg

io_setup.o:	io_setup.c io_setup.h
	gcc -O3 -Wall -I$(HOME_DIR)/usr/include -c io_setup.c 

energy.o:	energy.c energy.h
	gcc -O3 -Wall -I$(HOME_DIR)/usr/include -c energy.c

install: nrg 
	cp nrg $(HOME_DIR)/bin/nrg

clean:
	rm *.o
