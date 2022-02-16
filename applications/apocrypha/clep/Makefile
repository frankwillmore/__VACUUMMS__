#!/usr/bin/make

HOME=/home1/00479/willmore

INSTALL_DIR=$(HOME)/usr/bin
LIBRARY_DIR=$(HOME)/usr/lib
INCLUDE_DIR=$(HOME)/usr/include

clep:	clep.c 
	gcc clep.c -O2 -o clep -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -Wall -lftw -lm -lftw_energy

test:	test.c 
	gcc test.c -O2 -o test -lftw_energy -L/home1/00479/willmore/usr/lib -Wall -I/home1/00479/willmore/usr/include
#	gcc test.c -O2 -o test -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -I. -lftw_energy

install:	clep 
	cp clep $(INSTALL_DIR) 

clean:
	rm -f *.o clep
