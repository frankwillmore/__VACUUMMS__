#!/usr/bin/make


INSTALL_DIR=~/usr/local/bin

INCLUDES=-I~/usr/include -I.
LIBS=-L~/lib -L/opt/apps/cuda/3.1/cuda/lib64/

CC=nvcc
CFLAGS= -O2 -lm -lcudart 

CommandLineOptions.o:	CommandLineOptions.c fv.h
	$(CC) $(CFLAGS) CommandLineOptions.c -c $(INCLUDES) $(LIBS)

readPolymerConfiguration.o:	readPolymerConfiguration.c fv.h
	$(CC) $(CFLAGS) readPolymerConfiguration.c -c $(INCLUDES) $(LIBS)

fv:	fv.cu fv.h CommandLineOptions.o readPolymerConfiguration.o 
	$(CC) $(CFLAGS) fv.cu CommandLineOptions.o readPolymerConfiguration.o -o fv $(INCLUDES) $(LIBS) 

clean:
	rm -f *.o 
	rm -f *.linkinfo
	rm -f fv

install: fv
	mv fv ${INSTALL_DIR}
