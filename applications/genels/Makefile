#!/usr/bin/make

HOME=/home/00479/willmore/home/

INCLUDES=-I$(HOME)/usr/include -I.
LIBS=-L$(HOME)/lib -L/opt/apps/cuda/2.2/cuda/lib/
CC=nvcc
CFLAGS= -O2 -lpthread -lm -lcudart -D_REENTRANT 
#CFLAGS= -g -deviceemu -lpthread -lm -lcudart -D_REENTRANT

ThreadMain.o:	ThreadMain.c genels.h
	$(CC) $(CFLAGS) ThreadMain.c -c $(INCLUDES) $(LIBS)

CommandLineOptions.o:	CommandLineOptions.c genels.h
	$(CC) $(CFLAGS) CommandLineOptions.c -c $(INCLUDES) $(LIBS)

readPolymerConfiguration.o:	readPolymerConfiguration.c genels.h
	$(CC) $(CFLAGS) readPolymerConfiguration.c -c $(INCLUDES) $(LIBS)

io.o:	io.c genels.h
	$(CC) $(CFLAGS) io.c -c $(INCLUDES) $(LIBS)

Verlet.o:	Verlet.c genels.h
	$(CC) $(CFLAGS) Verlet.c -c $(INCLUDES) $(LIBS)

CUDAEnergy.o:	CUDAEnergy.cu genels.h
	$(CC) $(CFLAGS) CUDAEnergy.cu -c $(INCLUDES) $(LIBS)

genels:	genels.c genels.h ThreadMain.o CommandLineOptions.o readPolymerConfiguration.o Verlet.o CUDAEnergy.o io.o
	$(CC) $(CFLAGS) genels.c CUDAEnergy.o Verlet.o CommandLineOptions.o readPolymerConfiguration.o ThreadMain.o io.o -o genels $(INCLUDES) $(LIBS) 

genels-debug:	genels.c genels.h ThreadMain.o CommandLineOptions.o readPolymerConfiguration.o Verlet.o CUDAEnergy.o io.o
	$(CC) -g -deviceemu $(CFLAGS) genels.c CUDAEnergy.o Verlet.o CommandLineOptions.o readPolymerConfiguration.o ThreadMain.o io.o -o genels $(INCLUDES) $(LIBS) 

clean:
	rm *.o 
	rm *.linkinfo
	rm genels
	rm genels-emu

readit:	readit.c genels.h
	$(CC) $(CFLAGS) readit.c -o readit $(INCLUDES) $(LIBS)
