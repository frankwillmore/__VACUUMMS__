#!/usr/bin/make

HOME=/home/00479/willmore/home/

INCLUDES=-I$(HOME)/usr/include -I.
LIBS=-L$(HOME)/lib -L/opt/apps/cuda/2.2/cuda/lib/
CC=nvcc
CFLAGS= -O2 -lpthread -lm -lcudart -lftw -D_REENTRANT 
#CFLAGS= -g -deviceemu -lpthread -lm -lcudart -lftw -D_REENTRANT
#-DNDEBUG if ignoring assertions

MersenneTwister.o:	MersenneTwister.c pmdgpu.h
	$(CC) $(CFLAGS) MersenneTwister.c -c $(INCLUDES) $(LIBS)

ThreadMain.o:	ThreadMain.c pmdgpu.h
	$(CC) $(CFLAGS) ThreadMain.c -c $(INCLUDES) $(LIBS)

Configuration.o:	Configuration.c pmdgpu.h
	$(CC) $(CFLAGS) Configuration.c -c $(INCLUDES) $(LIBS)

CommandLineOptions.o:	CommandLineOptions.c pmdgpu.h
	$(CC) $(CFLAGS) CommandLineOptions.c -c $(INCLUDES) $(LIBS)

Verlet.o:	Verlet.c pmdgpu.h
	$(CC) $(CFLAGS) Verlet.c -c $(INCLUDES) $(LIBS)

Energy.o:	Energy.c pmdgpu.h
	$(CC) $(CFLAGS) Energy.c -c $(INCLUDES) $(LIBS)

Rotation.o:	Rotation.c pmdgpu.h
	$(CC) $(CFLAGS) Rotation.c -c $(INCLUDES) $(LIBS)

CUDAEnergy.o:	CUDAEnergy.cu pmdgpu.h
	$(CC) $(CFLAGS) CUDAEnergy.cu -c $(INCLUDES) $(LIBS)

pmdgpu:	pmdgpu.c pmdgpu.h MersenneTwister.o ThreadMain.o CommandLineOptions.o Configuration.o Verlet.o CUDAEnergy.o Energy.o
	mpicc -O2 -lpthread -lcudart -lftw -D_REENTRANT pmdgpu.c CUDAEnergy.o Verlet.o Configuration.o CommandLineOptions.o ThreadMain.o MersenneTwister.o Energy.o -o pmdgpu $(INCLUDES) $(LIBS) 

clean:
	rm *.o 
	rm *.linkinfo
	rm pmdgpu

