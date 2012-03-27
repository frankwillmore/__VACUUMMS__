#!/usr/bin/make

HOME=/ranger/share/home/00479/willmore/

INCLUDES=-I$(HOME)/usr/include -I.
#LIBS=-L$(HOME)/lib -L/opt/apps/cuda/2.2/cuda/lib/
#LIBS=-L$(HOME)/lib -L/opt/apps/cuda/3.0/cuda/lib/
LIBS=-L$(HOME)/lib -L/opt/apps/cuda/3.0/cuda/lib64/

CC=nvcc
#CFLAGS= -O0 -g -G -lpthread -lm -lcudart -lftw -D_REENTRANT 
#CFLAGS= -g -deviceemu -lpthread -lm -lcudart -lftw -D_REENTRANT
CFLAGS= -O3 -lpthread -lm -lcudart -lftw -D_REENTRANT 
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

pmdgpu:	pmdgpu.c pmdgpu.h MersenneTwister.o ThreadMain.o CommandLineOptions.o Configuration.o Verlet.o CUDAEnergy.o Energy.o Rotation.o
	mpicc -g -O0 -lpthread -lcudart -lftw -D_REENTRANT pmdgpu.c CUDAEnergy.o Verlet.o Configuration.o CommandLineOptions.o ThreadMain.o MersenneTwister.o Energy.o Rotation.o -o pmdgpu $(INCLUDES) $(LIBS) 

clean:
	rm -f *.o 
	rm -f *.linkinfo
	rm -f pmdgpu
	rm -f pmdgpu-emu

install:	pmdgpu
	cp pmdgpu $(SCRATCH)/bin/

install-emu:	pmdgpu-emu
	cp pmdgpu-emu $(SCRATCH)/bin/
