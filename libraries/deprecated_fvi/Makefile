#!/usr/bin/make

#  Note that library routines are packaged in source files prefixed with ftw_

HOME=/home1/00479/willmore

LIBRARY_DIR=$(HOME)/usr/lib
INCLUDE_DIR=$(HOME)/usr/include
INSTALL_DIR=$(HOME)/usr/bin

#INCLUDES=-I$(INCLUDE_DIR) 
#CUDA_LIB=/opt/apps/cuda/3.1/cuda/lib64/
#LIBS=-L$(LIBRARY_DIR) -L$(CUDA_LIB)

# Building an .so with CUDA
# NOTE:  To specify -fPIC, first specify -Xcompiler

NVCC=nvcc
NVCCFLAGS= -lm -lcudart 
#NVCCFLAGS= -G -g -lm -lcudart 

DEBUG=-G -g

CC=gcc

ftw_gfg2fvi.o:	ftw_gfg2fvi.cu ftw_gfg2fvi.h
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -c ftw_gfg2fvi.cu -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm $(DEBUG)

libftw_fvi.so:       ftw_gfg2fvi.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -shared -o libftw_fvi.so ftw_gfg2fvi.o -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm $(DEBUG)

example:	example.c ftw_gfg2fvi.o 
	$(CC) -g example.c -o example -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lftw_fvi -lm

install-lib:        libftw_fvi.so
	mv libftw_fvi.so $(LIBRARY_DIR)

install-header:	ftw_gfg2fvi.h ftw_gfgEnergy.h
	cp ftw_gfg2fvi.h $(INCLUDE_DIR)

install:	install-header install-lib 





# Energy lib, non-cuda
ftw_gfgEnergy.o:	ftw_gfgEnergy.c ftw_gfgEnergy.h
	$(CC) -Wall -O2 -fPIC -c ftw_gfgEnergy.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) 

libftw_energy.so:       ftw_gfgEnergy.o ftw_gfgEnergy.o 
	$(CC) -Wall -fPIC -shared -o libftw_energy.so ftw_gfgEnergy.o -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm

install-energy-header:	ftw_gfgEnergy.h
	cp ftw_gfgEnergy.h $(INCLUDE_DIR)

install-energy: libftw_energy.so install-energy-header
	cp libftw_energy.so $(LIBRARY_DIR)

clean:
	rm -f *.o 
	rm -f *.linkinfo
	rm -f libftw_fvi.so
	rm -f libftw_energy.so

