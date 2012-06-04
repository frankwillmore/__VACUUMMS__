#!/usr/bin/make

#  Note that library routines are packaged in source files prefixed with ftw_

HOME=/home1/00479/willmore

LIBRARY_DIR=$(HOME)/usr/lib
INCLUDE_DIR=$(HOME)/usr/include
INSTALL_DIR=$(HOME)/usr/bin


# Not using these... going to let the cuda modules do the work.
#INCLUDES=-I$(INCLUDE_DIR) 
#CUDA_LIB=/opt/apps/cuda/3.1/cuda/lib64/
#LIBS=-L$(LIBRARY_DIR) -L$(CUDA_LIB)

# On building an .so with CUDA
# NOTE:  To specify -fPIC, first specify -Xcompiler

NVCC=nvcc
NVCCFLAGS= -lm -lcudart 
#NVCCFLAGS= -G -g -lm -lcudart 

CC=gcc

# This is just some junk/utils for cuda apps
ftw_cuda_utils.o:	ftw_cuda_utils.cu ftw_cuda_utils.h
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -c ftw_cuda_utils.cu -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm 

# The free volume application
ftw_gfg2fvi.o:	ftw_gfg2fvi.cu ftw_gfg2fvi.h
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -c ftw_gfg2fvi.cu -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm

# Then the library...

libftw_cuda.so:       ftw_cuda_utils.o ftw_gfg2fvi.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -shared -o libftw_cuda.so ftw_cuda_utils.o ftw_gfg2fvi.o -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm 
	rm -f ftw_cuda_utils.o ftw_gfg2fvi.o

# Installation instructions

install-lib:        libftw_cuda.so
	mv libftw_cuda.so $(LIBRARY_DIR)

install-headers:	ftw_gfg2fvi.h ftw_cuda_utils.h
	cp ftw_gfg2fvi.h ftw_cuda_utils.h $(INCLUDE_DIR)

install:	install-headers install-lib 

clean:
	rm -f *.o 
	rm -f *.linkinfo
	rm -f libftw_cuda.so

