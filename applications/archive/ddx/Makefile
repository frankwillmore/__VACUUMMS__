#!/usr/bin/make

HOME=/home1/00479/willmore

INSTALL_DIR=$(HOME)/usr/bin
LIBRARY_DIR=$(HOME)/usr/lib
INCLUDE_DIR=$(HOME)/usr/include

ddx:	ddx.h ddx.c io_setup.o 
	gcc io_setup.o ddx.c -O3 -o ddx -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm 

trace_ddx:	ddx.h trace_ddx.c io_setup.o 
	gcc io_setup.o trace_ddx.c -O3 -o trace_ddx -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm 

# Discrete version -- requires CUDA and GPU-enabled node
dddx:	dddx.h dddx.c 
	gcc dddx.c -O3 -o dddx -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw_fvi -lftw -lm -lpthread -D_REENTRANT

# io_setup used only for ddx version, not dddx
io_setup.o:	io_setup.c io_setup.h
	gcc -Wall -O3 -c io_setup.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR)

install:	ddx trace_ddx
	mv ddx trace_ddx $(INSTALL_DIR) 

install-cuda:	dddx
	mv dddx $(INSTALL_DIR)

clean:
	rm -f *.o ddx dddx trace_ddx cull 
