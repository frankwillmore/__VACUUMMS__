#!/usr/bin/make

# Set MAX_NUM_MOLECULES to override default of 64k, e.g. 4M:
# DEFINES:=-DMAX_NUM_MOLECULES=4194304

# Defualt install is to user home

HOME_DIR:=$$HOME

INSTALL_DIR:=$(HOME_DIR)/usr/bin
LIBRARY_DIR:=$(HOME_DIR)/usr/lib
INCLUDE_DIR:=$(HOME_DIR)/usr/include

CC:=icc
CFLAGS:=-Wall -O3 -lftw

ddx:	ddx.h ddx.c io_setup.o 
	$(CC) $(CFLAGS) io_setup.o ddx.c -o ddx -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) 

pddx:	ddx.h pddx.c 
	$(CC) $(CFLAGS) -pthread pddx.c -o pddx -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) $(DEFINES)

# trace_ddx outlines the path followed from insertion to minimum
trace_ddx:	ddx.h trace_ddx.c io_setup.o 
	$(CC) $(CFLAGS) io_setup.o trace_ddx.c -o trace_ddx -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) 

io_setup.o:	io_setup.c io_setup.h
	$(CC) $(CCFLAGS) -c io_setup.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR)

install:	pddx 
	mv pddx $(INSTALL_DIR) 

clean:
	rm -f *.o pddx ddx dddx trace_ddx cull 

