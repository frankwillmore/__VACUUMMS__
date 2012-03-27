#!/bin/make

# Shared-object library code for libftw.so

CC=gcc
CFLAGS=-O2 -Wall -fPIC

HOME=/home1/00479/willmore/
INCLUDE_DIR=$(HOME)/usr/include
LIBRARY_DIR=$(HOME)/usr/lib
INSTALL_DIR=$(HOME)/usr/lib

# The header files here contain definitions of ftw-kernel codes
# And the c files are their implementation
# The master copies of the header files are kept HERE, and installed with 
# 'make install-headers'
#
# The ftw_types.h is maintained here, as well.
# All library routines here are prefixed with ftw_

ftw_rng.o:	ftw_rng.c ftw_rng.h 
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c ftw_rng.c

ftw_rng2.o:	ftw_rng2.c ftw_rng2.h 
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c ftw_rng2.c

ftw_std.o:	ftw_std.c ftw_std.h
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c ftw_std.c

ftw_param.o:	ftw_param.c ftw_param.h
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c ftw_param.c

ftw_config_parser.o:	ftw_config_parser.c ftw_config_parser.h
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c ftw_config_parser.c

ftw_cav_parser.o:	ftw_cav_parser.c ftw_cav_parser.h
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c ftw_cav_parser.c

# ftw_command_line_parser is deprecated in favor of ftw_param (as of March 24, 2011...)
# Originally written as command_line_parser, I'm leaving it in the build tree until I clean up usage.
# Any instances won't be able to find the original name, but it's the same API
# ftw_command_line_parser.o:	ftw_command_line_parser.c ftw_command_line_parser.h
#	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c ftw_command_line_parser.c

libftw.so: ftw_types.h ftw_rng.o ftw_std.o ftw_param.o ftw_rng2.o ftw_cav_parser.o ftw_config_parser.o
	$(CC) $(CFLAGS) -shared -o libftw.so *.o 

ftw_types:	ftw_types.h

install:	install-headers install-lib

install-lib:    libftw.so 
	mv libftw.so $(INSTALL_DIR)

clean:
	rm -f *.o libftw.so

test:	
	$(CC) test.c -I$(HOME)/usr/include -L/home/00479/willmore/usr/lib -I. -lftw -lm

install-headers: ftw_types.h ftw_command_line_parser.h ftw_rng.h ftw_rng2.h ftw_std.h ftw_param.h ftw_config_parser.h ftw_cav_parser.h 
	cp ftw_types.h ftw_command_line_parser.h ftw_rng.h ftw_rng2.h ftw_std.h ftw_param.h ftw_config_parser.h ftw_cav_parser.h $(INCLUDE_DIR)

